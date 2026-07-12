"""
Reproduction of "Mapping Networks" (Sen & Mukherjee, CVPR 2026)
Section 3.1.1 — Image Classification on MNIST / FashionMNIST.  CUDA version.

================================ USAGE ========================================
  # 1) Baseline CNN2 (LeNet-inspired, ~108K params)
  python mapping_networks_cuda.py --stage baseline --arch cnn2

  # 2) Mapping Network, single-latent-vector training (Ours*, paper Sec 2.4.1)
  python mapping_networks_cuda.py --stage map --arch cnn2 --d 1024 --epochs 100
  python mapping_networks_cuda.py --stage map --arch cnn2 --d 2048 --epochs 100

  # FashionMNIST:  add  --dataset fmnist
  # Larger target (AlexNet-inspired CNN1, ~538K params):  --arch cnn1
  # Resume automatically from checkpoint (delete ckpt_*.pt to restart).

Paper reference results (Table 1, MNIST):
  CNN2 baseline  108,618 params -> 98.69%
  Ours* d=1024     1,024 params -> 97.88%
  Ours* d=2048     2,048 params -> 98.66%
================================================================================

Components implemented (paper Sec. 2.2 / 2.3):
 * Trainable latent vector z in R^d                                (Sec 2.2.1)
 * Fixed orthogonally-initialized mapping weights W0 (d x P),
   additively modulated by z:  w_ij <- w_ij + alpha * z_i          (Eq. 20)
   =>  theta_hat = sigma(z @ W0 + alpha*||z||^2 + b)               (Eq. 21)
   (modulation folded analytically into the matvec; W_eff is never
    materialized, so memory stays at one d x P buffer)
 * Fixed bias b = flattened standard init theta_0 of the target, so
   theta_hat(z~0) starts on a well-scaled point of the weight manifold.
 * Deterministic reshape into per-layer tensors                    (Eq. 22)
 * Target network used for feed-forward only; gradients flow
   exclusively through the mapping (z, and loss coefficients)      (Eq. 23)
 * Mapping Loss  L = L_task + l1*L_stab + l2*L_smooth + l3*L_align (Eq. 25)
     L_stab   (Eq. 27): E||f_{theta(z+eps)}(x) - f_{theta(z)}(x)||^2
     L_smooth (Eq. 28): Jacobian-norm penalty, finite-difference estimate
                        ||theta(z+eps)-theta(z)||^2 / sigma^2 (reuses the
                        stability perturbation)
     L_align  (Eq. 29): 1 - cos(z, W_m), W_m = row-mean of modulated weights
   Coefficients are trainable via softplus (paper says "trainable"; note that
   pure loss minimization drives them toward 0 — observed in our runs too).
"""
import argparse, gzip, json, os, struct, time, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

# ------------------------------------------------------------------ data
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "http://yann.lecun.com/exdb/mnist/",
]
FMNIST_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

def download(dataset, root):
    d = os.path.join(root, dataset); os.makedirs(d, exist_ok=True)
    for f in FILES:
        path = os.path.join(d, f)
        if os.path.exists(path): continue
        sources = [FMNIST_BASE + f] if dataset == "fmnist" else [m + f for m in MIRRORS]
        for url in sources:
            try:
                print(f"downloading {url}"); urllib.request.urlretrieve(url, path); break
            except Exception as e:
                print(f"  failed ({e}), trying next mirror")
        if not os.path.exists(path):
            raise RuntimeError(f"could not download {f}")
    return d

def load_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        dims = struct.unpack(">" + "I" * (magic & 0xFF), f.read(4 * (magic & 0xFF)))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)

def get_data(dataset, root, device):
    d = download(dataset, root)
    mean, std = ((0.1307, 0.3081) if dataset == "mnist" else (0.2860, 0.3530))
    def prep(img, lab):
        X = torch.tensor(load_idx(img).astype(np.float32) / 255.).unsqueeze(1)
        X = (X - mean) / std
        return X.to(device), torch.tensor(load_idx(lab).astype(np.int64)).to(device)
    Xtr, Ytr = prep(f"{d}/{FILES[0]}", f"{d}/{FILES[1]}")
    Xte, Yte = prep(f"{d}/{FILES[2]}", f"{d}/{FILES[3]}")
    return Xtr, Ytr, Xte, Yte   # MNIST fits in GPU memory (~200MB) — keep resident

def batches(X, Y, bs, shuffle=True):
    idx = torch.randperm(len(X), device=X.device) if shuffle else torch.arange(len(X), device=X.device)
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]; yield X[j], Y[j]

# --------------------------------------------------------------- targets
class CNN2(nn.Module):
    """LeNet-inspired, EXACT architecture from supplemental Table 2:
    Conv1 1->16 3x3 (no pad, 28->26), pool -> 13
    Conv2 16->32 3x3 (no pad, 13->11), pool -> 5
    FC1 800->128, FC2 128->10.  Total = 108,618 (matches paper exactly)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)

class CNN1(nn.Module):
    """AlexNet-inspired, EXACT architecture from supplemental Table 1:
    Conv1 1->32 3x3 pad1, pool -> 14; Conv2 32->64 3x3 pad1, pool -> 7;
    Conv3 64->128 3x3 pad1; Conv4 128->128 3x3 pad1, pool -> 3;
    FC1 1152->256, FC2 256->10.  Total = 537,994 (matches paper exactly)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)

ARCHS = {"cnn1": CNN1, "cnn2": CNN2}

@torch.no_grad()
def accuracy(fwd, X, Y, bs=2000):
    return 100. * sum((fwd(xb).argmax(1) == yb).sum().item()
                      for xb, yb in batches(X, Y, bs, False)) / len(X)

# --------------------------------------------------------- mapping network
class MappingNetwork(nn.Module):
    def __init__(self, target, d, alpha=1e-4, device="cuda", scale="uniform"):
        super().__init__()
        self.target = target.to(device)
        for p in self.target.parameters(): p.requires_grad_(False)
        self.shapes = [(n, p.shape, p.numel()) for n, p in target.named_parameters()]
        self.P, self.d, self.alpha = sum(n for *_, n in self.shapes), d, alpha

        # trainable latent vector (Sec 2.2.1)
        self.z = nn.Parameter(torch.randn(d, device=device) * 0.1)

        # fixed mapping weights with exactly orthonormal rows (Sec 2.2.2),
        # via W0 = (G G^T)^{-1/2} G  — much cheaper than QR for d << P
        G = torch.randn(d, self.P, device=device)
        ev, evec = torch.linalg.eigh(G @ G.T)
        W0 = evec @ torch.diag(ev.clamp_min(1e-8).rsqrt()) @ evec.T @ G
        self.register_buffer("W0", W0)
        # perturbation scaling, two modes for controlled comparison:
        #  uniform: col_scale = 1 (original config; reached 97.79% @ d=2048,
        #           lr=1e-2 — note fc1 holds ~95% of P, so uniform scale is
        #           effectively already matched to the dominant layer)
        #  layer:   scale each column by its layer's init std, equalizing the
        #           RELATIVE per-layer update rate (~lr per step under Adam).
        #           Requires lr >= 1e-2 to avoid underfitting.
        if scale == "layer":
            col_scale = torch.cat([p.detach().std().clamp_min(1e-3).expand(p.numel())
                                   for p in target.parameters()]).to(device)
            col_scale = col_scale * (self.P / d) ** 0.5
        else:
            col_scale = torch.ones(self.P, device=device)
        self.register_buffer("col_scale", col_scale)
        # fixed bias = target's standard init (theta_0)
        self.register_buffer("b", torch.cat([p.detach().flatten()
                                             for p in target.parameters()]))
        self.register_buffer("W0_rowmean", W0.mean(dim=1))
        # trainable Mapping-Loss coefficients (Eq. 25), softplus(-2) ~ 0.127
        self.rho = nn.Parameter(torch.full((3,), -2.0, device=device))

    def generate(self, z):                                   # Eq. 20-21
        pert = (z @ self.W0 + self.alpha * (z * z).sum()) * self.col_scale
        return torch.tanh(pert + self.b)

    def forward_with(self, theta_hat, x):                    # Eq. 22-23
        out, p = {}, 0
        for name, shape, n in self.shapes:
            out[name] = theta_hat[p:p + n].view(shape); p += n
        return functional_call(self.target, out, (x,))

    def lambdas(self): return F.softplus(self.rho)

# ---------------------------------------------------------------- training
def train_baseline(args, Xtr, Ytr, Xte, Yte):
    torch.manual_seed(args.seed)
    net = ARCHS[args.arch]().to(args.device)
    n = sum(p.numel() for p in net.parameters())
    print(f"[{args.arch}] trainable params = {n:,}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    log = []
    for ep in range(1, args.base_epochs + 1):
        net.train(); t0 = time.time()
        for xb, yb in batches(Xtr, Ytr, args.bs):
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
        net.eval()
        tr, te = accuracy(net, Xtr, Ytr), accuracy(net, Xte, Yte)
        log.append(dict(epoch=ep, train_acc=tr, test_acc=te))
        print(f"[{args.arch}] ep{ep:3d}  train {tr:.2f}%  test {te:.2f}%  "
              f"({time.time()-t0:.1f}s)", flush=True)
    return dict(params=n, log=log)

def train_mapping(args, Xtr, Ytr, Xte, Yte):
    torch.manual_seed(args.seed)
    mn = MappingNetwork(ARCHS[args.arch]().to(args.device), d=args.d,
                        alpha=args.alpha, device=args.device, scale=args.scale)
    trainable = [mn.z, mn.rho]
    n_tr = sum(p.numel() for p in trainable)
    print(f"[Ours* {args.arch} d={args.d}] target P={mn.P:,}  "
          f"trainable={n_tr:,}  ({mn.P/args.d:.0f}x reduction)")
    opt = torch.optim.Adam(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 1e-2)
    ckpt = f"ckpt_{args.arch}_{args.dataset}_d{args.d}_{args.scale}.pt"
    log, start = [], 1
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=args.device)
        mn.z.data.copy_(st["z"]); mn.rho.data.copy_(st["rho"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        log, start = st["log"], st["epoch"] + 1
        print(f"resumed from epoch {st['epoch']}")

    best = max((l["test_acc"] for l in log), default=0.)
    for ep in range(start, args.epochs + 1):
        t0 = time.time()
        for step, (xb, yb) in enumerate(batches(Xtr, Ytr, args.bs)):
            opt.zero_grad()
            theta = mn.generate(mn.z)
            logits = mn.forward_with(theta, xb)
            loss = F.cross_entropy(logits, yb)               # Eq. 26  L_task
            if step % args.reg_every == 0:                   # regularizers
                eps = torch.randn_like(mn.z) * args.sigma
                theta_p = mn.generate(mn.z + eps)
                logits_p = mn.forward_with(theta_p, xb)
                L_stab = (logits_p - logits).pow(2).mean()             # Eq. 27
                L_smooth = (theta_p - theta).pow(2).sum() / (args.sigma**2 * mn.P)  # Eq. 28
                Wm = mn.W0_rowmean + mn.alpha * mn.z                   # Eq. 29
                L_align = 1 - F.cosine_similarity(mn.z, Wm, dim=0)
                if args.fix_lam > 0:      # fixed coefficients (anti-collapse)
                    lam = torch.full((3,), args.fix_lam, device=args.device)
                else:                     # trainable per paper (tends to ~0)
                    lam = mn.lambdas()
                loss = loss + lam[0]*L_stab + lam[1]*L_smooth + lam[2]*L_align
            loss.backward(); opt.step()
        sched.step()
        with torch.no_grad():
            theta = mn.generate(mn.z)
            fwd = lambda x: mn.forward_with(theta, x)
            tr, te = accuracy(fwd, Xtr, Ytr), accuracy(fwd, Xte, Yte)
        best = max(best, te)
        lam = [round(v, 4) for v in mn.lambdas().tolist()]
        log.append(dict(epoch=ep, train_acc=tr, test_acc=te, lambdas=lam,
                        lr=sched.get_last_lr()[0]))
        print(f"[Ours* d={args.d}] ep{ep:3d}  train {tr:.2f}%  test {te:.2f}%  "
              f"best {best:.2f}%  lam={lam}  lr={sched.get_last_lr()[0]:.2e}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        torch.save(dict(z=mn.z.data, rho=mn.rho.data, opt=opt.state_dict(),
                        sched=sched.state_dict(), log=log, epoch=ep), ckpt)
    return dict(trainable=n_tr, target_P=mn.P, best_test=best, log=log)

# -------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baseline", "map", "all"], default="all")
    ap.add_argument("--arch", choices=["cnn1", "cnn2"], default="cnn2")
    ap.add_argument("--dataset", choices=["mnist", "fmnist"], default="mnist")
    ap.add_argument("--d", type=int, default=1024, help="latent dimension")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--base-epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2,
                    help="initial lr for z (cosine-decayed)")
    ap.add_argument("--alpha", type=float, default=1e-4, help="modulation scale (Eq.20)")
    ap.add_argument("--sigma", type=float, default=1e-2, help="latent perturbation std")
    ap.add_argument("--reg-every", type=int, default=4,
                    help="compute Lstab/Lsmooth/Lalign every k steps (1=every step)")
    ap.add_argument("--fix-lam", type=float, default=0.,
                    help=">0: use fixed loss coefficients instead of trainable "
                         "(trainable ones collapse to ~0); try 0.05-0.1")
    ap.add_argument("--scale", choices=["uniform", "layer"], default="uniform",
                    help="latent perturbation scaling (see MappingNetwork)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}"
          + (f" ({torch.cuda.get_device_name(0)})" if args.device == "cuda" else ""))

    Xtr, Ytr, Xte, Yte = get_data(args.dataset, args.data_root, args.device)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    if args.stage in ("all", "baseline"):
        results[f"{args.arch}_{args.dataset}_baseline"] = \
            train_baseline(args, Xtr, Ytr, Xte, Yte)
        json.dump(results, open(args.out, "w"), indent=2)
    if args.stage in ("all", "map"):
        results[f"ours_{args.arch}_{args.dataset}_d{args.d}_{args.scale}"] = \
            train_mapping(args, Xtr, Ytr, Xte, Yte)
        json.dump(results, open(args.out, "w"), indent=2)
    print("DONE — results saved to", args.out)

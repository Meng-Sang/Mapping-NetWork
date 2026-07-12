# Reproducing Mapping Networks — Sec. 3.1.1 Image Classification (MNIST)

This project reproduces the image-classification experiment (Sec. 3.1.1) of the CVPR 2026 paper **"Mapping Networks"** (Sen & Mukherjee). Instead of directly training the ~108K parameters of a target CNN, only a low-dimensional latent vector z (1024 / 2048 dims) is trained; fixed, orthogonally initialized mapping weights generate all target-network parameters from z. We compare accuracy and overfitting against the directly trained baseline.

## 1. What the Code Does

The single-file script `mapping_networks_cuda.py` implements the full training pipeline of the paper:

**Baseline stage** (`--stage baseline`): directly trains the target CNN (Adam, lr = 1e-3) with the exact architectures from the paper's supplemental Tables 1/2, serving as the reference.

**Mapping Network stage** (`--stage map`, the paper's Single Latent Vector Training, SLVT):
- A trainable latent vector z ∈ R^d (Sec. 2.2.1); z and the 3 loss coefficients are the **only** trainable parameters (2,051 in total for d = 2048);
- Fixed mapping weights W₀ ∈ R^{d×P} with exactly orthonormal rows via Löwdin symmetric orthogonalization (Sec. 2.2.2); the mapping bias b is set to the target network's standard initialization θ₀;
- Additive modulation w_ij ← w_ij + αz_i (Eq. 20), folded analytically into the matrix–vector product so the modulated matrix is never materialized;
- The generated θ̂ = σ(W·z + b) is deterministically partitioned and reshaped into the target network's layers (Eq. 21–23); the target network performs forward passes only, and gradients flow exclusively back to z (Eq. 24);
- Mapping Loss (Eq. 25–29): task cross-entropy + stability + smoothness (finite-difference estimate of the Jacobian norm) + alignment, with softplus-parameterized trainable coefficients;
- Per-epoch checkpointing with automatic resume, cosine learning-rate schedule, and per-epoch train/test accuracy logged to `results.json`.

Additional ablation switches: `--scale {uniform,layer}` (per-layer perturbation scaling), `--mod {scalar,quad}` (literal Eq. 20 scalar modulation vs. the general Theorem-2 form M(z) = Bz with quadratic latent features), `--act {tanh,identity}`, and `--fix-lam` (fixed regularization coefficients, preventing the trainable λ's from collapsing to 0).

## 2. Dataset, Architectures, and Hyperparameters

**Dataset**: MNIST (60,000 train / 10,000 test, 28×28 grayscale), standard normalization (0.1307, 0.3081); FashionMNIST supported via `--dataset fmnist`. The script downloads the data automatically from multiple mirrors — no torchvision dependency.

**Target architectures** (exactly matching the paper's supplemental material, parameter counts verified):

| Network | Structure | # Params |
|---|---|---|
| CNN2 (LeNet-style) | Conv 1→16 3×3 (no pad) → pool → Conv 16→32 3×3 (no pad) → pool → FC 800→128 → FC 128→10 | **108,618** ✓ |
| CNN1 (AlexNet-style) | Conv 32/64/128/128 (3×3, pad 1) + 3×pool → FC 1152→256 → FC 256→10 | **537,994** ✓ |

**Hyperparameters** (the paper does not disclose any training details for the mapping network; the following are this reproduction's choices):

| Item | Value |
|---|---|
| Latent dimension d | 1024 / 2048 (as in the paper) |
| z initialization | N(0, 0.1²) |
| Mapping weights W₀ | Gaussian init, then (GGᵀ)^(−1/2)G row orthogonalization, frozen |
| Mapping bias b | Target network's standard initialization θ₀, frozen |
| Modulation scale α | 1e-4 |
| Activation σ | tanh |
| Optimizer | Adam, lr = 1e-2, cosine decay (eta_min = lr×0.1) |
| Batch size | 128 |
| Epochs | 200 (baseline: 10–15) |
| Perturbation std (L_stab/L_smooth) | 1e-2, regularizers computed every 4 steps |
| λ initialization | softplus(−2) ≈ 0.127, trainable |

## 3. Reproduction Results

### Accuracy (MNIST test set)

| Method | Trainable Params | Paper | This Repro | Gap |
|---|---|---|---|---|
| CNN2 baseline | 108,618 | 98.69% | **99.02%** (ep 10) | +0.33% |
| Ours* d=2048 (exact arch, scalar mod, 200 ep) | 2,051 | 98.66% | **97.17%** | −1.49% |
| Ours* d=2048 (early reconstructed arch, 100 ep)* | 2,051 | 98.66% | 97.79% | −0.87% |

\* The early CNN2 was an approximate architecture reconstructed to match the parameter count (108,374 params), shown for reference.

Key observations:

- **The overfitting-reduction claim is successfully reproduced**: the Mapping Network's train/test gap is essentially zero (97.18% / 96.98%, with test occasionally above train), whereas the baseline shows a ~0.6% gap (99.65% / 99.02%). This confirms the paper's core claim: a ~53× compression of trainable degrees of freedom structurally removes the capacity to memorize training noise.
- **A ~1.5% absolute-accuracy gap remains unclosed**, and it manifests as underfitting (train accuracy only 97.2%). Attribution: (a) the paper discloses none of the mapping network's training hyperparameters (σ, α, lr, epochs); (b) the trainable λ's inevitably collapse to 0 under pure loss minimization (measured: lam = [0, 0, 0]), so the 2–3% regularization gain claimed in the paper is not reproducible as described; (c) the modulation as literally written in Eq. 20 collapses algebraically to the scalar α‖z‖² (rank-0), which cannot possibly carry the 2–4% modulation gain claimed in Table 7 — suggesting the actual implementation follows Theorem 2's general per-weight modulation M(z) = Bz (containing quadratic terms in z; see `--mod quad`). This equation–implementation inconsistency is the most likely source of the reproduction gap. Our 97.17–97.79% is consistent with the known results of random-subspace training (Li et al. 2018, the paper's ref [16]).

### Speed (single GPU)

| Stage | Time per Epoch | Notes |
|---|---|---|
| CNN2 baseline | ~0.7 s | Direct training of 108K params |
| Ours* d=2048 | ~4.6–4.7 s | Each step includes θ̂ generation (d×P matvec) + target forward |

Note: the Mapping Network's **per-epoch training cost is ~6.6× the baseline's**, and W₀ must reside in GPU memory (d = 2048 × P ≈ 108K → ~0.9 GB fp32; doubled with `--mod quad`). What this method reduces is trainable degrees of freedom and the information content of the model (the trained result is fully reconstructable from z + a random seed, ~8 KB) — not training time or runtime memory. This contradicts the "reducing training time" phrasing in the paper's abstract; the paper's own conclusion acknowledges SLVT's memory overhead.

### Commands

```bash
python mapping_networks_cuda.py --stage baseline --arch cnn2
python mapping_networks_cuda.py --stage map --d 2048 --epochs 200            # main experiment
python mapping_networks_cuda.py --stage map --d 2048 --epochs 200 --mod quad # quadratic-modulation hypothesis test
python mapping_networks_cuda.py --stage map --d 1024 --epochs 200            # matches the paper's 97.88% row
```

Dependencies: `torch`, `numpy`. Switching `--mod / --scale / --act / --d` creates separate checkpoints; when resuming under the same name, hyperparameters must stay consistent.

## 4. Citation

```bibtex
@InProceedings{Sen_2026_CVPR,
    author    = {Sen, Lord and Mukherjee, Shyamapada},
    title     = {Mapping Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
                 and Pattern Recognition (CVPR)},
    year      = {2026},
    pages     = {36215--36223}
}
```

arXiv preprint: [arXiv:2602.19134](https://arxiv.org/abs/2602.19134). Random-subspace reference method: Li et al., *Measuring the Intrinsic Dimension of Objective Landscapes*, arXiv:1804.08838 (the paper's ref [16]).

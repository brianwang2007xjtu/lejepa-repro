# Phase 1 — Audit of *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*

Paper: arXiv:2511.08544v3 (14 Nov 2025), Balestriero & LeCun.
Official code: <https://github.com/galilai-group/lejepa> (redirects to `rbalestr-lab/lejepa`), commit `c293d29`.

This document is the evidence base for every implementation decision in this repository.
Each claim is tagged with its source: **[P]** paper, **[C]** official code, **[M]** `MINIMAL.md`.

---

## 0. Environment detection report

Measured on 2026-10-30.

| Item | Value |
|---|---|
| Access | rented AutoDL container (SSH) |
| Host OS | Ubuntu 22.04.5 LTS, kernel 5.15.0-94-generic |
| CPU | 2× Intel Xeon Platinum 8352S, 128 logical cores |
| RAM | 503 GB |
| GPU | **1× NVIDIA GeForce RTX 4090 D, 24 576 MiB** (sm_89) |
| Driver / CUDA runtime | 595.71.05 / CUDA 13.2 |
| `nvcc` | not installed (irrelevant: PyTorch ships its own CUDA runtime) |
| Python | 3.12.3 (miniconda at `/root/miniconda3`) |
| PyTorch | 2.8.0+cu128, `torch.cuda.is_available() == True` |
| torchvision | 0.23.0+cu128 |
| Installed for this project | timm, einops, pytest, scipy, tensorboard 2.20.0, PyYAML 6.0.2 |
| **Writable disk** | `/root/autodl-tmp` = **50 GB**; `/` (overlay) = 30 GB |
| Read-only shared data | `/root/autodl-pub` (4 TB AutoFS): contains `ImageNet/ILSVRC2012/*.tar`, `ImageNet100/imagenet100.zip` (14.4 GB) |

**Critical capacity facts that constrain the experiment plan**

* `ILSVRC2012_img_train.tar` is **147.9 GB** and `ILSVRC2012_img_val.tar` is 6.7 GB.
  Extracting ImageNet-1K needs ≈150 GB; the only writable volume is 50 GB.
  → **ImageNet-1K cannot be materialised on this machine.**
* Even if it could, the paper's ImageNet-1K recipe is ViT-L/14 @ 238 px, batch 512, 100 epochs [P §6.4, launch scripts].
  A single 24 GB RTX 4090 cannot hold batch 512 × 8 views × 238 px ViT-L in bf16 (≈60 GB of activations).
* `ImageNet100/imagenet100.zip` (14.4 GB, 100 classes, ≈126 k train / 5 k val) **does fit** and is a standard SSL benchmark.

These facts drive the three-tier plan in §7 and are restated in `REPRODUCTION_REPORT.md`.

---

## 1. What the official repository actually contains

`find` over the cloned repo (`.git` excluded):

```
MINIMAL.md                     <- 130-line end-to-end reference (ViT-S/8 + Imagenette)
README.md                      <- hyper-parameters, augmentation, linear-probe protocol
lejepa/univariate/             <- epps_pulley.py, cramer_von_mises.py, watson.py,
                                  anderson_darling.py, jarque_bera.py, shapiro_wilk.py,
                                  moments.py, entropy.py, likelihood.py
lejepa/multivariate/           <- slicing.py (SlicingUnivariateTest), bhep, bhep_m, comb, hz, hv
scripts/launch_inet10.py       <- generates a sweep over 50 timm backbones
scripts/launch_*_ablation.md   <- raw hydra command lines for the paper's ablations
figures/*.py                   <- figure reproduction scripts
tests/*.py                     <- tests for the statistical tests
```

**There is no training script.** Every `launch_*.md` invokes `python scripts/je.py --config-dir scripts/configs ...`,
but neither `scripts/je.py` nor `scripts/configs/` exists in the repository, and the code depends on the
external `stable_pretraining` package (README: *"we provide a pretraining skeleton script using `stable_pretraining`"*).
Therefore the **only executable, complete definition of the training loop published by the authors is
`MINIMAL.md` plus Algorithms 1 and 2 in the paper.** Everything in this repository is reconstructed from
those three sources; no `lejepa` package code is imported at any point.

Consequence: the exact `je.py` used for the ImageNet-1K numbers is not public. Where `MINIMAL.md` and the
paper's algorithms differ (§5), we follow the paper's algorithms and record the difference.

---

## 2. Method, as stated in the paper

### 2.1 Design principle (Section 3)

Among distributions with a fixed scalar covariance constraint, the isotropic Gaussian $\mathcal N(0, I_K)$
uniquely minimises the integrated square bias of both $k$-NN and kernel downstream predictors
(Theorem 1), and minimises the variance of a ridge linear probe (Lemmas 1–2).
Hence the training target for the embedding distribution is $\mathcal N(0, I_K)$.

### 2.2 SIGReg (Definition 2)

$$\mathrm{SIGReg}_T(\mathcal A, \{f_\theta(x_n)\}_{n=1}^N)\;\triangleq\;\frac{1}{|\mathcal A|}\sum_{a\in\mathcal A}T\big(\{a^\top f_\theta(x_n)\}_{n=1}^N\big)$$

with $\mathcal A=\{a_1,\dots,a_M\}\subset S^{K-1}$ unit-norm random directions, and $T$ the Epps–Pulley statistic.
The paper replaces the $\max_{a\in\mathcal A}$ of the global test (Eq. 4) with a **mean** to avoid sparse
gradients [P §4.2].

### 2.3 Epps–Pulley statistic (Section 4.2.3)

$$\mathrm{EP}=N\int_{-\infty}^{\infty}\big|\hat\varphi_X(t)-\varphi(t)\big|^2 w(t)\,dt,\qquad
\hat\varphi_X(t)=\frac1N\sum_{j=1}^N e^{itX_j},\qquad \varphi(t)=e^{-t^2/2}$$

with Gaussian weight $w$. The paper's prose says $w(t)=e^{-t^2/\sigma^2}$ with $\sigma=1$;
Algorithm 1 and the code use $w(t)=e^{-t^2/2}$ (see D3).

Because $\int |\hat\varphi_X-\varphi|^2 w = \mathrm{MMD}^2$ for the kernel whose Fourier transform is $w$,
taking $w(t)=e^{-t^2/2}$ makes the exact integral equal to the **Gaussian-kernel MMD² with bandwidth 1**:

$$\mathrm{EP} \;=\; N\cdot \mathbb E\,k(X,X') + N\cdot\mathbb E\,k(Y,Y') - 2N\cdot\mathbb E\,k(X,Y),
\qquad k(u,v)=e^{-(u-v)^2/2},\; Y\sim\mathcal N(0,1).$$

This identity is used in `tests/test_epps_pulley.py` as an independent, closed-form check of the quadrature.

### 2.4 Prediction / invariance loss (Section 5.1, Eq. 5–7)

With $V_g$ global views (indices $1..V_g$) and $V_l$ local views, $V=V_g+V_l$:

$$\mathcal L_{\text{pred}}=\frac{1}{V_g}\sum_{v=1}^{V_g}\frac1V\sum_{v'=1}^{V}\|z_{n,v}-z_{n,v'}\|_2^2
=\frac1V\sum_{v'=1}^{V}\Big\|\underbrace{\frac{1}{V_g}\sum_{v=1}^{V_g}z_{n,v}}_{\mu_n}-z_{n,v'}\Big\|_2^2$$

The equality is the standard bias/variance decomposition (proved in [P §B.6]): the mean over all
view-pairs collapses to "distance of every view to the mean of the **global** views".

### 2.5 Total objective (Eq. LeJEPA)

$$\mathcal L_{\text{LeJEPA}}=\frac{\lambda}{V}\sum_{v=1}^{V}\mathrm{SIGReg}\big(\{z_{n,v}\}_{n=1}^B\big)
+(1-\lambda)\,\mathcal L_{\text{pred}}$$

$\lambda$ is the **only** hyper-parameter. No stop-gradient, no teacher, no predictor, no EMA, no prototypes,
no schedulers on anything but the LR [P §1, §5.1, Alg. 2].

### 2.6 Why directional tests suffice (Section 4.3)

Theorem 5 bounds the expected discrepancy over random directions by $C(K,\alpha)|\mathcal A|^{-2\alpha/(K-1)}$
for Sobolev-regular $p_\theta\in H^\alpha$; directions are **resampled at every step**, so the cumulative
number of directions grows linearly with training (§"SGD beats the curse of dimensionality",
Figure 7: resampling beats a fixed direction set).

---

## 3. Reference implementation, algorithm by algorithm

### Algorithm 1 — SIGReg + Epps–Pulley [P, p. 10]

```python
def SIGReg(x, global_step, num_slices=256):
    dev = dict(device=x.device)
    g = torch.Generator(**dev); g.manual_seed(global_step)     # synced across ranks
    proj_shape = (x.size(1), num_slices)                       # (K, M)
    A = torch.randn(proj_shape, generator=g, **dev)
    A /= A.norm(p=2, dim=0)                                    # unit-norm COLUMNS
    t = torch.linspace(-5, 5, 17, **dev)
    exp_f = torch.exp(-0.5 * t ** 2)
    x_t = (x @ A).unsqueeze(2) * t                             # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = x.size(0) * world_size
    return torch.trapz(err, t, dim=1) * N
```

Facts extracted:

| Item | Value | Source |
|---|---|---|
| direction distribution | $\mathcal N(0,I_K)$ i.i.d. per entry | [P Alg.1], [C `slicing.py`] |
| normalisation | each **column** divided by its L2 norm → $a_m\in S^{K-1}$ | [P Alg.1], [C], [M] |
| shape of $A$ | $(K, M)$ i.e. `(proj_dim, num_slices)` | [C] |
| projection | $Y = XA\in\mathbb R^{N\times M}$ | [C] |
| resampling | **every forward pass**, seeded by `global_step` | [P Alg.1], [C] |
| sampling RNG | dedicated `torch.Generator`, seeded with the step counter, **not** the global RNG | [C `slicing.py`] |
| integration | trapezoid, $M$ stats averaged (`reduction="mean"`) | [C], [M] |
| N-scaling | statistic is multiplied by $N$ (and by world size under DDP) | [P Alg.1], [C] |
| DDP | ECF averaged across ranks **before** the statistic is formed | [P Alg.1], [C] |
| grad through $A$ | none; $A$ is built under `torch.no_grad()` | [C] |

### Algorithm 2 — LeJEPA [P, p. 11]

```python
def LeJEPA(global_views, all_views, lambd):
    g_emb = forward(torch.cat(global_views))      # (V_g * B, K)
    a_emb = forward(torch.cat(all_views))         # (V   * B, K)
    centers = g_emb.view(-1, bs, K).mean(0)       # (B, K)  mean over GLOBAL views
    a_emb   = a_emb.view(-1, bs, K)               # (V, B, K)
    sim     = (centers - a_emb).square().mean()   # scalar: mean over V, B, K
    sigreg  = mean(SIGReg(emb, global_step) for emb in a_emb)   # mean over V
    return (1 - lambd) * sim + lambd * sigreg
```

Two deviations from Eq. (5)–(7) live in this code and **must be preserved**, because $\lambda=0.05$ was
tuned against *this* scale:

1. `centers` uses only the **global** views (consistent with Eq. 7; `MINIMAL.md` uses all views because
   there all views are interchangeable).
2. `.square().mean()` divides by $K$ as well as by $V,B$; Eq. (5) has no $1/K$. This keeps the
   prediction loss $O(1)$ when `projector_dim` changes from 64 to 2048, which is what makes the
   $(\lambda,\text{proj-dim})$ ablation in Table 1d meaningful.

### Projector / backbone [M]

```python
self.backbone = timm.create_model("vit_small_patch8_224", pretrained=False,
                                  num_classes=512, drop_path_rate=0.1, img_size=128)
self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
def forward(x):                       # x: (N, V, C, H, W)
    N, V = x.shape[:2]
    emb = self.backbone(x.flatten(0, 1))          # (N*V, 512)
    return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)   # (V, N, D)
```

* `num_classes=512` is used as an **embedding width**, i.e. the timm head is a `Linear(384, 512)` on the CLS token.
* Tensor layout throughout the algorithm is $(V,B,D)$ — view-major — so `mean(0)` is the view mean.
* `torchvision.ops.MLP` applies `norm_layer` **and** GELU only to `hidden_channels[:-1]`; the final
  layer is a bare `Linear` (+ Dropout).  So the projector is
  `Linear(512,2048)-BN-GELU-Linear(2048,2048)-BN-GELU-Linear(2048,D)` — a **linear output**.
  Verified against the installed `torchvision` source, not from memory (see D8).

### Data pipeline, ImageNet-1K [README "Data augmentation and views"]

Two global views + six local views, identical photometric transforms, different crop scales:

| | Global (×2) | Local (×6) |
|---|---|---|
| RandomResizedCrop | 224×224, scale (0.3, 1.0) | 98×98, scale (0.05, 0.3) |
| RandomHorizontalFlip | p=0.5 | p=0.5 |
| ColorJitter (p=0.8) | 0.4 / 0.4 / 0.2 / 0.1 | same |
| RandomGrayscale | p=0.2 | p=0.2 |
| GaussianBlur | p=0.5 | p=0.5 |
| RandomSolarize | p=0.2, thr=128 | p=0.2, thr=128 |
| Normalize | ImageNet mean/std | same |

`MINIMAL.md` for Imagenette instead uses a **single** augmentation stack for all $V$ views
(`RandomResizedCrop(128, scale=(0.08,1.0))`, ColorJitter(0.8,0.8,0.8,0.2) p=0.8, RandomGrayscale p=0.2,
GaussianBlur(7,(0.1,2.0)), Solarize(128) p=0.2, flip), i.e. $V_g=V$.

### Optimisation [README + launch configs]

| | ImageNet-1K (ViT-L/14) | Imagenette (MINIMAL) |
|---|---|---|
| optimiser | AdamW | AdamW |
| lr | 5e-4 | 2e-3 |
| weight decay | 5e-2 (ViT), 5e-4 (ResNet) | 5e-2 (encoder), 1e-7 (probe) |
| schedule | linear warmup + cosine, final lr = lr/1000 | warmup 1 epoch (start_factor 1e-2) + cosine to 1e-3 |
| precision | bf16 | bf16 (`autocast`) |
| batch | 512 | 256 |
| epochs | 100 | 800 |
| λ | 0.05 | 0.02 |
| views | $V_g{=}2$, $V_l{=}6$ ($V{=}8$) | $V{=}4$ |
| res | 238 / 98 (launch) vs 224 / 96 (paper text) | 128 / 128 |
| embedding/proj dim | 512 / 512 | 512 / 16 |
| drop_path | 0.1 | 0.1 |
| slices $M$ | 1024 (recommended), 1000 in launch scripts | 256 |
| integration | 17 pts, $[-5,5]$ (recommended) / $[-3,3]$ (code default) | 17 pts, $[0,3]$ |

Linear probe [README]: concatenate the CLS token of the **last two layers**, LayerNorm, AdamW with
wd = 1e-6, same warmup+cosine schedule as pretraining.

### Reported numbers to compare against

| Setting | Metric | Value | Source |
|---|---|---|---|
| ViT-L/14, IN-1K, 100 ep, $V_g{=}2,V_l{=}6$, λ=.05 | frozen linear probe | 74.24 % | [P Table 1b] |
| same, batch 512 | frozen linear probe | 74.72 % | [P Table 1c] |
| same, 1024 slices | frozen linear probe | 74.81 % | [P Table 1d] |
| ViT-L/14 IN-1K 100ep | online linear probe | 77.1 % | [P §6.4] |
| ConvNeXtV2-H IN-1K 100ep | online linear probe | 78.5 % | [P §6.4] |
| ViT-H/14 IN-1K | frozen linear probe | 79 % | [P abstract] |
| ViT-S/8 + Imagenette, 800 ep, ViT-S/8 | online probe | **90.7 %** | [M] |
| ViT-S/8, IN-100, 400 ep, 3-layer proj, no SWA/predictor | frozen linear probe | 81.07 % | [P Table 4] |
| ResNet-50, IN-100, 400 ep, λ=.05, 8 views | frozen linear probe | 84.32 % | [P Table 7] |
| 50 timm models <20 M, IN-10, frozen linear probe | top-1 | 91.5–95 % | [P Fig. 9] |

---

## 4. Paper ↔ official code correspondence

| Component | Paper | Code | Verdict |
|---|---|---|---|
| Direction sampling | $\mathcal N(0,I_K)$, per-step resample | `torch.randn((K,M), generator=g)` seeded by `global_step` | **match** |
| Direction normalisation | $a\in S^{K-1}$ | `A /= A.norm(p=2, dim=0)` | **match** |
| Projection | $a^\top f_\theta(x_n)$ | `x @ A` | **match** |
| ECF | $\frac1N\sum e^{itX_j}$ | `(1j*x_t).exp().mean(0)` | **match** |
| Target CF | $e^{-t^2/2}$ | `torch.exp(-0.5*t**2)` | **match** |
| Weight | $e^{-t^2/\sigma^2}$, σ=1 (prose) / $e^{-t^2/2}$ (Alg. 1) | `* exp_f` = $e^{-t^2/2}$ | **paper prose ≠ paper algorithm; code follows Alg. 1** |
| Quadrature | trapezoid, 17 knots | `torch.trapz` over $[-5,5]$ (Alg. 1) / symmetric doubling on $[0,3]$ (repo) | **same rule, different domain** |
| Aggregation | $\frac1{|\mathcal A|}\sum_a T$ | `stats.mean()` | **match** |
| N scaling | $N\int$ | `* N * world_size` | **match** |
| DDP | ECF all-reduced | `all_reduce(cos_mean)`, `all_reduce(sin_mean)` | **match** |
| Invariance | Eq. 5–7 | `(centers - a_emb).square().mean()` | **match** (with the $1/K$ of Alg. 2) |
| Total loss | $\lambda\,\mathrm{SIGReg}+(1-\lambda)\mathcal L_{pred}$ | idem | **match** |
| Views | $V_g=2,V_l=6$ | dataset yields $V$ crops; `n_views=8` | **match** |

---

## 5. Discrepancies found (with evidence)

**D1 — The ImageNet training script is not public.**
All ablation launchers call `scripts/je.py --config-dir scripts/configs`, neither of which is in the repo;
`setup.py`/`pyproject.toml` only package the statistics. The reproducible definition of the training loop
is `MINIMAL.md` + Alg. 1–2. *Impact:* the ImageNet numbers cannot be bit-reproduced, only re-implemented.

**D2 — The paper's main ViT runs use an SWA teacher for the centre, although the method is presented as teacher-free.**
[P §6.1]: *"While a teacher-student configuration does provide a small performance boost for ViT models …
it is not necessary to prevent collapse. In our setup, we apply SWA on the encoder producing $\mu$ in Equation (6)."*
Every `launch_*.md` passes `++teacher_student=true`. Table 4 quantifies the gap on IN-100 / ViT-S/8 /
3-layer: 81.07 % without, 83.63 % with SWA (+2.6 pts). *Impact:* a faithful ImageNet reproduction would
include SWA; our default follows Alg. 2 (no SWA) and we expose `--swa` as an ablation. This is the single
largest methodological caveat in the reproduction.

**D3 — Epps–Pulley weight function.**
[P §4.2.3] states $w(t)=e^{-t^2/\sigma^2}$ with $\sigma=1$, i.e. $e^{-t^2}$; [P Alg. 1] and [C] use $e^{-t^2/2}$.
*Resolution:* follow Alg. 1 / code ($e^{-t^2/2}$); it is also the only choice that makes EP equal to the
Gaussian-kernel MMD² of bandwidth 1 (used as our correctness test).

**D4 — Integration domain.**
[P Alg. 1] $t\in[-5,5]$; [P Table 1a / §6.1] recommend $[-5,5]$; [C `EppsPulley.__init__`] default `t_max=3`;
[M] uses `linspace(0,3,17)`. Table 1a shows the difference is ≈0.5 pt (74.68 vs 74.80 at 2048 slices).
*Resolution:* `t_max=5.0` default (paper recommendation), configurable, ablated over {1,3,5}.

**D5 — 1/K in the invariance loss.**
Eq. (5) is a squared $\ell_2$ norm; Alg. 2 uses `.mean()` over $V,B,K$. *Resolution:* follow Alg. 2.
Justification: it makes the λ balance independent of `projector_dim`, which is required for Table 1d to be
interpretable at proj-dim 64…1024.

**D6 — Centre over global vs. all views.**
Alg. 2 and Eq. (7) use the global views only; [M] averages over all $V$ views, which is equivalent there
because all views share one augmentation stack ($V_g=V$). *Resolution:* implement `center = mean(global views)`
and set $V_g=V$ for the Imagenette configuration.

**D7 — Resolution.**
[P "Experiment Details 1"]: 224 global / 96 local. Launch configs: `resolution=238`, `local_resolution=98`.
96 is not a multiple of 14 while 98 is, and 238 = 17·14. *Resolution:* resolution must be a multiple of the
patch size; default to the paper text (224/96) for patch-16/8 models and to 238/98 for patch-14 models.

**D8 — Projector unspecified (resolved by reading the torchvision source).**
The paper only ablates projector depth (1/2/3 layers, Table 4) and never gives the norm/activation.
[M] uses `MLP(·,[2048,2048,D], norm_layer=BatchNorm1d)`.  Reading
`torchvision/ops/feature_pyramid_network.py` shows `MLP` attaches `norm_layer` and the activation
only for `hidden_channels[:-1]`; the final layer is `Linear` + `Dropout` with **no norm and no
activation**.  The reference projector is therefore
`Linear-BN-GELU -> Linear-BN-GELU -> Linear`, i.e. a **linear output head**.
*Resolution:* defaults `hidden_norm="bn", final_norm="none", final_act="none"`.
This matters: a GELU on the output layer is bounded below and rectifying, so the embedding
distribution could never be Gaussian and SIGReg would plateau at a positive floor.  A first
training attempt that used `final_act="gelu"` was discarded for exactly that reason
(see §8, failed runs).

**D9 — Number of slices.**
[P Alg. 1] default 256; [P §6.1 / Table 1a] recommend 1024; [M] 256; README quick-start 1024.
*Resolution:* 1024 for main runs, ablated over {128,256,512,1024,4096}.

**D10 — λ default.**
0.05 recommended for ViT/IN-1K; [M] uses 0.02 for Imagenette. Both are paper/code-supported, used accordingly.

---

## 6. What we deliberately do **not** add

Because the paper's central claim is "heuristics-free", the implementation must not contain, and the tests
must assert the absence of: stop-gradient, EMA teacher, predictor/student-teacher asymmetry, prototypes,
whitening layers, register tokens, gradient clipping, LR schedulers on anything but the LR
(weight decay is constant), and any extra regulariser (e.g. variance/covariance terms).
`tests/test_invariance.py::test_no_stop_gradient` asserts that $\partial\mathcal L/\partial z_{n,v}$ is
non-zero for **every** view, which fails immediately if a stop-gradient is reintroduced.

---

## 7. Reproduction plan and feasibility

Three tiers, kept explicitly separate in all reporting.

**Tier 1 — Algorithmic verification (fully faithful).**
Unit + numerical tests of EP, SIGReg, the invariance loss, gradients, bf16/fp32, batch-size stability,
synthetic SIGReg-on-free-embeddings experiment reproducing [P Fig. 6] (recover an isotropic Gaussian from
an adversarial "X" 2-D subspace inside $\mathbb R^{512}$ with $M=16$ directions).

**Tier 2 — Reference-matched run (fully faithful to a published number).**
Imagenette + ViT-S/8 @128, $V=4$, λ=0.02, proj 16, lr 2e-3, bs 256, 800 epochs [M].
Citable target: **90.7 % online probe**. Single 4090, ≈2 h per the authors.

**Tier 3 — Scale-limited approximate reproduction (explicitly not a full reproduction).**
ImageNet-**100** + ViT-S/16 @224 (2 global 224 + 6 local 96), λ=0.05, bs≤256, 100–200 epochs,
then frozen-backbone linear probe and kNN. Reference point: [P Table 4] ViT-S/8 IN-100 400 ep = 81.07 %
(no SWA) — different patch size and shorter schedule, so only order-of-magnitude agreement is expected.

**Not attempted, with reasons:** ImageNet-1K (needs 150 GB disk, ≥8×A100 for the published schedule);
ViT-L/14 & ViT-H/14 (do not fit batch 512×8 views in 24 GB); 60-architecture sweep.

**Ablations (Tier 2/3 scale):** λ ∈ {0.01,0.02,0.05,0.1,0.2}; slices ∈ {128,256,512,1024};
integration {t_max∈{1,3,5}} × {17 points}; proj-dim ∈ {16,64,256,512}; views ∈ {2,4,8} and
$(V_g,V_l)$ ∈ {(2,6),(4,4),(1,7)}; batch ∈ {128,256}; EP quadrature knots ∈ {5,17,41}; SWA on/off.
Each ablation changes exactly one factor.

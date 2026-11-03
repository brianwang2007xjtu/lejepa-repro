# LeJEPA — Independent Reproduction

A from-scratch, independent reproduction of

> **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics**
> Randall Balestriero, Yann LeCun — [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)

**No `lejepa` package code is imported anywhere in this repository.** The Epps–Pulley
statistic, SIGReg, the prediction/invariance loss, the multi-view pipeline and the training
loop are all re-derived from the paper's equations and validated against closed-form
mathematics. The official repository is used only as *documentation* to audit the paper
(see [`docs/AUDIT.md`](docs/AUDIT.md)).

---

## The method in four equations

| | |
|---|---|
| **Epps–Pulley** goodness-of-fit vs $\mathcal N(0,1)$ | $\mathrm{EP}=N\int_{-\infty}^{\infty}\lvert\hat\varphi_X(t)-\varphi(t)\rvert^2 w(t)\,dt$, $\;w(t)=e^{-t^2/2}$ |
| **SIGReg** (Definition 2) | $\frac1{\lvert\mathcal A\rvert}\sum_{a\in\mathcal A}\mathrm{EP}(\{a^\top z_n\})$, $\;a\in S^{K-1}$ resampled every step |
| **Prediction / invariance** (Eq. 6–7) | $\frac1V\sum_{v'}\lVert \mu_n-z_{n,v'}\rVert^2$, $\;\mu_n=\frac1{V_g}\sum_{v\le V_g}z_{n,v}$ |
| **LeJEPA** | $\lambda\,\mathrm{SIGReg}+(1-\lambda)\,\mathcal L_{\text{pred}}$ |

No stop-gradient, no EMA teacher, no predictor, no prototypes, no gradient clipping.
`tests/test_invariance.py::test_no_stop_gradient_every_view_receives_gradient` asserts this.

## Layout

```
configs/     debug | in100_* | imagenette | imagenet_vit (paper scale)
datasets/    transforms.py (multi-crop), imagenet.py (view datasets + loaders)
models/      backbone.py (timm ViT + 2-layer-CLS readout), projector.py (MLP + view reshaping)
losses/      epps_pulley.py, sigreg.py, invariance.py
engine/      trainer.py, optimizer.py (warmup+cosine), distributed.py (DDP), swa.py
evaluation/  linear_probe.py (frozen LP + weighted kNN)
scripts/     train.py, evaluate.py, benchmark.py, prepare_in100.py
tests/       test_epps_pulley | test_sigreg | test_invariance | test_gradients | test_optimizer
docs/        AUDIT.md (phase-1 audit) | ENVIRONMENT.md | DATASETS.md | HANDOFF.md
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # torch, torchvision, timm, numpy, PyYAML, tensorboard
pip install pytest scipy                 # tests only
```

Verified with Python 3.12.3, torch 2.8.0+cu128, torchvision 0.23.0, timm 1.0.30,
1× RTX 4090 D (24 GB). See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Data

```bash
# ImageNet-100: archive has train only; build val from the ImageNet-1K val tar
python scripts/prepare_in100.py \
  --raw  /path/to/in100_raw/imagenet100 \
  --out  /root/autodl-tmp/data/imagenet100 \
  --val-tar /path/to/ILSVRC2012_img_val.tar \
  --devkit  /path/to/ILSVRC2012_devkit_t12.tar.gz
```

Details and the ImageNet-1K situation: [`docs/DATASETS.md`](docs/DATASETS.md).

## Train

```bash
# Experiment A — pipeline debug on synthetic data (no dataset needed)
python scripts/train.py --config configs/debug.yaml --dry-run

# Experiment B — ImageNet-100, ViT-S/16 @128, V=4 (MINIMAL-scale recipe)
python scripts/train.py --config configs/in100_vits16_128.yaml

# Experiment C — ImageNet-100, ViT-S/16, 2x224 + 6x96 views, paper recipe (λ=0.05)
python scripts/train.py --config configs/imagenet100_vits16.yaml

# resume / override / DDP
python scripts/train.py --config configs/in100_vits16_128.yaml --resume auto
python scripts/train.py --config configs/in100_vits16_128.yaml --set lambd=0.05 sigreg.num_slices=256
torchrun --nproc_per_node=4 scripts/train.py --config configs/imagenet_vit.yaml
```

Every run writes `<run_dir>/config.json`, `metrics.jsonl`, `metrics.csv`, `tb/` and
`checkpoints/{last,final,epoch_XXXX}.pt`. A checkpoint restores the model, optimiser,
scheduler, the SIGReg direction-generator step counter **and** the RNG states, so a resumed
run continues the exact same direction sequence.

## Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/in100_vits16_128/checkpoints/final.pt \
  --data-root /root/autodl-tmp/data/imagenet100 --data-name imagenet100 \
  --num-classes 100 --resolution 128 --feature-source last2cls \
  --probe-epochs 100 --batch-size 256
```

`--feature-source last2cls` is the paper's protocol (concatenated CLS of the last two blocks
+ LayerNorm, AdamW wd=1e-6); `embedding` probes the backbone head output instead. Weighted
k-NN (k=20, cosine, T=0.07) is reported alongside.

## Test

```bash
python -m pytest tests/ -v          # 77 passed, 1 skipped on a 4090
python -m pytest tests/ -q -m "not slow"
```

The tests check *values*, not just shapes:

* `test_constant_sample_matches_quadrature_reference` — EP against adaptive quadrature (≤2e-4 rel. error)
* `test_mmd_identity_random_samples` — EP/N = √(2π)·MMD²_gaussian-kernel, closed form
* `test_standard_normal_expectation_is_exact_constant` — E[EP] = √(2π)(1−1/√3) = 1.0594 under H₀
* `test_collapse_is_a_saddle_not_an_attractor` — EP has zero gradient at the collapsed point, which is a local **maximum**
* `test_pairwise_equals_eq7_plus_global_variance` — the corrected Eq. (5)→(7) identity (the paper's Appendix B.6 is wrong by a variance term)
* `test_recovers_isotropic_gaussian_from_adversarial_x` — the paper's Figure 6 experiment
* `test_peak_lr_equals_configured_lr` — regression test for a scheduler bug that silently divided the LR by 100

## Reproduction status (measured)

| run | setting | frozen linear probe (IN-100) | k-NN |
|---|---|---|---|
| **Exp C** | 8 views (2x224 + 6x96), ViT-S/16, λ=0.05, 50 ep | **69.02 %** | 60.24 % |
| **Exp B** | 4 views @128, ViT-S/16, λ=0.02, 40 ep | **49.52 %** | 42.78 % |

Ablations (12 ep each, ±2–3 pt noise — see [`docs/ABLATIONS.md`](docs/ABLATIONS.md)):
baseline λ=.02 **30.9** · λ=.05 30.4 · M=64 28.6 · λ=.005 27.2 · **λ=.2 4.9** · **V=2 3.2**.

SIGReg converged to **1.54** against a theoretical floor of **1.0594**; the projector-space
effective rank rose 8.3 → 15.8 (of 16) with **no collapse** in any correctly-configured run.

Pretrained weights (8 runs, ~109 MB each, slimmed to weights + config) are released
separately — see [`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md) and
[`docs/checkpoint_manifest.json`](docs/checkpoint_manifest.json) for SHA-256 checksums.

See [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md) for the full results, the comparison
table with the paper, the failure analyses, and an explicit three-tier classification
(full / approximate / algorithmic-only). **This is an approximate reproduction: ImageNet-1K
and ViT-L/14 were not run.**

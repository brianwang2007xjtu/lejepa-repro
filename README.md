# LeJEPA: Independent Reproduction

**English** | [简体中文](README.zh-CN.md)

An independent implementation of *LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics* (Balestriero & LeCun, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)).

No code from the `lejepa` package is imported. The Epps–Pulley statistic, SIGReg, the random
slicing scheme, the prediction loss and the training loop were re-derived from the paper's
equations and checked against closed-form results.

The method is reproduced. The paper's ImageNet-1K results are not — see [Scope](#scope).

---

## Scope

| | |
|---|---|
| **Reproduced** | objective and training recipe; the statistical machinery, verified against analytic references; end-to-end pretraining on ImageNet-100 |
| **Not reproduced** | ImageNet-1K, ViT-L/14 and ViT-H/14, the paper's reported accuracies |

ImageNet-1K was not run because the training archive is 147.9 GB and the only writable volume
on the reproduction host was 50 GB. ViT-L/14 at batch 512 × 8 views of 238 px needs roughly
60 GB of activations and does not fit on a 24–32 GB GPU. Details and the three-tier
classification are in [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md).

## Results

ImageNet-100, ViT-S/16, bf16, AdamW, linear warm-up with cosine decay. Evaluation is a frozen
linear probe on the concatenated CLS token of the last two blocks, following the paper's README.

| Run | Views | λ | Epochs | Linear probe | k-NN |
|---|---|---|---|---|---|
| **Exp C** | 8 (2×224 + 6×96) | 0.05 | 50 | **69.02 %** | 60.24 % |
| **Exp B** | 4 (128 px) | 0.02 | 40 | **49.52 %** | 42.78 % |

SIGReg on Exp B converged from 9.89 to **1.54**, against a theoretical floor of 1.0594 for a
perfect isotropic Gaussian. The projector-space effective rank rose from 8.3 to 15.8 out of 16,
and no representation collapse occurred in any correctly configured run.

Ablations, 12 epochs each, with roughly ±2–3 points of run-to-run noise (see
[docs/ABLATIONS.md](docs/ABLATIONS.md)):

| Variant | Linear probe |
|---|---|
| baseline, λ = 0.02 | 30.9 % |
| λ = 0.05 | 30.4 % |
| M = 64 slices | 28.6 % |
| λ = 0.005 | 27.2 % |
| λ = 0.2 | 4.9 % |
| V = 2 views | 3.2 % |

The last two are large enough to be conclusive. λ = 0.2 and V = 2 both leave the embedding
distribution correctly Gaussian while destroying view invariance, which is the failure mode the
λ trade-off is supposed to control. Differences among the first four are inside the noise and
should not be read as a ranking.

Pretrained weights for all eight runs are published as release assets; see
[docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md).

## Method

| | |
|---|---|
| Epps–Pulley statistic | $N\int \lvert\hat\varphi_X(t)-\varphi(t)\rvert^2 w(t)\,dt$, with $w(t)=e^{-t^2/2}$ |
| SIGReg | $\frac{1}{\lvert\mathcal A\rvert}\sum_{a\in\mathcal A}\mathrm{EP}(\{a^\top z_n\})$, directions $a\in S^{K-1}$ redrawn every step |
| Prediction loss | $\frac{1}{V}\sum_{v'}\lVert\mu_n-z_{n,v'}\rVert^2$, with $\mu_n=\frac{1}{V_g}\sum_{v\le V_g}z_{n,v}$ |
| Total | $\lambda\,\mathrm{SIGReg}+(1-\lambda)\,\mathcal L_{\text{pred}}$ |

There is no stop-gradient, no EMA teacher, no predictor and no gradient clipping.
`tests/test_invariance.py::test_no_stop_gradient_every_view_receives_gradient` asserts that
every view receives a non-zero gradient.

## Layout

```
losses/       epps_pulley.py  sigreg.py  invariance.py
models/       backbone.py (timm ViT, dynamic image size)  projector.py (MLP, view reshaping)
datasets/     transforms.py (multi-crop)  imagenet.py (view datasets, loaders)
engine/       trainer.py  optimizer.py  distributed.py
evaluation/   linear_probe.py (frozen probe + weighted k-NN)
scripts/      train.py  evaluate.py  prepare_in100.py  ablation_statistics.py
configs/      debug  imagenette  in100_vits16_128  imagenet100_vits16  imagenet_vit
tests/        78 tests across five files
docs/         AUDIT.md  ABLATIONS.md  DATASETS.md  ENVIRONMENT.md  RELEASE_NOTES.md
```

## Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install pytest scipy          # tests only
```

Verified with Python 3.12.3, torch 2.8.0+cu128, torchvision 0.23.0, timm 1.0.30 on an
RTX 4080 SUPER. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Data

The public ImageNet-100 archive contains the 126,689 training images only. The validation split
is built from the ImageNet-1K validation set:

```bash
python scripts/prepare_in100.py \
  --raw     /path/to/in100_raw/imagenet100 \
  --out     /path/to/imagenet100 \
  --val-tar /path/to/ILSVRC2012_img_val.tar \
  --devkit  /path/to/ILSVRC2012_devkit_t12.tar.gz
```

This yields 100 classes with 126,689 training and 5,000 validation images. Class indices are
resolved through the devkit, so no hard-coded wnid ordering is assumed.

## Training

```bash
# pipeline check on synthetic data: forward, backward, step, save, resume
python scripts/train.py --config configs/debug.yaml --dry-run

# Exp B: 4 views at 128 px
python scripts/train.py --config configs/in100_vits16_128.yaml

# Exp C: 8 views, the paper's recipe
python scripts/train.py --config configs/imagenet100_vits16.yaml --set epochs=50 batch_size=192

# resume, override a single leaf, or run on 4 GPUs
python scripts/train.py --config configs/in100_vits16_128.yaml --resume auto
python scripts/train.py --config configs/in100_vits16_128.yaml --set lambd=0.05
torchrun --nproc_per_node=4 scripts/train.py --config configs/imagenet_vit.yaml
```

Each run writes `config.json`, `metrics.jsonl`, `metrics.csv`, a TensorBoard directory and
periodic checkpoints. A checkpoint restores the model, optimiser, scheduler, the SIGReg
direction-generator step counter and the RNG states, so a resumed run continues the same
direction sequence.

`configs/imagenet_vit.yaml` records the exact ImageNet-1K configuration from the paper. It is
kept for completeness and requires hardware this project did not have.

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/in100_vits16_128/checkpoints/final.pt \
  --data-root /path/to/imagenet100 --data-name imagenet100 \
  --num-classes 100 --resolution 128 --feature-source last2cls \
  --probe-epochs 100 --batch-size 256
```

`last2cls` follows the paper's protocol (concatenated CLS of the last two blocks, LayerNorm
inside the probe, AdamW with weight decay 1e-6). `embedding` probes the backbone head output
instead. Weighted k-NN (k = 20, cosine, T = 0.07) is reported alongside.

## Tests

```bash
python -m pytest tests/ -v
```

78 tests. They check numerical values rather than shapes:

| Test | What it establishes |
|---|---|
| `test_constant_sample_matches_quadrature_reference` | the statistic matches adaptive quadrature to 2e-4 relative error |
| `test_mmd_identity_random_samples` | $\mathrm{EP}/N=\sqrt{2\pi}\,\mathrm{MMD}^2$ for the Gaussian kernel, from a closed form |
| `test_standard_normal_expectation_is_exact_constant` | $E[\mathrm{EP}]=\sqrt{2\pi}(1-1/\sqrt3)=1.0594$ under the null |
| `test_collapse_is_a_saddle_not_an_attractor` | the collapsed point has zero gradient and is a local maximum |
| `test_paper_eq5_equals_eq7_plus_global_variance` | the corrected Eq. (5) → (7) identity |
| `test_recovers_isotropic_gaussian_from_adversarial_x` | the paper's Figure 6 experiment |
| `test_peak_lr_equals_configured_lr` | a scheduler regression that otherwise divides the learning rate by 100 |

## Notes on the paper

Two issues were found while implementing and are documented in [docs/AUDIT.md](docs/AUDIT.md)
together with eight further places where the paper and the official code disagree:

1. **Eq. (5) is not equal to Eq. (6)/(7).** Appendix B.6 asserts
   $\frac{1}{V_g}\sum_v\lVert z_v\rVert^2=\lVert\bar z\rVert^2$, which holds only when the
   global views are already identical. The correct identity is
   $\text{Eq.}(5)=\text{Eq.}(7)+\frac{1}{V_g}\sum_{v\le V_g}\lVert z_v-\bar z\rVert^2$.
   Algorithm 2 implements Eq. (7), so this project trains with Eq. (7).
2. **The fully collapsed point is a saddle, not an attractor.** The Epps–Pulley statistic has
   exactly zero gradient there, but every perturbation lowers it, so collapse is unstable.
   This is the mechanism by which SIGReg prevents collapse.

One limitation of the method was also observed: a Rademacher ±1 embedding scores 97.3 at D = 2
but 1.02 at D = 256, indistinguishable from a correct embedding. A 1-D projection of a product
measure becomes Gaussian by the central limit theorem, so at high dimension every directional
marginal looks correct even though the joint law is not. SIGReg constrains marginals, not the
joint distribution.

## Limitations

- No ImageNet-1K, no ViT-L/14 or ViT-H/14, so no number in this repository is comparable to the
  paper's reported accuracies.
- Exp B and Exp C differ from the paper in dataset, patch size, batch size and epoch budget;
  the 19.5-point gap between them cannot be attributed to the number of views alone.
- The ablation sweep runs 12 epochs per variant and therefore supports only order-of-magnitude
  conclusions.
- Collapse diagnostics monitor the projector output, while evaluation reads backbone features.
  A backbone-space diagnostic script exists (`scripts/diagnose_checkpoints.py`) but was not
  completed, because the CPU quota available at the time was 0.5 cores.

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{balestriero2025lejepa,
  title  = {LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author = {Balestriero, Randall and LeCun, Yann},
  year   = {2025},
  eprint = {2511.08544},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

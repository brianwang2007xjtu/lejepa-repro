# Ablations

Two separate families, because they answer different questions and cost wildly different amounts.

| Family | Needs a GPU? | What it establishes |
|---|---|---|
| **Statistical** (`scripts/ablation_statistics.py`) | no | properties of the estimator itself: discrimination, variance vs $M$, quadrature error, projection normalisation |
| **Training** (`scripts/run_ablations.sh`) | yes | does the *end-to-end* result depend on $\lambda$, $M$, $V$? |

The statistical family is run first and for free; it also tells us which training ablations are
not worth paying for.

---

## 1. Statistical ablations — completed

Command: `python scripts/ablation_statistics.py --out-dir results/ablations --skip-slow`
Output: `results/ablations/statistical_ablations.json`

### 1.1 Discrimination (N=2048, D=64, M=1024, t_max=5, 17 knots)

| embedding | SIGReg |
|---|---|
| isotropic Gaussian | **0.993** |
| anisotropic (one dim ×4) | **23.75** |
| rank-8 correlated (dimensional collapse) | **58.76** |
| constant | **2201.6** |
| half-Gaussian (`|N(0,1)|`) | **971.9** |
| Rademacher ±1, D=64 | 1.258 |

Three orders of magnitude of separation between a healthy embedding and a collapsed one.

### 1.2 Number of slices $M$ (embeddings held fixed, 30 draws)

| M | mean | std | rel. std |
|---|---|---|---|
| 16 | 0.980 | 0.225 | **23.0 %** |
| 64 | 0.999 | 0.101 | 10.2 % |
| 256 | 1.018 | 0.058 | 5.7 % |
| 1024 | 1.025 | 0.029 | 2.8 % |
| 4096 | 1.018 | 0.016 | **1.5 %** |

The **mean is unbiased** at every $M$ — more slices only reduce variance. This is the statistical
counterpart of the paper's Figure 7 claim that resampling makes small $|A|$ sufficient.

### 1.3 Quadrature error vs the closed-form MMD identity

Reference: $\mathrm{EP}/N=\sqrt{2\pi}\,\mathrm{MMD}^2$ with kernel $e^{-(u-v)^2/2}$.

| t_max | P=5 | P=9 | **P=17** | P=33 | P=65 |
|---|---|---|---|---|---|
| 1 | 96.8 % | 97.0 % | **97.1 %** | 97.1 % | 97.1 % |
| 3 | 9.48 % | 0.84 % | **0.64 %** | 0.59 % | 0.58 % |
| **5** | 20.8 % | 2.24 % | **0.007 %** | 0.007 % | 0.007 % |
| 8 | 95.7 % | 15.9 % | **0.10 %** | 0.007 % | 0.007 % |

The paper's recommended $(t_{\max}=5, P=17)$ is the sweet spot: **0.007 %** error. The official
class's default of $t_{\max}=3$ costs 0.64 %, and $t_{\max}=1$ is catastrophically wrong (97 %)
regardless of how many knots are used — the truncation, not the resolution, is the problem.

### 1.4 Projection normalisation

| directions | SIGReg |
|---|---|
| unit-norm columns (paper) | **0.990** |
| raw $\mathcal N(0,I)$ columns | 1104.7 |

Unnormalised directions scale each projection by $\sim\sqrt{D}$, so the statistic no longer tests
$\mathcal N(0,1)$. Normalisation is not a detail.

### 1.5 A limitation the paper does not discuss: detection power collapses with dimension

Rademacher $\pm1$ embeddings (a strongly non-Gaussian product measure), sweeping $D$:

| D | SIGReg | kurtosis of one projection |
|---|---|---|
| 2 | **97.27** | 1.84 |
| 4 | 27.36 | 1.79 |
| 8 | 6.80 | 2.42 |
| 16 | 2.05 | 2.81 |
| 64 | 1.28 | 2.89 |
| 256 | **1.02** | 2.83 |

A 1-D projection of a $\pm1$ product is a weighted sum of independent $\pm1$ signs, which is
asymptotically Gaussian by the central limit theorem. So at large $D$ **every directional marginal
looks Gaussian even though the joint law is not**, and SIGReg reports ~1.0 (indistinguishable from
a correct embedding). This is the practical face of the paper's Theorem 5: the number of
directions needed scales with $K$ and with the Sobolev smoothness of $p_\theta$. It matters because
it means SIGReg constrains *marginals*, not the full joint law — a direction for future work.

---

## 2. Training ablations — scope B

Six runs, each **15 epochs** on ImageNet-100 with the cheap Experiment-B recipe
(ViT-S/16 @128, 4 views, bs 256, 494 steps/epoch, ~2.7 min/epoch), followed by a 60-epoch
frozen linear probe. ≈43 min per run, ≈4.4 h total.

| run | factor changed | baseline |
|---|---|---|
| `abl_base_lamb0.02` | — (baseline) | λ=0.02 |
| `abl_lamb0.005` | λ | 0.02 |
| `abl_lamb0.05` | λ | 0.02 |
| `abl_lamb0.2` | λ | 0.02 |
| `abl_slices64` | SIGReg slices M | 256 |
| `abl_views2` | number of views V | 4 |

One factor changes per run; everything else is pinned.

### Dropped, and why

| dropped run | would have measured | why it was cut |
|---|---|---|
| `t_max=5` | integration domain | **already answered exactly** in §1.3 (0.007 % vs 0.64 % quadrature error); paper Table 1a shows a 0.1 pt accuracy difference |
| `proj_dim=512` | projector width | paper Table 1d already ranks widths; the ~1.5 pt gap is inside this sweep's ±2–3 pt noise |
| `slices=1024` | more slices | paper reports +0.6 pt; §1.2 already quantifies the variance reduction exactly |

### The noise caveat (stated up front, not buried)

At 15 epochs the frozen linear probe has **run-to-run noise of roughly ±2–3 points**. This sweep
can therefore only expose *large* effects — for example λ=0.005 collapsing or λ=0.2 destroying the
invariance term. It must **not** be used to rank close variants, and no conclusion of the form
"λ=0.05 is 0.8 pt better than λ=0.02" is supportable from it. A clean λ sweep at the paper's
budget (400 epochs on ImageNet-100) would cost ~27 GPU-hours per point and was out of scope.

---

## Reproducing

```bash
# statistical (CPU, ~2 min)
python scripts/ablation_statistics.py --out-dir results/ablations

# training (GPU; ~4.4 h for the 6 runs)
EPOCHS=15 bash scripts/run_ablations.sh
# results accumulate in results/ablations/sweep.jsonl + one JSON per run
```

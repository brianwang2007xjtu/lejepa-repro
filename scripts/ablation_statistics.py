"""Statistical ablations of the SIGReg / Epps–Pulley machinery — no training required.

These experiments test the paper's *statistical* claims directly, on fixed embeddings, so
they need no GPU and no dataset and give exact, repeatable answers:

1. ``num_slices`` (paper Table 1a / §6.1, and Figure 7)
   * estimator mean and standard deviation of SIGReg vs M,
   * the **resample-vs-fixed** comparison of Figure 7: the expected directional statistic at
     the end of training, for directions resampled every step versus kept fixed.
2. ``t_max`` / ``n_points`` (paper Table 1a)
   * the quadrature error of the Epps–Pulley statistic against its closed form,
     ``EP/N = sqrt(2*pi) * MMD^2`` with the Gaussian kernel ``exp(-(u-v)^2/2)``.
3. Projection normalisation
   * unit-norm columns (the paper's choice) versus unnormalised Gaussian columns.
4. Discrimination
   * SIGReg on isotropic / anisotropic / rank-deficient / constant / two-point embeddings.

Usage::

    python scripts/ablation_statistics.py --out-dir results/ablations
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import torch

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from losses.epps_pulley import EppsPulley  # noqa: E402
from losses.sigreg import SIGReg, sample_projection_directions  # noqa: E402

SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- 1. num_slices
def ablation_num_slices(z: torch.Tensor, reps: int = 30) -> List[Dict]:
    """Estimator mean/std vs M, holding the embeddings fixed."""
    out = []
    for m in (16, 32, 64, 128, 256, 512, 1024, 4096):
        sig = SIGReg(num_slices=m, t_max=5.0, n_points=17, seed=0)
        vals = torch.tensor([sig(z).item() for _ in range(reps)])
        rec = {
            "num_slices": m,
            "mean": vals.mean().item(),
            "std": vals.std().item(),
            "std_over_mean": (vals.std() / vals.mean()).item(),
        }
        out.append(rec)
        print(f"  M={m:5d}  mean={rec['mean']:.4f}  std={rec['std']:.4f}  "
              f"rel_std={rec['std_over_mean']:.4f}")
    return out


def ablation_resample_vs_fixed(d: int = 64, n: int = 512, steps: int = 400) -> Dict:
    """Paper Figure 7: expected directional statistic vs M, resampled vs fixed.

    We optimise the *embeddings* (not a network) towards SIGReg for `steps` gradient steps,
    once with the directions redrawn at every step and once with a single frozen direction
    set, and record the final statistic.  Matching the paper's finding, resampling should
    reach a lower statistic for the same M because the cumulative number of directions grows
    with training time.
    """
    torch.manual_seed(0)
    z0 = torch.randn(n, d)

    results = {}
    for mode in ("resample", "fixed"):
        per_m = []
        for m in (16, 64, 256):
            torch.manual_seed(1)
            z = torch.nn.Parameter(z0.clone())
            if mode == "fixed":
                a = sample_projection_directions(d, m, z.device)
                sig = SIGReg(num_slices=m, t_max=5.0, n_points=17, seed=None)
                # freeze the directions by monkey-patching the sampler
                frozen = [a]

                def frozen_sampler(*_a, **_k):
                    return frozen[0]

                sig._frozen = frozen_sampler  # marker only; real freeze below
                orig_forward = sig.forward

                def forward_fixed(zz, _sig=sig, _a=a):
                    proj = zz @ _a
                    return _sig.univariate_test(proj).mean()

                sig.forward = forward_fixed  # type: ignore[assignment]
            else:
                sig = SIGReg(num_slices=m, t_max=5.0, n_points=17, seed=0)

            opt = torch.optim.Adam([z], lr=1e-2)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = sig(z)
                loss.backward()
                opt.step()
            with torch.no_grad():
                # evaluate every run with the SAME fresh, large direction set so the numbers
                # are comparable (the training-time statistic is not a fair yardstick)
                ev = SIGReg(num_slices=4096, t_max=5.0, n_points=17, seed=123)
                per_m.append(ev(z).item())
        results[mode] = per_m
        print(f"  {mode:9s}: " + ", ".join(
            f"M={m}:{v:.4f}" for m, v in zip((16, 64, 256), per_m)))
    return {
        "M": [16, 64, 256],
        "resample": results["resample"],
        "fixed": results["fixed"],
    }


# --------------------------------------------------------------------------- 2. quadrature
def mmd2_vs_standard_normal(x: torch.Tensor) -> float:
    x = x.reshape(-1)
    d = x[:, None] - x[None, :]
    term_xx = torch.exp(-0.5 * d.square()).mean().item()
    term_yy = 1.0 / math.sqrt(3.0)
    term_xy = (torch.exp(-0.25 * x.square()) / math.sqrt(2.0)).mean().item()
    return term_xx + term_yy - 2.0 * term_xy


def ablation_quadrature() -> List[Dict]:
    """Quadrature error vs (t_max, n_points), measured against the closed-form MMD identity."""
    torch.manual_seed(0)
    out = []
    # a mixture of shapes so the error is not tuned to one distribution
    samples = {
        "N(0,1)": torch.randn(4096),
        "N(0,4)": torch.randn(4096) * 2.0,
        "uniform": (torch.rand(4096) - 0.5) * math.sqrt(12),
    }
    for t_max in (1.0, 3.0, 5.0, 8.0):
        for p in (5, 9, 17, 33, 65):
            errs = []
            for name, x in samples.items():
                ep = EppsPulley(t_max=t_max, n_points=p)(x.reshape(-1, 1)).item()
                ref = x.numel() * SQRT_2PI * mmd2_vs_standard_normal(x)
                errs.append(abs(ep - ref) / abs(ref))
            rec = {"t_max": t_max, "n_points": p,
                   "max_rel_err": max(errs), "mean_rel_err": sum(errs) / len(errs)}
            out.append(rec)
            print(f"  t_max={t_max:4.1f} P={p:3d}  max_rel_err={rec['max_rel_err']:.4%}  "
                  f"mean={rec['mean_rel_err']:.4%}")
    return out


# --------------------------------------------------------------------------- 3. projection normalisation
def ablation_projection_norm(d: int = 64, n: int = 1024, m: int = 256) -> Dict:
    """Unit-norm columns (paper) vs unnormalised Gaussian columns."""
    torch.manual_seed(0)
    z = torch.randn(n, d)
    ep = EppsPulley(t_max=5.0, n_points=17)

    a = sample_projection_directions(d, m, z.device, generator=None)
    unit = ep(z @ a).mean().item()

    g = torch.Generator().manual_seed(0)
    raw = torch.randn(d, m, generator=g)
    unnorm = ep(z @ raw).mean().item()

    print(f"  unit-norm columns : SIGReg={unit:.4f}")
    print(f"  raw N(0,I) columns: SIGReg={unnorm:.4f}  (scales with sqrt(D), not a valid test)")
    return {"unit_norm": unit, "unnormalised": unnorm}


# --------------------------------------------------------------------------- 4. discrimination
def ablation_discrimination(d: int = 64, n: int = 2048) -> Dict:
    torch.manual_seed(0)
    sig = SIGReg(num_slices=1024, t_max=5.0, n_points=17, seed=0)
    cases = {}

    cases["isotropic_gaussian"] = torch.randn(n, d)

    s = torch.ones(d)
    s[0] = 4.0
    cases["anisotropic_1dim_x4"] = torch.randn(n, d) * s

    latent = torch.randn(n, 8)
    corr = latent @ torch.randn(8, d) / math.sqrt(8)
    cases["rank8_correlated"] = corr / corr.std(dim=0, keepdim=True)

    cases["constant_ones"] = torch.ones(n, d)
    cases["two_point"] = (torch.randint(0, 2, (n, d)).float() * 2 - 1)
    cases["half_gaussian"] = torch.randn(n, d).abs()

    out = {}
    for name, z in cases.items():
        v = sig(z).item()
        out[name] = v
        print(f"  {name:24s} SIGReg={v:10.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/ablations")
    ap.add_argument("--skip-slow", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    report: Dict = {}

    print("\n[1] discrimination")
    report["discrimination"] = ablation_discrimination()

    print("\n[2] num_slices (estimator statistics)")
    torch.manual_seed(0)
    report["num_slices"] = ablation_num_slices(torch.randn(2048, 64))

    print("\n[3] quadrature error vs closed-form MMD  (paper Table 1a)")
    report["quadrature"] = ablation_quadrature()

    print("\n[4] projection normalisation")
    report["projection_norm"] = ablation_projection_norm()

    if not args.skip_slow:
        print("\n[5] resample vs fixed directions  (paper Figure 7)")
        report["resample_vs_fixed"] = ablation_resample_vs_fixed()

    path = os.path.join(args.out_dir, "statistical_ablations.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

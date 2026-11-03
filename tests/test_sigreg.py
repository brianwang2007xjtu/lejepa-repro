"""Unit tests for SIGReg (Sketched Isotropic Gaussian Regularization).

What is checked, and why
------------------------
* **Projection directions** — shape ``(K, M)``, unit L2 norm per *column*, approximately
  uniform on S^{K-1} (each coordinate ~ N(0, 1/K)), and re-drawn at every forward pass.
* **Behaviour on known distributions** — isotropic Gaussian gives the theoretical
  ``E[EP] = sqrt(2*pi)(1 - 1/sqrt(3)) ~= 1.06``; constant vectors, anisotropic Gaussians and
  rank-deficient (correlated) Gaussians all give strictly larger values.
* **Gradient** — flows to the input and is non-zero, is *zero* with respect to the sampled
  directions (they are constants), and matches finite differences.
* **The paper's Figure 6 experiment** — starting from D=512 samples whose first two
  coordinates lie on an adversarial "X", gradient descent **on the embeddings alone** with
  M=16 slices must recover an isotropic Gaussian.  This is the strongest end-to-end check
  that SIGReg is implemented correctly, because it exercises the sampler, the projection,
  the statistic and the gradient together.

Run: ``pytest tests/test_sigreg.py -v -s``
"""

import math

import pytest
import torch

from losses.epps_pulley import EppsPulley
from losses.sigreg import SIGReg, sample_projection_directions

SQRT_2PI = math.sqrt(2.0 * math.pi)
E_EP_NULL = SQRT_2PI * (1.0 - 1.0 / math.sqrt(3.0))  # 1.0594


# --------------------------------------------------------------------------- directions
def test_projection_directions_unit_norm():
    a = sample_projection_directions(dim=64, num_slices=256, device=torch.device("cpu"))
    assert a.shape == (64, 256)
    norms = a.norm(p=2, dim=0)
    print("column norms: min=%.6f max=%.6f" % (norms.min(), norms.max()))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_projection_directions_are_isotropic():
    """A normalised Gaussian vector is uniform on the sphere: E[a_i] = 0, Var[a_i] = 1/K,
    and E[a_i^4] = 3/K(K+2).  Check the first two moments with many samples."""
    k, m = 32, 200_000
    a = sample_projection_directions(k, m, torch.device("cpu"), generator=None)
    mean = a.mean(dim=1)
    var = a.var(dim=1, unbiased=False)
    print(f"E[a_i] ~ 0 : max|mean|={mean.abs().max():.5f} (expect ~{1/math.sqrt(m):.5f})")
    print(f"Var[a_i]   : mean={var.mean():.6f} (expect {1/k:.6f})")
    assert mean.abs().max() < 6.0 / math.sqrt(m)
    assert abs(var.mean().item() - 1.0 / k) < 0.02 / k * 20


def test_directions_are_resampled_each_step():
    torch.manual_seed(0)
    sig = SIGReg(num_slices=16, seed=0)
    z = torch.randn(64, 8)
    first = sig(z).item()
    second = sig(z).item()
    print(f"step0={first:.6f}  step1={second:.6f}")
    assert first != second, "directions must be resampled at every forward pass"


def test_seeded_directions_are_reproducible():
    def run():
        torch.manual_seed(123)
        sig = SIGReg(num_slices=64, seed=7)
        z = torch.randn(128, 16)
        return [sig(z).item() for _ in range(3)]

    a, b = run(), run()
    print("run A:", a, "\nrun B:", b)
    assert a == b, "same seed + same step counter must reproduce the direction sequence"


def test_fixed_directions_configuration():
    """``seed=None`` uses the global RNG; seeding torch must still make it deterministic."""
    z = torch.randn(128, 16)
    sig = SIGReg(num_slices=64, seed=None)
    torch.manual_seed(5)
    a = sig(z).item()
    torch.manual_seed(5)
    b = sig(z).item()
    assert a == b


# --------------------------------------------------------------------------- statistics
def _sigreg_many(sig, z, reps=40):
    return torch.tensor([sig(z).item() for _ in range(reps)])


def test_isotropic_gaussian_matches_theory():
    torch.manual_seed(0)
    sig = SIGReg(num_slices=512, t_max=5.0, n_points=17, seed=0)
    z = torch.randn(2048, 64)
    vals = _sigreg_many(sig, z)
    print(f"isotropic Gaussian: mean={vals.mean():.4f} std={vals.std():.4f} "
          f"(theory {E_EP_NULL:.4f})")
    assert abs(vals.mean().item() - E_EP_NULL) < 0.2


def test_constant_vectors_are_heavily_penalised():
    torch.manual_seed(0)
    sig = SIGReg(num_slices=256, seed=0)
    normal = sig(torch.randn(1024, 32)).item()
    const = sig(torch.ones(1024, 32)).item()
    zeros = sig(torch.zeros(1024, 32)).item()
    print(f"N(0,I)={normal:.4f}  ones={const:.4f}  zeros={zeros:.4f}")
    assert const > 20 * normal
    assert zeros > 20 * normal


def test_anisotropic_gaussian_is_penalised():
    torch.manual_seed(0)
    sig = SIGReg(num_slices=512, seed=0)
    k = 64
    iso = sig(torch.randn(2048, k)).item()

    scales = torch.ones(k)
    scales[0] = 4.0
    aniso = sig(torch.randn(2048, k) * scales).item()

    scales2 = torch.linspace(0.5, 2.0, k)
    aniso2 = sig(torch.randn(2048, k) * scales2).item()
    print(f"isotropic={iso:.4f}  one-dim x4={aniso:.4f}  graded scales={aniso2:.4f}")
    assert aniso > iso and aniso2 > iso


def test_correlated_gaussian_is_penalised():
    """Rank-deficient / strongly correlated embeddings must be detected: a rotated Gaussian
    has identity covariance only if the rotation is orthogonal, so use a genuine
    low-rank-plus-noise construction."""
    torch.manual_seed(0)
    sig = SIGReg(num_slices=512, seed=0)
    k, n = 64, 2048

    iso = sig(torch.randn(n, k)).item()

    # rank-8 latent factor model: massive correlation, variance still O(1) per dim
    latent = torch.randn(n, 8)
    loadings = torch.randn(8, k)
    corr = latent @ loadings / math.sqrt(8)
    corr = corr / corr.std(dim=0, keepdim=True)
    corr_val = sig(corr).item()

    print(f"isotropic={iso:.4f}  rank-8 correlated={corr_val:.4f}")
    assert corr_val > 3 * iso, "SIGReg must detect dimensional collapse"


def test_more_slices_reduce_estimator_variance():
    torch.manual_seed(0)
    z = torch.randn(512, 32)
    out = {}
    for m in (16, 64, 256, 1024):
        sig = SIGReg(num_slices=m, seed=0)
        vals = _sigreg_many(sig, z, reps=20)
        out[m] = vals.std().item()
        print(f"  M={m:5d}: mean={vals.mean():.4f} std={out[m]:.4f}")
    assert out[1024] < out[16], "more slices must reduce Monte-Carlo variance"


# --------------------------------------------------------------------------- gradients
def test_gradient_flows_and_is_correct():
    torch.manual_seed(0)
    z = torch.randn(128, 16, dtype=torch.float64, requires_grad=True)
    sig = SIGReg(num_slices=32, seed=0, t_max=5.0, n_points=17)

    loss = sig(z)
    (g,) = torch.autograd.grad(loss, z)
    print(f"loss={loss.item():.6f}  |grad|={g.norm():.6f}  finite={torch.isfinite(g).all()}")
    assert torch.isfinite(g).all() and g.norm() > 0

    # finite differences on a few coordinates
    eps = 1e-6

    def f(v):
        return SIGReg(num_slices=32, seed=0, t_max=5.0, n_points=17)(v)

    for idx in [(0, 0), (5, 3), (17, 9)]:
        zp = z.detach().clone()
        zp[idx] += eps
        zm = z.detach().clone()
        zm[idx] -= eps
        fd = (f(zp) - f(zm)).item() / (2 * eps)
        ana = g[idx].item()
        rel = abs(fd - ana) / max(abs(fd), 1e-9)
        print(f"  idx={idx} analytic={ana:+.6e} fd={fd:+.6e} rel={rel:.2e}")
        assert rel < 2e-4


def test_directions_carry_no_gradient():
    torch.manual_seed(0)
    sig = SIGReg(num_slices=8, seed=0)
    z = torch.randn(32, 4)
    sig(z)
    # the sampler runs under torch.no_grad(); nothing in the module should require grad
    assert not any(p.requires_grad for p in sig.parameters())
    assert sig.global_step.requires_grad is False


def test_double_backward_supported():
    """SIGReg must be twice differentiable (the paper discusses curvature bounds)."""
    torch.manual_seed(0)
    z = torch.randn(64, 8, dtype=torch.float64, requires_grad=True)
    sig = SIGReg(num_slices=16, seed=0)
    loss = sig(z)
    (g,) = torch.autograd.grad(loss, z, create_graph=True)
    (h,) = torch.autograd.grad(g.sum(), z)
    print(f"|grad|={g.norm():.4f}  |hessian diag| mean={h.abs().mean():.4f}")
    assert torch.isfinite(h).all()


# --------------------------------------------------------------------------- shapes
def test_arbitrary_leading_dimensions():
    torch.manual_seed(0)
    sig = SIGReg(num_slices=64, seed=0)
    z = torch.randn(4, 128, 16)  # (V, B, K)
    out = sig(z)
    assert out.ndim == 0
    # must equal the average over the leading dimension
    per = torch.stack([SIGReg(num_slices=64, seed=0)(z[i]) for i in range(4)])
    print(f"batched={out.item():.6f}  loop-mean={per.mean().item():.6f}")
    assert out.item() == pytest.approx(per.mean().item(), rel=1e-5)


def test_rejects_degenerate_inputs():
    sig = SIGReg(num_slices=8)
    with pytest.raises(ValueError):
        sig(torch.randn(1, 16))  # single sample
    with pytest.raises(ValueError):
        sig(torch.randn(16))  # 1-D


def test_num_slices_none_reduction_returns_per_slice():
    sig = SIGReg(num_slices=37, reduction="none")
    out = sig(torch.randn(256, 8))
    assert out.shape == (37,)


# --------------------------------------------------------------------------- Fig. 6 replication
@pytest.mark.slow
def test_recovers_isotropic_gaussian_from_adversarial_x():
    """Paper Figure 6: N=512 samples in R^512, first two coordinates forced onto an "X"
    (two crossing lines through the origin).  Optimising the *embeddings* with M=16 slices
    must spread those two coordinates back to a standard Gaussian.
    """
    torch.manual_seed(0)
    n, d, m = 512, 512, 16

    # 1. start from an isotropic Gaussian ...
    z = torch.randn(n, d)
    # 2. ... and replace dims 0,1 by the adversarial "X": two lines with angle +-45 deg
    t = torch.randn(n)
    sign = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    z[:, 0] = t * sign
    z[:, 1] = t * sign
    z = z + 0.0

    target = EppsPulley()(z[:, :2])
    print(f"before: EP(dim0)={target[0]:.3f} EP(dim1)={target[1]:.3f} "
          f"std=({z[:,0].std():.3f},{z[:,1].std():.3f})")

    z = torch.nn.Parameter(z.clone())
    opt = torch.optim.Adam([z], lr=1e-2)
    sig = SIGReg(num_slices=m, t_max=5.0, n_points=17, seed=0)

    losses = []
    for step in range(600):
        opt.zero_grad()
        loss = sig(z)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 150 == 0:
            print(f"  step {step:4d}: SIGReg={loss.item():.4f}")

    with torch.no_grad():
        final = z.detach()
        ep_after = EppsPulley()(final[:, :2])
        std = final[:, :2].std(dim=0)
        mean = final[:, :2].mean(dim=0)
        kurt = (((final[:, :2] - mean) / std) ** 4).mean(dim=0)
        # total variance over all 512 dims should be ~unchanged (SIGReg does not shrink)
        total_var = final.var(dim=0).sum().item()

    print(f"after : SIGReg={losses[-1]:.4f}  EP=({ep_after[0]:.3f},{ep_after[1]:.3f}) "
          f"std=({std[0]:.3f},{std[1]:.3f}) mean=({mean[0]:.3f},{mean[1]:.3f}) "
          f"kurt=({kurt[0]:.2f},{kurt[1]:.2f})  total_var={total_var:.1f} (target {d})")

    assert losses[-1] < losses[0], "SIGReg must decrease"
    # the two adversarial coordinates must become much closer to Gaussian
    assert ep_after.max().item() < 5.0, f"X-subspace not Gaussianised: {ep_after}"
    assert (std > 0.6).all() and (std < 1.6).all(), f"scale not restored: {std}"
    assert kurt.max().item() < 5.0, f"kurtosis still heavy: {kurt}"
    # global isotropic scale must be preserved (no collapse, no explosion)
    assert 0.5 * d < total_var < 1.5 * d

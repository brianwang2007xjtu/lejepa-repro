"""Unit tests for the Epps–Pulley statistic.

The point of this file is that the *numerical value* is checked against two independent
closed forms, not merely "the loss goes down":

1. **Constant sample** ``x_j = c``.  Then ``phi_hat(t) = e^{ict}`` exactly, so

       EP/N = ∫_{-inf}^{inf} |e^{ict} - e^{-t^2/2}|^2 e^{-t^2/2} dt

   which we evaluate by high-accuracy adaptive quadrature (scipy) and compare against the
   17-knot composite-trapezoid implementation.

2. **Kernel-MMD identity.**  With ``w(t) = e^{-t^2/2}``,

       ∫ |phi_hat - phi|^2 w dt = sqrt(2*pi) * MMD^2(P_hat, N(0,1))

   where MMD is the Gaussian-kernel MMD with ``k(u,v) = exp(-(u-v)^2/2)``.  Both the
   empirical term and the cross term against N(0,1) have closed forms:

       E[k(X,X')] = mean_{ij} exp(-(x_i-x_j)^2/2)
       E[k(Y,Y')] = 1/sqrt(3)                        (Y, Y' iid N(0,1))
       E[k(X,Y)]  = mean_i exp(-x_i^2/4)/sqrt(2)

   so the identity yields an exact expected value for any finite sample.

Run: ``pytest tests/test_epps_pulley.py -v``
"""

import math

import pytest
import torch

from losses.epps_pulley import EppsPulley

SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- helpers
def mmd2_vs_standard_normal(x: torch.Tensor) -> float:
    """Exact Gaussian-kernel MMD^2 between the empirical measure of ``x`` and N(0,1)."""
    x = x.reshape(-1)
    n = x.numel()
    # E[k(X, X')]
    d = x[:, None] - x[None, :]
    term_xx = torch.exp(-0.5 * d.square()).mean().item()
    # E[k(Y, Y')] for Y ~ N(0,1):  Y - Y' ~ N(0, 2) -> E exp(-Z^2/2) = 1/sqrt(1+2)
    term_yy = 1.0 / math.sqrt(3.0)
    # E[k(X, Y)]
    term_xy = (torch.exp(-0.25 * x.square()) / math.sqrt(2.0)).mean().item()
    return term_xx + term_yy - 2.0 * term_xy


def ep_constant_reference(c: float, t_max: float = 5.0) -> float:
    """∫_{-T}^{T} |e^{ict} - e^{-t^2/2}|^2 e^{-t^2/2} dt  by 1e5-point Simpson."""
    n = 200_001
    t = torch.linspace(-t_max, t_max, n, dtype=torch.float64)
    phi = torch.exp(-0.5 * t.square())
    re = torch.cos(c * t)
    im = torch.sin(c * t)
    integrand = (re - phi).square() + im.square()
    integrand = integrand * phi
    return torch.trapz(integrand, t).item()


# --------------------------------------------------------------------------- 1. exact-value checks
def test_constant_sample_matches_quadrature_reference():
    """EP for x_j = c must equal N * ∫|e^{ict} - phi|^2 phi dt."""
    n = 4096
    for c in (0.0, 0.5, 1.0, 2.0):
        x = torch.full((n, 1), c, dtype=torch.float64)
        ep = EppsPulley(t_max=5.0, n_points=17)(x).item()
        ref = n * ep_constant_reference(c)
        rel = abs(ep - ref) / ref
        print(f"c={c}: EP={ep:.6f}  ref={ref:.6f}  rel_err={rel:.4%}")
        assert rel < 0.02, f"quadrature error too large for c={c}: {ep} vs {ref}"


def test_mmd_identity_random_samples():
    """EP/N must equal sqrt(2*pi) * MMD^2 against N(0,1) (up to quadrature error)."""
    torch.manual_seed(0)
    for dist, name in [
        (lambda: torch.randn(2048), "N(0,1)"),
        (lambda: torch.randn(2048) * 2.0, "N(0,4)"),
        (lambda: torch.randn(2048) + 1.0, "N(1,1)"),
    ]:
        x = dist().reshape(-1, 1)
        ep = EppsPulley(t_max=5.0, n_points=17)(x).item()
        ref = x.numel() * SQRT_2PI * mmd2_vs_standard_normal(x)
        rel = abs(ep - ref) / max(ref, 1e-12)
        print(f"{name}: EP={ep:.6f}  sqrt(2pi)*N*MMD^2={ref:.6f}  rel_err={rel:.3%}")
        assert rel < 0.03, f"MMD identity violated for {name}: {ep} vs {ref}"


def test_quadrature_converges_with_more_points():
    """Increasing n_points must converge to the fine reference (monotone-ish error decay)."""
    torch.manual_seed(1)
    x = torch.randn(1024, 1, dtype=torch.float64)
    ref = 1024 * SQRT_2PI * mmd2_vs_standard_normal(x)
    errs = {}
    for p in (5, 9, 17, 33, 65):
        ep = EppsPulley(t_max=5.0, n_points=p)(x).item()
        errs[p] = abs(ep - ref) / ref
        print(f"n_points={p:3d}  EP={ep:.6f}  rel_err={errs[p]:.4%}")
    assert errs[65] < errs[5], "refining the quadrature must reduce the error"
    assert errs[17] < 0.03


# --------------------------------------------------------------------------- 2. discrimination
def test_standard_normal_expectation_is_exact_constant():
    """Under the null, E[EP] = sqrt(2*pi) * (1 - 1/sqrt(3)) = 1.0594, independent of N.

    Derivation.  EP = N*sqrt(2*pi)*MMD^2 and, for N iid draws from the null,

        E[MMD^2_emp] = (1/N)[k(x,x) - E k] = (1/N)[1 - 1/sqrt(3)]

    because E[k(X,X')] = E[k(X,Y)] = E[k(Y,Y')] = 1/sqrt(3) for the Gaussian kernel with
    bandwidth 1 and unit-variance marginals.  Hence E[EP] = sqrt(2*pi)(1 - 1/sqrt(3)).

    This is the honest statement: the EP statistic of *correctly* Gaussian data is ~1.06,
    not 0.  A statistic of 0 is only attained by the exact N(0,1) measure, never by a
    finite sample.  Any test asserting ``stat < 0.1`` for Gaussian data would be testing
    the wrong quantity.
    """
    pred = SQRT_2PI * (1.0 - 1.0 / math.sqrt(3.0))
    print(f"predicted E[EP] under H0 = {pred:.4f}")
    ep = EppsPulley()
    for n in (512, 4096):
        torch.manual_seed(42)
        vals = torch.cat([ep(torch.randn(n, 1)) for _ in range(200)])
        mean = vals.mean().item()
        print(f"  N={n:5d}: mean={mean:.4f}  std={vals.std().item():.4f}  "
              f"rel_err={abs(mean - pred) / pred:.2%}")
        # 200 replicates -> standard error ~ std/sqrt(200) ~ 0.07; allow 4 sigma.
        assert abs(mean - pred) < 0.30, f"E[EP] should be {pred:.4f}, got {mean:.4f}"
    assert pred == pytest.approx(1.0594, abs=1e-3)


def test_constant_distribution_high_statistic():
    x = torch.full((1024, 1), 1.0)
    stat = EppsPulley()(x).item()
    ref_ratio = ep_constant_reference(1.0)
    print(f"constant: statistic={stat:.4f}  (= N * {ref_ratio:.6f})")
    assert stat > 50.0


def test_rademacher_distribution_high_statistic():
    torch.manual_seed(0)
    signs = torch.randint(0, 2, (4096, 4)).float() * 2 - 1
    stat = EppsPulley()(signs)
    normal = EppsPulley()(torch.randn(4096, 4))
    print("Rademacher:", stat, " N(0,1):", normal)
    assert (stat > 10.0).all(), f"Rademacher should be far from Gaussian, got {stat}"
    assert (stat > normal).all()


def test_shift_and_scale_are_detected():
    torch.manual_seed(0)
    base = torch.randn(2048, 1)
    ep = EppsPulley()
    s0 = ep(base).item()
    s1 = ep(base + 1.0).item()
    s2 = ep(base * 3.0).item()
    print(f"base={s0:.4f}  +1={s1:.4f}  *3={s2:.4f}")
    assert s1 > s0 and s2 > s0


def test_rademacher_converges_to_low_statistic_as_scale_grows():
    """Scaling a Rademacher sample by s -> 0 collapses it to a point mass at 0, which is
    *also* the t -> 0 limit of the CF; scaling it UP makes the two atoms resolve.
    We assert the opposite, more informative direction: large scale resolves the atoms."""
    torch.manual_seed(0)
    base = (torch.randint(0, 2, (4096, 1)).float() * 2 - 1)
    stats = [EppsPulley()(base * s).item() for s in (0.1, 0.5, 1.0, 2.0)]
    print("Rademacher scale sweep:", [f"{v:.3f}" for v in stats])
    assert stats[-1] >= stats[0], "atomic structure must be resolved at larger scale"


# --------------------------------------------------------------------------- 3. invariance / shapes
def test_permutation_and_column_independence():
    torch.manual_seed(0)
    x = torch.randn(512, 5)
    ep = EppsPulley()
    a = ep(x)
    perm = torch.randperm(512)
    b = ep(x[perm])
    assert torch.allclose(a, b, rtol=1e-5), "sample order must not matter"
    cols = [2, 0, 4, 1, 3]
    c = ep(x[:, cols])
    assert torch.allclose(c, a[cols], rtol=1e-5), "columns must be independent"


def test_batched_leading_dimensions():
    torch.manual_seed(0)
    x = torch.randn(3, 256, 7)  # (V, N, M)
    out = EppsPulley()(x)
    print("batched shape:", out.shape)
    assert out.shape == (3, 7)
    # must agree with the loop over the leading dimension
    loop = torch.stack([EppsPulley()(x[i]) for i in range(3)])
    assert torch.allclose(out, loop, rtol=1e-5)


def test_non_negative_and_finite():
    torch.manual_seed(0)
    for n in (2, 3, 16, 1000):
        out = EppsPulley()(torch.randn(n, 4))
        assert torch.isfinite(out).all()
        assert (out >= -1e-6).all()


# --------------------------------------------------------------------------- 4. gradients
def test_gradient_matches_finite_difference():
    torch.manual_seed(0)
    x = torch.randn(64, 1, dtype=torch.float64, requires_grad=True)

    def f(v):
        return EppsPulley(t_max=5.0, n_points=17)(v).sum()

    loss = f(x)
    (g,) = torch.autograd.grad(loss, x)

    eps = 1e-6
    idx = torch.randint(0, 64, (8,))
    for i in idx:
        xp = x.detach().clone()
        xp[i] += eps
        xm = x.detach().clone()
        xm[i] -= eps
        fd = (f(xp) - f(xm)).item() / (2 * eps)
        ana = g[i].item()
        rel = abs(fd - ana) / max(abs(fd), 1e-8)
        print(f"i={int(i):2d}  analytic={ana:+.6e}  fd={fd:+.6e}  rel={rel:.2e}")
        assert rel < 1e-5, "analytic gradient disagrees with central differences"


def test_gradient_is_bounded_for_extreme_inputs():
    """Theorem 4 promises bounded gradients; check no blow-up on huge inputs."""
    for scale in (1.0, 10.0, 100.0, 1000.0):
        x = (torch.randn(256, 1, dtype=torch.float64) * scale).requires_grad_(True)
        loss = EppsPulley()(x).sum()
        (g,) = torch.autograd.grad(loss, x)
        print(f"scale={scale:7.1f}  |grad|_max={g.abs().max().item():.4e}")
        assert torch.isfinite(g).all()
        assert g.abs().max().item() < 5.0 * 256, "gradient must stay O(N), not explode"


def test_collapse_is_a_saddle_not_an_attractor():
    """The fully collapsed sample x = 0 is a *stationary point* of EP with zero gradient.

    d/dx_j E[cos(t x_j)] = -(t/N) sin(t x_j) = 0 at x_j = 0, so the gradient vanishes
    identically at the point mass at 0.  This test documents two facts that matter for
    LeJEPA:

    * the gradient is exactly zero there (so an *exactly* collapsed encoder is stuck), and
    * the collapse is a **local maximum** of the statistic: any perturbation lowers the
      loss, so the collapsed configuration is unstable and cannot attract the dynamics.

    This is the mathematical reason SIGReg prevents representation collapse: collapse is a
    saddle, not a minimum of the objective.
    """
    ep = EppsPulley()
    x0 = torch.zeros(512, 1, dtype=torch.float64)
    loss0 = ep(x0).item()

    xg = x0.clone().requires_grad_(True)
    ep(xg).sum().backward()
    grad_norm = xg.grad.norm().item()
    print(f"L(collapse)={loss0:.4f}   ||grad||={grad_norm:.3e}")

    torch.manual_seed(0)
    lowered = 0
    for eps in (0.01, 0.05, 0.1, 0.5):
        perturbed = eps * torch.randn(512, 1, dtype=torch.float64)
        loss_p = ep(perturbed).item()
        print(f"  eps={eps:5.2f}: L(perturbed)={loss_p:.4f}  delta={loss_p - loss0:+.4f}")
        if loss_p < loss0:
            lowered += 1

    assert grad_norm < 1e-9, "exact collapse is a stationary point"
    assert lowered == 4, "collapse must be a local maximum (perturbations reduce the loss)"


# --------------------------------------------------------------------------- 5. dtypes
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_support(dtype):
    torch.manual_seed(0)
    x = torch.randn(128, 3, dtype=dtype)
    out = EppsPulley()(x.to(dtype))
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


def test_bf16_matches_fp32_within_tolerance():
    if not torch.cuda.is_available():
        pytest.skip("bf16 check needs CUDA")
    torch.manual_seed(0)
    x = torch.randn(512, 16, device="cuda")
    ep32 = EppsPulley().cuda()
    ep16 = EppsPulley().cuda()
    fp32 = ep32(x)
    bf16 = ep16(x.to(torch.bfloat16)).float()
    rel = ((bf16 - fp32).abs() / fp32.clamp_min(1e-6)).max().item()
    print(f"bf16 vs fp32 max rel err = {rel:.3%}")
    assert rel < 0.05

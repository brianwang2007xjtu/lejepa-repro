"""Unit tests for the LeJEPA prediction / invariance loss.

The identity that must hold (paper Eq. 5 -> Eq. 7, §B.6)::

    (1/V_g) sum_{v=1}^{V_g} (1/V) sum_{v'} ||z_v - z_{v'}||^2
        == (1/V) sum_{v'} || mean_{v<=V_g} z_v - z_{v'} ||^2

Both sides are computed on hand-built tensors so the numbers are checkable by hand.
A worked example is included in ``test_hand_computed_two_views`` where the answer is
obtained analytically on paper.

Also asserted: no stop-gradient (the gradient with respect to *every* view is non-zero),
correct handling of `n_global_views == V` and `n_global_views == 1`, and exact agreement
between the "mean over features" and "sum over features" variants.

Run: ``pytest tests/test_invariance.py -v -s``
"""

import pytest
import torch

from losses.invariance import LeJEPAObjective, prediction_loss
from losses.sigreg import SIGReg


# --------------------------------------------------------------------------- hand examples
def test_hand_computed_two_views():
    """V = V_g = 2, B = 1, K = 1.

        z_0 = 1, z_1 = 3  ->  mu = 2, distances 1 and 1, mean = 1
    """
    z = torch.tensor([[[1.0]], [[3.0]]])  # (V=2, B=1, K=1)
    out = prediction_loss(z, n_global_views=2)
    print("two views, hand value 1.0 ->", out.item())
    assert out.item() == pytest.approx(1.0)


def test_hand_computed_three_views():
    """V = V_g = 3, K = 1: z = [0, 1, 5] -> mu = 2, sq dists 4, 1, 9 -> mean 14/3."""
    z = torch.tensor([[[0.0]], [[1.0]], [[5.0]]])
    out = prediction_loss(z, n_global_views=3)
    print("three views, hand value 14/3 ->", out.item())
    assert out.item() == pytest.approx(14.0 / 3.0)


def test_hand_computed_with_local_views():
    """V_g = 1 global (z=0) and V_l = 2 local (z=-2, z=4).  mu = 0 (the single global view),
    so the loss is mean over the 3 views of (0, 4, 16) = 20/3."""
    z = torch.tensor([[[0.0]], [[-2.0]], [[4.0]]])
    out = prediction_loss(z, n_global_views=1)
    print("1 global + 2 local, hand value 20/3 ->", out.item())
    assert out.item() == pytest.approx(20.0 / 3.0)


def test_hand_computed_multidim():
    """K = 2, V = V_g = 2:
        z_0 = (0,0), z_1 = (2,4) -> mu = (1,2)
        d0 = (1,2), d1 = (-1,-2), both with sum-of-squares 5, mean-of-squares 2.5
    """
    z = torch.tensor([[[0.0, 0.0]], [[2.0, 4.0]]])
    out = prediction_loss(z, n_global_views=2, feature_reduce="mean")
    out_sum = prediction_loss(z, n_global_views=2, feature_reduce="sum")
    print(f"mean-over-K={out.item()}  sum-over-K={out_sum.item()}")
    assert out.item() == pytest.approx(2.5)
    assert out_sum.item() == pytest.approx(5.0)


# --------------------------------------------------------------------------- Eq.5 vs Eq.7
@pytest.mark.parametrize("v", [2, 4, 8])
@pytest.mark.parametrize("vg", [1, 2, 4])
def test_paper_eq5_equals_eq7_plus_global_variance(v, vg):
    """The paper's Eq. (5) is **not** equal to Eq. (7); it exceeds it by the variance of the
    global views.

    Paper Appendix B.6 claims ``(1/Vg) sum_v ||z_v||^2 == ||zbar||^2`` (its Eq. 21).  That
    identity only holds when the global views are already identical.  The correct statement is

        (1/Vg) sum_v ||z_v||^2 = ||zbar||^2 + (1/Vg) sum_v ||z_v - zbar||^2

    so

        Eq.(5) = Eq.(7) + (1/Vg) sum_v ||z_v - zbar||^2 .

    Algorithm 2 of the paper implements Eq. (7), i.e. the paper's *code* is consistent with
    Eq. (6)/(7) and not with Eq. (5).  We train with Eq. (7) (see ``prediction_loss``) and
    keep the Eq. (5) branch only for this test.
    """
    if vg > v:
        pytest.skip("Vg > V")
    torch.manual_seed(0)
    b, k = 16, 32
    z = torch.randn(v, b, k, dtype=torch.float64)

    eq5 = prediction_loss(z, vg, pairwise=True, feature_reduce="sum").item()
    eq7 = prediction_loss(z, vg, pairwise=False, feature_reduce="sum").item()
    var_g = (z[:vg] - z[:vg].mean(dim=0)).square().sum(-1).sum().item() / (vg * b)
    print(f"V={v} Vg={vg}: Eq5={eq5:.6f}  Eq7={eq7:.6f}  Var_g={var_g:.6f}  "
          f"residual={eq5 - eq7 - var_g:.2e}")
    assert eq5 == pytest.approx(eq7 + var_g, rel=1e-12)


def test_eq5_and_eq7_coincide_for_a_single_global_view():
    torch.manual_seed(0)
    z = torch.randn(4, 16, 8, dtype=torch.float64)
    a = prediction_loss(z, 1, pairwise=True, feature_reduce="sum").item()
    b = prediction_loss(z, 1, pairwise=False, feature_reduce="sum").item()
    print(f"Vg=1: Eq5={a:.8f} Eq7={b:.8f}")
    assert a == pytest.approx(b, rel=1e-12)


def test_eq5_and_eq7_coincide_when_global_views_are_identical():
    """If every global view is the same vector, Var_g = 0 and both forms agree."""
    torch.manual_seed(0)
    g = torch.randn(16, 8, dtype=torch.float64)
    locals_ = torch.randn(6, 16, 8, dtype=torch.float64)
    z = torch.cat([g.unsqueeze(0).repeat(2, 1, 1), locals_], dim=0)
    a = prediction_loss(z, 2, pairwise=True, feature_reduce="sum").item()
    b = prediction_loss(z, 2, pairwise=False, feature_reduce="sum").item()
    print(f"identical globals: Eq5={a:.8f} Eq7={b:.8f}")
    assert a == pytest.approx(b, rel=1e-12)


def test_global_views_only_define_the_centre():
    """Changing a *local* view must not change mu; changing a global view must."""
    z = torch.randn(4, 8, 16, dtype=torch.float64)
    mu_before = z[:2].mean(dim=0)
    z2 = z.clone()
    z2[3] += 100.0  # perturb a local view
    mu_after = z2[:2].mean(dim=0)
    assert torch.allclose(mu_before, mu_after)

    z3 = z.clone()
    z3[0] += 100.0  # perturb a global view
    assert not torch.allclose(mu_before, z3[:2].mean(dim=0))


# --------------------------------------------------------------------------- scales
def test_quadratic_in_scale():
    z = torch.randn(4, 8, 16)
    a = prediction_loss(z, 2).item()
    b = prediction_loss(2 * z, 2).item()
    print(f"L(z)={a:.6f}  L(2z)={b:.6f}  ratio={b/a:.4f}")
    assert b == pytest.approx(4 * a, rel=1e-5)


def test_scale_invariant_to_feature_dim():
    """With the Algorithm-2 reduction (mean over K) the loss must not grow with K."""
    torch.manual_seed(0)
    vals = {}
    for k in (16, 64, 256):
        z = torch.randn(4, 8, k)
        vals[k] = prediction_loss(z, 2).item()
        print(f"  K={k:3d}: L={vals[k]:.5f}")
    assert max(vals.values()) / min(vals.values()) < 1.3


def test_constructor_validation():
    with pytest.raises(ValueError):
        prediction_loss(torch.randn(4, 8), 2)  # 2-D input
    with pytest.raises(ValueError):
        prediction_loss(torch.randn(4, 8, 16), 5)  # Vg > V


# --------------------------------------------------------------------------- no stop-grad
def test_no_stop_gradient_every_view_receives_gradient():
    """If a stop-gradient were present on any view, its gradient would be exactly zero."""
    torch.manual_seed(0)
    sig = SIGReg(num_slices=64, seed=0)
    obj = LeJEPAObjective(sigreg=sig, lambd=0.05, n_global_views=2)
    z = torch.randn(4, 8, 16, requires_grad=True)
    loss = obj(z)
    (g,) = torch.autograd.grad(loss, z, retain_graph=False)
    per_view = g.abs().sum(dim=(1, 2))
    print("per-view |grad|:", per_view.tolist())
    assert (per_view > 0).all(), "every view must receive gradient (no stop-gradient)"


def test_prediction_gradient_is_linear_in_z():
    """d/dz of a quadratic form is linear; check against the analytic expression for the
    prediction term alone, with lamdbda=0."""
    torch.manual_seed(0)
    v, b, k = 4, 8, 16
    vg = 2
    z = torch.randn(v, b, k, dtype=torch.float64, requires_grad=True)

    loss = prediction_loss(z, n_global_views=vg)
    (g,) = torch.autograd.grad(loss, z)

    # analytic: L = (1/(V B K)) sum_{v',b,k} (mu - z)^2 with mu = mean of globals
    with torch.no_grad():
        mu = z[:vg].mean(dim=0)  # (B, K)
        d = mu.unsqueeze(0) - z  # (V, B, K)
        # dL/dz_v' = -2 d_v' / (V B K)          for local views
        # dL/dz_g   = +2 (sum_{v'} d_v') / (V B K Vg) for global views
        scale = 2.0 / (v * b * k)
        g_analytic = -scale * d
        g_analytic[:vg] += scale / vg * d.sum(dim=0, keepdim=True)
    err = (g - g_analytic).abs().max().item()
    print(f"max |autograd - analytic| = {err:.3e}")
    assert err < 1e-12


# --------------------------------------------------------------------------- full objective
def test_objective_weighting():
    """L = (1-lam)*pred + lam*sigreg must hold exactly for a controllable fake SIGReg."""
    torch.manual_seed(0)
    z = torch.randn(4, 8, 16)

    class FakeSIG(torch.nn.Module):
        def forward(self, x):
            return torch.tensor(2.0)

    pred = prediction_loss(z, 2).item()
    for lam in (0.0, 0.05, 0.5, 1.0):
        obj = LeJEPAObjective(FakeSIG(), lambd=lam, n_global_views=2)
        got = obj(z).item()
        want = (1 - lam) * pred + lam * 2.0
        print(f"  lam={lam:4.2f}: got={got:.6f} want={want:.6f}")
        assert got == pytest.approx(want, rel=1e-6)


def test_objective_components_shapes():
    torch.manual_seed(0)
    sig = SIGReg(num_slices=32, seed=0)
    obj = LeJEPAObjective(sig, lambd=0.05, n_global_views=2)
    out = obj(torch.randn(8, 4, 16), return_components=True)
    print("components:", out.prediction.item(), out.sigreg.item())
    assert out.prediction_per_view.shape == (8,)
    assert out.sigreg_per_view.shape == (8,)
    assert torch.isfinite(out.loss)


def test_lambda_validation():
    with pytest.raises(ValueError):
        LeJEPAObjective(SIGReg(), lambd=1.5)

"""LeJEPA prediction / invariance loss and the full LeJEPA objective.

Prediction loss, paper Eq. (5)–(7)
----------------------------------
With ``V_g`` global views (indices 0 .. V_g-1), ``V_l`` local views and ``V = V_g + V_l``::

    L_pred = (1/V_g) sum_{v=1}^{V_g} (1/V) sum_{v'=1}^{V} || z_{n,v} - z_{n,v'} ||_2^2   (5)
           = (1/V) sum_{v'} || (1/V_g) sum_{v=1}^{V_g} z_{n,v} - z_{n,v'} ||_2^2          (6)
           = (1/V) sum_{v'} || mu_n - z_{n,v'} ||_2^2                                      (7)

    mu_n = (1/V_g) sum_{v=1}^{V_g} z_{n,v}          "global centre" of sample n

Equation (5) -> (6) is the bias/variance decomposition proved in the paper's §B.6: the mean
over *all* pairs (global v, any v') of squared distances equals the mean over v' of the
squared distance to the mean of the globals.  Both forms are implemented here
(``pairwise=True`` / ``False``) precisely so that a unit test can check the identity.

Reduction
---------
Algorithm 2 of the paper writes ``sim = (centers - a_emb).square().mean()``, i.e. it averages
over ``V``, ``B`` **and** ``K``.  Eq. (5) itself has no ``1/K``.  We follow Algorithm 2 —
the code is what the recommended ``lambda`` was tuned against, and the ``1/K`` is what makes
the (lambda, projector_dim) trade-off in Table 1d comparable across ``projector_dim`` in
{64, ..., 1024}.  ``feature_reduce="sum"`` recovers the literal Eq. (5).

No stop-gradient
----------------
Every view is differentiated.  ``tests/test_invariance.py`` asserts that ``dL/dz`` is
non-zero for all V views, which would fail if a stop-gradient were reintroduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

__all__ = ["prediction_loss", "LeJEPAObjective", "LeJEPAOutput"]


def prediction_loss(
    z: Tensor,
    n_global_views: int,
    pairwise: bool = False,
    feature_reduce: str = "mean",
) -> Tensor:
    """LeJEPA prediction loss.

    Parameters
    ----------
    z : Tensor
        View embeddings, shape ``(V, B, K)``.  Views ``0 .. n_global_views-1`` must be the
        global views (the paper fixes that index convention, §5.1).
    n_global_views : int
        ``V_g``.  ``V_g = V`` means every view is global.
    pairwise : bool
        If True compute the literal Eq. (5) (O(V^2)); if False compute Eq. (6)/(7) (O(V)).
        The two agree to floating-point precision and the test-suite checks it.
    feature_reduce : {"mean", "sum"}
        Whether to average the squared norm over the K feature dimensions.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (V, B, K), got {tuple(z.shape)}")
    n_views = z.size(0)
    if not 0 < n_global_views <= n_views:
        raise ValueError(f"n_global_views={n_global_views} out of range for V={n_views}")
    if feature_reduce not in ("mean", "sum"):
        raise ValueError(f"unknown feature_reduce {feature_reduce!r}")

    global_views = z[:n_global_views]  # (V_g, B, K)

    if pairwise:
        # Eq. (5): (1/V_g) sum_v (1/V) sum_v' ||z_v - z_v'||^2
        #   diff[v, v', b, k] = z_v - z_v'
        diff = global_views.unsqueeze(1) - z.unsqueeze(0)  # (V_g, V, B, K)
        squared = diff.square()
        if feature_reduce == "mean":
            squared = squared.mean(dim=-1)  # (V_g, V, B)
        else:
            squared = squared.sum(dim=-1)  # (V_g, V, B)
        return squared.mean()

    # Eq. (6)/(7): (1/V) sum_v' || mu - z_v' ||^2,  mu = mean over the global views
    mu = global_views.mean(dim=0)  # (B, K)
    diff = mu.unsqueeze(0) - z  # (V, B, K)
    squared = diff.square()
    if feature_reduce == "mean":
        squared = squared.mean(dim=-1)  # (V, B)
    else:
        squared = squared.sum(dim=-1)  # (V, B)
    return squared.mean()


@dataclass
class LeJEPAOutput:
    """Container for the decomposed objective (all scalars unless noted)."""

    loss: Tensor
    prediction: Tensor
    sigreg: Tensor
    prediction_per_view: Tensor  # (V,) diagnostic: mean squared distance of each view to mu
    sigreg_per_view: Tensor  # (V,) diagnostic: SIGReg of each view's embedding


class LeJEPAObjective(torch.nn.Module):
    """``L = (1 - lambda) * L_pred + lambda * SIGReg`` (paper Eq. "LeJEPA").

    The SIGReg term is applied to **each view separately** and averaged over the V views,
    exactly as in Algorithm 2::

        sigreg = mean(SIGReg(emb) for emb in a_emb)     # a_emb: (V, B, K)

    Parameters
    ----------
    sigreg : nn.Module
        A :class:`~losses.sigreg.SIGReg` instance (injected so that ablations can swap the
        statistical test).
    lambd : float
        The single trade-off hyper-parameter ``lambda`` in [0, 1].
    n_global_views : int
        Number of leading views that are global.
    """

    def __init__(
        self,
        sigreg: torch.nn.Module,
        lambd: float = 0.05,
        n_global_views: int = 2,
        feature_reduce: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= lambd <= 1.0:
            raise ValueError(f"lambda must be in [0, 1], got {lambd}")
        self.sigreg = sigreg
        self.lambd = float(lambd)
        self.n_global_views = int(n_global_views)
        self.feature_reduce = feature_reduce

    def forward(self, z: Tensor, return_components: bool = False):
        """Compute the objective on ``z`` of shape ``(V, B, K)``."""
        pred = prediction_loss(
            z,
            n_global_views=self.n_global_views,
            feature_reduce=self.feature_reduce,
        )

        # SIGReg per view, then averaged over views (Algorithm 2).  ``z[v]`` is (B, K).
        per_view = torch.stack([self.sigreg(z[v]) for v in range(z.size(0))])
        sig = per_view.mean()

        # lambda weights SIGReg, (1 - lambda) weights the prediction term.
        loss = (1.0 - self.lambd) * pred + self.lambd * sig

        if not return_components:
            return loss

        with torch.no_grad():
            mu = z[: self.n_global_views].mean(dim=0)
            pred_per_view = (mu.unsqueeze(0) - z).square().mean(dim=(1, 2))
            sig_per_view = per_view.detach()

        return LeJEPAOutput(
            loss=loss,
            prediction=pred.detach(),
            sigreg=sig.detach(),
            prediction_per_view=pred_per_view.detach(),
            sigreg_per_view=sig_per_view,
        )

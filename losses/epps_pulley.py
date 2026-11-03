"""Epps–Pulley goodness-of-fit statistic against the standard Gaussian.

Mathematical definition (paper Eq. "Epps–Pulley", §4.2.3)::

    EP = N * ∫_{-inf}^{inf} |phi_hat_X(t) - phi(t)|^2 w(t) dt

    phi_hat_X(t) = (1/N) sum_j exp(i t X_j)      empirical characteristic function
    phi(t)       = exp(-t^2 / 2)                 CF of N(0, 1)
    w(t)         = exp(-t^2 / 2)                 Gaussian window

Expanding the squared modulus with w(t) = exp(-t^2/2)::

    |phi_hat - phi|^2 = (Re phi_hat - phi)^2 + (Im phi_hat)^2
                     = (mean_j cos(t X_j) - phi(t))^2 + (mean_j sin(t X_j))^2

because phi(t) is real and Im phi(t) = 0 for the standard Gaussian.

Quadrature
----------
We exploit the symmetry of the integrand (both |phi_hat - phi|^2 and w are even) and
integrate only over t >= 0, doubling all interior weights.  With ``n_points`` knots on
[0, t_max] and ``dt = t_max / (n_points - 1)`` the composite-trapezoid weights are::

    w_0 = dt            (t = 0, shared by both halves of the symmetric sum)
    w_i = 2 * dt        (interior)
    w_{n-1} = dt        (t = t_max, endpoint)

so the statistic is::

    EP = N * sum_i w_i * exp(-t_i^2 / 2) * [(cos_mean_i - phi_i)^2 + sin_mean_i^2]

This is exactly the official implementation's construction (``weights * phi`` with
``weights[[0, -1]] = dt``), which is itself an improvement over Algorithm 1's plain
trapezoid on [-5, 5] ("improved quadrature for free", MINIMAL.md).

Consistency check
-----------------
For w(t) = exp(-t^2/2) the exact integral equals the maximum-mean-discrepancy squared of
the Gaussian kernel k(u,v) = exp(-(u-v)^2/2) with bandwidth 1::

    ∫ |phi_hat - phi|^2 exp(-t^2/2) dt = MMD^2(P_hat, N(0,1))

so EP/N -> MMD^2 as the quadrature is refined.  ``tests/test_epps_pulley.py`` verifies this
identity numerically against the closed-form kernel expectation E[k(Y,Y')] = 1/sqrt(3).

Shape convention
----------------
Input ``x`` has shape ``(..., N, M)``: ``N`` samples (dim ``-2``) and ``M`` independent
1-D problems (dim ``-1``, i.e. the SIGReg slices).  The output has shape ``(..., M)``,
each entry being the statistic of one univariate problem.  The ``N`` factor is applied
*inside* this module, matching Eq. (Epps–Pulley) and the official implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["EppsPulley"]


def _all_reduce(x: Tensor, op: str = "AVG") -> Tensor:
    """Average ``x`` across DDP ranks (identity when not distributed).

    The ECF must be averaged over *all* samples of the global batch before the
    statistic is formed, otherwise every rank would test a different empirical
    distribution and the gradients would disagree.  Using the autograd-aware
    ``torch.distributed.nn`` variant keeps the gradient flowing to every rank.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        import torch.distributed.nn as dist_nn

        reduce_op = getattr(dist_nn.ReduceOp, op.upper())
        return dist_nn.all_reduce(x, reduce_op)
    return x


class EppsPulley(nn.Module):
    """Epps–Pulley normality statistic (differentiable, O(N) time and memory).

    Parameters
    ----------
    t_max:
        Upper end of the (symmetric) integration domain, i.e. the integral runs over
        ``[-t_max, t_max]``.  The paper recommends 5 (Table 1a / §6.1) while the
        official class defaults to 3; both are exposed in the configs.
    n_points:
        Number of quadrature knots on ``[0, t_max]``.  ``n_points=17`` is the paper's
        value; the symmetric doubling yields ``2 * n_points - 1`` effective knots on
        ``[-t_max, t_max]``.
    """

    def __init__(self, t_max: float = 5.0, n_points: int = 17) -> None:
        super().__init__()
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        if t_max <= 0:
            raise ValueError(f"t_max must be > 0, got {t_max}")

        self.t_max = float(t_max)
        self.n_points = int(n_points)

        # Integration knots on the positive half-line: t_0 = 0, ..., t_{P-1} = t_max.
        t = torch.linspace(0.0, self.t_max, self.n_points, dtype=torch.float32)
        dt = self.t_max / (self.n_points - 1)

        # Composite-trapezoid weights, doubled for the symmetric half (see module docstring).
        weights = torch.full((self.n_points,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt

        # phi(t) = exp(-t^2 / 2): the CF of N(0, 1) AND the Gaussian window w(t).
        phi = torch.exp(-0.5 * t.square())

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        # Pre-multiplying by phi folds the window w(t) into the quadrature weights.
        self.register_buffer("weights", weights * phi)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the statistic for each column of ``x``.

        Parameters
        ----------
        x : Tensor
            Shape ``(..., N, M)``.  ``N`` = samples, ``M`` = independent 1-D problems.

        Returns
        -------
        Tensor
            Shape ``(..., M)``, the statistic ``N * ∫ |phi_hat - phi|^2 w dt``.
        """
        if x.ndim < 2:
            raise ValueError(f"expected (..., N, M) input, got shape {tuple(x.shape)}")

        n_samples = x.size(-2)
        if n_samples == 0:
            raise ValueError("EppsPulley received an empty sample dimension")

        # (..., N, M, 1) * (P,) -> (..., N, M, P) : t * x_j for every sample j and knot.
        x_t = x.unsqueeze(-1) * self.t

        # Empirical characteristic function, averaged over the N samples:
        #   Re phi_hat(t) = mean_j cos(t x_j),  Im phi_hat(t) = mean_j sin(t x_j)
        cos_mean = torch.cos(x_t).mean(dim=-3)  # (..., M, P)
        sin_mean = torch.sin(x_t).mean(dim=-3)  # (..., M, P)

        # DDP: the ECF is a mean over samples, so it is the ECF that must be synchronised.
        cos_mean = _all_reduce(cos_mean)
        sin_mean = _all_reduce(sin_mean)

        # |phi_hat - phi|^2 with phi real.
        err = (cos_mean - self.phi).square() + sin_mean.square()  # (..., M, P)

        # Weighted quadrature then the N scaling of Eq. (Epps–Pulley).
        statistic = (err * self.weights).sum(dim=-1) * n_samples  # (..., M)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            statistic = statistic * torch.distributed.get_world_size()
        return statistic

    def extra_repr(self) -> str:
        return f"t_max={self.t_max}, n_points={self.n_points}"

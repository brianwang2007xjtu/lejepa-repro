"""Sketched Isotropic Gaussian Regularization (SIGReg).

Definition 2 of the paper::

    SIGReg_T(A, {f_theta(x_n)}_{n=1}^N) = (1/|A|) sum_{a in A} T({a^T f_theta(x_n)}_{n=1}^N)

i.e. a multivariate goodness-of-fit test towards N(0, I_K), built by (i) projecting the
K-dimensional embeddings on |A| = M random unit directions a_m in S^{K-1}, (ii) running a
univariate test T on every 1-D projection, (iii) averaging the M statistics.

Why this works, in one paragraph
--------------------------------
By the hyperspherical Cramér–Wold theorem (Lemma 3), P = Q iff a^T#P = a^T#Q for every
a in S^{K-1}.  Replacing the max over directions of Eq. (4) by a mean (to avoid sparse
gradients, §4.2), and resampling A at every optimisation step (so the cumulative number of
directions grows linearly in training time — "SGD beats the curse of dimensionality",
Fig. 7), yields a consistent test at O(N*M) cost.

Projection sampler (Algorithm 1, and `lejepa/multivariate/slicing.py`)
---------------------------------------------------------------------
*  ``A = torch.randn(K, M)``          i.i.d. standard normal entries
*  ``A /= A.norm(p=2, dim=0)``        each **column** rescaled to unit L2 norm
*  ``Y = X @ A``                      shape (N, M)

so the columns of A are uniform on the sphere S^{K-1} (a Gaussian vector normalised to unit
norm is exactly uniform on the sphere).  Sampling noise must NOT carry gradients; the
directions are constants of the optimisation step, not parameters.

Determinism / DDP
-----------------
The official implementation draws A from a dedicated ``torch.Generator`` seeded with a
synchronised ``global_step`` counter that is incremented after every forward pass.  This
gives (i) different directions at every step, exactly as the paper requires, (ii) identical
directions on every rank at a given step, and (iii) exact reproducibility from a checkpoint
(the counter is a registered buffer).  We reproduce that scheme; the plain ``torch.randn``
path (used by MINIMAL.md) is available through ``seed=None``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .epps_pulley import EppsPulley

__all__ = ["SIGReg", "sample_projection_directions"]


def sample_projection_directions(
    dim: int,
    num_slices: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Draw ``num_slices`` unit-norm directions in R^dim, returned as ``(dim, num_slices)``.

    Each column is an i.i.d. standard-normal vector divided by its L2 norm, i.e. uniform
    on the unit sphere S^{dim-1}.  Division by the norm is done in float32 to avoid
    under/overflow when the caller runs in bf16.
    """
    if dim <= 0 or num_slices <= 0:
        raise ValueError(f"dim and num_slices must be positive, got {dim}, {num_slices}")

    a = torch.randn((dim, num_slices), device=device, dtype=torch.float32, generator=generator)

    # Normalise each column: a_m <- a_m / ||a_m||_2.  Clamp guards the (probability-zero)
    # case of an all-zero draw, which would otherwise produce NaNs.
    norms = a.norm(p=2, dim=0, keepdim=True).clamp_min(torch.finfo(a.dtype).tiny)
    a = a / norms

    return a.to(dtype)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization loss.

    Parameters
    ----------
    num_slices:
        Number of random 1-D projections M = |A|.  Paper default 256 (Alg. 1) and
        recommended 1024 (§6.1 / Table 1a).
    t_max, n_points:
        Forwarded to :class:`~losses.epps_pulley.EppsPulley`.
    reduction:
        ``"mean"`` (Definition 2, the default) averages the M statistics; ``"none"``
        returns the ``(..., M)`` tensor, which the tests use to inspect individual slices.
    seed:
        If given, directions are drawn from a private generator seeded with
        ``seed + global_step``; if ``None``, the global torch RNG is used.
    """

    def __init__(
        self,
        num_slices: int = 1024,
        t_max: float = 5.0,
        n_points: int = 17,
        reduction: str = "mean",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction {reduction!r}")
        self.num_slices = int(num_slices)
        self.reduction = reduction
        self.seed = seed

        self.univariate_test = EppsPulley(t_max=t_max, n_points=n_points)

        # Step counter, synchronised across ranks and checkpointed with the module so that
        # resuming training continues the same direction sequence.
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator: Optional[torch.Generator] = None
        self._generator_device: Optional[torch.device] = None

    # ------------------------------------------------------------------ internals
    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        """Return a device-local generator reseeded with ``seed`` (cached across calls)."""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(int(seed))
        return self._generator

    def _sync_step(self) -> int:
        """Return the step counter, taking the max across DDP ranks to keep them in phase."""
        step = self.global_step
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            step = step.clone()
            torch.distributed.all_reduce(step, op=torch.distributed.ReduceOp.MAX)
        return int(step.item())

    # ------------------------------------------------------------------ forward
    def forward(self, z: Tensor) -> Tensor:
        """Compute SIGReg on embeddings ``z`` of shape ``(..., N, K)``.

        ``N`` is the number of samples used to estimate the ECF and ``K`` the embedding
        dimension.  Arbitrary leading dimensions are supported, e.g. ``(V, B, K)`` for the
        V views of a LeJEPA batch: the result is then the average over views and slices.
        """
        if z.ndim < 2:
            raise ValueError(f"expected (..., N, K) input, got shape {tuple(z.shape)}")

        dim = z.size(-1)
        n_samples = z.size(-2)
        if n_samples < 2:
            # The ECF of a single point is |e^{itx}| = 1 everywhere; the statistic would be
            # well defined but meaningless (and its gradient exactly zero).  Fail loudly.
            raise ValueError(f"SIGReg needs at least 2 samples, got {n_samples}")

        # --- 1. random directions (no gradient: they are optimisation-step constants) ---
        with torch.no_grad():
            if self.seed is None:
                a = sample_projection_directions(dim, self.num_slices, z.device, z.dtype)
            else:
                gen = self._get_generator(z.device, self.seed + self._sync_step())
                a = sample_projection_directions(dim, self.num_slices, z.device, z.dtype, gen)
            self.global_step.add_(1)

        # --- 2. project: (..., N, K) @ (K, M) -> (..., N, M) ---
        projected = z @ a

        # --- 3. univariate Epps–Pulley per slice -> (..., M) ---
        stats = self.univariate_test(projected)

        if self.reduction == "mean":
            return stats.mean()
        if self.reduction == "sum":
            return stats.sum()
        return stats

    def extra_repr(self) -> str:
        return (
            f"num_slices={self.num_slices}, t_max={self.univariate_test.t_max}, "
            f"n_points={self.univariate_test.n_points}, reduction={self.reduction}"
        )

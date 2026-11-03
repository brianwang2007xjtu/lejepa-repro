"""Projection head for LeJEPA.

The paper does not pin down the projector: Table 4 only ablates its **depth**
(1-layer / 2-layer / 3-layer).  The reference implementation (``MINIMAL.md``) uses::

    torchvision.ops.MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

Verified against ``torchvision/ops/feature_pyramid_network.py``: ``MLP`` appends
``norm_layer`` + activation only for ``hidden_channels[:-1]``; the **final** layer is
``Linear`` (+ ``Dropout``) with **no normalisation and no activation**.  Therefore
``MINIMAL.md``'s projector is

    Linear(512, 2048) -> BN -> GELU -> Linear(2048, 2048) -> BN -> GELU -> Linear(2048, D)

which is what ``hidden_norm="bn", final_norm="none", final_act="none"`` reproduces.

Why the output layer must stay linear
-------------------------------------
SIGReg drives the projector output towards ``N(0, I_K)``.  A GELU on the output layer has a
lower bound (-0.17) and rectifies, so the output distribution can never be Gaussian: the
statistic then plateaus at a strictly positive floor and the regulariser fights the
architecture instead of the collapse.  The defaults are therefore the linear output above;
``final_act="gelu"`` remains available only as an ablation (it reproduces the shape of a
misreading of ``MINIMAL.md`` and is useful to quantify the damage).

Shapes
------
``(B, embedding_dim) -> (B, projector_dim)``.  The view/batch folding used by LeJEPA
(``(N, V, ...) -> (N*V, ...) -> (V, N, ...)``) lives in :class:`LeJEPAEncoder`, not here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

__all__ = ["Projector", "LeJEPAEncoder", "build_projector"]


def _make_norm(kind: str, dim: int, use_sync_bn: bool = False) -> nn.Module:
    if kind == "none" or kind is None:
        return nn.Identity()
    if kind == "bn":
        bn = nn.BatchNorm1d(dim)
        if use_sync_bn:
            bn = nn.SyncBatchNorm(dim)
        return bn
    if kind == "ln":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm kind {kind!r}")


def _make_act(kind: str) -> nn.Module:
    if kind in ("none", None):
        return nn.Identity()
    if kind == "gelu":
        return nn.GELU()
    if kind == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"unknown activation {kind!r}")


class Projector(nn.Module):
    """MLP projector ``embedding_dim -> hidden_dims[0] -> ... -> projector_dim``.

    Parameters
    ----------
    embedding_dim:
        Backbone output width.
    projector_dim:
        Output width (the ``K`` of SIGReg).  Paper default 512; ``MINIMAL.md`` uses 16 for
        Imagenette.
    hidden_dims:
        Hidden widths applied between the input and output layers.  ``[2048, 2048]`` with
        ``projector_dim=K`` gives the 3-layer MLP of ``MINIMAL.md``.
    hidden_norm, final_norm, final_act:
        See the module docstring.
    """

    def __init__(
        self,
        embedding_dim: int,
        projector_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        hidden_norm: str = "bn",
        final_norm: str = "bn",
        final_act: str = "gelu",
        use_sync_bn: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims or [])

        self.embedding_dim = int(embedding_dim)
        self.projector_dim = int(projector_dim)

        layers: List[nn.Module] = []
        in_dim = self.embedding_dim
        for hidden in hidden_dims:
            layers += [
                nn.Linear(in_dim, hidden, bias=bias),
                _make_norm(hidden_norm, hidden, use_sync_bn),
                _make_act("gelu"),
            ]
            in_dim = hidden
        # Final layer: the "depth" quoted in Table 4 counts the linear layers, so
        # hidden_dims=[] -> 1-layer, [1024] -> 2-layer, [2048,2048] -> 3-layer.
        layers += [
            nn.Linear(in_dim, self.projector_dim, bias=bias),
            _make_norm(final_norm, self.projector_dim, use_sync_bn),
            _make_act(final_act),
        ]
        self.net = nn.Sequential(*layers)

    @property
    def depth(self) -> int:
        return sum(isinstance(m, nn.Linear) for m in self.net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, projector_dim={self.projector_dim}, "
            f"depth={self.depth}"
        )


class LeJEPAEncoder(nn.Module):
    """Backbone + projector, with the view/batch reshaping of Algorithm 2.

    Why views are handled as a *sequence*
    -------------------------------------
    LeJEPA uses 2 global views at 224 px and 6 local views at 96 px (paper §6.1).  A tensor of
    shape ``(N, V, C, H, W)`` cannot represent that, and ``torch.cat(all_views)`` in
    Algorithm 2 only works because the authors forward one resolution group at a time.  We
    therefore accept a sequence of ``V`` tensors, each ``(N, C, H, W)``, group the views by
    spatial shape, concatenate *within* each group, run one backbone call per group, and
    reassemble.  For a uniform-resolution batch (the Imagenette configuration) this collapses
    to a single backbone call, so there is no efficiency loss.

    Returns
    -------
    ``emb``  : ``(V, N, embedding_dim)`` — view-major, fed to the optional online probe.
    ``proj`` : ``(V, N, projector_dim)`` — view-major, so ``proj.mean(0)`` is the view mean,
               which is the convention of Algorithm 2 (``a_emb.view(V, bs, K)``).
    """

    def __init__(self, backbone: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    @staticmethod
    def _as_view_list(views) -> List[Tensor]:
        if torch.is_tensor(views):
            if views.ndim == 5:  # (N, V, C, H, W) — uniform resolution
                return [views[:, i] for i in range(views.shape[1])]
            if views.ndim == 4:  # (N, C, H, W) — single view
                return [views]
            raise ValueError(f"cannot interpret tensor of shape {tuple(views.shape)} as views")
        views = list(views)
        if not views:
            raise ValueError("received an empty list of views")
        n = views[0].shape[0]
        for i, v in enumerate(views):
            if v.ndim != 4:
                raise ValueError(f"view {i} must be (N, C, H, W), got {tuple(v.shape)}")
            if v.shape[0] != n:
                raise ValueError(f"view {i} has batch {v.shape[0]} != {n}")
        return views

    def forward(self, views):
        views = self._as_view_list(views)
        n_views = len(views)

        # Group views that share a spatial shape so each backbone call sees a rectangular batch.
        groups: Dict[tuple, List[int]] = {}
        for i, v in enumerate(views):
            groups.setdefault(tuple(v.shape[1:]), []).append(i)

        embs: List[Tensor] = [torch.empty(0)] * n_views
        for shape, idxs in groups.items():
            batch = torch.cat([views[i] for i in idxs], dim=0)  # (len(idxs)*N, C, H, W)
            out = self.backbone(batch)  # (len(idxs)*N, E)
            for i, chunk in zip(idxs, out.chunk(len(idxs), dim=0)):
                embs[i] = chunk

        emb = torch.stack(embs, dim=0)  # (V, N, E)
        v, n, e = emb.shape
        proj = self.projector(emb.reshape(v * n, e)).reshape(v, n, -1)  # (V, N, K)
        return emb, proj


def build_projector(cfg: dict) -> Projector:
    """Build a projector from a config dict (see ``configs/*.yaml``)."""
    return Projector(
        embedding_dim=cfg.get("embedding_dim", 512),
        projector_dim=cfg.get("projector_dim", 512),
        hidden_dims=cfg.get("hidden_dims", [2048, 2048]),
        hidden_norm=cfg.get("hidden_norm", "bn"),
        final_norm=cfg.get("final_norm", "bn"),
        final_act=cfg.get("final_act", "gelu"),
        use_sync_bn=cfg.get("use_sync_bn", False),
    )

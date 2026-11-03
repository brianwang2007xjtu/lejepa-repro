"""Vision Transformer backbone wrapper (timm) for LeJEPA.

The encoder maps an image batch ``(B, 3, H, W)`` to a ``(B, embedding_dim)`` vector.
We follow the reference implementation (``MINIMAL.md``) exactly::

    timm.create_model(model_name, pretrained=False, num_classes=embedding_dim,
                      drop_path_rate=..., img_size=...)

so the "embedding" is ``head(CLS)`` — a single learnable ``Linear(embed_dim, embedding_dim_out)``
applied to the CLS token, with ``embedding_dim_out = embedding_dim`` from the config.

Two read-out paths are exposed:

``forward``
    ``(B, 3, H, W) -> (B, embedding_dim)`` — the tensor fed to the projector during pretraining.
``forward_features``
    ``(B, 3, H, W) -> (B, 2 * encoder_dim)`` — concatenation of the CLS token of the **last
    two transformer blocks**, which is the frozen linear-probe protocol of the paper's README
    (*"concatenation of the CLS token from the last two layers"*).

No EMA/teacher copy is created here: LeJEPA has no teacher (paper §5.1, Algorithm 2).  An
optional SWA copy for the paper's §6.1 variant is built explicitly in ``engine/swa.py`` and is
off by default.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

__all__ = ["TimmBackbone", "build_backbone"]


class TimmBackbone(nn.Module):
    """Thin wrapper around a timm model exposing LeJEPA's two read-out paths.

    Parameters
    ----------
    model_name:
        Any timm model name, e.g. ``"vit_small_patch8_224"``, ``"vit_small_patch16_224"``,
        ``"vit_large_patch14_224"``.
    embedding_dim:
        Output width of :meth:`forward` (the ``num_classes`` argument of timm, used as an
        embedding width rather than a classifier width).
    img_size:
        Optional override of the positional-embedding grid resolution.
    drop_path_rate:
        Stochastic-depth rate. 0.1 in all reference configs.
    pretrained:
        Must stay False for self-supervised pretraining from scratch.
    """

    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 512,
        img_size: Optional[int] = None,
        drop_path_rate: float = 0.0,
        pretrained: bool = False,
        dynamic_img_size: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.model_name = model_name
        self.embedding_dim = int(embedding_dim)

        kwargs = dict(
            pretrained=pretrained,
            num_classes=self.embedding_dim,
            drop_path_rate=drop_path_rate,
        )
        if img_size is not None:
            kwargs["img_size"] = int(img_size)
        # LeJEPA feeds 224 px global and 96 px local views to the same encoder.  A ViT with a
        # fixed positional grid rejects the second shape, so the positional embedding must be
        # interpolated per input resolution (`dynamic_img_size`).  This is standard practice
        # for multi-crop SSL (DINO, I-JEPA) and does not add parameters.
        if dynamic_img_size:
            kwargs["dynamic_img_size"] = True

        self.encoder: nn.Module = timm.create_model(model_name, **kwargs)

        # Width of a single transformer block output (before the head).
        self.encoder_dim: int = int(getattr(self.encoder, "embed_dim", self.embedding_dim))
        self.has_class_token: bool = bool(getattr(self.encoder, "class_token", True))
        self._blocks: Optional[nn.ModuleList] = getattr(self.encoder, "blocks", None)

    # ------------------------------------------------------------------ helpers
    def _pool(self, tokens: Tensor) -> Tensor:
        """CLS token if the model has one, otherwise the mean patch token."""
        if self.has_class_token:
            return tokens[:, 0]
        return tokens.mean(dim=1)

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor) -> Tensor:
        """``(B, 3, H, W) -> (B, embedding_dim)``."""
        return self.encoder(x)

    @torch.no_grad()
    def forward_features(self, x: Tensor) -> Tensor:
        """``(B, 3, H, W) -> (B, 2 * encoder_dim)``: CLS of the last two blocks, concatenated.

        This is the paper's frozen-evaluation read-out.  Falls back to the model's own
        pooled features for architectures without ``.blocks`` (e.g. CNNs).
        """
        if self._blocks is None or len(self._blocks) < 2:
            feats = self.encoder.forward_features(x)
            if feats.ndim == 4:  # (B, C, H, W) conv feature map
                feats = feats.mean(dim=(2, 3))
            elif feats.ndim == 3:  # (B, N, C) token sequence
                feats = self._pool(feats)
            return feats

        captured: List[Tensor] = []

        def hook(_module, _inputs, output):
            captured.append(self._pool(output))

        handles = [
            self._blocks[-2].register_forward_hook(hook),
            self._blocks[-1].register_forward_hook(hook),
        ]
        try:
            self.encoder.forward_features(x)
        finally:
            for h in handles:
                h.remove()

        if len(captured) != 2:  # pragma: no cover - defensive
            raise RuntimeError(f"expected 2 captured block outputs, got {len(captured)}")
        return torch.cat(captured, dim=-1)

    # ------------------------------------------------------------------ misc
    def freeze(self) -> "TimmBackbone":
        """Disable gradients (used by the frozen-backbone evaluations)."""
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze(self) -> "TimmBackbone":
        for p in self.parameters():
            p.requires_grad_(True)
        return self

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"model_name={self.model_name}, embedding_dim={self.embedding_dim}, "
            f"encoder_dim={self.encoder_dim}, params={self.num_parameters/1e6:.1f}M"
        )


def build_backbone(cfg: dict) -> TimmBackbone:
    """Build a backbone from a config dict (see ``configs/*.yaml``)."""
    return TimmBackbone(
        model_name=cfg["model_name"],
        embedding_dim=cfg.get("embedding_dim", 512),
        img_size=cfg.get("img_size"),
        drop_path_rate=cfg.get("drop_path_rate", 0.0),
        pretrained=cfg.get("pretrained", False),
        dynamic_img_size=cfg.get("dynamic_img_size", True),
    )

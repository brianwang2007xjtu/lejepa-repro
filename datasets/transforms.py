"""Multi-view augmentation pipelines.

Two recipes are implemented, and they are **not** interchangeable — they come from
different sources in the paper's ecosystem:

``"lejepa"`` (default)
    The ImageNet-1K recipe of the official README ("Data augmentation and views"):
    every view receives the *same* photometric transforms, and views differ only by
    their RandomResizedCrop resolution and scale,

        global: RandomResizedCrop(224, scale=(0.30, 1.0))
        local : RandomResizedCrop( 98, scale=(0.05, 0.30))
        both  : HFlip(p=.5), ColorJitter(.4,.4,.2,.1) p=.8, Gray(p=.2),
                GaussianBlur p=.5, Solarize(128) p=.2, Normalize

``"minimal"``
    The single-stack recipe of ``MINIMAL.md`` used for the Imagenette reference run:
    ``RandomResizedCrop(128, scale=(0.08,1.0))``, ColorJitter(.8,.8,.8,.2) p=.8,
    Gray(.2), GaussianBlur(7,(0.1,2.0)) p=.5, Solarize(128) p=.2, HFlip, Normalize.
    All V views use the identical pipeline (``V_g = V``).

The evaluation transform is the standard ``Resize(s) -> CenterCrop(res) -> Normalize``.
Nothing here re-implements normalisation statistics: ImageNet mean/std are used for every
dataset, as in the reference code.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = [
    "build_view_transform",
    "build_eval_transform",
    "MultiViewTransform",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]


def _to_tensor_norm() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def build_view_transform(
    resolution: int,
    scale: Tuple[float, float],
    recipe: str = "lejepa",
    color_jitter: Optional[Sequence[float]] = None,
    blur_p: float = 0.5,
    solarize_p: float = 0.2,
    grayscale_p: float = 0.2,
) -> v2.Compose:
    """Build one augmentation stack (a single view of an image)."""
    if recipe == "lejepa":
        color_jitter = color_jitter or (0.4, 0.4, 0.2, 0.1)
        ops: List = [
            v2.RandomResizedCrop(resolution, scale=scale, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(*color_jitter)], p=0.8),
            v2.RandomGrayscale(p=grayscale_p),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=blur_p),
            v2.RandomApply([v2.RandomSolarize(threshold=128.0)], p=solarize_p),
            _to_tensor_norm(),
        ]
    elif recipe == "minimal":
        # MINIMAL.md: one stack reused for every view; RandomApply default p = 0.5.
        ops = [
            v2.RandomResizedCrop(resolution, scale=scale, antialias=True),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128.0)], p=0.2),
            v2.RandomHorizontalFlip(p=0.5),
            _to_tensor_norm(),
        ]
    else:
        raise ValueError(f"unknown recipe {recipe!r}")
    return v2.Compose(ops)


def build_eval_transform(resolution: int = 224, resize: Optional[int] = None) -> v2.Compose:
    """Deterministic evaluation transform used by linear probe / kNN."""
    resize = resize or int(round(resolution * 256 / 224))
    return v2.Compose(
        [
            v2.Resize(resize, antialias=True),
            v2.CenterCrop(resolution),
            _to_tensor_norm(),
        ]
    )


class MultiViewTransform:
    """Produce ``V = V_g + V_l`` views of one PIL image as a ``(V, C, H, W)`` float tensor.

    The first ``n_global`` views use the global transform, the remaining ``n_local`` views
    the local one — the index convention that LeJEPA's loss relies on (§5.1: *"we set the
    first 1..V_g indices of each z_{n,v} as the global views"*).
    """

    def __init__(
        self,
        n_global: int,
        n_local: int,
        global_resolution: int = 224,
        local_resolution: int = 96,
        global_scale: Tuple[float, float] = (0.3, 1.0),
        local_scale: Tuple[float, float] = (0.05, 0.3),
        recipe: str = "lejepa",
    ) -> None:
        self.n_global = int(n_global)
        self.n_local = int(n_local)
        self.n_views = self.n_global + self.n_local

        self.global_tf = build_view_transform(global_resolution, global_scale, recipe=recipe)
        if self.n_local > 0:
            self.local_tf = build_view_transform(local_resolution, local_scale, recipe=recipe)
        else:
            self.local_tf = None

    def __call__(self, img) -> Tuple[Tensor, ...]:
        """Return a tuple of V tensors, each (C, H, W).

        A *tuple* rather than a stacked tensor because global (224 px) and local (96 px) views
        have different spatial shapes; ``torch.stack`` would fail.  The default DataLoader
        collate turns this into a list of V batched tensors, which the encoder groups by
        spatial shape.
        """
        views = [self.global_tf(img) for _ in range(self.n_global)]
        if self.n_local > 0:
            views += [self.local_tf(img) for _ in range(self.n_local)]
        return tuple(views)

    def __repr__(self) -> str:
        return (
            f"MultiViewTransform(n_global={self.n_global}, n_local={self.n_local}, "
            f"n_views={self.n_views})"
        )

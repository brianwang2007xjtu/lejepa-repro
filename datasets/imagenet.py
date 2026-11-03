"""Datasets for LeJEPA pretraining and evaluation.

Every pretraining dataset returns ``(views, label)`` where ``views`` has shape
``(V, 3, H, W)`` and ``label`` is an ``int64`` scalar, so a default ``DataLoader`` collates
to ``(B, V, 3, H, W)`` and ``(B,)`` — the layout consumed by
:class:`models.projector.LeJEPAEncoder`.

Supported sources
-----------------
``imagenette``   ImageNette (10 classes), the dataset of ``MINIMAL.md``.
``imagenet100``  ImageNet-100 (100 classes), the paper's "inet100" benchmark (Table 4/7).
``imagenet``     ImageNet-1K from an ImageFolder tree.
``synthetic``    Random tensors — used by Experiment A (pipeline debug, no real data).
``fake_images``  Random-in-memory PIL images through the *real* augmentation pipeline, so the
                 data-loading path is exercised without any dataset on disk.

All data is read lazily through ``torch.utils.data.Dataset``; nothing is ever moved to the
GPU in bulk.  Labels are returned by the pretraining loaders only to feed the optional
online probe / diagnostics and are **never** used in the self-supervised objective.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from .transforms import MultiViewTransform, build_eval_transform

__all__ = [
    "MultiViewImageFolder",
    "SyntheticViewDataset",
    "FakeImageDataset",
    "build_pretrain_dataset",
    "build_eval_dataset",
    "build_pretrain_loader",
    "build_eval_loader",
    "resolve_split_dir",
]

_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "validation", "valid", "test"),
}


def resolve_split_dir(root: str, split: str) -> str:
    """Find the directory holding ``split`` inside ``root``.

    Handles the layouts produced by the usual ImageNet/Imagenette/ImageNet-100 archives::

        root/{train,val}/class/*.JPEG        (split at depth 1)
        root/{train,test}/class/*.JPEG
        root/class/*.JPEG                    (split already resolved)
    """
    candidates = _SPLIT_ALIASES.get(split, (split,))
    for name in candidates:
        for cand in (os.path.join(root, name),):
            if os.path.isdir(cand):
                return cand
    # maybe ``root`` already *is* the split directory
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if subdirs and not any(os.path.isdir(os.path.join(root, d, subdirs[0])) for d in subdirs):
        return root
    raise FileNotFoundError(
        f"could not find split {split!r} under {root!r}; tried {candidates}. "
        f"Contents: {subdirs[:10]}"
    )


class MultiViewImageFolder(Dataset):
    """ImageFolder that returns V augmented views of every image."""

    def __init__(
        self,
        root: str,
        transform: MultiViewTransform,
        split: str = "train",
        max_samples: Optional[int] = None,
        **folder_kwargs,
    ) -> None:
        self.split_dir = resolve_split_dir(root, split)
        self.base = ImageFolder(self.split_dir, **folder_kwargs)
        self.transform = transform
        self.max_samples = max_samples
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx

    def __len__(self) -> int:
        n = len(self.base)
        return min(n, self.max_samples) if self.max_samples else n

    def __getitem__(self, index: int):
        img, label = self.base[index]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)


class SyntheticViewDataset(Dataset):
    """Random-tensor dataset for pipeline debugging (Experiment A)."""

    def __init__(
        self,
        length: int = 256,
        num_classes: int = 10,
        n_views: int = 4,
        resolution: int = 64,
        channels: int = 3,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.num_classes = num_classes
        self.n_views = n_views
        self.shape = (n_views, channels, resolution, resolution)
        # Fixed per-index noise so that "samples" are stable across epochs (a constantly
        # regenerating random dataset would hide data-pipeline bugs).
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randn((length, *self.shape), generator=g)
        self.labels = torch.randint(0, num_classes, (length,), generator=g)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]


class FakeImageDataset(Dataset):
    """Random PIL images pushed through the real multi-view augmentation pipeline.

    Used to validate augmentation, collation and the full train step without touching the
    filesystem.  ``structured=True`` draws images with a class-dependent colour so that a
    probe can in principle learn something (useful as a smoke test for the evaluation code).
    """

    def __init__(
        self,
        length: int = 512,
        num_classes: int = 10,
        transform: Optional[MultiViewTransform] = None,
        size: int = 160,
        structured: bool = True,
        seed: int = 0,
    ) -> None:
        from PIL import Image

        self.length = length
        self.num_classes = num_classes
        self.transform = transform
        self.size = size
        self.structured = structured
        self._Image = Image
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (length,), generator=g)
        # one base colour per class
        self.colours = torch.randint(32, 224, (num_classes, 3), generator=g)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        import numpy as np

        label = int(self.labels[index])
        rng = np.random.default_rng(index)
        if self.structured:
            base = self.colours[label].to(torch.int64).numpy()
            arr = np.clip(
                base[None, None, :] + rng.normal(0, 18, (self.size, self.size, 3)), 0, 255
            ).astype("uint8")
        else:
            arr = rng.integers(0, 256, (self.size, self.size, 3), dtype="uint8")
        img = self._Image.fromarray(arr, mode="RGB")
        views = self.transform(img) if self.transform is not None else img
        return views, torch.tensor(label, dtype=torch.long)


# --------------------------------------------------------------------------- factories
def build_pretrain_dataset(cfg: dict, transform: MultiViewTransform) -> Dataset:
    """Instantiate a pretraining dataset from a config dict."""
    name = cfg["name"].lower()
    if name == "synthetic":
        return SyntheticViewDataset(
            length=cfg.get("length", 256),
            num_classes=cfg.get("num_classes", 10),
            n_views=transform.n_views,
            resolution=cfg.get("resolution", 64),
            seed=cfg.get("seed", 0),
        )
    if name == "fake_images":
        return FakeImageDataset(
            length=cfg.get("length", 512),
            num_classes=cfg.get("num_classes", 10),
            transform=transform,
            size=cfg.get("size", 160),
            structured=cfg.get("structured", True),
            seed=cfg.get("seed", 0),
        )
    root = cfg["root"]
    return MultiViewImageFolder(
        root=root,
        transform=transform,
        split=cfg.get("split", "train"),
        max_samples=cfg.get("max_samples"),
    )


def build_eval_dataset(cfg: dict, resolution: int = 224) -> Dataset:
    """Instantiate a single-view evaluation dataset (one deterministic view per image)."""
    tf = build_eval_transform(resolution, cfg.get("resize", None))
    name = cfg["name"].lower()
    if name == "synthetic":
        return SyntheticViewDataset(
            length=cfg.get("length", 256),
            num_classes=cfg.get("num_classes", 10),
            n_views=1,
            resolution=cfg.get("resolution", resolution),
            seed=cfg.get("seed", 1000),
        )
    if name == "fake_images":
        return FakeImageDataset(
            length=cfg.get("length", 256),
            num_classes=cfg.get("num_classes", 10),
            transform=tf,
            size=cfg.get("size", resolution),
            structured=cfg.get("structured", True),
            seed=cfg.get("seed", 1000),
        )
    return ImageFolder(resolve_split_dir(cfg["root"], cfg.get("split", "val")), transform=tf)


def build_pretrain_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 8,
    seed: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """DataLoader with worker-persistent, per-worker seeding.

    Random seeds are made a deterministic function of the global seed so that a run is
    reproducible from its config alone, while different workers get different streams.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    def worker_init(worker_id: int) -> None:
        import random

        import numpy as np

        s = seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_init,
        generator=g,
    )


def build_eval_loader(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

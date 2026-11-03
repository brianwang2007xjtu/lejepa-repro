"""Optimiser and learning-rate schedule.

Paper / reference settings (see ``docs/AUDIT.md`` §3):

* AdamW everywhere.
* ``lr = 5e-4`` for ImageNet-1K ViT, ``2e-3`` for the Imagenette reference run.
* ``weight_decay = 5e-2`` for ViTs, ``5e-4`` for ResNets, ``1e-7`` for the online probe.
* **Constant** weight decay — the paper is explicit that the only scheduled quantity is the
  learning rate ("no hyper-parameter schedulers"), and the README states the final LR is
  ``initial_lr / 1000``.
* Linear warm-up then cosine annealing.

The schedule is expressed as a per-group multiplier so that the encoder and the (optional)
online probe can follow different rules, mirroring ``MINIMAL.md``'s two parameter groups with
``SequentialLR([LinearLR, CosineAnnealingLR])``.

Reproducing ``MINIMAL.md`` exactly
----------------------------------
``MINIMAL.md`` builds ``CosineAnnealingLR(opt, T_max=..., eta_min=1e-3)`` with an encoder peak
LR of ``2e-3``, i.e. the encoder LR decays only to 50 % of its peak — *not* to 1/1000.
Because that is the code behind the published 90.7 % Imagenette number, the value is exposed
as ``min_lr_ratio`` and the Imagenette config sets it to 0.5.  All other configs use 1e-3
(the README's ``initial_lr / 1000``).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

__all__ = ["build_param_groups", "build_optimizer", "build_scheduler", "describe_optimizer"]


def build_param_groups(
    module: nn.Module,
    lr: float,
    weight_decay: float,
    skip_params: Optional[Iterable[str]] = None,
) -> List[Dict]:
    """Split parameters into decay / no-decay groups.

    Following standard practice (and the reference implementation, which relies on
    ``torchvision.ops.MLP``), parameters of normalisation layers and all biases are excluded
    from weight decay.  This is the only "free" convention we adopt; it is not a change to
    any paper hyper-parameter.
    """
    skip = tuple(skip_params or ())
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if any(s in name for s in skip):
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay, "name": "decay"})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": "no_decay"})
    if not groups:
        raise ValueError("no trainable parameters found")
    return groups


def build_optimizer(
    param_groups: List[Dict],
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> AdamW:
    return AdamW(param_groups, betas=betas, eps=eps)


def _warmup_cosine_lambda(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
    start_factor: float = 0.01,
) -> float:
    """Multiplier in ``[min_lr_ratio * start_factor, 1]``.

    * ``step < warmup_steps``: linear from ``start_factor`` to 1.
    * afterwards: cosine from 1 down to ``min_lr_ratio``.
    """
    if warmup_steps > 0 and step < warmup_steps:
        alpha = step / max(1, warmup_steps)
        return start_factor + (1.0 - start_factor) * alpha
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 1e-3,
    start_factor: float = 0.01,
    group_min_lr_ratios: Optional[Dict[str, float]] = None,
) -> LambdaLR:
    """Per-group warm-up + cosine schedule.

    ``group_min_lr_ratios`` maps a parameter-group ``name`` to its own final/peak ratio, so
    the online probe can keep a constant LR while the encoder decays.

    Implementation note (this used to be a bug)
    -------------------------------------------
    ``LambdaLR`` calls ``step()`` *inside* ``__init__``, which immediately rewrites every
    ``param_group["lr"]`` to ``lr * lr_lambda(0) = lr * start_factor``.  Any wrapper that
    reads ``base_lr`` from ``param_groups`` *after* construction therefore captures
    ``lr * 0.01`` and silently trains with a 100x too small learning rate.

    We avoid that entirely by handing PyTorch one lambda *per group* (a documented feature of
    ``LambdaLR``) instead of post-hoc patching the group LRs.
    """
    ratios: List[float] = []
    for group in optimizer.param_groups:
        name = group.get("name", "")
        if group_min_lr_ratios and name in group_min_lr_ratios:
            ratios.append(group_min_lr_ratios[name])
        else:
            ratios.append(min_lr_ratio)

    def make_lambda(ratio: float):
        def lr_lambda(step: int) -> float:
            return _warmup_cosine_lambda(
                step, warmup_steps, total_steps, ratio, start_factor
            )

        return lr_lambda

    return LambdaLR(optimizer, lr_lambda=[make_lambda(r) for r in ratios])


def describe_optimizer(optimizer: AdamW) -> str:
    lines = []
    for i, g in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in g["params"])
        lines.append(
            f"  group {i} ({g.get('name', '?')}): {n/1e6:.2f}M params, "
            f"lr={g['lr']:.2e}, wd={g['weight_decay']:.2e}"
        )
    return "\n".join(lines)

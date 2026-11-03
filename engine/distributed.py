"""Distributed-training helpers (DDP scaffolding; single-GPU is the default path).

The training code is written so that the *same* script runs unchanged on 1 GPU or N GPUs
launched with ``torchrun``.  Nothing in the algorithm special-cases the world size: SIGReg's
ECF is all-reduced (see ``losses/epps_pulley.py``), BatchNorm can be swapped for
SyncBatchNorm, and the direction sampler is seeded with a globally-synchronised step counter.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

__all__ = [
    "init_distributed",
    "cleanup_distributed",
    "is_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "wrap_ddp",
    "convert_sync_batchnorm",
    "barrier",
    "all_reduce_mean",
]


def init_distributed(backend: Optional[str] = None) -> tuple[int, int, int]:
    """Initialise the process group from ``torchrun`` environment variables.

    Returns ``(rank, world_size, local_rank)``.  When ``RANK``/``WORLD_SIZE`` are absent the
    function is a no-op returning ``(0, 1, 0)``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return 0, 1, 0

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def wrap_ddp(module: nn.Module, local_rank: int = 0, find_unused_parameters: bool = False) -> nn.Module:
    """Wrap ``module`` in DDP when a process group exists, else return it unchanged."""
    if not is_distributed():
        return module
    device_ids = [local_rank] if torch.cuda.is_available() else None
    return DDP(
        module,
        device_ids=device_ids,
        output_device=device_ids[0] if device_ids else None,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=True,
    )


def convert_sync_batchnorm(module: nn.Module) -> nn.Module:
    """Replace BatchNorm with SyncBatchNorm (needed when the projector uses BN + DDP)."""
    if not is_distributed():
        return module
    return nn.SyncBatchNorm.convert_sync_batchnorm(module)


def barrier() -> None:
    if is_distributed():
        dist.barrier()


@torch.no_grad()
def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """Average a scalar/vector tensor across ranks (used for logging)."""
    if not is_distributed():
        return x
    out = x.clone().detach()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / get_world_size()

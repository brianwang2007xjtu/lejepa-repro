"""Train LeJEPA from a YAML config.

Usage
-----
    python scripts/train.py --config configs/imagenette.yaml
    python scripts/train.py --config configs/imagenette.yaml --set epochs=5 batch_size=64
    python scripts/train.py --config configs/imagenette.yaml --resume auto
    torchrun --nproc_per_node=4 scripts/train.py --config configs/imagenet_vit.yaml

``--set key=value`` overrides any leaf of the config (dotted paths allowed, e.g.
``--set sigreg.num_slices=256``).  Values are parsed as YAML, so ``lambd=0.05`` becomes a
float and ``true`` a bool.

Everything about a run is determined by its config + seed; the resolved config is written to
``<run_dir>/config.json`` next to the metrics so an experiment can always be re-launched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeJEPA pretraining")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    p.add_argument("--resume", default=None, help="'auto' | 'last' | path to a checkpoint")
    p.add_argument("--run-name", default=None, help="override run_name")
    p.add_argument("--output-dir", default=None, help="override output_dir")
    p.add_argument("--dry-run", action="store_true", help="build everything, run 3 steps, exit")
    return p.parse_args()


def load_config(path: str, overrides: list[str]) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)

    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    cfg.setdefault("output_dir", "checkpoints")
    cfg.setdefault("run_name", os.path.splitext(os.path.basename(args.config))[0])
    if args.dry_run:
        cfg["epochs"] = 1
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["length"] = cfg["data"].get("length", 64)
        cfg["batch_size"] = min(cfg["batch_size"], 8)
        cfg["log_every"] = 1

    from engine.distributed import cleanup_distributed, init_distributed
    from engine.trainer import Trainer

    rank, world_size, local_rank = init_distributed()

    from engine.distributed import is_main_process

    if is_main_process():
        print(f"config: {json.dumps(cfg, indent=2, default=str)}")

    trainer = Trainer(cfg)

    if args.resume:
        path = args.resume
        if path in ("auto", "last"):
            path = os.path.join(trainer.run_dir, "checkpoints", "last.pt")
        if os.path.exists(path):
            trainer.load_checkpoint(path)
        elif is_main_process():
            print(f"[resume] {path} not found; starting from scratch")

    if args.dry_run:
        trainer.run_dir = os.path.join(trainer.output_dir, cfg["run_name"] + "_dryrun")
        if is_main_process():
            os.makedirs(trainer.run_dir, exist_ok=True)
            _dry_run(trainer)
        cleanup_distributed()
        return

    trainer.train()
    cleanup_distributed()


def _dry_run(trainer) -> None:
    """Forward + backward + optimiser step + checkpoint + restore, on 3 real batches."""
    from engine.trainer import MetricsLogger

    trainer.logger = MetricsLogger(trainer.run_dir, enable_tensorboard=False)
    it = iter(trainer.train_loader)
    for i in range(3):
        views, labels = next(it)
        metrics, _ = trainer.train_step(views, labels)
        print(f"  step {i}: " + ", ".join(f"{k}={float(v):.5f}" for k, v in metrics.items()))
        trainer.logger.log(metrics, i)

    path = os.path.join(trainer.run_dir, "checkpoints", "dryrun.pt")
    trainer.save_checkpoint(path)
    print(f"  saved {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    step_before = trainer.state.step
    gs_before = int(trainer.sigreg.global_step)
    trainer.load_checkpoint(path)
    assert trainer.state.step == step_before, "step counter did not round-trip"
    assert int(trainer.sigreg.global_step) == gs_before, "SIGReg step counter did not round-trip"
    print(f"  restored: step={trainer.state.step} sigreg_step={int(trainer.sigreg.global_step)}")

    metas = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"  peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    trainer.logger.close()


if __name__ == "__main__":
    main()

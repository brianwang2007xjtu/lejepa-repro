"""Evaluate a LeJEPA checkpoint: frozen-backbone linear probe + k-NN.

Usage
-----
    python scripts/evaluate.py --checkpoint checkpoints/in100_vits16/checkpoints/final.pt \
        --data-root /root/autodl-tmp/data/imagenet100 --data-name imagenet100 \
        --num-classes 100 --feature-source last2cls --probe-epochs 100

Feature protocols
-----------------
``last2cls``   (default) concatenation of the CLS token of the last two blocks, then a
               LayerNorm inside the probe — the official README protocol.
``embedding``  the backbone's ``embedding_dim`` output, i.e. what ``MINIMAL.md`` probes.

The results (accuracy, wall-clock, feature dimension, parameter count) are written to
``<checkpoint_dir>/eval_<protocol>.json`` so they can be collected into the comparison table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Frozen-backbone evaluation of LeJEPA")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--data-name", default="imagenette")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--feature-source", default="last2cls", choices=["last2cls", "embedding"])
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--precision", default="bf16")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"checkpoint: {args.checkpoint}")
    print(f"  pretrained for {ckpt['state']['epoch']} epochs, step {ckpt['state']['step']}")
    print(f"  lambda={cfg.get('lambd')} views={cfg['data']['n_global_views']}+"
          f"{cfg['data']['n_local_views']} sigreg={cfg.get('sigreg')}")

    from models.backbone import build_backbone
    from datasets.imagenet import build_eval_dataset, build_eval_loader
    from evaluation.linear_probe import run_evaluation

    backbone = build_backbone(cfg["model"]).to(device)
    # Checkpoints store the *flat* state_dict of LeJEPAEncoder, so the backbone weights are
    # the entries prefixed with "backbone." (the projector is not needed for evaluation).
    bb_state = {
        k[len("backbone.") :]: v
        for k, v in ckpt["model"].items()
        if k.startswith("backbone.")
    }
    if not bb_state:
        raise SystemExit(f"no 'backbone.*' keys in checkpoint; got {list(ckpt['model'])[:5]}")
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}")
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}")
    print(f"  loaded {len(bb_state)} backbone tensors")
    backbone.freeze()
    backbone.eval()
    n_params = backbone.num_parameters
    print(f"  backbone={cfg['model']['model_name']} ({n_params/1e6:.1f}M params)")

    dc = {
        "name": args.data_name,
        "root": args.data_root,
        "num_classes": args.num_classes,
    }
    train_ds = build_eval_dataset({**dc, "split": "train"}, resolution=args.resolution)
    val_ds = build_eval_dataset({**dc, "split": "val"}, resolution=args.resolution)
    print(f"  train={len(train_ds)} val={len(val_ds)} res={args.resolution}")

    train_loader = build_eval_loader(train_ds, args.batch_size, args.num_workers)
    val_loader = build_eval_loader(val_ds, args.batch_size, args.num_workers)

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    results = run_evaluation(
        backbone,
        train_loader,
        val_loader,
        num_classes=args.num_classes,
        device=device,
        feature_source=args.feature_source,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        knn_k=args.knn_k,
        max_batches=args.max_val_batches,
        verbose=True,
    )

    results.update(
        {
            "checkpoint": os.path.abspath(args.checkpoint),
            "pretrain_epochs": ckpt["state"]["epoch"],
            "backbone": cfg["model"]["model_name"],
            "backbone_params": n_params,
            "feature_source": args.feature_source,
            "dataset": args.data_name,
            "resolution": args.resolution,
            "lambd": cfg.get("lambd"),
            "n_views": cfg["data"]["n_global_views"] + cfg["data"]["n_local_views"],
        }
    )

    print("\n=== RESULTS ===")
    for k in ("linear_top1", "linear_top5", "knn_acc", "feat_dim"):
        if k in results:
            print(f"  {k:16s}: {results[k]}")
    print(f"  {'linear (s)':16s}: {results.get('linear_probe_seconds', 0):.1f}")
    print(f"  {'extract (s)':16s}: {results.get('feature_extract_seconds', 0):.1f}")

    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        f"eval_{args.feature_source}_{args.data_name}.json",
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""Backbone-space diagnostics for a checkpoint (CPU-friendly, small sample).

Why this exists: the trainer monitors the *projector* output, but the linear probe reads
*backbone* CLS features.  A run can therefore show a perfectly isotropic projector
distribution (eff. rank ~16/16) while the backbone features are useless.  This script measures
the space that the evaluation actually uses.

    python scripts/diagnose_checkpoints.py --data-root ... --n 2000 ckpt1 ckpt2 ...
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def eff_rank(z):
    z = z.float()
    zc = z - z.mean(0, keepdim=True)
    cov = zc.T @ zc / max(1, z.size(0) - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum())), float(ev.max() / ev.min()), float(z.std(0).min())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/imagenet100")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results/backbone_diagnostics.json")
    args = ap.parse_args()

    from models.backbone import build_backbone
    from datasets.imagenet import build_eval_dataset
    from datasets.transforms import build_eval_transform
    from torch.utils.data import DataLoader

    ds = build_eval_dataset({"name": "imagenet100", "root": args.data_root,
                             "split": "train", "num_classes": 100}, resolution=args.resolution)
    # build_eval_dataset returns a plain torchvision ImageFolder, which stores the file list
    # in `.samples`; truncating it gives a cheap, fixed diagnostic subset.
    ds.samples = ds.samples[: args.n]
    ds.targets = ds.targets[: args.n]
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)

    out = {}
    for ck in args.checkpoints:
        payload = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        bb = build_backbone(cfg["model"])
        sd = {k[len("backbone."):]: v for k, v in payload["model"].items() if k.startswith("backbone.")}
        bb.load_state_dict(sd, strict=False)
        bb.eval().float()

        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                feats.append(bb.forward_features(x))
                labels.append(y)
        f = torch.cat(feats); lab = torch.cat(labels)
        er, cond, stdmin = eff_rank(f)
        # class separability: how much of the total variance is between-class
        mu = f.mean(0, keepdim=True)
        tot = ((f - mu) ** 2).sum()
        betw = torch.zeros_like(tot)
        for c in lab.unique():
            fc = f[lab == c]
            betw = betw + fc.size(0) * ((fc.mean(0) - mu.squeeze(0)) ** 2).sum()
        out[os.path.basename(os.path.dirname(os.path.dirname(ck)))] = {
            "feat_dim": f.size(1), "effective_rank": er, "cond_number": cond,
            "min_feature_std": stdmin, "between_class_var_ratio": float(betw / tot),
        }
        print(f"{ck}: eff_rank={er:.2f}/{f.size(1)} cond={cond:.1f} "
              f"std_min={stdmin:.3f} between-class var ratio={float(betw/tot):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)

if __name__ == "__main__":
    main()

"""Micro-benchmark: measure step time / memory for a given config (no real data needed).

    python scripts/benchmark.py --config configs/imagenet100_vits16.yaml --batch-size 128 --steps 20
"""
from __future__ import annotations
import argparse, os, sys, time
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    bs = args.batch_size or cfg["batch_size"]
    dc = cfg["data"]
    v, rg, rl = dc["n_global_views"] + dc["n_local_views"], dc["global_resolution"], dc["local_resolution"]

    from models.backbone import build_backbone
    from models.projector import build_projector, LeJEPAEncoder
    from losses.sigreg import SIGReg
    from losses.invariance import LeJEPAObjective

    dev = torch.device("cuda")
    enc = LeJEPAEncoder(build_backbone(cfg["model"]), build_projector(cfg["projector"])).to(dev)
    sig = SIGReg(num_slices=cfg["sigreg"]["num_slices"], t_max=cfg["sigreg"]["t_max"]).to(dev)
    obj = LeJEPAObjective(sig, lambd=cfg["lambd"], n_global_views=dc["n_global_views"]).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg["optim"]["lr"])

    views = [torch.randn(bs, 3, rg, rg, device=dev) for _ in range(dc["n_global_views"])]
    views += [torch.randn(bs, 3, rl, rl, device=dev) for _ in range(dc["n_local_views"])]
    x = views   # list of (B,3,H,W), possibly different resolutions

    torch.cuda.reset_peak_memory_stats()
    ts = []
    for i in range(args.steps):
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, proj = enc(x)
            loss = obj(proj)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    warm = ts[args.warmup:]
    mean = sum(warm) / len(warm)
    print(f"bs={bs} views={v} res={rg}/{rl}  mean={mean*1000:.1f} ms  "
          f"({bs/mean:.1f} samples/s)  peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  -> 100 epochs on 126,689 images: {100*126689/bs*mean/3600:.2f} h")

if __name__ == "__main__":
    main()

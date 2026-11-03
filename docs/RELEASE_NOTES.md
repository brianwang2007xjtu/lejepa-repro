# LeJEPA Reproduction — Pretrained Checkpoints

Weights for the runs reported in [`REPRODUCTION_REPORT.md`](../REPRODUCTION_REPORT.md).

Attach all eight `.pt` files plus `manifest.json` to a GitHub release (each is ~109 MB, well
under GitHub's 2 GB per-asset limit). This file doubles as the release description.

These are **slim** checkpoints: model weights + the exact training config + run state.
The AdamW optimiser state, scheduler state and RNG snapshots have been stripped, which takes
each file from 312 MB to ~109 MB. They are therefore for **evaluation / inspection**, not for
resuming training. To resume training, use the full checkpoint produced by the training run
(identical to these plus optimiser state).

## Files

| file | run | epochs | frozen linear probe (IN-100) |
|---|---|---|---|
| `expC_in100_vits16_8view.pt` | **Exp C** — 8 views (2×224 + 6×96), ViT-S/16, λ=0.05, bs 192 | 50 | **69.02 %** top-1 / 91.00 % top-5 |
| `expB_in100_vits16_128.pt` | **Exp B** — 4 views @128, ViT-S/16, λ=0.02, bs 256 | 40 | **49.52 %** top-1 / 78.88 % top-5 |
| `abl_base_lamb0.02.pt` | ablation baseline, λ=0.02 | 12 | 30.92 % |
| `abl_lamb0.05.pt` | ablation, λ=0.05 | 12 | 30.38 % |
| `abl_slices64.pt` | ablation, M=64 slices (baseline 256) | 12 | 28.64 % |
| `abl_lamb0.005.pt` | ablation, λ=0.005 (partial dimensional collapse) | 12 | 27.24 % |
| `abl_lamb0.2.pt` | ablation, λ=0.2 (distribution healthy, features useless) | 12 | 4.88 % |
| `abl_views2.pt` | ablation, V=2 views (baseline 4) | 12 | 3.24 % |

All runs: ImageNet-100, ViT-S/16, bf16, AdamW, linear warm-up + cosine.
Ablation numbers carry **±2–3 pt run-to-run noise** at the 12-epoch budget — see
`docs/ABLATIONS.md`. Only the λ=0.2 and V=2 failures are large enough to be conclusive.

## Loading

```python
import torch
from models.backbone import build_backbone

ckpt = torch.load("expC_in100_vits16_8view.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["config"]              # the exact config of the run (see configs/*.yaml)

backbone = build_backbone(cfg["model"])
state = {k[len("backbone."):]: v for k, v in ckpt["model"].items()
         if k.startswith("backbone.")}
backbone.load_state_dict(state)
backbone.eval()

# frozen linear probe read-out: CLS of the last two blocks, concatenated (dim 768)
feats = backbone.forward_features(images)      # (B, 3, H, W) -> (B, 768)
```

The full `LeJEPAEncoder` (backbone + projector) is `ckpt["model"]` verbatim, so
`models.projector.LeJEPAEncoder(...).load_state_dict(ckpt["model"])` also works.

**Resolution matters.** Exp C was trained with 2 global views at 224 px and 6 local views at
96 px; evaluate it at 224. Exp B and the ablations were trained at 128 px; evaluate at 128.

## Verifying the download

`manifest.json` lists size and a truncated SHA-256 for every file:

```bash
shasum -a 256 *.pt | cut -c1-16
```

## What these weights are not

They are **not** an ImageNet-1K reproduction. ImageNet-1K could not be materialised on the
reproduction host (the train archive alone is 147.9 GB; the writable volume was 50 GB) and
ViT-L/14 at batch 512 × 8 views does not fit on a 24–32 GB GPU. Exp B and Exp C are
ImageNet-100 runs with reduced schedules. See the report's §3 for the three-tier
classification (full / approximate / algorithmic-only).

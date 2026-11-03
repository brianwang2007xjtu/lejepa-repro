# Datasets

## ImageNet-100 (used for Experiments B and C)

The public `imagenet100.zip` (14.4 GB, at `/root/autodl-pub/ImageNet100/`) contains
**only the training split** — 100 class folders, 126 689 JPEGs, no validation set.

Standard SSL practice (and the only way to stay comparable with the paper's "inet100"
numbers) is to take the ImageNet-1K **validation** set and keep the images belonging to the
100 selected classes. `scripts/prepare_in100.py` does this:

```bash
python scripts/prepare_in100.py \
  --raw  /root/autodl-tmp/data/in100_raw/imagenet100 \
  --out  /root/autodl-tmp/data/imagenet100 \
  --val-tar /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar \
  --devkit  /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz
```

Result:

```
/root/autodl-tmp/data/imagenet100/
├── train/  100 classes, 126 689 images
└── val/    100 classes,   5 000 images   (50 per class)
```

Implementation notes worth knowing:

* `ILSVRC2012_devkit_t12/data/meta.mat` contains **1860** synsets, of which the **first 1000**
  are the classification classes in exactly the order used by
  `ILSVRC2012_validation_ground_truth.txt` (1-based indices). The remaining 860 are auxiliary
  WordNet entries for the detection task and must be dropped.
* The val tar is streamed **once**; only members whose class is in the 100 selected wnids are
  written, so extraction costs ≈700 MB rather than 6.7 GB.

## Imagenette (not used — archive unreachable)

`MINIMAL.md` uses ImageNette (`frgfm/imagenette`, 160 px). The `fast-ai` S3 archive is not
reachable from this host (`s3.amazonaws.com` returns 307 to a blocked endpoint, the download
times out) and the HuggingFace dataset repo only mirrors metadata, not image bytes
(`huggingface.co` unreachable; `hf-mirror.com` serves the API but the data points at S3).
ImageNette is also a 10-class *subset of ImageNet-1K*, and extracting 10 classes out of the
147.9 GB train tar would require a full sequential scan of a network-mounted archive.

Consequence: Experiment B uses ImageNet-100 with the recipe hyper-parameters of `MINIMAL.md`
(λ=0.02, proj_dim=16, lr=2e-3, single augmentation stack, V=4 all-global) instead of
ImageNette, so the published **90.7 %** number is *not* directly comparable. This is recorded
as a limitation, not silently substituted.

## ImageNet-1K (not possible here)

| Requirement | Available |
|---|---|
| ≈150 GB disk to extract train+val (train tar alone is 147.9 GB) | 50 GB writable |
| batch 512 × 8 views × 238 px ViT-L/14 in bf16 ≈ 60 GB activations | 24 GB |
| paper schedule: 100 epochs, ≥8 GPUs for a comparable wall-clock | 1 GPU |

`configs/imagenet_vit.yaml` documents the exact paper configuration and is intentionally
runnable only on adequate hardware.

## Data loading contract

A pretraining dataset returns `(views, label)` where `views` is a **tuple of V tensors**, each
`(3, H, W)`. A tuple — not a stacked tensor — because global (224 px) and local (96/98 px)
views have different spatial shapes. The default `DataLoader` collate turns this into a list
of V batched tensors, and `LeJEPAEncoder` groups views by spatial shape before calling the
backbone. Labels are used **only** by the optional online probe, never by the SSL objective.

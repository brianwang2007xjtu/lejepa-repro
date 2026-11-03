# Environment

Measured on 2026-10-30 on a rented AutoDL container.

## Hardware

| Item | Value |
|---|---|
| GPU | **1 × NVIDIA GeForce RTX 4090 D, 24 564 MiB** (sm_89, Ada) |
| Driver / CUDA runtime | 595.71.05 / CUDA 13.2 |
| CPU | Intel Xeon Platinum 8352S, `nproc` = 128 |
| **CPU quota (cgroup v2)** | `cpu.max = 1600000 100000` → **16 effective cores** |
| RAM | 503 GB |
| Writable disk | `/root/autodl-tmp` = **50 GB** (XFS quota), `/` overlay = 30 GB |
| Read-only shared data | `/root/autodl-pub` (4 TB AutoFS) |

The 16-core cgroup quota matters: with `num_workers=48` the dataloader thrashes and step time
*degrades* (0.94 s vs 0.29 s). 16 workers is optimal for the augmentation stack used here.

## Software

| Package | Version |
|---|---|
| Python | 3.12.3 (`/root/miniconda3`) |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| timm | 1.0.30 |
| numpy / scipy | 2.3.2 / 1.18.1 |
| einops | 0.8.2 |
| tensorboard | 2.20.0 |
| PyYAML | 6.0.2 |
| pytest | (installed for this project) |

`nvcc` is absent; irrelevant, PyTorch bundles its own CUDA runtime. `pip` is configured to the
aliyun mirror (`/etc/pip.conf`).

## Reproduce the environment

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm numpy PyYAML tensorboard Pillow
pip install pytest scipy            # tests only
```

## Measured throughput (single 4090, bf16, bs as noted)

| Config | step time | peak GPU mem | 100 epochs on IN-100 |
|---|---|---|---|
| ViT-S/16, 4 views @128, bs 256 | 0.29 s | 12.5 GB | ≈ 2.3 h |
| ViT-S/16, 2@224 + 6@96, bs 128 | 0.19 s (compute only) | 14.3 GB | ≈ 5.2 h |
| ViT-S/8, 4 views @128, bs 256 | **OOM** | > 24 GB | — |

The ViT-S/8 row is the reason the Imagenette reference config (`configs/imagenette.yaml`) is
kept as documentation but not run: 4 views × 256 images × 256 patch tokens does not fit in
24 GB. `MINIMAL.md`'s "single GPU, ~2 h" therefore implies a ≥40 GB card.

## Disk budget

ImageNet-1K train tar is 147.9 GB and needs ≈150 GB extracted; the only writable volume is
50 GB. **ImageNet-1K cannot be materialised here** — this is the binding constraint on the
reproduction tier, not GPU FLOPs.

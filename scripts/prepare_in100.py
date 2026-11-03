"""Build the ImageNet-100 directory tree (train + val) used by the experiments.

The public ``imagenet100.zip`` archive contains **only the 126,689 training images** — there
is no validation split.  The standard protocol (used by every SSL paper that reports
ImageNet-100) is to take the ImageNet-1K *validation* set and keep the images belonging to
the 100 selected classes.  This script does exactly that:

1. move ``<raw>/imagenet100`` -> ``<out>/train`` (ImageFolder layout, 100 class dirs);
2. stream ``ILSVRC2012_img_val.tar`` once and extract only the members whose class is in the
   100 selected wnids, into ``<out>/val/<wnid>/``.

The class index -> wnid mapping is read from the official devkit (``meta.mat``), so no
hard-coded ordering assumptions are made.

    python scripts/prepare_in100.py \
        --raw /root/autodl-tmp/data/in100_raw/imagenet100 \
        --out /root/autodl-tmp/data/imagenet100 \
        --val-tar /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar \
        --devkit /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile


def load_wnid_order(devkit_tar: str, workdir: str) -> list[str]:
    """Return the 1000 wnids in the devkit's class-index order (1-based indices in val.txt)."""
    import scipy.io

    os.makedirs(workdir, exist_ok=True)
    with tarfile.open(devkit_tar, "r:gz") as tf:
        tf.extractall(workdir)
    meta_path = os.path.join(workdir, "ILSVRC2012_devkit_t12", "data", "meta.mat")
    meta = scipy.io.loadmat(meta_path, squeeze_me=True, struct_as_record=False)
    synsets = meta["synsets"]
    # The devkit's meta.mat contains 1860 synsets, of which the FIRST 1000 are the
    # ImageNet-1K classification classes in exactly the order used by
    # ILSVRC2012_validation_ground_truth.txt (1-based class indices).  The remaining 860
    # are auxiliary WordNet entries used by the detection task and must be dropped.
    all_wnids = [str(s.WNID) for s in synsets]
    wnids = all_wnids[:1000]
    assert len(wnids) == 1000, f"expected 1000 classification synsets, got {len(wnids)}"
    assert len(set(wnids)) == 1000, "duplicate wnids in the first 1000 synsets"
    return wnids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="extracted imagenet100/ directory")
    ap.add_argument("--out", required=True, help="destination root (train/ val/ live here)")
    ap.add_argument("--val-tar", required=True)
    ap.add_argument("--devkit", required=True)
    ap.add_argument("--workdir", default="/root/autodl-tmp/data/_devkit")
    ap.add_argument("--limit-per-class", type=int, default=50,
                    help="max val images per class (50 gives the standard 5000-image split)")
    args = ap.parse_args()

    train_out = os.path.join(args.out, "train")
    if os.path.isdir(args.raw):
        classes = sorted(
            d for d in os.listdir(args.raw) if os.path.isdir(os.path.join(args.raw, d))
        )
        print(f"[1/4] found {len(classes)} class directories in {args.raw}")
        if len(classes) != 100:
            print(f"  WARNING: expected 100 classes, found {len(classes)}")
        os.makedirs(args.out, exist_ok=True)
        if os.path.isdir(train_out):
            raise SystemExit(f"{train_out} already exists; refusing to overwrite")
        print(f"[2/4] moving {args.raw} -> {train_out}")
        shutil.move(args.raw, train_out)
    elif os.path.isdir(train_out):
        # idempotent re-run: the train split was already moved into place
        classes = sorted(
            d for d in os.listdir(train_out) if os.path.isdir(os.path.join(train_out, d))
        )
        print(f"[1/4] raw dir absent; reusing existing train split with "
              f"{len(classes)} classes at {train_out}")
        print("[2/4] train split already in place")
    else:
        raise SystemExit(f"neither {args.raw} nor {train_out} exists")

    # ---- class-index -> wnid
    wnids = load_wnid_order(args.devkit, args.workdir)
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}
    selected = {w: wnid_to_idx[w] for w in classes if w in wnid_to_idx}
    missing = [w for w in classes if w not in wnid_to_idx]
    if missing:
        raise SystemExit(f"classes not present in the devkit: {missing[:5]}")
    print(f"[3/4] mapped {len(selected)} classes to ImageNet-1K indices")

    gt_path = os.path.join(args.workdir, "ILSVRC2012_devkit_t12", "data",
                           "ILSVRC2012_validation_ground_truth.txt")
    with open(gt_path) as f:
        gt = [int(line) - 1 for line in f.read().split()]  # 0-based class index
    print(f"  validation ground truth: {len(gt)} entries")

    # ---- stream the val tar exactly once, extracting only the selected classes
    val_out = os.path.join(args.out, "val")
    os.makedirs(val_out, exist_ok=True)
    for w in selected:
        os.makedirs(os.path.join(val_out, w), exist_ok=True)

    per_class: dict[str, int] = {w: 0 for w in selected}
    idx_to_wnid = {i: w for w, i in selected.items()}

    written = 0
    with tarfile.open(args.val_tar, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            stem = os.path.splitext(os.path.basename(member.name))[0]
            try:
                n = int(stem.split("_")[-1]) - 1  # ILSVRC2012_val_00000001.JPEG -> 0
            except ValueError:
                continue
            cls_idx = gt[n]
            w = idx_to_wnid.get(cls_idx)
            if w is None:
                continue
            if per_class[w] >= args.limit_per_class:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            dst = os.path.join(val_out, w, os.path.basename(member.name))
            with open(dst, "wb") as out:
                shutil.copyfileobj(f, out)
            per_class[w] += 1
            written += 1
            if written % 500 == 0:
                print(f"    extracted {written} val images ...", flush=True)
            if all(v >= args.limit_per_class for v in per_class.values()):
                break

    print(f"[4/4] wrote {written} validation images to {val_out}")
    counts = sorted(per_class.values())
    print(f"  per-class counts: min={counts[0]} max={counts[-1]}")
    n_train = sum(len(os.listdir(os.path.join(train_out, c))) for c in classes)
    print(f"  total train images: {n_train}")


if __name__ == "__main__":
    main()

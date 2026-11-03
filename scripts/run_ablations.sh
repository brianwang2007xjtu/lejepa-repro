#!/usr/bin/env bash
# Sequential training-ablation sweep.  Each run = train (short schedule) + frozen linear probe.
#
# Usage:  nohup scripts/run_ablations.sh > logs/ablations.out 2>&1 &
#
# Design notes
# ------------
# * Base recipe is the *cheap* Experiment-B configuration (ViT-S/16 @128, 4 views, bs 256,
#   ~82 s/epoch) because a 15-epoch ablation only makes sense if it is affordable; the
#   multi-view Experiment C recipe costs 4.5 min/epoch.
# * 15 epochs per run is SHORT.  Differences below ~2-3 points of linear-probe accuracy are
#   inside the run-to-run noise at this budget and must not be over-interpreted.  The sweep is
#   designed to expose *large* effects (e.g. lambda = 0), not to rank close variants.
# * Exactly ONE factor changes per run; everything else is pinned to the baseline.
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/lejepa-reproduction

EPOCHS=${EPOCHS:-15}
OUT=results/ablations
mkdir -p "$OUT" logs

run_one () {
  local name="$1"; shift
  local overrides="$*"
  echo "=== [$name] overrides: $overrides ==="
  if [ -f "checkpoints/$name/checkpoints/final.pt" ]; then
    echo "  already done, skipping training"
  else
    python scripts/train.py --config configs/in100_vits16_128.yaml \
      --set epochs="$EPOCHS" save_every=999 log_every=40 eval.enabled=false num_workers=16 \
      $overrides --run-name "$name" > "logs/$name.out" 2>&1
    if [ $? -ne 0 ]; then echo "  TRAIN FAILED"; tail -5 "logs/$name.out"; return 1; fi
  fi
  python -u scripts/evaluate.py \
    --checkpoint "checkpoints/$name/checkpoints/final.pt" \
    --data-root /root/autodl-tmp/data/imagenet100 --data-name imagenet100 \
    --num-classes 100 --resolution 128 --feature-source last2cls \
    --probe-epochs 60 --batch-size 256 --num-workers 16 \
    --output "$OUT/${name}.json" > "logs/${name}_eval.out" 2>&1
  python - "$name" "$OUT/${name}.json" "$overrides" <<'PY'
import json, sys
name, path, ov = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(path))
rec = {"name": name, "overrides": ov, "epochs": None,
       "linear_top1": r["linear_top1"], "linear_top5": r["linear_top5"],
       "knn_acc": r["knn_acc"], "feat_dim": r["feat_dim"]}
with open("results/ablations/sweep.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"  {name}: linear_top1={rec['linear_top1']:.2f}  knn={rec['knn_acc']:.2f}")
PY
}

# ---------------- baseline (lambda = 0.02, the MINIMAL.md default)
run_one abl_base_lamb0.02        "lambd=0.02"

# ---------------- lambda sweep (paper Figure 8 / Table 7)
run_one abl_lamb0.005            "lambd=0.005"
run_one abl_lamb0.05             "lambd=0.05"
run_one abl_lamb0.2              "lambd=0.2"

# ---------------- number of slices: lower bound only (paper Table 1a / §4.3)
#   abl_slices1024 was DROPPED: the paper reports only +0.6 pt for 1024 vs 256 on IN-1K,
#   which is inside this sweep's noise, and scripts/ablation_statistics.py already quantifies
#   the variance reduction exactly (rel. std 23% at M=16 -> 1.5% at M=4096).
run_one abl_slices64             "sigreg.num_slices=64"

# ---------------- number of views (paper Table 1b)
run_one abl_views2               "data.n_global_views=2 projector.projector_dim=16"

# DROPPED for the B-scope sweep (see docs/ABLATIONS.md):
#   abl_proj512  -- paper Table 1d already ranks projector widths; +-2-3 pt noise here
#                   cannot resolve the ~1.5 pt difference.
#   abl_tmax5    -- answered exactly and for free by scripts/ablation_statistics.py:
#                   quadrature error vs closed-form MMD is 0.007% at t_max=5 and 0.64% at
#                   t_max=3; the paper's Table 1a shows a 0.1 pt accuracy difference.

echo "=== ALL ABLATIONS DONE ==="
python - <<'PY'
import json, os
p = "results/ablations/sweep.jsonl"
if os.path.exists(p):
    rows = [json.loads(l) for l in open(p)]
    print(f"{'run':28s} {'top1':>7s} {'top5':>7s} {'knn':>7s}")
    for r in sorted(rows, key=lambda x: -x["linear_top1"]):
        print(f"{r['name']:28s} {r['linear_top1']:7.2f} {r['linear_top5']:7.2f} {r['knn_acc']:7.2f}")
PY

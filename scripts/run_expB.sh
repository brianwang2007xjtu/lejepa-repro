#!/usr/bin/env bash
# Launch Experiment B (ImageNet-100, ViT-S/16 @128, V=4, 40 epochs) detached from the shell.
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/lejepa-reproduction
mkdir -p logs

RUN=expB_in100_vits16_128
if pgrep -f "run_name_placeholder_${RUN}" > /dev/null 2>&1; then
  echo "already running"; exit 0
fi

rm -rf "checkpoints/${RUN}"
setsid nohup python scripts/train.py \
  --config configs/in100_vits16_128.yaml \
  --set epochs=40 save_every=10 eval.enabled=true eval.every=5 num_workers=24 \
  --run-name "${RUN}" > "logs/${RUN}.out" 2>&1 < /dev/null &
echo "launched pid $! -> logs/${RUN}.out"

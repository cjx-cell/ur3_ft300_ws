#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pi0-env
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/ubuntu/ur3_ft300_ws
OUTDIR="/home/ubuntu/ur3_ft300_ws/ai-models/samoe/v_$(date +%m%d_%H%M)"
exec python -m lerobot.scripts.lerobot_train \
  --policy.path=/tmp/sa_moe_config \
  --dataset.repo_id=cjx-cell/ur3_peg_in_hole \
  --batch_size=1 \
  --output_dir="$OUTDIR" \
  --policy.push_to_hub=false \
  2>&1 | tee /tmp/sa_moe_train.log

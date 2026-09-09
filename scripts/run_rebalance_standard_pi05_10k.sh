#!/bin/bash
# Fine-tune the corrected Pi0.5 baseline with episode-start frame weighting.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
PYTHON_ENV="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
SOURCE_CHECKPOINT="$WS_DIR/outputs/train/pi05_relative_h50_global_task_continue_10k/checkpoints/010000/pretrained_model"
DATASET_ROOT="$WS_DIR/pap_moe_framework/datasets/lerobot_overfit_5ep_relative_h50_global_task"
OUTDIR="$WS_DIR/outputs/train/pi05_relative_h50_global_task_rebalanced_10k"

if [ ! -f "$SOURCE_CHECKPOINT/model.safetensors" ]; then
  echo "ERROR: source checkpoint is missing: $SOURCE_CHECKPOINT" >&2
  exit 2
fi
if [ ! -f "$DATASET_ROOT/GLOBAL_TASK_VIEW.json" ]; then
  echo "ERROR: global-task dataset view is missing: $DATASET_ROOT" >&2
  exit 2
fi
if [ -e "$OUTDIR" ]; then
  echo "ERROR: output directory already exists: $OUTDIR" >&2
  echo "Refusing to overwrite or silently resume it." >&2
  exit 2
fi

export PYTHONPATH=/home/ubuntu/lerobot/src
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "================================================================="
echo "  Pi0.5 Initial-Motion Rebalanced Fine-Tuning"
echo "  Source:   $SOURCE_CHECKPOINT"
echo "  Dataset:  $DATASET_ROOT"
echo "  Steps:    10000"
echo "  Batch:    1"
echo "  Sampling: frame_index < 50 receives 10x weight"
echo "            expected startup-frame share: ~46%"
echo "  Output:   $OUTDIR"
echo "================================================================="

cd "$WS_DIR"
exec "$PYTHON_ENV" -m lerobot.scripts.lerobot_train \
  --policy.path="$SOURCE_CHECKPOINT" \
  --dataset.repo_id=pap_moe/ur3_peg_in_hole_overfit_5ep_relative_h50_global_task \
  --dataset.root="$DATASET_ROOT" \
  --batch_size=1 \
  --steps=10000 \
  --initial_frame_sampling_count=50 \
  --initial_frame_sampling_weight=10.0 \
  --num_workers=0 \
  --save_freq=2500 \
  --log_freq=20 \
  --output_dir="$OUTDIR" \
  --policy.push_to_hub=false

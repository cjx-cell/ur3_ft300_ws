#!/usr/bin/env bash
# Targeted startup-frame rebalance from the optimized-data 10k baseline.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
SOURCE_CHECKPOINT="$WS_DIR/outputs/train/pi05_v9_absolute_1000step_20260804_201520/checkpoints/010000/pretrained_model"
DATASET_ROOT="$WS_DIR/pap_moe_framework/datasets/lerobot_v9_stagegate_pilot_9ep_global_task_absolute"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$WS_DIR/outputs/train/pi05_v9_absolute_startup_rebalanced_3k_$RUN_TAG"
LOG_FILE="$WS_DIR/artifacts/pi05_v9_absolute_startup_rebalanced_3k_$RUN_TAG.log"

for required in \
  "$SOURCE_CHECKPOINT/model.safetensors" \
  "$SOURCE_CHECKPOINT/policy_preprocessor.json" \
  "$DATASET_ROOT/meta/info.json"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 2
  fi
done

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LEROBOT_REBUILD_PROCESSORS || true

echo "Pi0.5 v9 startup-frame rebalanced fine-tuning"
echo "  initialization: $SOURCE_CHECKPOINT"
echo "  dataset:        $DATASET_ROOT"
echo "  startup rule:   frame_index < 25, weight=10"
echo "  updates:        3000"
echo "  peak LR:        1e-5"
echo "  output:         $OUTPUT_DIR"
echo "  log:            $LOG_FILE"

cd "$WS_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path="$SOURCE_CHECKPOINT" \
  --policy.optimizer_lr=1e-5 \
  --dataset.repo_id=pap_moe/pi05_v9_absolute_global_task \
  --dataset.root="$DATASET_ROOT" \
  --batch_size=1 \
  --steps=3000 \
  --initial_frame_sampling_count=25 \
  --initial_frame_sampling_weight=10 \
  --num_workers=0 \
  --save_checkpoint=true \
  --save_freq=1500 \
  --log_freq=50 \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=false \
  2>&1 | tee "$LOG_FILE"

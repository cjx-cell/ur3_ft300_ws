#!/usr/bin/env bash
# Closed-loop-oriented Pi0.5 action-expert repair with semantic phase balance.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
SOURCE_CHECKPOINT="${PI05_SOURCE_CHECKPOINT:-$WS_DIR/outputs/train/pi05_v9_absolute_30000step_20260804_231244/checkpoints/030000/pretrained_model}"
DATASET_ROOT="${PI05_DATASET_ROOT:-$WS_DIR/pap_moe_framework/datasets/lerobot_v9_ground_train_24ep_absolute}"
STEPS="${PI05_ACTION_STEPS:-20000}"
BATCH_SIZE="${PI05_ACTION_BATCH_SIZE:-1}"
SAVE_FREQ="${PI05_ACTION_SAVE_FREQ:-2500}"
PEAK_LR="${PI05_ACTION_LR:-1e-5}"
GLOBAL_TASK="pick up the peg and insert it into the hole"

for value in "$STEPS" "$BATCH_SIZE" "$SAVE_FREQ"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: steps, batch size, and save frequency must be positive integers" >&2
    exit 2
  fi
done
for required in \
  "$SOURCE_CHECKPOINT/model.safetensors" \
  "$SOURCE_CHECKPOINT/policy_preprocessor.json" \
  "$DATASET_ROOT/meta/info.json" \
  "$DATASET_ROOT/meta/tasks.parquet"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 2
  fi
done

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$WS_DIR/outputs/train/pi05_v9_semantic_rebalanced_action_${STEPS}step_$RUN_TAG"
LOG_FILE="$WS_DIR/artifacts/pi05_v9_semantic_rebalanced_action_${STEPS}step_$RUN_TAG.log"

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# The source checkpoint predates global_task. Rebuild with current dataset
# statistics so the saved processor contains the fixed deployment prompt.
export LEROBOT_REBUILD_PROCESSORS=1

mkdir -p "$WS_DIR/artifacts" "$WS_DIR/outputs/train"
echo "Pi0.5 v9 semantic-rebalanced action repair"
echo "  source:     $SOURCE_CHECKPOINT"
echo "  dataset:    $DATASET_ROOT"
echo "  prompt:     $GLOBAL_TASK"
echo "  sampling:   8 semantic phases, balanced"
echo "  startup:    frame_index < 50 receives 5x within phase"
echo "  steps:      $STEPS"
echo "  batch:      $BATCH_SIZE"
echo "  peak LR:    $PEAK_LR"
echo "  output:     $OUTPUT_DIR"
echo "  log:        $LOG_FILE"

cd "$WS_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path="$SOURCE_CHECKPOINT" \
  --policy.global_task="$GLOBAL_TASK" \
  --policy.optimizer_lr="$PEAK_LR" \
  --dataset.repo_id=pap_moe/pi05_v9_semantic_rebalanced_action \
  --dataset.root="$DATASET_ROOT" \
  --semantic_task_balanced_sampling=true \
  --semantic_task_count=8 \
  --initial_frame_sampling_count=50 \
  --initial_frame_sampling_weight=5 \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers=0 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=50 \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=false \
  2>&1 | tee "$LOG_FILE"

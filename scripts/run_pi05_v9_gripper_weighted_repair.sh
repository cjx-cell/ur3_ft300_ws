#!/usr/bin/env bash
# Short, targeted Pi0.5 repair for the binary gripper action dimension.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
SOURCE_CHECKPOINT="${PI05_SOURCE_CHECKPOINT:-$WS_DIR/outputs/train/pi05_v9_semantic_rebalanced_action_20000step_20260808_103251/checkpoints/020000/pretrained_model}"
DATASET_ROOT="${PI05_DATASET_ROOT:-$WS_DIR/pap_moe_framework/datasets/lerobot_v9_ground_train_24ep_absolute}"
STEPS="${PI05_REPAIR_STEPS:-2500}"
SAVE_FREQ="${PI05_REPAIR_SAVE_FREQ:-2500}"
SAVE_CHECKPOINT="${PI05_REPAIR_SAVE_CHECKPOINT:-true}"
BATCH_SIZE="${PI05_REPAIR_BATCH_SIZE:-1}"
PEAK_LR="${PI05_REPAIR_PEAK_LR:-2e-6}"
DECAY_LR="${PI05_REPAIR_DECAY_LR:-5e-7}"
WARMUP_STEPS="${PI05_REPAIR_WARMUP_STEPS:-100}"
GRIPPER_DIM_WEIGHT="${PI05_GRIPPER_DIM_WEIGHT:-4.0}"
GRIPPER_OPEN_WEIGHT="${PI05_GRIPPER_OPEN_WEIGHT:-2.0}"
GLOBAL_TASK="pick up the peg and insert it into the hole"

for value in "$STEPS" "$SAVE_FREQ" "$BATCH_SIZE" "$WARMUP_STEPS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: steps, save frequency, batch size and warmup must be positive integers" >&2
    exit 2
  fi
done
if [[ "$SAVE_CHECKPOINT" != "true" && "$SAVE_CHECKPOINT" != "false" ]]; then
  echo "ERROR: PI05_REPAIR_SAVE_CHECKPOINT must be true or false" >&2
  exit 2
fi
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

RUN_TAG="${PI05_REPAIR_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$WS_DIR/outputs/train/pi05_v9_gripper_weighted_repair_${STEPS}step_$RUN_TAG"
LOG_FILE="$WS_DIR/artifacts/pi05_v9_gripper_weighted_repair_${STEPS}step_$RUN_TAG.log"

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_REBUILD_PROCESSORS=1

mkdir -p "$WS_DIR/artifacts" "$WS_DIR/outputs/train"
echo "Pi0.5 v9 gripper-weighted targeted repair"
echo "  source:              $SOURCE_CHECKPOINT"
echo "  dataset:             $DATASET_ROOT"
echo "  prompt:              $GLOBAL_TASK"
echo "  sampling:            8 semantic phases balanced; startup 5x"
echo "  gripper index:       6 (normalized open <= 0)"
echo "  gripper dim weight:  $GRIPPER_DIM_WEIGHT"
echo "  gripper open weight: $GRIPPER_OPEN_WEIGHT"
echo "  steps:               $STEPS"
echo "  LR:                  $PEAK_LR -> $DECAY_LR; warmup=$WARMUP_STEPS"
echo "  output:              $OUTPUT_DIR"
echo "  log:                 $LOG_FILE"

cd "$WS_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path="$SOURCE_CHECKPOINT" \
  --policy.global_task="$GLOBAL_TASK" \
  --policy.gripper_action_index=6 \
  --policy.gripper_loss_weight="$GRIPPER_DIM_WEIGHT" \
  --policy.gripper_open_loss_weight="$GRIPPER_OPEN_WEIGHT" \
  --policy.gripper_open_threshold_normalized=0.0 \
  --policy.optimizer_lr="$PEAK_LR" \
  --policy.scheduler_decay_lr="$DECAY_LR" \
  --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$STEPS" \
  --dataset.repo_id=pap_moe/pi05_v9_gripper_weighted_repair \
  --dataset.root="$DATASET_ROOT" \
  --semantic_task_balanced_sampling=true \
  --semantic_task_count=8 \
  --initial_frame_sampling_count=50 \
  --initial_frame_sampling_weight=5 \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers=0 \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=50 \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=false \
  2>&1 | tee "$LOG_FILE"

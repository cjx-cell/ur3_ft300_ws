#!/usr/bin/env bash
# Train a conservative release/retract detector on the frozen 20k Pi0.5 policy.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
SOURCE_CHECKPOINT="${PI05_SOURCE_CHECKPOINT:-$WS_DIR/outputs/train/pi05_v9_semantic_rebalanced_action_20000step_20260808_103251/checkpoints/020000/pretrained_model}"
DATASET_ROOT="${PI05_DATASET_ROOT:-$WS_DIR/pap_moe_framework/datasets/lerobot_v9_ground_train_24ep_absolute}"
STEPS="${PI05_RELEASE_STEPS:-500}"
SAVE_FREQ="${PI05_RELEASE_SAVE_FREQ:-500}"
BATCH_SIZE="${PI05_RELEASE_BATCH_SIZE:-1}"
PEAK_LR="${PI05_RELEASE_PEAK_LR:-3e-4}"
DECAY_LR="${PI05_RELEASE_DECAY_LR:-3e-5}"
WARMUP_STEPS="${PI05_RELEASE_WARMUP_STEPS:-50}"
POSITIVE_WEIGHT="${PI05_RELEASE_POSITIVE_WEIGHT:-4.0}"
THRESHOLD="${PI05_RELEASE_THRESHOLD:-0.8}"
PHASE_START_COUNT="${PI05_RELEASE_PHASE_START_COUNT:-10}"
PHASE_START_WEIGHT="${PI05_RELEASE_PHASE_START_WEIGHT:-10.0}"
GLOBAL_TASK="pick up the peg and insert it into the hole"

for value in \
  "$STEPS" "$SAVE_FREQ" "$BATCH_SIZE" "$WARMUP_STEPS" "$PHASE_START_COUNT"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: steps, save frequency, batch size and warmup must be positive integers" >&2
    exit 2
  fi
done
for required in \
  "$SOURCE_CHECKPOINT/model.safetensors" \
  "$DATASET_ROOT/meta/info.json" \
  "$DATASET_ROOT/meta/tasks.parquet"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 2
  fi
done

RUN_TAG="${PI05_RELEASE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$WS_DIR/outputs/train/pi05_v9_release_override_${STEPS}step_$RUN_TAG"
LOG_FILE="$WS_DIR/artifacts/pi05_v9_release_override_${STEPS}step_$RUN_TAG.log"

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_REBUILD_PROCESSORS=1

mkdir -p "$WS_DIR/artifacts" "$WS_DIR/outputs/train"
echo "Pi0.5 v9 release-only gripper override training"
echo "  source:           $SOURCE_CHECKPOINT"
echo "  dataset:          $DATASET_ROOT"
echo "  prompt:           $GLOBAL_TASK"
echo "  online inputs:    VLM + current/fast/slow force + state history"
echo "  trainable:        release head only"
echo "  sampling:         8 semantic phases exactly balanced"
echo "  boundary boost:   first $PHASE_START_COUNT x $PHASE_START_WEIGHT"
echo "  positive weight:  $POSITIVE_WEIGHT"
echo "  threshold:        $THRESHOLD"
echo "  steps:            $STEPS"
echo "  LR:               $PEAK_LR -> $DECAY_LR; warmup=$WARMUP_STEPS"
echo "  output:           $OUTPUT_DIR"
echo "  log:              $LOG_FILE"

cd "$WS_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path="$SOURCE_CHECKPOINT" \
  --policy.global_task="$GLOBAL_TASK" \
  --policy.gripper_action_index=6 \
  --policy.use_release_gripper_override=true \
  --policy.train_release_head_only=true \
  --policy.release_head_hidden_dim=256 \
  --policy.release_head_positive_weight="$POSITIVE_WEIGHT" \
  --policy.release_head_probability_threshold="$THRESHOLD" \
  --policy.optimizer_lr="$PEAK_LR" \
  --policy.scheduler_decay_lr="$DECAY_LR" \
  --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$STEPS" \
  --dataset.repo_id=pap_moe/pi05_v9_release_override \
  --dataset.root="$DATASET_ROOT" \
  --semantic_task_balanced_sampling=true \
  --semantic_task_count=8 \
  --semantic_phase_start_sampling_count="$PHASE_START_COUNT" \
  --semantic_phase_start_sampling_weight="$PHASE_START_WEIGHT" \
  --initial_frame_sampling_count=0 \
  --initial_frame_sampling_weight=1 \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers=0 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=25 \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=false \
  2>&1 | tee "$LOG_FILE"

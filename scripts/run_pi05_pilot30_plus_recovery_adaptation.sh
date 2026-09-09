#!/usr/bin/env bash
# Fine-tune the pure Pi0.5 action expert on pilot30 + validated recovery data.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
SOURCE_CHECKPOINT="${PI05_SOURCE_CHECKPOINT:-$WS_DIR/outputs/train/pi05_v9_absolute_1000step_20260818_233017/checkpoints/010000/pretrained_model}"
DATASET_ROOT="${PI05_RECOVERY_DATASET:-$WS_DIR/pap_moe_framework/datasets/lerobot_pi05_pilot30_plus_recovery_v4_global_prompt_full}"
STEPS="${PI05_RECOVERY_TRAIN_STEPS:-3000}"
BATCH_SIZE="${PI05_RECOVERY_BATCH_SIZE:-1}"
PEAK_LR="${PI05_RECOVERY_PEAK_LR:-2.5e-5}"
DECAY_LR="${PI05_RECOVERY_DECAY_LR:-2.5e-6}"
SCHEDULER_DECAY_STEPS="${PI05_RECOVERY_SCHEDULER_DECAY_STEPS:-$STEPS}"
SCHEDULER_WARMUP_STEPS="${PI05_RECOVERY_SCHEDULER_WARMUP_STEPS:-1000}"
SAVE_FREQ="${PI05_RECOVERY_SAVE_FREQ:-500}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PI05_RECOVERY_OUTPUT_DIR:-$WS_DIR/outputs/train/pi05_pilot30_plus_recovery_v1_$RUN_TAG}"
LOG_FILE="$WS_DIR/artifacts/pi05_pilot30_plus_recovery_train_$RUN_TAG.log"
CONFIG_PATH="$SOURCE_CHECKPOINT/train_config.json"
SKILL_TRANSITION_WINDOW="${PI05_SKILL_TRANSITION_WINDOW:-0}"
SKILL_TRANSITION_WEIGHT="${PI05_SKILL_TRANSITION_WEIGHT:-1.0}"
SKILL_TRANSITION_FROM="${PI05_SKILL_TRANSITION_FROM:-}"
SKILL_TRANSITION_TO="${PI05_SKILL_TRANSITION_TO:-}"
PRESERVE_SOURCE_PROCESSOR_STATS="${PI05_PRESERVE_SOURCE_PROCESSOR_STATS:-false}"
GRIPPER_TRANSITION_WINDOW="${PI05_GRIPPER_TRANSITION_WINDOW:-0}"
GRIPPER_TRANSITION_WEIGHT="${PI05_GRIPPER_TRANSITION_WEIGHT:-1.0}"
INITIAL_FRAME_COUNT="${PI05_INITIAL_FRAME_COUNT:-0}"
INITIAL_FRAME_WEIGHT="${PI05_INITIAL_FRAME_WEIGHT:-1.0}"
RECOVERY_EPISODE_START_INDEX="${PI05_RECOVERY_EPISODE_START_INDEX:-}"
RECOVERY_START_FRAME_COUNT="${PI05_RECOVERY_START_FRAME_COUNT:-0}"
RECOVERY_START_FRAME_WEIGHT="${PI05_RECOVERY_START_FRAME_WEIGHT:-1.0}"
DATASET_TOLERANCE_S="${PI05_DATASET_TOLERANCE_S:-0.001}"
TRAIN_SEED="${PI05_TRAIN_SEED:-1000}"
PEFT_RANK="${PI05_RECOVERY_PEFT_RANK:-0}"
ACTION_PREFIX_LOSS_HORIZON="${PI05_ACTION_PREFIX_LOSS_HORIZON:-}"
ACTION_PREFIX_LOSS_WEIGHT="${PI05_ACTION_PREFIX_LOSS_WEIGHT:-1.0}"

for required in "$CONFIG_PATH" "$SOURCE_CHECKPOINT/model.safetensors" "$DATASET_ROOT/meta/info.json" "$DATASET_ROOT/meta/stats.json"; do
  [[ -f "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "ERROR: output already exists: $OUTPUT_DIR" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: PI05_RECOVERY_BATCH_SIZE must be a positive integer" >&2; exit 2; }
[[ "$SCHEDULER_DECAY_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: PI05_RECOVERY_SCHEDULER_DECAY_STEPS must be a positive integer" >&2; exit 2; }
[[ "$SCHEDULER_WARMUP_STEPS" =~ ^[0-9]+$ ]] || { echo "ERROR: PI05_RECOVERY_SCHEDULER_WARMUP_STEPS must be a non-negative integer" >&2; exit 2; }
[[ "$PEFT_RANK" =~ ^[0-9]+$ ]] || { echo "ERROR: PI05_RECOVERY_PEFT_RANK must be a non-negative integer" >&2; exit 2; }
[[ "$RECOVERY_START_FRAME_COUNT" =~ ^[0-9]+$ ]] || { echo "ERROR: PI05_RECOVERY_START_FRAME_COUNT must be a non-negative integer" >&2; exit 2; }
if [[ "$RECOVERY_START_FRAME_COUNT" -gt 0 && -z "$RECOVERY_EPISODE_START_INDEX" ]]; then
  echo "ERROR: PI05_RECOVERY_EPISODE_START_INDEX is required when recovery-start sampling is enabled" >&2
  exit 2
fi
MATERIALIZATION_RECEIPT="$DATASET_ROOT/meta/d1_materialization.json"
if [[ -f "$MATERIALIZATION_RECEIPT" ]]; then
  TRAIN_EPISODES="${PI05_TRAIN_EPISODES:-$(jq -c '.split_episode_indices.train' "$MATERIALIZATION_RECEIPT")}" 
else
  TRAIN_EPISODES="${PI05_TRAIN_EPISODES:-null}"
fi

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# The first robot-domain adaptation rebuilds normalization statistics. Later
# cumulative DAgger rounds must preserve the source checkpoint's processors so
# a nearly constant joint cannot jump to a different normalized coordinate.
if [[ "$PRESERVE_SOURCE_PROCESSOR_STATS" == "true" ]]; then
  unset LEROBOT_REBUILD_PROCESSORS || true
  export LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS=1
elif [[ "$PRESERVE_SOURCE_PROCESSOR_STATS" == "false" ]]; then
  export LEROBOT_REBUILD_PROCESSORS=1
  unset LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS || true
else
  echo "ERROR: PI05_PRESERVE_SOURCE_PROCESSOR_STATS must be true or false" >&2
  exit 2
fi

echo "Pure Pi0.5 rollout-recovery adaptation"
echo "  source:  $SOURCE_CHECKPOINT"
echo "  dataset: $DATASET_ROOT"
echo "  steps:   $STEPS"
echo "  batch:   $BATCH_SIZE"
echo "  lr:      $PEAK_LR -> $DECAY_LR"
echo "  scheduler decay steps: $SCHEDULER_DECAY_STEPS"
echo "  scheduler warmup steps: $SCHEDULER_WARMUP_STEPS"
echo "  output:  $OUTPUT_DIR"
echo "  log:     $LOG_FILE"
echo "  local phase transition: ±${SKILL_TRANSITION_WINDOW} frames x${SKILL_TRANSITION_WEIGHT}"
echo "  preserve source processor stats: $PRESERVE_SOURCE_PROCESSOR_STATS"
echo "  gripper transition sampling: ±${GRIPPER_TRANSITION_WINDOW} x${GRIPPER_TRANSITION_WEIGHT}"
echo "  initial frame sampling: first ${INITIAL_FRAME_COUNT} x${INITIAL_FRAME_WEIGHT}"
echo "  recovery-start sampling: episodes>=${RECOVERY_EPISODE_START_INDEX:-off}, first ${RECOVERY_START_FRAME_COUNT} positive frames x${RECOVERY_START_FRAME_WEIGHT}"
echo "  video timestamp tolerance: ${DATASET_TOLERANCE_S}s"
echo "  train seed: ${TRAIN_SEED}"
echo "  executed-prefix loss: horizon=${ACTION_PREFIX_LOSS_HORIZON:-off}, weight=${ACTION_PREFIX_LOSS_WEIGHT}"
if [[ "$PEFT_RANK" -gt 0 ]]; then
  echo "  adaptation: LeRobot LoRA rank ${PEFT_RANK} (base policy frozen)"
else
  echo "  adaptation: checkpoint training preset (no PEFT)"
fi
echo "  train episodes: $TRAIN_EPISODES"

SKILL_TRANSITION_ARGS=()
if [[ -n "$SKILL_TRANSITION_FROM" || -n "$SKILL_TRANSITION_TO" ]]; then
  if [[ -z "$SKILL_TRANSITION_FROM" || -z "$SKILL_TRANSITION_TO" ]]; then
    echo "ERROR: PI05_SKILL_TRANSITION_FROM and PI05_SKILL_TRANSITION_TO must be set together" >&2
    exit 2
  fi
  SKILL_TRANSITION_ARGS+=(
    --skill_progress_transition_from_phase="$SKILL_TRANSITION_FROM"
    --skill_progress_transition_to_phase="$SKILL_TRANSITION_TO"
  )
  echo "  local phase filter: ${SKILL_TRANSITION_FROM}->${SKILL_TRANSITION_TO}"
fi

RECOVERY_START_ARGS=(
  --semantic_recovery_start_sampling_count="$RECOVERY_START_FRAME_COUNT"
  --semantic_episode_sampling_weight="$RECOVERY_START_FRAME_WEIGHT"
)
if [[ -n "$RECOVERY_EPISODE_START_INDEX" ]]; then
  RECOVERY_START_ARGS+=(
    --semantic_episode_sampling_start_index="$RECOVERY_EPISODE_START_INDEX"
  )
fi

PEFT_ARGS=()
if [[ "$PEFT_RANK" -gt 0 ]]; then
  # Use LeRobot's policy-native PI0.5 targets.  The base checkpoint remains
  # bit-identical and the adapter learns only a low-rank recovery residual,
  # which bounds catastrophic drift of already-working grasp/transport skills.
  PEFT_ARGS+=(
    --peft.method_type=LORA
    --peft.r="$PEFT_RANK"
    --peft.full_training_modules='[]'
  )
fi

ACTION_PREFIX_ARGS=()
if [[ -n "$ACTION_PREFIX_LOSS_HORIZON" ]]; then
  [[ "$ACTION_PREFIX_LOSS_HORIZON" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: PI05_ACTION_PREFIX_LOSS_HORIZON must be a positive integer" >&2
    exit 2
  }
  ACTION_PREFIX_ARGS+=(
    --policy.action_prefix_loss_horizon="$ACTION_PREFIX_LOSS_HORIZON"
    --policy.action_prefix_loss_weight="$ACTION_PREFIX_LOSS_WEIGHT"
  )
fi

cd "$WS_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --config_path="$CONFIG_PATH" \
  --resume=false \
  --policy.pretrained_path="$SOURCE_CHECKPOINT" \
  --dataset.repo_id=pap_moe/pi05_pilot30_plus_recovery_v4 \
  --dataset.root="$DATASET_ROOT" \
  --dataset.episodes="$TRAIN_EPISODES" \
  --dataset_sample_weight_key=d1.sample_weight \
  --tolerance_s="$DATASET_TOLERANCE_S" \
  --seed="$TRAIN_SEED" \
  --batch_size="$BATCH_SIZE" \
  --output_dir="$OUTPUT_DIR" \
  --steps="$STEPS" \
  --optimizer.lr="$PEAK_LR" \
  --scheduler.peak_lr="$PEAK_LR" \
  --scheduler.decay_lr="$DECAY_LR" \
  --scheduler.num_warmup_steps="$SCHEDULER_WARMUP_STEPS" \
  --scheduler.num_decay_steps="$SCHEDULER_DECAY_STEPS" \
  --policy.optimizer_lr="$PEAK_LR" \
  --policy.scheduler_decay_lr="$DECAY_LR" \
  --policy.scheduler_warmup_steps="$SCHEDULER_WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS" \
  --skill_progress_transition_sampling_window="$SKILL_TRANSITION_WINDOW" \
  --skill_progress_transition_sampling_weight="$SKILL_TRANSITION_WEIGHT" \
  --gripper_transition_sampling_window="$GRIPPER_TRANSITION_WINDOW" \
  --gripper_transition_sampling_weight="$GRIPPER_TRANSITION_WEIGHT" \
  --initial_frame_sampling_count="$INITIAL_FRAME_COUNT" \
  --initial_frame_sampling_weight="$INITIAL_FRAME_WEIGHT" \
  "${RECOVERY_START_ARGS[@]}" \
  "${SKILL_TRANSITION_ARGS[@]}" \
  "${PEFT_ARGS[@]}" \
  "${ACTION_PREFIX_ARGS[@]}" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=50 \
  2>&1 | tee "$LOG_FILE"

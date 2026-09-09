#!/usr/bin/env bash
# Train one unified Pi0.5 policy without stage routing, auxiliary heads, or FSM.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
DATASET_ROOT="${PI05_DATASET_ROOT:-$WS_DIR/pap_moe_framework/datasets/lerobot_d2_skill_progress_native_v15_pi05_hybrid}"
INPUT_MODEL="${1:-${PI05_SOURCE_CHECKPOINT:-$WS_DIR/outputs/train/pi05_v9_semantic_rebalanced_action_20000step_20260808_103251/checkpoints/020000/pretrained_model}}"
STEPS="${2:-${PI05_BASELINE_STEPS:-5000}}"
BATCH_SIZE="${3:-${PI05_BASELINE_BATCH_SIZE:-2}}"
SAVE_FREQ="${PI05_BASELINE_SAVE_FREQ:-1000}"
SEED="${PI05_BASELINE_SEED:-1000}"
PEAK_LR="${PI05_BASELINE_LR:-5e-6}"
DECAY_LR="${PI05_BASELINE_DECAY_LR:-$(awk -v lr="$PEAK_LR" 'BEGIN { print lr * 0.5 }')}"
USE_D1_WEIGHTS="${PI05_USE_D1_WEIGHTS:-true}"
GRIPPER_TRANSITION_WINDOW="${PI05_GRIPPER_TRANSITION_WINDOW:-0}"
GRIPPER_TRANSITION_WEIGHT="${PI05_GRIPPER_TRANSITION_WEIGHT:-1.0}"
INITIAL_FRAME_COUNT="${PI05_INITIAL_FRAME_COUNT:-0}"
INITIAL_FRAME_WEIGHT="${PI05_INITIAL_FRAME_WEIGHT:-1.0}"
GRIPPER_ACTION_INDEX="${PI05_GRIPPER_ACTION_INDEX:-6}"
RECOVERY_EPISODE_START="${PI05_RECOVERY_EPISODE_START:-}"
RECOVERY_START_COUNT="${PI05_RECOVERY_START_COUNT:-0}"
RECOVERY_EXTRA_WEIGHT="${PI05_RECOVERY_EXTRA_WEIGHT:-1.0}"
GRIPPER_LOSS_WEIGHT="${PI05_GRIPPER_LOSS_WEIGHT:-1.0}"
GRIPPER_OPEN_LOSS_WEIGHT="${PI05_GRIPPER_OPEN_LOSS_WEIGHT:-1.0}"
GLOBAL_TASK="${PI05_GLOBAL_TASK:-pick up the peg and insert it into the hole}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PI05_BASELINE_OUTPUT_DIR:-$WS_DIR/outputs/train/pi05_d2_true_baseline_$RUN_TAG}"

[[ -f "$DATASET_ROOT/meta/d1_materialization.json" ]] || { echo "ERROR: materialized baseline dataset is missing: $DATASET_ROOT" >&2; exit 2; }
[[ -f "$INPUT_MODEL/config.json" ]] || { echo "ERROR: source checkpoint is missing: $INPUT_MODEL" >&2; exit 2; }
[[ "$USE_D1_WEIGHTS" == "true" || "$USE_D1_WEIGHTS" == "false" ]] || { echo "ERROR: PI05_USE_D1_WEIGHTS must be true or false" >&2; exit 2; }
TRAIN_EPISODES="${PI05_TRAIN_EPISODES:-$(jq -c '.split_episode_indices.train' "$DATASET_ROOT/meta/d1_materialization.json")}"
export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Continuation training must retain the pretrained Pi0.5 normalization domain.
# The physical materialization deliberately matches that domain.
export LEROBOT_REBUILD_PROCESSORS="${LEROBOT_REBUILD_PROCESSORS:-0}"
export LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS="${LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS:-1}"

echo "Pure Pi0.5 baseline training"
echo "  dataset: $DATASET_ROOT"
echo "  source:  $INPUT_MODEL"
echo "  output:  $OUTPUT_DIR"
echo "  steps/batch/lr/save: $STEPS / $BATCH_SIZE / $PEAK_LR / $SAVE_FREQ"
echo "  seed: $SEED"
echo "  D1 weighted sampling: $USE_D1_WEIGHTS"
echo "  train episodes: $TRAIN_EPISODES"
echo "  gripper transition sampling: +/-$GRIPPER_TRANSITION_WINDOW frames x$GRIPPER_TRANSITION_WEIGHT"
echo "  initial-frame preservation: first $INITIAL_FRAME_COUNT frames x$INITIAL_FRAME_WEIGHT"
echo "  extra recovery sampling: episode>=${RECOVERY_EPISODE_START:-disabled}, first=$RECOVERY_START_COUNT x$RECOVERY_EXTRA_WEIGHT"
echo "  unified-flow gripper loss: x$GRIPPER_LOSS_WEIGHT (open x$GRIPPER_OPEN_LOSS_WEIGHT)"

SAMPLE_WEIGHT_ARGS=()
if [[ "$USE_D1_WEIGHTS" == "true" ]]; then
  SAMPLE_WEIGHT_ARGS+=(--dataset_sample_weight_key=d1.sample_weight)
fi
GRIPPER_SAMPLING_ARGS=()
if [[ "$GRIPPER_TRANSITION_WINDOW" -gt 0 ]]; then
  # Sampling metadata only: this does not enable the deterministic gripper head.
  GRIPPER_SAMPLING_ARGS+=(--policy.gripper_action_index="$GRIPPER_ACTION_INDEX")
fi
RECOVERY_SAMPLING_ARGS=()
if [[ -n "$RECOVERY_EPISODE_START" ]]; then
  RECOVERY_SAMPLING_ARGS+=(
    --semantic_episode_sampling_start_index="$RECOVERY_EPISODE_START"
    --semantic_recovery_start_sampling_count="$RECOVERY_START_COUNT"
    --semantic_episode_sampling_weight="$RECOVERY_EXTRA_WEIGHT"
  )
fi

exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path="$INPUT_MODEL" \
  --policy.global_task="$GLOBAL_TASK" \
  --policy.gripper_loss_weight="$GRIPPER_LOSS_WEIGHT" \
  --policy.gripper_open_loss_weight="$GRIPPER_OPEN_LOSS_WEIGHT" \
  --policy.optimizer_lr="$PEAK_LR" \
  --policy.scheduler_decay_lr="$DECAY_LR" \
  --dataset.repo_id=pap_moe/pi05_d2_true_baseline \
  --dataset.root="$DATASET_ROOT" \
  --dataset.episodes="$TRAIN_EPISODES" \
  "${SAMPLE_WEIGHT_ARGS[@]}" \
  "${GRIPPER_SAMPLING_ARGS[@]}" \
  "${RECOVERY_SAMPLING_ARGS[@]}" \
  --gripper_transition_sampling_window="$GRIPPER_TRANSITION_WINDOW" \
  --gripper_transition_sampling_weight="$GRIPPER_TRANSITION_WEIGHT" \
  --initial_frame_sampling_count="$INITIAL_FRAME_COUNT" \
  --initial_frame_sampling_weight="$INITIAL_FRAME_WEIGHT" \
  --batch_size="$BATCH_SIZE" \
  --seed="$SEED" \
  --steps="$STEPS" \
  --num_workers=0 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=50 \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=false

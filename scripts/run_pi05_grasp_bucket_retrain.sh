#!/usr/bin/env bash
# Freeze one explicit cumulative DAgger bucket and retrain once.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
SOURCE_CHECKPOINT="${PI05_SOURCE_CHECKPOINT:-$WS_DIR/outputs/train/pi05_grasp_bucket_v3_preserve_v2_stats_20260819_2055/checkpoints/003000/pretrained_model}"
MIN_RECOVERIES="${PI05_MIN_BUCKET_RECOVERIES:-4}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
BUCKET_NAME="${PI05_BUCKET_NAME:-pi05_pilot30_plus_demo_manifold_recovery4_v1}"
MANIFEST="${PI05_BUCKET_MANIFEST:-$WS_DIR/pap_moe_framework/datasets/manifests/${BUCKET_NAME}.json}"
DATASET_ROOT="${PI05_BUCKET_DATASET_ROOT:-$WS_DIR/pap_moe_framework/datasets/lerobot_${BUCKET_NAME}}"
OUTPUT_DIR="${PI05_BUCKET_OUTPUT_DIR:-$WS_DIR/outputs/train/${BUCKET_NAME}_$RUN_TAG}"
RECOVERY_ROOT="$WS_DIR/pap_moe_framework/datasets/raw_rollout_recovery_v3_multitask_full_modalities"
read -r -a RECOVERY_EPISODES <<<"${PI05_RECOVERY_EPISODES:-pi05_ep15001_grasp_lift_20260820_175524 pi05_ep15001_grasp_lift_20260820_180156 pi05_ep15001_grasp_lift_20260820_180649 pi05_ep15001_grasp_lift_20260820_181223}"
read -r -a INCLUDE_CHECKPOINTS <<<"${PI05_INCLUDE_CHECKPOINTS:-}"

if ! [[ "$MIN_RECOVERIES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PI05_MIN_BUCKET_RECOVERIES must be a positive integer" >&2
  exit 2
fi
[[ -f "$SOURCE_CHECKPOINT/model.safetensors" ]] || { echo "ERROR: missing source checkpoint" >&2; exit 2; }
[[ ! -e "$DATASET_ROOT" ]] || { echo "ERROR: dataset output already exists: $DATASET_ROOT" >&2; exit 2; }

cd "$WS_DIR"
RECOVERY_ARGS=()
for episode in "${RECOVERY_EPISODES[@]}"; do
  RECOVERY_ARGS+=(--recovery-episode "$RECOVERY_ROOT/$episode/data.npz")
done
CHECKPOINT_LINEAGE_ARGS=()
for checkpoint in "${INCLUDE_CHECKPOINTS[@]}"; do
  [[ -n "$checkpoint" ]] || continue
  CHECKPOINT_LINEAGE_ARGS+=(--include-checkpoint "$checkpoint")
done
"$PYTHON_BIN" scripts/build_pi05_pilot30_recovery_manifest.py \
  --checkpoint "$SOURCE_CHECKPOINT" \
  "${CHECKPOINT_LINEAGE_ARGS[@]}" \
  "${RECOVERY_ARGS[@]}" \
  --output "$MANIFEST"

recovery_count="$($PYTHON_BIN - "$MANIFEST" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["counts"]["formal_recovery_episodes"])
PY
)"
if (( recovery_count < MIN_RECOVERIES )); then
  echo "ERROR: grasp/lift bucket has $recovery_count validated recoveries; need at least $MIN_RECOVERIES" >&2
  exit 3
fi

"$PYTHON_BIN" pap_moe_framework/scripts/materialize_d1_lerobot.py \
  --manifest "$MANIFEST" \
  --output "$DATASET_ROOT" \
  --gripper-contract gazebo_physical \
  --global-task-prompt
"$PYTHON_BIN" pap_moe_framework/scripts/materialize_d1_lerobot.py \
  --manifest "$MANIFEST" \
  --output "$DATASET_ROOT" \
  --gripper-contract gazebo_physical \
  --global-task-prompt \
  --audit-only

env \
  PI05_SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT" \
  PI05_RECOVERY_DATASET="$DATASET_ROOT" \
  PI05_RECOVERY_OUTPUT_DIR="$OUTPUT_DIR" \
  PI05_RECOVERY_TRAIN_STEPS="${PI05_RECOVERY_TRAIN_STEPS:-3000}" \
  bash scripts/run_pi05_pilot30_plus_recovery_adaptation.sh

echo "Completed one grasp/lift failure-bucket retraining round: $OUTPUT_DIR"

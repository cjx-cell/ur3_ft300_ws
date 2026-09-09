#!/usr/bin/env bash
# Run one corrected D1 stage for full PAP-MoE only.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
GROUP="${1:-}"
STAGE="${2:-}"
INPUT_MODEL="${3:-}"
STEPS="${4:-}"
BATCH_SIZE="${5:-2}"
DATASET_ROOT="$WS_DIR/pap_moe_framework/datasets/lerobot_d1_v3"
PI05_BACKBONE="$WS_DIR/outputs/train/pi05_v9_semantic_rebalanced_action_20000step_20260808_103251/checkpoints/020000/pretrained_model"

case "$GROUP" in
  PAP-VS) SENSOR_MODE=vs ;;
  PAP-VSF) SENSOR_MODE=vsf ;;
  *) echo "Usage: $0 {PAP-VS|PAP-VSF} STAGE [INPUT_MODEL] [STEPS] [BATCH_SIZE]" >&2; exit 2 ;;
esac
if [[ -z "$INPUT_MODEL" ]]; then
  if [[ "$STAGE" != "subtask" ]]; then
    echo "ERROR: INPUT_MODEL may be omitted only for subtask" >&2
    exit 2
  fi
  INPUT_MODEL="$PI05_BACKBONE"
fi
if [[ ! -f "$DATASET_ROOT/meta/d1_materialization.json" ]]; then
  echo "ERROR: corrected D1 v3 has not been materialized" >&2
  exit 2
fi

export PAP_MOE_CONTROLLED_SENSOR_MODE="$SENSOR_MODE"
args=("$STAGE" "$INPUT_MODEL" "$DATASET_ROOT")
if [[ -n "$STEPS" ]]; then args+=("$STEPS" "$BATCH_SIZE"); fi
exec "$WS_DIR/scripts/run_pap_moe_v9_stage_train.sh" "${args[@]}"

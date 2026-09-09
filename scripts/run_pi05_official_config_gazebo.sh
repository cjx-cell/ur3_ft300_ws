#!/usr/bin/env bash
# Evaluate a Pi0.5 checkpoint with the official LeRobot inference horizon.
# This wrapper intentionally disables every project-specific policy add-on.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
CHECKPOINT="${PI05_CHECKPOINT:-$WS_DIR/outputs/train/pi05_grasp_bucket_v3_preserve_v2_stats_20260819_2055/checkpoints/003000/pretrained_model}"
GUI="${1:-true}"
EPISODE="${2:-15001}"

if [[ ! -f "$CHECKPOINT/model.safetensors" && ! -f "$CHECKPOINT/adapter_model.safetensors" ]]; then
  echo "ERROR: checkpoint is missing: $CHECKPOINT" >&2
  exit 2
fi

echo "LeRobot-official Pi0.5 inference-contract evaluation"
echo "  checkpoint:       $CHECKPOINT"
echo "  predicted steps:  50"
echo "  executed steps:   50"
echo "  RTC:              disabled"
echo "  flow noise:       fresh per replan"
echo "  action ensemble:  1"
echo "  arm post-filter:  disabled"
echo "  auxiliary heads:  disabled by checkpoint config"
echo "  episode:          $EPISODE"

# POLICY_ACTION_CHUNK_MAX_STEP_RAD=10 is intentionally above every legal UR3
# joint-range transition, making the inherited project-side clamp an identity.
# The generic bridge requires a positive value and rejects zero.
PI05_CHECKPOINT="$CHECKPOINT" \
PI05_STATE_GRIPPER_MODE=continuous_radians_0_0.8 \
PI05_FIXED_NOISE_PER_REPLAN=false \
PI05_ENSEMBLE_SIZE=1 \
PI05_ACTION_CHUNK_SIZE=50 \
PI05_MAX_ARM_STEP_RAD=0.0 \
PI05_PHYSICAL_ARM_RESIDUAL_SCALE=1.0 \
PI05_HOLD_GRIPPER_PER_REPLAN=false \
PI05_RTC_ENABLED=false \
PI05_DYNAMIC_TASK_PROMPT=false \
PI05_ACTION_OUT_ADAPTER= \
PI05_LORA_ADAPTER= \
PI05_STAGE_LORA_ADAPTER= \
PI05_FINAL_STAGE_LORA_ADAPTER= \
PI05_GRASP_STAGE_LORA_ADAPTER= \
PI05_POST_GRASP_LORA_ADAPTER= \
PI05_INSERTION_LORA_ADAPTER= \
PI05_CONTACT_LORA_ADAPTER= \
PI05_INSERTION_FEEDBACK_ADAPTER= \
PI05_TRACE_FIRST_CHUNKS=20 \
PI05_TRACE_MULTIMODAL=false \
PI05_RECORD_VIDEO=true \
PI05_INFERENCE_READY_POLLS=1800 \
POLICY_ACTION_CHUNK_MAX_STEP_RAD=10.0 \
exec "$WS_DIR/scripts/run_pi05_v9_absolute_gazebo_eval.sh" "$GUI" "$EPISODE"

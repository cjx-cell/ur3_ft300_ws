#!/usr/bin/env bash
# Collect one full-modal Pi0.5 rollout -> align-precontact recovery episode.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
ROS_SETUP_FILE="/opt/ros/humble/setup.bash"
SETUP_FILE="$WS_DIR/install/setup.bash"
ROS_PYTHON="/usr/bin/python3"
EVAL_SCRIPT="$WS_DIR/scripts/run_pi05_v9_absolute_gazebo_eval.sh"
RECORDER="$WS_DIR/pap_moe_framework/scripts/record_rollout_recovery.py"
EXPERT="$WS_DIR/pap_moe_framework/scripts/run_privileged_rollout_recovery_expert.py"
VALIDATOR="$WS_DIR/pap_moe_framework/scripts/validate_rollout_recovery.py"
CHECKPOINT="${PI05_CHECKPOINT:-$WS_DIR/outputs/train/pi05_grasp_bucket_v3_preserve_v2_stats_20260819_2055/checkpoints/003000/pretrained_model}"
SEED="${PI05_SEED:-0}"
TRIGGER_DWELL_S="${PI05_ALIGN_TRIGGER_DWELL_S:-15.0}"
TRIGGER_XY_M="${PI05_ALIGN_TRIGGER_XY_M:-0.050}"
TRIGGER_MIN_PEG_Z_M="${PI05_ALIGN_TRIGGER_MIN_PEG_Z_M:-0.940}"
SUCCESS_XY_M="${PI05_ALIGN_SUCCESS_XY_M:-0.008}"
RECOVERY_TIMEOUT_S="${PI05_RECOVERY_TIMEOUT_S:-270}"
GUI="${1:-false}"
EPISODE="${2:-15001}"

[[ "$GUI" == "true" || "$GUI" == "false" ]] || { echo "ERROR: GUI must be true or false" >&2; exit 2; }
[[ -f "$CHECKPOINT/model.safetensors" ]] || { echo "ERROR: checkpoint missing: $CHECKPOINT" >&2; exit 2; }

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
COLLECTION_DIR="$WS_DIR/artifacts/pi05_ep${EPISODE}_align_recovery_$RUN_TAG"
SESSION_DIR="$COLLECTION_DIR/session"
OUTCOME_FILE="$COLLECTION_DIR/outcome.json"
EPISODE_DIR="$WS_DIR/pap_moe_framework/datasets/raw_rollout_recovery_v3_multitask_full_modalities/pi05_ep${EPISODE}_align_precontact_$RUN_TAG"
OUTPUT_FILE="$EPISODE_DIR/data.npz"
mkdir -p "$COLLECTION_DIR" "$EPISODE_DIR"

set +u
source "$ROS_SETUP_FILE"
source "$SETUP_FILE"
set -u
export PYTHONPATH="$WS_DIR${PYTHONPATH:+:$PYTHONPATH}"
ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    kill -0 -- "-$pid" 2>/dev/null && kill -TERM -- "-$pid" 2>/dev/null || true
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "Pure Pi0.5 episode-${EPISODE} align-precontact recovery collection"
echo "  checkpoint:     $CHECKPOINT"
echo "  rollout:        official 50 predicted / 50 executed, fresh noise"
echo "  trigger:        closed grasp, peg_z>=${TRIGGER_MIN_PEG_Z_M}m, xy<=${TRIGGER_XY_M}m for ${TRIGGER_DWELL_S}s"
echo "  expert success: xy<=${SUCCESS_XY_M}m, keep gripper command at 0.8"
echo "  seed:           $SEED"
echo "  output:         $OUTPUT_FILE"

env \
  PI05_CHECKPOINT="$CHECKPOINT" \
  PI05_STATE_GRIPPER_MODE=continuous_radians_0_0.8 \
  PI05_SEED="$SEED" \
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
  PI05_TRACE_FIRST_CHUNKS=30 \
  PI05_TRACE_MULTIMODAL=false \
  PI05_RECORD_VIDEO=true \
  PI05_MAX_EPISODE_DURATION_S=300 \
  PI05_INFERENCE_READY_POLLS=1800 \
  PI05_ROLLOUT_RECOVERY_SESSION_DIR="$SESSION_DIR" \
  PI05_LAUNCH_MOVEIT_FOR_RECOVERY=true \
  POLICY_ACTION_CHUNK_MAX_STEP_RAD=10.0 \
  setsid "$EVAL_SCRIPT" "$GUI" "$EPISODE" >"$COLLECTION_DIR/eval.log" 2>&1 &
EVAL_PID=$!
PIDS+=("$EVAL_PID")

echo "Waiting for recovery IPC session..."
for _ in $(seq 1 900); do
  [[ -f "$SESSION_DIR/session.json" ]] && break
  if ! kill -0 "$EVAL_PID" 2>/dev/null; then
    echo "ERROR: evaluation exited before creating recovery session" >&2
    tail -100 "$COLLECTION_DIR/eval.log" >&2
    exit 4
  fi
  sleep 0.5
done
[[ -f "$SESSION_DIR/session.json" ]] || { echo "ERROR: recovery session timed out" >&2; exit 4; }

"$ROS_PYTHON" "$RECORDER" \
  --session-dir "$SESSION_DIR" \
  --outcome-file "$OUTCOME_FILE" \
  --output "$OUTPUT_FILE" \
  --source-policy-checkpoint "$CHECKPOINT" \
  --episode-id "${EPISODE}-${RUN_TAG}" \
  --recovery-phase align_precontact \
  --pre-takeover-context-s 1.0 \
  >"$COLLECTION_DIR/recorder.log" 2>&1 &
RECORDER_PID=$!
PIDS+=("$RECORDER_PID")

"$ROS_PYTHON" "$EXPERT" \
  --session-dir "$SESSION_DIR" \
  --outcome-file "$OUTCOME_FILE" \
  --recovery-phase align_precontact \
  --trigger-after-s "$TRIGGER_DWELL_S" \
  --trigger-xy-m "$TRIGGER_XY_M" \
  --trigger-min-peg-z-m "$TRIGGER_MIN_PEG_Z_M" \
  --success-xy-m "$SUCCESS_XY_M" \
  --align-only \
  --timeout-s "$RECOVERY_TIMEOUT_S" \
  >"$COLLECTION_DIR/expert.log" 2>&1 &
EXPERT_PID=$!
PIDS+=("$EXPERT_PID")

while kill -0 "$EXPERT_PID" 2>/dev/null; do
  if ! kill -0 "$EVAL_PID" 2>/dev/null && [[ ! -f "$OUTCOME_FILE" ]]; then
    kill -TERM "$EXPERT_PID" "$RECORDER_PID" 2>/dev/null || true
    wait "$EXPERT_PID" 2>/dev/null || true
    wait "$RECORDER_PID" 2>/dev/null || true
    echo "ERROR: evaluation exited before recovery completed" >&2
    tail -100 "$COLLECTION_DIR/eval.log" >&2 || true
    exit 5
  fi
  sleep 0.5
done

set +e
wait "$EXPERT_PID"; EXPERT_STATUS=$?
wait "$RECORDER_PID"; RECORDER_STATUS=$?
set -e
kill -TERM -- "-$EVAL_PID" 2>/dev/null || true
for _ in $(seq 1 100); do kill -0 "$EVAL_PID" 2>/dev/null || break; sleep 0.1; done
kill -0 "$EVAL_PID" 2>/dev/null && kill -KILL -- "-$EVAL_PID" 2>/dev/null || true
wait "$EVAL_PID" 2>/dev/null || true

if (( EXPERT_STATUS != 0 || RECORDER_STATUS != 0 )); then
  echo "ERROR: recovery collection failed (expert=$EXPERT_STATUS recorder=$RECORDER_STATUS)" >&2
  tail -100 "$COLLECTION_DIR/expert.log" >&2 || true
  tail -100 "$COLLECTION_DIR/recorder.log" >&2 || true
  exit 5
fi

"$ROS_PYTHON" "$VALIDATOR" "$OUTPUT_FILE" | tee "$COLLECTION_DIR/validation.log"

GAZEBO_LOG_DIR="$(sed -n 's/^  logs:[[:space:]]*//p' "$COLLECTION_DIR/eval.log" | tail -1)"
VIDEO_FILE="$GAZEBO_LOG_DIR/pi05_pure_rollout.mp4"
"$ROS_PYTHON" - "$OUTPUT_FILE" "$EPISODE_DIR/recovery_timing.json" "$VIDEO_FILE" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

episode_path, output_path, video_path = map(Path, sys.argv[1:])
with np.load(episode_path, allow_pickle=True) as episode:
    takeover = np.flatnonzero(np.asarray(episode["intervention_mask"], dtype=bool))
    if takeover.size == 0:
        raise SystemExit("validated recovery has no intervention frame")
    index = int(takeover[0])
    timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
    payload = {
        "takeover_index": index,
        "takeover_timestamp_s": float(timestamps[index]),
        "takeover_from_episode_start_s": float(timestamps[index] - timestamps[0]),
        "approximate_video_time_s": float(timestamps[index] - timestamps[0]),
        "video_file": str(video_path),
        "video_exists": video_path.is_file(),
        "timing_note": "Video time is approximate; takeover_index and timestamp are authoritative.",
    }
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Validated baseline align recovery episode: $OUTPUT_FILE"

#!/usr/bin/env bash
# Collect one full-modal grasp/lift recovery from a Pi0.5 L0 rollout.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
ROS_SETUP_FILE="/opt/ros/humble/setup.bash"
SETUP_FILE="$WS_DIR/install/setup.bash"
ROS_PYTHON="/usr/bin/python3"
EVAL_SCRIPT="$WS_DIR/scripts/run_pi05_v9_absolute_gazebo_eval.sh"
RECORDER="$WS_DIR/pap_moe_framework/scripts/record_rollout_recovery.py"
EXPERT="$WS_DIR/pap_moe_framework/scripts/run_privileged_rollout_recovery_expert.py"
VALIDATOR="$WS_DIR/pap_moe_framework/scripts/validate_rollout_recovery.py"
CHECKPOINT="${PI05_CHECKPOINT:-$WS_DIR/outputs/train/pi05_v9_absolute_1000step_20260818_233017/checkpoints/010000/pretrained_model}"
ROLLOUT_LORA_ADAPTER="${PI05_ROLLOUT_LORA_ADAPTER:-}"
RECOVERY_PHASE="${PI05_RECOVERY_PHASE:-grasp_lift}"
TRIGGER_AFTER_S="${PI05_RECOVERY_TRIGGER_AFTER_S:-35.0}"
TRIGGER_XY_M="${PI05_RECOVERY_TRIGGER_XY_M:-0.020}"
TRIGGER_MIN_HEIGHT_M="${PI05_RECOVERY_TRIGGER_MIN_HEIGHT_M:-0.160}"
TRIGGER_CLOSED_RAD="${PI05_RECOVERY_TRIGGER_CLOSED_RAD:-0.300}"
TRIGGER_MAX_LIFT_M="${PI05_RECOVERY_TRIGGER_MAX_LIFT_M:-0.005}"
TRIGGER_SUSTAIN_S="${PI05_RECOVERY_TRIGGER_SUSTAIN_S:-3.0}"
MAX_RECOVERABLE_DROP_M="${PI05_RECOVERY_MAX_RECOVERABLE_DROP_M:-0.020}"
MAX_RECOVERABLE_XY_SHIFT_M="${PI05_RECOVERY_MAX_RECOVERABLE_XY_SHIFT_M:-0.012}"
# Episode 15001 fixture is exactly the v8 scene (peg 0.105/0.258, hole
# -0.108/0.236). Never silently use the newer v9 random-scene coordinates.
REFERENCE_ROOT="${PI05_RECOVERY_REFERENCE_ROOT:-$WS_DIR/pap_moe_framework/datasets/raw_v8_exactfit_scripted}"
REFERENCE_BRIDGE_STEPS="${PI05_RECOVERY_REFERENCE_BRIDGE_STEPS:-20}"
# Keep lateral alignment accurate while avoiding the old near-static expert
# trajectory.  At 10 Hz, 3 mm over 10 samples gives 3 mm/s lateral motion;
# insertion uses 8 mm over 8 samples (10 mm/s), with the exact-fit lower
# section retaining the 2x interpolation safeguard (5 mm/s).
RECOVERY_CHUNK_SIZE="${PI05_RECOVERY_EXPERT_CHUNK_SIZE:-30}"
ALIGNMENT_CHUNK_SIZE="${PI05_RECOVERY_ALIGNMENT_CHUNK_SIZE:-10}"
INSERTION_CHUNK_SIZE="${PI05_RECOVERY_INSERTION_CHUNK_SIZE:-8}"
GRASP_PICKUP_CHUNK_SIZE="${PI05_RECOVERY_GRASP_PICKUP_CHUNK_SIZE:-60}"
DESCENT_XY_GATE_M="${PI05_RECOVERY_DESCENT_XY_GATE_M:-0.0005}"
CHUNK_DESCENT_M="${PI05_RECOVERY_CHUNK_DESCENT_M:-0.008}"
GRASP_PROBE_CHUNKS="${PI05_RECOVERY_GRASP_PROBE_CHUNKS:-2}"
GRASP_PROBE_STEP_M="${PI05_RECOVERY_GRASP_PROBE_STEP_M:-0.006}"
GRASP_PROBE_SUCCESS_LIFT_M="${PI05_RECOVERY_GRASP_PROBE_SUCCESS_LIFT_M:-0.008}"
GRASP_XY_GATE_M="${PI05_RECOVERY_GRASP_XY_GATE_M:-0.012}"
GRASP_Z_GATE_M="${PI05_RECOVERY_GRASP_Z_GATE_M:-0.010}"
GRASP_TRIGGER_MODE="${PI05_RECOVERY_GRASP_TRIGGER_MODE:-gripper_misaligned}"
RECORD_VIDEO="${PI05_RECORD_VIDEO:-true}"
ROLLOUT_ACTION_CHUNK_SIZE="${PI05_RECOVERY_ROLLOUT_ACTION_CHUNK_SIZE:-10}"
ROLLOUT_RTC_ENABLED="${PI05_RECOVERY_RTC_ENABLED:-true}"
ROLLOUT_RTC_HORIZON="${PI05_RECOVERY_RTC_HORIZON:-$ROLLOUT_ACTION_CHUNK_SIZE}"
ROLLOUT_MAX_ARM_STEP_RAD="${PI05_RECOVERY_ROLLOUT_MAX_ARM_STEP_RAD:-0.0}"
# Match the LeRobot-official evaluation contract.  The ROS bridge requires a
# positive value, so 10.0 acts as an identity for every legal UR3 transition.
# A 0.025 controller-side clamp changes the reached state before the next RTC
# replan and therefore makes recovery collection off-contract with evaluation.
CONTROLLER_MAX_STEP_RAD="${PI05_RECOVERY_CONTROLLER_MAX_STEP_RAD:-10.0}"
PRE_TAKEOVER_CONTEXT_S="${PI05_RECOVERY_PRE_TAKEOVER_CONTEXT_S:-1.0}"
# The expert process starts while the 4B policy checkpoint is still loading.
# A full-task episode can then contain multiple policy/expert handoffs and a
# deliberately bounded insertion. 270 s can therefore expire during a valid
# descent even though Gazebo itself is still within its episode budget.
RECOVERY_TIMEOUT_S="${PI05_RECOVERY_TIMEOUT_S:-900}"
IPC_WAIT_POLLS="${PI05_RECOVERY_IPC_WAIT_POLLS:-1800}"
GUI="${1:-false}"
EPISODE="${2:-15008}"

OPTIONAL_EXPERT_ARGS=()
if [[ -n "${PI05_RECOVERY_POLICY_ALIGNMENT_ENTRY_M:-}" ]]; then
  OPTIONAL_EXPERT_ARGS+=(
    --rejoin-policy-alignment-entry-m
    "$PI05_RECOVERY_POLICY_ALIGNMENT_ENTRY_M"
  )
fi
if [[ -n "${PI05_RECOVERY_POLICY_CONTACT_FORCE_N:-}" ]]; then
  OPTIONAL_EXPERT_ARGS+=(
    --rejoin-policy-contact-force-n
    "$PI05_RECOVERY_POLICY_CONTACT_FORCE_N"
  )
fi
if [[ -n "${PI05_RECOVERY_POLICY_ALIGNMENT_STALL_S:-}" ]]; then
  OPTIONAL_EXPERT_ARGS+=(
    --rejoin-policy-alignment-stall-s
    "$PI05_RECOVERY_POLICY_ALIGNMENT_STALL_S"
  )
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
COLLECTION_DIR="$WS_DIR/artifacts/pi05_ep${EPISODE}_grasp_recovery_$RUN_TAG"
SESSION_DIR="$COLLECTION_DIR/session"
OUTCOME_FILE="$COLLECTION_DIR/outcome.json"
if [[ "$RECOVERY_PHASE" == "full_task" ]]; then
  RECOVERY_ROOT="$WS_DIR/pap_moe_framework/datasets/raw_rollout_recovery_v4_multi_handoff_full_episode"
else
  RECOVERY_ROOT="$WS_DIR/pap_moe_framework/datasets/raw_rollout_recovery_v3_multitask_full_modalities"
fi
EPISODE_DIR="$RECOVERY_ROOT/pi05_ep${EPISODE}_${RECOVERY_PHASE}_$RUN_TAG"
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

echo "Pure Pi0.5 episode-${EPISODE} ${RECOVERY_PHASE} recovery collection"
echo "  checkpoint: $CHECKPOINT"
echo "  rollout LoRA:${ROLLOUT_LORA_ADAPTER:-disabled}"
if [[ "$GRASP_TRIGGER_MODE" == "time" ]]; then
  echo "  trigger:    ${TRIGGER_AFTER_S}s after first dispatched policy action"
elif [[ "$GRASP_TRIGGER_MODE" == "closed_no_lift" ]]; then
  echo "  trigger:    gripper>=${TRIGGER_CLOSED_RAD}rad, xy<=${TRIGGER_XY_M}m, peg lift<=${TRIGGER_MAX_LIFT_M}m sustained ${TRIGGER_SUSTAIN_S}s"
elif [[ "$GRASP_TRIGGER_MODE" == "gripper_misaligned" ]]; then
  echo "  trigger:    gripper>=${TRIGGER_CLOSED_RAD}rad while xy>${TRIGGER_XY_M}m"
else
  echo "  trigger:    xy<=${TRIGGER_XY_M}m, height>=${TRIGGER_MIN_HEIGHT_M}m"
fi
echo "  video:      $RECORD_VIDEO"
echo "  rollout action chunk: $ROLLOUT_ACTION_CHUNK_SIZE"
echo "  rollout RTC: $ROLLOUT_RTC_ENABLED (horizon=$ROLLOUT_RTC_HORIZON)"
echo "  rollout arm post-limit: $ROLLOUT_MAX_ARM_STEP_RAD"
echo "  controller step limit: $CONTROLLER_MAX_STEP_RAD"
echo "  trigger mode: $GRASP_TRIGGER_MODE"
echo "  policy execution/RTC horizon: ${ROLLOUT_ACTION_CHUNK_SIZE}/${ROLLOUT_RTC_HORIZON}"
echo "  policy RTC enabled: $ROLLOUT_RTC_ENABLED"
echo "  inference/controller arm limits: ${ROLLOUT_MAX_ARM_STEP_RAD}/${CONTROLLER_MAX_STEP_RAD} rad/step"
echo "  saved policy context before takeover: ${PRE_TAKEOVER_CONTEXT_S}s"
echo "  recovery timeout: ${RECOVERY_TIMEOUT_S}s"
echo "  recovery IPC wait: $((IPC_WAIT_POLLS / 2))s"
echo "  insertion gate/full descent chunk: ${DESCENT_XY_GATE_M}m / ${CHUNK_DESCENT_M}m"
echo "  expert chunk size (general/align/insert/grasp): ${RECOVERY_CHUNK_SIZE}/${ALIGNMENT_CHUNK_SIZE}/${INSERTION_CHUNK_SIZE}/${GRASP_PICKUP_CHUNK_SIZE}"
echo "  recovery target: exact-scene successful demonstration suffix"
echo "  output:     $OUTPUT_FILE"

# Keep recovery roll-in identical to the currently selected evaluation
# contract by default. All four values remain explicit overrides for controlled
# ablations, and are printed into the launch log for auditability.
env \
  PI05_CHECKPOINT="$CHECKPOINT" \
  PI05_SEED="${PI05_SEED:-0}" \
  PI05_FIXED_NOISE_PER_REPLAN="${PI05_FIXED_NOISE_PER_REPLAN:-false}" \
  PI05_ACTION_CHUNK_SIZE="$ROLLOUT_ACTION_CHUNK_SIZE" \
  PI05_MAX_ARM_STEP_RAD="$ROLLOUT_MAX_ARM_STEP_RAD" \
  PI05_PHYSICAL_ARM_RESIDUAL_SCALE=1.0 \
  PI05_RTC_ENABLED="$ROLLOUT_RTC_ENABLED" \
  PI05_RTC_EXECUTION_HORIZON="$ROLLOUT_RTC_HORIZON" \
  PI05_RTC_MAX_GUIDANCE_WEIGHT=10.0 \
  PI05_RTC_PREFIX_ATTENTION_SCHEDULE=EXP \
  PI05_MAX_EPISODE_DURATION_S=300 \
  PI05_RECORD_VIDEO="$RECORD_VIDEO" \
  PI05_ROLLOUT_RECOVERY_SESSION_DIR="$SESSION_DIR" \
  PI05_LAUNCH_MOVEIT_FOR_RECOVERY=true \
  PI05_DEMO_RECOVERY_EPISODE= \
  PI05_ACTION_OUT_ADAPTER= \
  PI05_LORA_ADAPTER="$ROLLOUT_LORA_ADAPTER" \
  PI05_STAGE_LORA_ADAPTER= \
  PI05_STAGE_LORA_REFERENCE= \
  PI05_FINAL_STAGE_LORA_ADAPTER= \
  PI05_FINAL_STAGE_LORA_REFERENCE= \
  PI05_GRASP_STAGE_LORA_ADAPTER= \
  PI05_GRASP_STAGE_LORA_REFERENCE= \
  PI05_POST_GRASP_LORA_ADAPTER= \
  PI05_POST_GRASP_LORA_REFERENCE= \
  PI05_INSERTION_LORA_ADAPTER= \
  PI05_INSERTION_LORA_REFERENCE= \
  PI05_CONTACT_LORA_ADAPTER= \
  PI05_CONTACT_LORA_REFERENCE= \
  PI05_INSERTION_FEEDBACK_ADAPTER= \
  PI05_GRIPPER_CLOSE_REFERENCE= \
  PI05_GRIPPER_OPEN_REFERENCE= \
  POLICY_ACTION_CHUNK_MAX_STEP_RAD="$CONTROLLER_MAX_STEP_RAD" \
  ROLLOUT_RECOVERY_GUARD_PREMATURE_RELEASE="${PI05_RECOVERY_GUARD_PREMATURE_RELEASE:-false}" \
  ROLLOUT_RECOVERY_GUARD_CLOSED_RAD="${PI05_RECOVERY_TRANSPORT_TRIGGER_OPEN_RAD:-0.55}" \
  ROLLOUT_RECOVERY_GUARD_MIN_LIFT_M="${PI05_RECOVERY_TRANSPORT_MIN_LIFT_M:-0.050}" \
  ROLLOUT_RECOVERY_GUARD_MIN_HOLE_DISTANCE_M="${PI05_RECOVERY_TRANSPORT_MIN_HOLE_DISTANCE_M:-0.080}" \
  setsid "$EVAL_SCRIPT" "$GUI" "$EPISODE" >"$COLLECTION_DIR/eval.log" 2>&1 &
EVAL_PID=$!
PIDS+=("$EVAL_PID")

echo "Waiting for recovery IPC session..."
for _ in $(seq 1 "$IPC_WAIT_POLLS"); do
  [[ -f "$SESSION_DIR/session.json" ]] && break
  if ! kill -0 "$EVAL_PID" 2>/dev/null; then
    echo "ERROR: evaluation exited before creating the recovery session" >&2
    tail -80 "$COLLECTION_DIR/eval.log" >&2
    exit 4
  fi
  sleep 0.5
done
[[ -f "$SESSION_DIR/session.json" ]] || { echo "ERROR: recovery session timed out" >&2; exit 4; }

RECORDER_ARGS=(
  --session-dir "$SESSION_DIR" \
  --outcome-file "$OUTCOME_FILE" \
  --output "$OUTPUT_FILE" \
  --source-policy-checkpoint "$CHECKPOINT" \
  --episode-id "${EPISODE}-$RUN_TAG" \
  --recovery-phase "$RECOVERY_PHASE"
)
if [[ "$RECOVERY_PHASE" != "full_task" ]]; then
  RECORDER_ARGS+=(--pre-takeover-context-s "$PRE_TAKEOVER_CONTEXT_S")
fi
"$ROS_PYTHON" "$RECORDER" "${RECORDER_ARGS[@]}" \
  >"$COLLECTION_DIR/recorder.log" 2>&1 &
RECORDER_PID=$!
PIDS+=("$RECORDER_PID")

"$ROS_PYTHON" "$EXPERT" \
  --session-dir "$SESSION_DIR" \
  --outcome-file "$OUTCOME_FILE" \
  --recovery-phase "$RECOVERY_PHASE" \
  --trigger-after-s "$TRIGGER_AFTER_S" \
  --grasp-trigger-mode "$GRASP_TRIGGER_MODE" \
  --grasp-trigger-xy-m "$TRIGGER_XY_M" \
  --grasp-trigger-min-height-m "$TRIGGER_MIN_HEIGHT_M" \
  --grasp-trigger-closed-rad "$TRIGGER_CLOSED_RAD" \
  --grasp-trigger-max-lift-m "$TRIGGER_MAX_LIFT_M" \
  --grasp-trigger-sustain-s "$TRIGGER_SUSTAIN_S" \
  --grasp-max-recoverable-drop-m "$MAX_RECOVERABLE_DROP_M" \
  --grasp-max-recoverable-xy-shift-m "$MAX_RECOVERABLE_XY_SHIFT_M" \
  --grasp-reference-episode "$REFERENCE_ROOT/pick_up_the_peg_and_insert_it_into_the_hole_episode_0001_success/data.npz" \
  --grasp-reference-episode "$REFERENCE_ROOT/pick_up_the_peg_and_insert_it_into_the_hole_episode_0002_success/data.npz" \
  --grasp-reference-episode "$REFERENCE_ROOT/pick_up_the_peg_and_insert_it_into_the_hole_episode_0003_success/data.npz" \
  --grasp-reference-episode "$REFERENCE_ROOT/pick_up_the_peg_and_insert_it_into_the_hole_episode_0004_success/data.npz" \
  --grasp-reference-bridge-steps "$REFERENCE_BRIDGE_STEPS" \
  --grasp-pickup-chunk-size "$GRASP_PICKUP_CHUNK_SIZE" \
  --grasp-reference-safe-open-steps 10 \
  --grasp-reference-max-anchor-l2-rad 1.5 \
  --transport-trigger-open-rad "${PI05_RECOVERY_TRANSPORT_TRIGGER_OPEN_RAD:-0.55}" \
  --transport-trigger-min-lift-m "${PI05_RECOVERY_TRANSPORT_MIN_LIFT_M:-0.050}" \
  --transport-trigger-min-hole-distance-m "${PI05_RECOVERY_TRANSPORT_MIN_HOLE_DISTANCE_M:-0.080}" \
  --transport-trigger-stall-s "${PI05_RECOVERY_TRANSPORT_STALL_S:-5.0}" \
  --transport-trigger-min-progress-m "${PI05_RECOVERY_TRANSPORT_MIN_PROGRESS_M:-0.010}" \
  --transport-success-xy-m "${PI05_RECOVERY_TRANSPORT_SUCCESS_XY_M:-0.060}" \
  --success-peg-z-m "${PI05_RECOVERY_SUCCESS_PEG_Z_M:-0.890}" \
  --rejoin-trigger-l2-rad "${PI05_RECOVERY_REJOIN_TRIGGER_L2_RAD:-0.20}" \
  --rejoin-min-policy-events "${PI05_RECOVERY_REJOIN_MIN_POLICY_EVENTS:-2}" \
  --rejoin-physical-weight-rad-per-m "${PI05_RECOVERY_REJOIN_PHYSICAL_WEIGHT_RAD_PER_M:-5.0}" \
  --rejoin-release-l2-rad "${PI05_RECOVERY_REJOIN_RELEASE_L2_RAD:-0.06}" \
  --rejoin-physical-entry-l2-rad "${PI05_RECOVERY_REJOIN_PHYSICAL_ENTRY_L2_RAD:-0.12}" \
  --rejoin-release-physical-m "${PI05_RECOVERY_REJOIN_RELEASE_PHYSICAL_M:-0.008}" \
  --rejoin-policy-alignment-min-descent-m "${PI05_RECOVERY_POLICY_ALIGNMENT_MIN_DESCENT_M:-0.003}" \
  "${OPTIONAL_EXPERT_ARGS[@]}" \
  --rejoin-stall-s "${PI05_RECOVERY_REJOIN_STALL_S:-6.0}" \
  --insertion-stall-s "${PI05_RECOVERY_INSERTION_STALL_S:-30.0}" \
  --insertion-min-progress-m "${PI05_RECOVERY_INSERTION_MIN_PROGRESS_M:-0.001}" \
  --rejoin-grasp-stall-policy-events "${PI05_RECOVERY_GRASP_STALL_POLICY_EVENTS:-30}" \
  --rejoin-lookahead-frames "${PI05_RECOVERY_REJOIN_LOOKAHEAD_FRAMES:-4}" \
  --rejoin-max-interventions "${PI05_RECOVERY_REJOIN_MAX_INTERVENTIONS:-20}" \
  --chunk-xy-step-m "${PI05_RECOVERY_CARTESIAN_XY_STEP_M:-0.003}" \
  --descent-xy-gate-m "$DESCENT_XY_GATE_M" \
  --chunk-descent-m "$CHUNK_DESCENT_M" \
  --chunk-size "$RECOVERY_CHUNK_SIZE" \
  --alignment-chunk-size "$ALIGNMENT_CHUNK_SIZE" \
  --insertion-chunk-size "$INSERTION_CHUNK_SIZE" \
  --grasp-probe-chunks "$GRASP_PROBE_CHUNKS" \
  --grasp-probe-step-m "$GRASP_PROBE_STEP_M" \
  --grasp-probe-success-lift-m "$GRASP_PROBE_SUCCESS_LIFT_M" \
  --grasp-xy-gate-m "$GRASP_XY_GATE_M" \
  --grasp-z-gate-m "$GRASP_Z_GATE_M" \
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
    tail -80 "$COLLECTION_DIR/eval.log" >&2 || true
    exit 5
  fi
  sleep 0.5
done

set +e
wait "$EXPERT_PID"; EXPERT_STATUS=$?
# A failed expert may have no valid episode to build. Do not let the recorder
# wait indefinitely for data that is deliberately being rejected.
if (( EXPERT_STATUS != 0 )); then
  kill -TERM "$RECORDER_PID" 2>/dev/null || true
fi
wait "$RECORDER_PID"; RECORDER_STATUS=$?
set -e
kill -TERM -- "-$EVAL_PID" 2>/dev/null || true
for _ in $(seq 1 200); do
  kill -0 "$EVAL_PID" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$EVAL_PID" 2>/dev/null; then
  echo "WARN: evaluation group did not exit after SIGTERM; escalating to SIGKILL" >&2
  kill -KILL -- "-$EVAL_PID" 2>/dev/null || true
fi
wait "$EVAL_PID" 2>/dev/null || true

if (( EXPERT_STATUS != 0 || RECORDER_STATUS != 0 )); then
  echo "ERROR: recovery collection failed (expert=$EXPERT_STATUS recorder=$RECORDER_STATUS)" >&2
  tail -80 "$COLLECTION_DIR/expert.log" >&2 || true
  tail -80 "$COLLECTION_DIR/recorder.log" >&2 || true
  exit 5
fi

"$ROS_PYTHON" "$VALIDATOR" "$OUTPUT_FILE" | tee "$COLLECTION_DIR/validation.log"

# Write an auditable handoff marker beside the raw episode. The recorder and
# video start close together but not on the exact same callback, so this is an
# approximate video time and the frame/timestamp remain the authoritative
# values for training-data inspection.
GAZEBO_LOG_DIR="$(sed -n 's/^  logs:[[:space:]]*//p' "$COLLECTION_DIR/eval.log" | tail -1)"
VIDEO_FILE="$GAZEBO_LOG_DIR/pi05_pure_rollout.mp4"
"$ROS_PYTHON" - "$OUTPUT_FILE" "$EPISODE_DIR/recovery_timing.json" "$VIDEO_FILE" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

episode_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
video_path = Path(sys.argv[3])
with np.load(episode_path, allow_pickle=True) as episode:
    intervention = np.asarray(episode["intervention_mask"], dtype=bool)
    takeover_indices = np.flatnonzero(intervention)
    if takeover_indices.size == 0:
        raise SystemExit("validated recovery has no intervention frame")
    takeover_index = int(takeover_indices[0])
    timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
    payload = {
        "takeover_index": takeover_index,
        "takeover_timestamp_s": float(timestamps[takeover_index]),
        "takeover_from_episode_start_s": float(timestamps[takeover_index] - timestamps[0]),
        "approximate_video_time_s": float(timestamps[takeover_index] - timestamps[0]),
        "video_file": str(video_path),
        "video_exists": video_path.is_file(),
        "timing_note": "Video time is approximate; takeover_index and timestamp are authoritative.",
    }
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
echo "Validated baseline recovery episode: $OUTPUT_FILE"

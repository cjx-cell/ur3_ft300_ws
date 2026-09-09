#!/usr/bin/env bash
# Collect one full-episode, multi-handoff PAP-MoE rollout recovery.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
ROS_SETUP_FILE="/opt/ros/humble/setup.bash"
SETUP_FILE="$WS_DIR/install/setup.bash"
ROS_PYTHON="/usr/bin/python3"
EVAL_SCRIPT="$WS_DIR/scripts/run_pap_moe_v9_gazebo_eval.sh"
RECORDER="$WS_DIR/pap_moe_framework/scripts/record_rollout_recovery.py"
EXPERT="$WS_DIR/pap_moe_framework/scripts/run_privileged_rollout_recovery_expert.py"
VALIDATOR="$WS_DIR/pap_moe_framework/scripts/validate_rollout_recovery.py"
# This roll-in reached 7.8 mm XY in pure-policy L0 and has repeatedly reached
# the aligned-high takeover gate.  The former 20260813_233827 default stalled
# about 19 cm from the peg and could not produce grasp-boundary recoveries.
DEFAULT_CHECKPOINT="$WS_DIR/outputs/train/pap_moe_v9_conditioner_20260814_184053/checkpoints/003000/pretrained_model"
CHECKPOINT="${PAP_MOE_CHECKPOINT:-$DEFAULT_CHECKPOINT}"
TRIGGER_AFTER_S="${PAP_MOE_RECOVERY_TRIGGER_AFTER_S:-22.0}"
# The current PAP roll-in hovers above the peg and the trajectory controller
# aborts before the predicted close reaches the measured gripper state.  A
# state-based closed-gripper trigger therefore never fires.  Trigger from the
# first policy action instead, after the reproducible hover failure is visible
# but while the scene is still recoverable.
TRIGGER_MODE="${PAP_MOE_RECOVERY_TRIGGER_MODE:-time}"
TRIGGER_XY_M="${PAP_MOE_RECOVERY_TRIGGER_XY_M:-0.012}"
TRIGGER_MIN_HEIGHT_M="${PAP_MOE_RECOVERY_TRIGGER_MIN_HEIGHT_M:-0.160}"
TRIGGER_CLOSED_RAD="${PAP_MOE_RECOVERY_TRIGGER_CLOSED_RAD:-0.05}"
REFERENCE_ROOT="${PAP_MOE_RECOVERY_REFERENCE_ROOT:-$WS_DIR/pap_moe_framework/datasets/raw_v8_exactfit_scripted}"
# Recorder samples asynchronously after the expert terminal check. One
# millimetre of settling rebound avoids rejecting an otherwise seated peg.
RECOVERY_SUCCESS_PEG_Z_M="${PAP_MOE_RECOVERY_SUCCESS_PEG_Z_M:-0.906}"
RTC_ENABLED="${PAP_MOE_RTC_ENABLED:-true}"
RTC_EXECUTION_HORIZON="${PAP_MOE_RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${PAP_MOE_RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
RTC_PREFIX_ATTENTION_SCHEDULE="${PAP_MOE_RTC_PREFIX_ATTENTION_SCHEDULE:-EXP}"
GUI="${1:-false}"
EPISODE="${2:-15001}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
COLLECTION_DIR="$WS_DIR/artifacts/pap_moe_ep${EPISODE}_full_recovery_$RUN_TAG"
SESSION_DIR="$COLLECTION_DIR/session"
OUTCOME_FILE="$COLLECTION_DIR/outcome.json"
EPISODE_DIR="$WS_DIR/pap_moe_framework/datasets/raw_rollout_recovery_v4_multi_handoff_full_episode/pap_moe_ep${EPISODE}_full_task_$RUN_TAG"
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

echo "PAP-MoE episode ${EPISODE} full-modal multi-handoff recovery collection"
echo "  checkpoint: $CHECKPOINT"
echo "  trigger:    mode=$TRIGGER_MODE time=${TRIGGER_AFTER_S}s xy<=${TRIGGER_XY_M}m height>=${TRIGGER_MIN_HEIGHT_M}m"
echo "  RTC:        enabled=$RTC_ENABLED horizon=$RTC_EXECUTION_HORIZON weight=$RTC_MAX_GUIDANCE_WEIGHT schedule=$RTC_PREFIX_ATTENTION_SCHEDULE"
echo "  output:     $OUTPUT_FILE"

# Keep the policy roll-in identical to the guarded pure-evaluation contract.
# Privileged expert commands use their own bounded bridge/chunk limits.
unset PAP_MOE_DEMO_RECOVERY_EPISODE
PAP_MOE_CHECKPOINT="$CHECKPOINT" \
PAP_MOE_ROLLOUT_RECOVERY_SESSION_DIR="$SESSION_DIR" \
PAP_MOE_LAUNCH_MOVEIT_FOR_RECOVERY=true \
PAP_MOE_RTC_ENABLED="$RTC_ENABLED" \
PAP_MOE_RTC_EXECUTION_HORIZON="$RTC_EXECUTION_HORIZON" \
PAP_MOE_RTC_MAX_GUIDANCE_WEIGHT="$RTC_MAX_GUIDANCE_WEIGHT" \
PAP_MOE_RTC_PREFIX_ATTENTION_SCHEDULE="$RTC_PREFIX_ATTENTION_SCHEDULE" \
PAP_MOE_EXECUTE_STEPS=10 \
PAP_MOE_FIXED_NOISE_PER_REPLAN=false \
PAP_MOE_EXPERT_MASK=1,1,1,1 \
PAP_MOE_DEMO_RECOVERY_AFTER_CHUNKS=0 \
PAP_MOE_MAX_EPISODE_DURATION_S=300 \
PAP_MOE_RECORD_VIDEO=true \
POLICY_ACTION_CHUNK_MAX_STEP_RAD=0.025 \
  setsid "$EVAL_SCRIPT" "$GUI" "$EPISODE" >"$COLLECTION_DIR/eval.log" 2>&1 &
EVAL_PID=$!
PIDS+=("$EVAL_PID")

echo "Waiting for recovery IPC session..."
for _ in $(seq 1 600); do
  [[ -f "$SESSION_DIR/session.json" ]] && break
  if ! kill -0 "$EVAL_PID" 2>/dev/null; then
    echo "ERROR: evaluation exited before creating the recovery session" >&2
    tail -80 "$COLLECTION_DIR/eval.log" >&2
    exit 4
  fi
  sleep 0.5
done
if [[ ! -f "$SESSION_DIR/session.json" ]]; then
  echo "ERROR: timed out waiting for recovery session" >&2
  exit 4
fi

"$ROS_PYTHON" "$RECORDER" \
  --session-dir "$SESSION_DIR" \
  --outcome-file "$OUTCOME_FILE" \
  --output "$OUTPUT_FILE" \
  --source-policy-checkpoint "$CHECKPOINT" \
  --episode-id "${EPISODE}-$RUN_TAG" \
  --recovery-phase full_task \
  >"$COLLECTION_DIR/recorder.log" 2>&1 &
RECORDER_PID=$!
PIDS+=("$RECORDER_PID")

"$ROS_PYTHON" "$EXPERT" \
  --session-dir "$SESSION_DIR" \
  --outcome-file "$OUTCOME_FILE" \
  --recovery-phase full_task \
  --trigger-after-s "$TRIGGER_AFTER_S" \
  --grasp-trigger-mode "$TRIGGER_MODE" \
  --grasp-trigger-xy-m "$TRIGGER_XY_M" \
  --grasp-trigger-min-height-m "$TRIGGER_MIN_HEIGHT_M" \
  --grasp-trigger-closed-rad "$TRIGGER_CLOSED_RAD" \
  --grasp-reference-episode "$REFERENCE_ROOT/pick_up_the_peg_and_insert_it_into_the_hole_episode_0001_success/data.npz" \
  --grasp-reference-episode "$REFERENCE_ROOT/pick_up_the_peg_and_insert_it_into_the_hole_episode_0002_success/data.npz" \
  --grasp-reference-episode "$REFERENCE_ROOT/pick_up_the_peg_and_insert_it_into_the_hole_episode_0003_success/data.npz" \
  --grasp-reference-episode "$REFERENCE_ROOT/pick_up_the_peg_and_insert_it_into_the_hole_episode_0004_success/data.npz" \
  --grasp-reference-max-anchor-l2-rad 1.5 \
  --grasp-pickup-chunk-size 60 \
  --chunk-size 30 \
  --alignment-chunk-size 10 \
  --insertion-chunk-size 8 \
  --chunk-xy-step-m 0.003 \
  --descent-xy-gate-m 0.0005 \
  --chunk-descent-m 0.008 \
  --success-peg-z-m "$RECOVERY_SUCCESS_PEG_Z_M" \
  --rejoin-grasp-stall-policy-events 10 \
  --rejoin-max-interventions 20 \
  --timeout-s 900 \
  >"$COLLECTION_DIR/expert.log" 2>&1 &
EXPERT_PID=$!
PIDS+=("$EXPERT_PID")

while kill -0 "$EXPERT_PID" 2>/dev/null; do
  if ! kill -0 "$EVAL_PID" 2>/dev/null && [[ ! -f "$OUTCOME_FILE" ]]; then
    kill -TERM "$EXPERT_PID" "$RECORDER_PID" 2>/dev/null || true
    wait "$EXPERT_PID" 2>/dev/null || true
    wait "$RECORDER_PID" 2>/dev/null || true
    echo "ERROR: evaluation exited before expert takeover/recovery completed" >&2
    tail -80 "$COLLECTION_DIR/eval.log" >&2 || true
    exit 5
  fi
  sleep 0.5
done

set +e
wait "$EXPERT_PID"
EXPERT_STATUS=$?
wait "$RECORDER_PID"
RECORDER_STATUS=$?
set -e
kill -TERM -- "-$EVAL_PID" 2>/dev/null || true
wait "$EVAL_PID" 2>/dev/null || true

if (( EXPERT_STATUS != 0 || RECORDER_STATUS != 0 )); then
  echo "ERROR: recovery collection failed (expert=$EXPERT_STATUS recorder=$RECORDER_STATUS)" >&2
  tail -80 "$COLLECTION_DIR/expert.log" >&2 || true
  tail -80 "$COLLECTION_DIR/recorder.log" >&2 || true
  exit 5
fi

"$ROS_PYTHON" "$VALIDATOR" "$OUTPUT_FILE" \
  | tee "$COLLECTION_DIR/validation.log"
echo "Validated recovery episode: $OUTPUT_FILE"

#!/usr/bin/env bash
# Guarded single-episode Gazebo evaluation for the final PAP-MoE v9 chain.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
ROS_SETUP_FILE="/opt/ros/humble/setup.bash"
SETUP_FILE="$WS_DIR/install/setup.bash"
PI_ENV_PYTHON="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
ROS_PYTHON="/usr/bin/python3"
PEG_SCRIPT_DIR="$WS_DIR/src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole"
INFERENCE_SCRIPT="$PEG_SCRIPT_DIR/ur3_pap_moe_peg_in_hole_inference.py"
ROS_SIDE_SCRIPT="$PEG_SCRIPT_DIR/ur3_pap_moe_peg_in_hole_ros_side.py"
OBSERVATION_HELPER="$PEG_SCRIPT_DIR/pap_moe_online_observation.py"
RESULT_WRITER="$WS_DIR/scripts/write_gazebo_eval_result.py"
VIDEO_RECORDER="$WS_DIR/scripts/record_gazebo_camera_video.py"
DEFAULT_CHECKPOINT="$WS_DIR/outputs/train/pap_moe_v9_conditioner_20260808_021028/checkpoints/001000/pretrained_model"
CHECKPOINT="${PAP_MOE_CHECKPOINT:-$DEFAULT_CHECKPOINT}"
INFERENCE_SEED="${PAP_MOE_SEED:-0}"
FIXED_NOISE_PER_REPLAN="${PAP_MOE_FIXED_NOISE_PER_REPLAN:-false}"
EXECUTE_STEPS="${PAP_MOE_EXECUTE_STEPS:-10}"
EXPERT_MASK="${PAP_MOE_EXPERT_MASK:-1,1,1,1}"
RTC_ENABLED="${PAP_MOE_RTC_ENABLED:-true}"
RTC_EXECUTION_HORIZON="${PAP_MOE_RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${PAP_MOE_RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
RTC_PREFIX_ATTENTION_SCHEDULE="${PAP_MOE_RTC_PREFIX_ATTENTION_SCHEDULE:-EXP}"
DEMO_RECOVERY_EPISODE="${PAP_MOE_DEMO_RECOVERY_EPISODE:-}"
DEMO_RECOVERY_AFTER_CHUNKS="${PAP_MOE_DEMO_RECOVERY_AFTER_CHUNKS:-0}"
ROLLOUT_RECOVERY_SESSION_DIR="${PAP_MOE_ROLLOUT_RECOVERY_SESSION_DIR:-}"
LAUNCH_MOVEIT_FOR_RECOVERY="${PAP_MOE_LAUNCH_MOVEIT_FOR_RECOVERY:-false}"
RECORD_VIDEO="${PAP_MOE_RECORD_VIDEO:-false}"
export PAP_MOE_MAX_EPISODE_DURATION_S="${PAP_MOE_MAX_EPISODE_DURATION_S:-100.0}"
export EXECUTE_STEPS

GUI="${1:-false}"
EPISODE="${2:-15001}"

if [[ "$GUI" != "true" && "$GUI" != "false" ]]; then
  echo "ERROR: GUI must be true or false, got: $GUI" >&2
  exit 2
fi
if ! [[ "$EXECUTE_STEPS" =~ ^[0-9]+$ ]] \
  || (( EXECUTE_STEPS < 1 || EXECUTE_STEPS > 50 )); then
  echo "ERROR: PAP_MOE_EXECUTE_STEPS must be an integer in [1, 50]" >&2
  exit 2
fi
if [[ "$FIXED_NOISE_PER_REPLAN" != "true" && "$FIXED_NOISE_PER_REPLAN" != "false" ]]; then
  echo "ERROR: PAP_MOE_FIXED_NOISE_PER_REPLAN must be true or false" >&2
  exit 2
fi
if [[ "$RTC_ENABLED" != "true" && "$RTC_ENABLED" != "false" ]]; then
  echo "ERROR: PAP_MOE_RTC_ENABLED must be true or false" >&2
  exit 2
fi
if [[ "$LAUNCH_MOVEIT_FOR_RECOVERY" != "true" && "$LAUNCH_MOVEIT_FOR_RECOVERY" != "false" ]]; then
  echo "ERROR: PAP_MOE_LAUNCH_MOVEIT_FOR_RECOVERY must be true or false" >&2
  exit 2
fi
if [[ "$RECORD_VIDEO" != "true" && "$RECORD_VIDEO" != "false" ]]; then
  echo "ERROR: PAP_MOE_RECORD_VIDEO must be true or false" >&2
  exit 2
fi
if [[ "$LAUNCH_MOVEIT_FOR_RECOVERY" == "true" && -z "$ROLLOUT_RECOVERY_SESSION_DIR" ]]; then
  echo "ERROR: MoveIt recovery launch requires PAP_MOE_ROLLOUT_RECOVERY_SESSION_DIR" >&2
  exit 2
fi
if ! [[ "$RTC_EXECUTION_HORIZON" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAP_MOE_RTC_EXECUTION_HORIZON must be a positive integer" >&2
  exit 2
fi
if ! [[ "$RTC_MAX_GUIDANCE_WEIGHT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
  echo "ERROR: PAP_MOE_RTC_MAX_GUIDANCE_WEIGHT must be positive" >&2
  exit 2
fi
case "$RTC_PREFIX_ATTENTION_SCHEDULE" in
  EXP|LINEAR|ONES|ZEROS) ;;
  *) echo "ERROR: unsupported PAP_MOE_RTC_PREFIX_ATTENTION_SCHEDULE" >&2; exit 2 ;;
esac

case "$EPISODE" in
  15001|15002|15003|15004) PEG_X=0.105; PEG_Y=0.258; HOLE_X=-0.108; HOLE_Y=0.236 ;;
  13001) PEG_X=-0.105; PEG_Y=0.258; HOLE_X=0.126;  HOLE_Y=0.227 ;;
  13002) PEG_X=0.015;  PEG_Y=0.241; HOLE_X=0.137;  HOLE_Y=0.265 ;;
  13003) PEG_X=0.066;  PEG_Y=0.272; HOLE_X=-0.070; HOLE_Y=0.251 ;;
  13004) PEG_X=0.091;  PEG_Y=0.229; HOLE_X=-0.141; HOLE_Y=0.250 ;;
  13005) PEG_X=0.104;  PEG_Y=0.228; HOLE_X=-0.074; HOLE_Y=0.222 ;;
  13006) PEG_X=0.140;  PEG_Y=0.226; HOLE_X=-0.011; HOLE_Y=0.227 ;;
  13008) PEG_X=-0.110; PEG_Y=0.273; HOLE_X=0.031;  HOLE_Y=0.265 ;;
  13009) PEG_X=-0.182; PEG_Y=0.237; HOLE_X=-0.042; HOLE_Y=0.232 ;;
  13010) PEG_X=0.189;  PEG_Y=0.222; HOLE_X=-0.141; HOLE_Y=0.233 ;;
  14003) PEG_X=0.015;  PEG_Y=0.244; HOLE_X=-0.170; HOLE_Y=0.227 ;;
  14012) PEG_X=0.159;  PEG_Y=0.250; HOLE_X=0.026;  HOLE_Y=0.254 ;;
  14013) PEG_X=-0.197; PEG_Y=0.223; HOLE_X=-0.040; HOLE_Y=0.275 ;;
  *)
    echo "ERROR: unsupported episode $EPISODE" >&2
    exit 2
    ;;
esac

for required in \
  "$ROS_SETUP_FILE" \
  "$SETUP_FILE" \
  "$CHECKPOINT/model.safetensors" \
  "$INFERENCE_SCRIPT" \
  "$ROS_SIDE_SCRIPT" \
  "$OBSERVATION_HELPER" \
  "$RESULT_WRITER"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required file is missing: $required" >&2
    exit 2
  fi
done
if [[ "$RECORD_VIDEO" == "true" && ! -e "$VIDEO_RECORDER" ]]; then
  echo "ERROR: video recorder is missing: $VIDEO_RECORDER" >&2
  exit 2
fi

"$PI_ENV_PYTHON" "$INFERENCE_SCRIPT" \
  --checkpoint "$CHECKPOINT" --validate-only

set +u
source "$ROS_SETUP_FILE"
source "$SETUP_FILE"
set -u

topic_has_publishers() {
  local info line
  info="$(ros2 topic info "$1" 2>/dev/null || true)"
  while IFS= read -r line; do
    [[ "$line" =~ ^Publisher[[:space:]]count:[[:space:]]([1-9][0-9]*)$ ]] && return 0
  done <<<"$info"
  return 1
}

if topic_has_publishers /clock; then
  echo "ERROR: an existing ROS/Gazebo simulation appears to be running (/clock exists)." >&2
  exit 2
fi

export PYTHONPATH="/home/ubuntu/lerobot/src:$PEG_SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_TAG="$(date +%Y%m%d_%H%M%S)_ep${EPISODE}"
LOG_DIR="$WS_DIR/artifacts/gazebo_pap_moe_v9_$RUN_TAG"
METRICS_FILE="$LOG_DIR/inference_metrics.csv"
mkdir -p "$LOG_DIR" /tmp/pap_moe_roslog
if [[ "$ROLLOUT_RECOVERY_SESSION_DIR" == "auto" ]]; then
  ROLLOUT_RECOVERY_SESSION_DIR="$LOG_DIR/rollout_recovery_session"
fi
export ROS_LOG_DIR=/tmp/pap_moe_roslog

rm -f \
  /tmp/ur3_inference_ready.txt \
  /tmp/ur3_joint_state.txt \
  /tmp/ur3_action.txt \
  /tmp/ur3_action_chunk.npy \
  /tmp/ur3_action_chunk_tmp.npy \
  /tmp/ur3_action_chunk_meta.json \
  /tmp/ur3_camera0.npy \
  /tmp/ur3_camera1.npy \
  /tmp/ur3_force.npy \
  /tmp/ur3_force_fast.npy \
  /tmp/ur3_force_slow.npy \
  /tmp/ur3_state_history.npy \
  /tmp/ur3_visual_quality.npy \
  /tmp/ur3_pap_moe_observation_meta.json

CHILD_PIDS=()
GAZEBO_PROCESS_GROUP=""
MOVEIT_PROCESS_GROUP=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$GAZEBO_PROCESS_GROUP" ]] \
    && kill -0 -- "-$GAZEBO_PROCESS_GROUP" 2>/dev/null; then
    kill -TERM -- "-$GAZEBO_PROCESS_GROUP" 2>/dev/null || true
  fi
  if [[ -n "$MOVEIT_PROCESS_GROUP" ]] \
    && kill -0 -- "-$MOVEIT_PROCESS_GROUP" 2>/dev/null; then
    kill -TERM -- "-$MOVEIT_PROCESS_GROUP" 2>/dev/null || true
  fi
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "PAP-MoE v9 guarded Gazebo evaluation"
echo "  checkpoint: $CHECKPOINT"
echo "  episode:    $EPISODE"
echo "  peg:        ($PEG_X, $PEG_Y)"
echo "  hole:       ($HOLE_X, $HOLE_Y)"
echo "  GUI:        $GUI"
echo "  seed:       $INFERENCE_SEED"
echo "  fixed noise:$FIXED_NOISE_PER_REPLAN"
echo "  execute:    $EXECUTE_STEPS/50 steps"
echo "  expert mask:$EXPERT_MASK"
echo "  RTC:        $RTC_ENABLED (horizon=$RTC_EXECUTION_HORIZON, weight=$RTC_MAX_GUIDANCE_WEIGHT, schedule=$RTC_PREFIX_ATTENTION_SCHEDULE)"
echo "  recovery:   ${ROLLOUT_RECOVERY_SESSION_DIR:-disabled}"
echo "  recovery IK:${LAUNCH_MOVEIT_FOR_RECOVERY}"
echo "  record video:$RECORD_VIDEO"
echo "  logs:       $LOG_DIR"

setsid ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py \
  gazebo_gui:="$GUI" launch_rviz:=false >"$LOG_DIR/gazebo.log" 2>&1 &
GAZEBO_PID=$!
GAZEBO_PROCESS_GROUP=$GAZEBO_PID
CHILD_PIDS+=("$GAZEBO_PID")

if [[ "$LAUNCH_MOVEIT_FOR_RECOVERY" == "true" ]]; then
  setsid ros2 launch ur3_ft300_moveit_config move_group.launch.py \
    >"$LOG_DIR/moveit.log" 2>&1 &
  MOVEIT_PID=$!
  MOVEIT_PROCESS_GROUP=$MOVEIT_PID
  CHILD_PIDS+=("$MOVEIT_PID")
  echo "Waiting for recovery-only MoveIt /compute_ik service..."
  for _ in $(seq 1 360); do
    ros2 service list 2>/dev/null | grep -Fxq /compute_ik && break
    sleep 0.5
  done
  if ! ros2 service list 2>/dev/null | grep -Fxq /compute_ik; then
    echo "ERROR: /compute_ik is unavailable; see $LOG_DIR/moveit.log" >&2
    exit 3
  fi
fi

echo "Waiting for joint states, force sensor, and both cameras..."
for _ in $(seq 1 180); do
  TOPICS="$(ros2 topic list 2>/dev/null || true)"
  if grep -Fxq /joint_states <<<"$TOPICS" \
    && grep -Fxq /force_torque_sensor_broadcaster/wrench <<<"$TOPICS" \
    && grep -Fxq /wrist_camera/color/image_raw <<<"$TOPICS" \
    && grep -Fxq /global_camera/color/image_raw <<<"$TOPICS"; then
    break
  fi
  sleep 0.5
done
TOPICS="$(ros2 topic list 2>/dev/null || true)"
for topic in \
  /joint_states \
  /force_torque_sensor_broadcaster/wrench \
  /wrist_camera/color/image_raw \
  /global_camera/color/image_raw; do
  if ! grep -Fxq "$topic" <<<"$TOPICS"; then
    echo "ERROR: timed out waiting for $topic; see $LOG_DIR/gazebo.log" >&2
    exit 3
  fi
done

VIDEO_PID=""
if [[ "$RECORD_VIDEO" == "true" ]]; then
  "$ROS_PYTHON" "$VIDEO_RECORDER" --output "$LOG_DIR/pap_moe_l0_rollout.mp4" \
    >"$LOG_DIR/video.log" 2>&1 &
  VIDEO_PID=$!
  CHILD_PIDS+=("$VIDEO_PID")
fi

INFERENCE_ARGS=(
  --checkpoint "$CHECKPOINT"
  --metrics-file "$METRICS_FILE"
  --seed "$INFERENCE_SEED"
  --execute-steps "$EXECUTE_STEPS"
  --expert-mask "$EXPERT_MASK"
)
if [[ "$FIXED_NOISE_PER_REPLAN" == "true" ]]; then
  INFERENCE_ARGS+=(--fixed-noise-per-replan)
fi
if [[ "$RTC_ENABLED" == "true" ]]; then
  INFERENCE_ARGS+=(
    --rtc
    --rtc-execution-horizon "$RTC_EXECUTION_HORIZON"
    --rtc-max-guidance-weight "$RTC_MAX_GUIDANCE_WEIGHT"
    --rtc-prefix-attention-schedule "$RTC_PREFIX_ATTENTION_SCHEDULE"
  )
fi
if [[ -n "$DEMO_RECOVERY_EPISODE" ]]; then
  INFERENCE_ARGS+=(
    --demo-recovery-episode "$DEMO_RECOVERY_EPISODE"
    --demo-recovery-after-chunks "$DEMO_RECOVERY_AFTER_CHUNKS"
  )
fi
"$PI_ENV_PYTHON" "$INFERENCE_SCRIPT" "${INFERENCE_ARGS[@]}" \
  >"$LOG_DIR/inference.log" 2>&1 &
INFERENCE_PID=$!
CHILD_PIDS+=("$INFERENCE_PID")

INFERENCE_READY_TIMEOUT_S="${PAP_MOE_INFERENCE_READY_TIMEOUT_S:-360}"
[[ "$INFERENCE_READY_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: PAP_MOE_INFERENCE_READY_TIMEOUT_S must be a positive integer" >&2
  exit 2
}
echo "Loading inference model (timeout=${INFERENCE_READY_TIMEOUT_S}s)..."
for _ in $(seq 1 $((INFERENCE_READY_TIMEOUT_S * 2))); do
  [[ -f /tmp/ur3_inference_ready.txt ]] && break
  if ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
    echo "ERROR: inference exited; see $LOG_DIR/inference.log" >&2
    exit 4
  fi
  sleep 0.5
done
if [[ ! -f /tmp/ur3_inference_ready.txt ]]; then
  echo "ERROR: inference readiness timed out; see $LOG_DIR/inference.log" >&2
  exit 4
fi

"$PI_ENV_PYTHON" - "$FIXED_NOISE_PER_REPLAN" "$RTC_ENABLED" "$RTC_EXECUTION_HORIZON" "$RTC_MAX_GUIDANCE_WEIGHT" "$RTC_PREFIX_ATTENTION_SCHEDULE" <<'PY'
import json
import os
import sys

with open("/tmp/ur3_inference_ready.txt", encoding="utf-8") as file:
    metadata = json.load(file)
expected_steps = int(os.environ["EXECUTE_STEPS"])
expected_fixed_noise = sys.argv[1] == "true"
expected_rtc = sys.argv[2] == "true"
expected = (
    "pap_moe",
    50,
    expected_steps,
    0.1,
    "pap_moe_v6",
    "continuous_radians_0_0.8",
    "continuous_radians_0_0.8",
    "continuous_radians_0_0.8",
)
actual = (
    metadata["policy_type"],
    metadata["predicted_action_steps"],
    metadata["executed_action_steps"],
    metadata["action_dt_s"],
    metadata["observation_schema"],
    metadata.get("gripper_action_mode"),
    metadata.get("state_gripper_mode"),
    metadata.get("state_history_gripper_mode"),
)
if actual != expected:
    raise SystemExit(f"ERROR: online contract mismatch: expected={expected}, actual={actual}")
if metadata.get("fixed_noise_per_replan") is not expected_fixed_noise:
    raise SystemExit(
        "ERROR: fixed-noise contract mismatch: "
        f"{metadata.get('fixed_noise_per_replan')} != {expected_fixed_noise}"
    )
if metadata.get("rtc_enabled") is not expected_rtc:
    raise SystemExit(
        f"ERROR: RTC mismatch: {metadata.get('rtc_enabled')} != {expected_rtc}"
    )
if expected_rtc:
    if metadata.get("rtc_execution_horizon") != int(sys.argv[3]):
        raise SystemExit("ERROR: RTC execution-horizon mismatch")
    if abs(metadata.get("rtc_max_guidance_weight", -1.0) - float(sys.argv[4])) > 1e-9:
        raise SystemExit("ERROR: RTC guidance-weight mismatch")
    if metadata.get("rtc_prefix_attention_schedule") != sys.argv[5]:
        raise SystemExit("ERROR: RTC attention-schedule mismatch")
    if metadata.get("rtc_action_dimensions") != "arm_only":
        raise SystemExit("ERROR: RTC must exclude the gripper dimension")
print(f"Inference ready: PAP-MoE, predict 50, execute {expected_steps}, dt=0.1 s")
PY

echo "Starting PAP-MoE ROS-side controller."
ROS_SIDE_ARGS=(
  --spawn --ep "$EPISODE"
  --action-chunk-size "$EXECUTE_STEPS"
  --peg-x "$PEG_X" --peg-y "$PEG_Y" --hole-x "$HOLE_X" --hole-y "$HOLE_Y"
)
if [[ -n "$ROLLOUT_RECOVERY_SESSION_DIR" ]]; then
  if [[ -e "$ROLLOUT_RECOVERY_SESSION_DIR/session.json" ]]; then
    echo "ERROR: recovery session already exists: $ROLLOUT_RECOVERY_SESSION_DIR" >&2
    exit 2
  fi
  ROS_SIDE_ARGS+=(
    --rollout-recovery-session-dir "$ROLLOUT_RECOVERY_SESSION_DIR"
    --rollout-recovery-source-checkpoint "$CHECKPOINT"
  )
fi
set +e
"$ROS_PYTHON" "$ROS_SIDE_SCRIPT" "${ROS_SIDE_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/ros_side.log"
CONTROLLER_STATUS=${PIPESTATUS[0]}
set -e
if [[ -n "$VIDEO_PID" ]] && kill -0 "$VIDEO_PID" 2>/dev/null; then
  kill -TERM "$VIDEO_PID" 2>/dev/null || true
  wait "$VIDEO_PID" 2>/dev/null || true
fi
set +e
"$PI_ENV_PYTHON" "$RESULT_WRITER" \
  --log "$LOG_DIR/ros_side.log" \
  --output "$LOG_DIR/result.json" \
  --policy pap_moe_v9 \
  --checkpoint "$CHECKPOINT" \
  --episode "$EPISODE" \
  --seed "$INFERENCE_SEED" \
  --execution-prefix "$EXECUTE_STEPS"
RESULT_STATUS=$?
set -e
if (( CONTROLLER_STATUS != 0 )); then
  echo "ERROR: ROS-side controller exited with status $CONTROLLER_STATUS" >&2
  exit "$CONTROLLER_STATUS"
fi
exit "$RESULT_STATUS"

#!/usr/bin/env bash
# Single-episode Gazebo reproduction for the v9 absolute-action baseline.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
ROS_SETUP_FILE="/opt/ros/humble/setup.bash"
SETUP_FILE="$WS_DIR/install/setup.bash"
PI_ENV_PYTHON="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
ROS_PYTHON="/usr/bin/python3"
PEG_SCRIPT_DIR="$WS_DIR/src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole"
INFERENCE_SCRIPT="${PI05_INFERENCE_SCRIPT:-$PEG_SCRIPT_DIR/ur3_baseline_peg_in_hole_inference.py}"
ROS_SIDE_SCRIPT="$PEG_SCRIPT_DIR/ur3_pi05_v9_absolute_peg_in_hole_ros_side.py"
RESULT_WRITER="$WS_DIR/scripts/write_gazebo_eval_result.py"
VIDEO_RECORDER="$WS_DIR/scripts/record_gazebo_camera_video.py"
CHECKPOINT="${PI05_CHECKPOINT:-$WS_DIR/outputs/train/pi05_v9_semantic_rebalanced_action_20000step_20260808_103251/checkpoints/020000/pretrained_model}"
STATE_GRIPPER_MODE="${PI05_STATE_GRIPPER_MODE:-continuous_radians_0_0.8}"
INFERENCE_SEED="${PI05_SEED:-0}"
FIXED_NOISE_PER_REPLAN="${PI05_FIXED_NOISE_PER_REPLAN:-false}"
ENSEMBLE_SIZE="${PI05_ENSEMBLE_SIZE:-1}"
MAX_ARM_STEP_RAD="${PI05_MAX_ARM_STEP_RAD:-0.0}"
PHYSICAL_ARM_RESIDUAL_SCALE="${PI05_PHYSICAL_ARM_RESIDUAL_SCALE:-1.0}"
TRACE_FIRST_CHUNKS="${PI05_TRACE_FIRST_CHUNKS:-5}"
TRACE_MULTIMODAL="${PI05_TRACE_MULTIMODAL:-false}"
ACTION_CHUNK_SIZE="${PI05_ACTION_CHUNK_SIZE:-10}"
HOLD_GRIPPER_PER_REPLAN="${PI05_HOLD_GRIPPER_PER_REPLAN:-false}"
RTC_ENABLED="${PI05_RTC_ENABLED:-false}"
RTC_EXECUTION_HORIZON="${PI05_RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${PI05_RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
RTC_PREFIX_ATTENTION_SCHEDULE="${PI05_RTC_PREFIX_ATTENTION_SCHEDULE:-EXP}"
DEMO_RECOVERY_EPISODE="${PI05_DEMO_RECOVERY_EPISODE:-}"
DEMO_RECOVERY_AFTER_CHUNKS="${PI05_DEMO_RECOVERY_AFTER_CHUNKS:-0}"
TASK_PROMPT="${PI05_TASK_PROMPT:-pick up the peg and insert it into the hole}"
DYNAMIC_TASK_PROMPT="${PI05_DYNAMIC_TASK_PROMPT:-false}"
ACTION_OUT_ADAPTER="${PI05_ACTION_OUT_ADAPTER:-}"
LORA_ADAPTER="${PI05_LORA_ADAPTER:-}"
STAGE_LORA_ADAPTER="${PI05_STAGE_LORA_ADAPTER:-}"
STAGE_LORA_REFERENCE="${PI05_STAGE_LORA_REFERENCE:-}"
STAGE_LORA_THRESHOLD="${PI05_STAGE_LORA_THRESHOLD:-0.15}"
FINAL_STAGE_LORA_ADAPTER="${PI05_FINAL_STAGE_LORA_ADAPTER:-}"
FINAL_STAGE_LORA_REFERENCE="${PI05_FINAL_STAGE_LORA_REFERENCE:-}"
FINAL_STAGE_LORA_THRESHOLD="${PI05_FINAL_STAGE_LORA_THRESHOLD:-0.15}"
GRASP_STAGE_LORA_ADAPTER="${PI05_GRASP_STAGE_LORA_ADAPTER:-}"
GRASP_STAGE_LORA_REFERENCE="${PI05_GRASP_STAGE_LORA_REFERENCE:-}"
GRASP_STAGE_LORA_THRESHOLD="${PI05_GRASP_STAGE_LORA_THRESHOLD:-0.10}"
POST_GRASP_LORA_ADAPTER="${PI05_POST_GRASP_LORA_ADAPTER:-}"
POST_GRASP_LORA_REFERENCE="${PI05_POST_GRASP_LORA_REFERENCE:-}"
POST_GRASP_LORA_THRESHOLD="${PI05_POST_GRASP_LORA_THRESHOLD:-0.15}"
INSERTION_LORA_ADAPTER="${PI05_INSERTION_LORA_ADAPTER:-}"
INSERTION_LORA_REFERENCE="${PI05_INSERTION_LORA_REFERENCE:-}"
INSERTION_LORA_THRESHOLD="${PI05_INSERTION_LORA_THRESHOLD:-0.10}"
CONTACT_LORA_ADAPTER="${PI05_CONTACT_LORA_ADAPTER:-}"
CONTACT_LORA_REFERENCE="${PI05_CONTACT_LORA_REFERENCE:-}"
CONTACT_LORA_THRESHOLD="${PI05_CONTACT_LORA_THRESHOLD:-0.10}"
INSERTION_FEEDBACK_ADAPTER="${PI05_INSERTION_FEEDBACK_ADAPTER:-}"
GRIPPER_CLOSE_REFERENCE="${PI05_GRIPPER_CLOSE_REFERENCE:-}"
GRIPPER_CLOSE_THRESHOLD="${PI05_GRIPPER_CLOSE_THRESHOLD:-0.0}"
GRIPPER_OPEN_REFERENCE="${PI05_GRIPPER_OPEN_REFERENCE:-}"
GRIPPER_OPEN_THRESHOLD="${PI05_GRIPPER_OPEN_THRESHOLD:-0.0}"
HOLD_ARM_ON_CLOSE="${PI05_HOLD_ARM_ON_CLOSE:-false}"
SIM_POSITION_GAIN="${PI05_SIM_POSITION_GAIN:-0.5}"
ROLLOUT_RECOVERY_SESSION_DIR="${PI05_ROLLOUT_RECOVERY_SESSION_DIR:-}"
LAUNCH_MOVEIT_FOR_RECOVERY="${PI05_LAUNCH_MOVEIT_FOR_RECOVERY:-false}"
RECORD_VIDEO="${PI05_RECORD_VIDEO:-false}"
INFERENCE_READY_POLLS="${PI05_INFERENCE_READY_POLLS:-900}"

GUI="${1:-true}"
EPISODE="${2:-15001}"

if [[ "$GUI" != "true" && "$GUI" != "false" ]]; then
  echo "ERROR: GUI must be true or false, got: $GUI" >&2
  exit 2
fi
if ! [[ "$INFERENCE_SEED" =~ ^-?[0-9]+$ ]]; then
  echo "ERROR: PI05_SEED must be an integer, got: $INFERENCE_SEED" >&2
  exit 2
fi
if [[ "$STATE_GRIPPER_MODE" != "binary_threshold_0.12" \
   && "$STATE_GRIPPER_MODE" != "continuous_0_1_closed_0.629rad" \
   && "$STATE_GRIPPER_MODE" != "continuous_radians_0_0.8" ]]; then
  echo "ERROR: unsupported PI05_STATE_GRIPPER_MODE: $STATE_GRIPPER_MODE" >&2
  exit 2
fi
if [[ "$FIXED_NOISE_PER_REPLAN" != "true" && "$FIXED_NOISE_PER_REPLAN" != "false" ]]; then
  echo "ERROR: PI05_FIXED_NOISE_PER_REPLAN must be true or false, got: $FIXED_NOISE_PER_REPLAN" >&2
  exit 2
fi
if ! [[ "$ENSEMBLE_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PI05_ENSEMBLE_SIZE must be a positive integer, got: $ENSEMBLE_SIZE" >&2
  exit 2
fi
if ! [[ "$MAX_ARM_STEP_RAD" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
  echo "ERROR: PI05_MAX_ARM_STEP_RAD must be non-negative, got: $MAX_ARM_STEP_RAD" >&2
  exit 2
fi
if ! [[ "$PHYSICAL_ARM_RESIDUAL_SCALE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
  echo "ERROR: PI05_PHYSICAL_ARM_RESIDUAL_SCALE must be numeric, got: $PHYSICAL_ARM_RESIDUAL_SCALE" >&2
  exit 2
fi
if ! [[ "$TRACE_FIRST_CHUNKS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: PI05_TRACE_FIRST_CHUNKS must be a non-negative integer, got: $TRACE_FIRST_CHUNKS" >&2
  exit 2
fi
if [[ "$TRACE_MULTIMODAL" != "true" && "$TRACE_MULTIMODAL" != "false" ]]; then
  echo "ERROR: PI05_TRACE_MULTIMODAL must be true or false" >&2
  exit 2
fi
if ! [[ "$ACTION_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PI05_ACTION_CHUNK_SIZE must be a positive integer, got: $ACTION_CHUNK_SIZE" >&2
  exit 2
fi
if [[ "$HOLD_GRIPPER_PER_REPLAN" != "true" && "$HOLD_GRIPPER_PER_REPLAN" != "false" ]]; then
  echo "ERROR: PI05_HOLD_GRIPPER_PER_REPLAN must be true or false" >&2
  exit 2
fi
if [[ "$RTC_ENABLED" != "true" && "$RTC_ENABLED" != "false" ]]; then
  echo "ERROR: PI05_RTC_ENABLED must be true or false" >&2
  exit 2
fi
if [[ "$DYNAMIC_TASK_PROMPT" != "true" && "$DYNAMIC_TASK_PROMPT" != "false" ]]; then
  echo "ERROR: PI05_DYNAMIC_TASK_PROMPT must be true or false" >&2
  exit 2
fi
if [[ "$LAUNCH_MOVEIT_FOR_RECOVERY" != "true" && "$LAUNCH_MOVEIT_FOR_RECOVERY" != "false" ]]; then
  echo "ERROR: PI05_LAUNCH_MOVEIT_FOR_RECOVERY must be true or false" >&2
  exit 2
fi
if [[ "$RECORD_VIDEO" != "true" && "$RECORD_VIDEO" != "false" ]]; then
  echo "ERROR: PI05_RECORD_VIDEO must be true or false" >&2
  exit 2
fi
if ! [[ "$INFERENCE_READY_POLLS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PI05_INFERENCE_READY_POLLS must be a positive integer" >&2
  exit 2
fi
if [[ "$LAUNCH_MOVEIT_FOR_RECOVERY" == "true" && -z "$ROLLOUT_RECOVERY_SESSION_DIR" ]]; then
  echo "ERROR: MoveIt recovery launch requires PI05_ROLLOUT_RECOVERY_SESSION_DIR" >&2
  exit 2
fi
if ! [[ "$RTC_EXECUTION_HORIZON" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PI05_RTC_EXECUTION_HORIZON must be a positive integer" >&2
  exit 2
fi
INFERENCE_CONSUMED_STEPS="$ACTION_CHUNK_SIZE"
if [[ "$RTC_ENABLED" == "true" ]]; then
  INFERENCE_CONSUMED_STEPS="$RTC_EXECUTION_HORIZON"
fi
if ! [[ "$RTC_MAX_GUIDANCE_WEIGHT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
  echo "ERROR: PI05_RTC_MAX_GUIDANCE_WEIGHT must be positive" >&2
  exit 2
fi
case "$RTC_PREFIX_ATTENTION_SCHEDULE" in
  EXP|LINEAR|ONES|ZEROS) ;;
  *) echo "ERROR: unsupported PI05_RTC_PREFIX_ATTENTION_SCHEDULE" >&2; exit 2 ;;
esac

case "$EPISODE" in
  # Exact v8 formal-success positions.  15001--15004 intentionally share
  # one scene but use four distinct demonstrated trajectory styles.
  15001|15002|15003|15004) PEG_X=0.105; PEG_Y=0.258; HOLE_X=-0.108; HOLE_Y=0.236 ;;
  15005) PEG_X=0.110; PEG_Y=0.256; HOLE_X=-0.116; HOLE_Y=0.239 ;;
  15006) PEG_X=0.122; PEG_Y=0.255; HOLE_X=-0.114; HOLE_Y=0.259 ;;
  15007) PEG_X=0.125; PEG_Y=0.242; HOLE_X=-0.105; HOLE_Y=0.238 ;;
  15008) PEG_X=0.111; PEG_Y=0.244; HOLE_X=-0.112; HOLE_Y=0.234 ;;
  # Third exact-fit pilot30 position group (episodes 0021--0030).
  15009) PEG_X=0.099; PEG_Y=0.261; HOLE_X=-0.128; HOLE_Y=0.236 ;;
  13001) PEG_X=-0.105; PEG_Y=0.258; HOLE_X=0.126; HOLE_Y=0.227 ;;
  13002) PEG_X=0.015;  PEG_Y=0.241; HOLE_X=0.137; HOLE_Y=0.265 ;;
  13003) PEG_X=0.066;  PEG_Y=0.272; HOLE_X=-0.070; HOLE_Y=0.251 ;;
  13004) PEG_X=0.091;  PEG_Y=0.229; HOLE_X=-0.141; HOLE_Y=0.250 ;;
  13005) PEG_X=0.104;  PEG_Y=0.228; HOLE_X=-0.074; HOLE_Y=0.222 ;;
  13006) PEG_X=0.140;  PEG_Y=0.226; HOLE_X=-0.011; HOLE_Y=0.227 ;;
  13008) PEG_X=-0.110; PEG_Y=0.273; HOLE_X=0.031; HOLE_Y=0.265 ;;
  13009) PEG_X=-0.182; PEG_Y=0.237; HOLE_X=-0.042; HOLE_Y=0.232 ;;
  13010) PEG_X=0.189;  PEG_Y=0.222; HOLE_X=-0.141; HOLE_Y=0.233 ;;
  # Held-out positions from the 24-episode ground-validation split.
  14003) PEG_X=0.015;  PEG_Y=0.244; HOLE_X=-0.170; HOLE_Y=0.227 ;;
  14012) PEG_X=0.159;  PEG_Y=0.250; HOLE_X=0.026;  HOLE_Y=0.254 ;;
  14013) PEG_X=-0.197; PEG_Y=0.223; HOLE_X=-0.040; HOLE_Y=0.275 ;;
  *)
    echo "ERROR: unsupported episode $EPISODE (pilot positions: 15001-15009; legacy ids remain available)" >&2
    exit 2
    ;;
esac

for required in "$ROS_SETUP_FILE" "$SETUP_FILE" "$INFERENCE_SCRIPT" "$ROS_SIDE_SCRIPT" "$RESULT_WRITER"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required file is missing: $required" >&2
    exit 2
  fi
done
if [[ ! -f "$CHECKPOINT/model.safetensors" && ! -f "$CHECKPOINT/adapter_model.safetensors" ]]; then
  echo "ERROR: checkpoint has neither model.safetensors nor adapter_model.safetensors: $CHECKPOINT" >&2
  exit 2
fi
if [[ "$RECORD_VIDEO" == "true" && ! -e "$VIDEO_RECORDER" ]]; then
  echo "ERROR: video recorder is missing: $VIDEO_RECORDER" >&2
  exit 2
fi
if [[ -n "$ACTION_OUT_ADAPTER" && ! -f "$ACTION_OUT_ADAPTER" ]]; then
  echo "ERROR: PI05_ACTION_OUT_ADAPTER is missing: $ACTION_OUT_ADAPTER" >&2
  exit 2
fi
if [[ -n "$LORA_ADAPTER" ]]; then
  for required in "$LORA_ADAPTER/adapter_config.json" "$LORA_ADAPTER/adapter_model.safetensors"; do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: PI05_LORA_ADAPTER file is missing: $required" >&2
      exit 2
    fi
  done
fi
if [[ -n "$STAGE_LORA_ADAPTER" ]]; then
  if [[ -z "$LORA_ADAPTER" || -z "$STAGE_LORA_REFERENCE" ]]; then
    echo "ERROR: stage LoRA requires PI05_LORA_ADAPTER and PI05_STAGE_LORA_REFERENCE" >&2
    exit 2
  fi
  read -r -a STAGE_LORA_REFERENCE_VALUES <<<"$STAGE_LORA_REFERENCE"
  if [[ "${#STAGE_LORA_REFERENCE_VALUES[@]}" -ne 6 ]]; then
    echo "ERROR: PI05_STAGE_LORA_REFERENCE must contain six joint values" >&2
    exit 2
  fi
  for required in "$STAGE_LORA_ADAPTER/adapter_config.json" "$STAGE_LORA_ADAPTER/adapter_model.safetensors"; do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: PI05_STAGE_LORA_ADAPTER file is missing: $required" >&2
      exit 2
    fi
  done
fi
if [[ -n "$FINAL_STAGE_LORA_ADAPTER" ]]; then
  if [[ -z "$STAGE_LORA_ADAPTER" || -z "$FINAL_STAGE_LORA_REFERENCE" ]]; then
    echo "ERROR: final stage requires stage adapter and final-stage reference" >&2
    exit 2
  fi
  read -r -a FINAL_STAGE_LORA_REFERENCE_VALUES <<<"$FINAL_STAGE_LORA_REFERENCE"
  if [[ "${#FINAL_STAGE_LORA_REFERENCE_VALUES[@]}" -ne 6 ]]; then
    echo "ERROR: PI05_FINAL_STAGE_LORA_REFERENCE must contain six values" >&2
    exit 2
  fi
  for required in "$FINAL_STAGE_LORA_ADAPTER/adapter_config.json" "$FINAL_STAGE_LORA_ADAPTER/adapter_model.safetensors"; do
    [[ -f "$required" ]] || { echo "ERROR: missing final-stage file: $required" >&2; exit 2; }
  done
fi
if [[ -n "$GRASP_STAGE_LORA_ADAPTER" ]]; then
  if [[ -z "$FINAL_STAGE_LORA_ADAPTER" || -z "$GRASP_STAGE_LORA_REFERENCE" ]]; then
    echo "ERROR: grasp stage requires final-stage adapter and grasp-stage reference" >&2
    exit 2
  fi
  read -r -a GRASP_STAGE_LORA_REFERENCE_VALUES <<<"$GRASP_STAGE_LORA_REFERENCE"
  [[ "${#GRASP_STAGE_LORA_REFERENCE_VALUES[@]}" -eq 6 ]] || { echo "ERROR: grasp-stage reference needs six values" >&2; exit 2; }
  for required in "$GRASP_STAGE_LORA_ADAPTER/adapter_config.json" "$GRASP_STAGE_LORA_ADAPTER/adapter_model.safetensors"; do
    [[ -f "$required" ]] || { echo "ERROR: missing grasp-stage file: $required" >&2; exit 2; }
  done
fi
if [[ -n "$POST_GRASP_LORA_ADAPTER" ]]; then
  if [[ -z "$STAGE_LORA_ADAPTER" || -z "$POST_GRASP_LORA_REFERENCE" ]]; then
    echo "ERROR: post-grasp adapter requires stage adapter and six-joint reference" >&2
    exit 2
  fi
  read -r -a POST_GRASP_LORA_REFERENCE_VALUES <<<"$POST_GRASP_LORA_REFERENCE"
  [[ "${#POST_GRASP_LORA_REFERENCE_VALUES[@]}" -eq 6 ]] || { echo "ERROR: post-grasp reference needs six values" >&2; exit 2; }
  for required in "$POST_GRASP_LORA_ADAPTER/adapter_config.json" "$POST_GRASP_LORA_ADAPTER/adapter_model.safetensors"; do
    [[ -f "$required" ]] || { echo "ERROR: missing post-grasp file: $required" >&2; exit 2; }
  done
fi
if [[ -n "$INSERTION_LORA_ADAPTER" ]]; then
  if [[ -z "$POST_GRASP_LORA_ADAPTER" || -z "$INSERTION_LORA_REFERENCE" ]]; then
    echo "ERROR: insertion adapter requires post-grasp adapter and six-joint reference" >&2
    exit 2
  fi
  read -r -a INSERTION_LORA_REFERENCE_VALUES <<<"$INSERTION_LORA_REFERENCE"
  [[ "${#INSERTION_LORA_REFERENCE_VALUES[@]}" -eq 6 ]] || { echo "ERROR: insertion reference needs six values" >&2; exit 2; }
  for required in "$INSERTION_LORA_ADAPTER/adapter_config.json" "$INSERTION_LORA_ADAPTER/adapter_model.safetensors"; do
    [[ -f "$required" ]] || { echo "ERROR: missing insertion file: $required" >&2; exit 2; }
  done
fi
if [[ -n "$CONTACT_LORA_ADAPTER" ]]; then
  if [[ -z "$INSERTION_LORA_ADAPTER" || -z "$CONTACT_LORA_REFERENCE" ]]; then
    echo "ERROR: contact adapter requires insertion adapter and six-joint reference" >&2
    exit 2
  fi
  read -r -a CONTACT_LORA_REFERENCE_VALUES <<<"$CONTACT_LORA_REFERENCE"
  [[ "${#CONTACT_LORA_REFERENCE_VALUES[@]}" -eq 6 ]] || { echo "ERROR: contact reference needs six values" >&2; exit 2; }
  for required in "$CONTACT_LORA_ADAPTER/adapter_config.json" "$CONTACT_LORA_ADAPTER/adapter_model.safetensors"; do
    [[ -f "$required" ]] || { echo "ERROR: missing contact file: $required" >&2; exit 2; }
  done
fi
if [[ -n "$INSERTION_FEEDBACK_ADAPTER" ]]; then
  if [[ -z "$CONTACT_LORA_ADAPTER" ]]; then
    echo "ERROR: insertion feedback adapter requires the contact-stage FSM" >&2
    exit 2
  fi
  test -s "$INSERTION_FEEDBACK_ADAPTER" || {
    echo "ERROR: missing insertion feedback adapter: $INSERTION_FEEDBACK_ADAPTER" >&2
    exit 2
  }
fi
if [[ -n "$GRIPPER_CLOSE_REFERENCE" ]]; then
  read -r -a GRIPPER_CLOSE_REFERENCE_VALUES <<<"$GRIPPER_CLOSE_REFERENCE"
  if [[ "${#GRIPPER_CLOSE_REFERENCE_VALUES[@]}" -ne 6 ]]; then
    echo "ERROR: PI05_GRIPPER_CLOSE_REFERENCE must contain six joint values" >&2
    exit 2
  fi
fi
if [[ -n "$GRIPPER_OPEN_REFERENCE" ]]; then
  read -r -a GRIPPER_OPEN_REFERENCE_VALUES <<<"$GRIPPER_OPEN_REFERENCE"
  if [[ "${#GRIPPER_OPEN_REFERENCE_VALUES[@]}" -ne 6 ]]; then
    echo "ERROR: PI05_GRIPPER_OPEN_REFERENCE must contain six joint values" >&2
    exit 2
  fi
fi
if [[ "$HOLD_ARM_ON_CLOSE" != "true" && "$HOLD_ARM_ON_CLOSE" != "false" ]]; then
  echo "ERROR: PI05_HOLD_ARM_ON_CLOSE must be true or false" >&2
  exit 2
fi

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
LOG_DIR="$WS_DIR/artifacts/gazebo_pi05_v9_absolute_$RUN_TAG"
mkdir -p "$LOG_DIR" /tmp/pap_moe_roslog
if [[ "$ROLLOUT_RECOVERY_SESSION_DIR" == "auto" ]]; then
  ROLLOUT_RECOVERY_SESSION_DIR="$LOG_DIR/rollout_recovery_session"
fi
export ROS_LOG_DIR=/tmp/pap_moe_roslog

rm -f /tmp/ur3_inference_ready.txt /tmp/ur3_joint_state.txt \
  /tmp/ur3_action.txt /tmp/ur3_action_chunk.npy \
  /tmp/ur3_action_chunk_tmp.npy /tmp/ur3_camera0.npy \
  /tmp/ur3_camera1.npy /tmp/ur3_force.npy \
  /tmp/ur3_force_fast.npy /tmp/ur3_force_slow.npy \
  /tmp/ur3_state_history.npy /tmp/ur3_visual_quality.npy \
  /tmp/ur3_pap_moe_observation_meta.json

CHILD_PIDS=()
CHILD_PROCESS_GROUPS=()
GAZEBO_PROCESS_GROUP=""
VIDEO_PID=""
terminate_process_group() {
  local group_id="$1"
  [[ -n "$group_id" ]] || return 0
  kill -0 -- "-$group_id" 2>/dev/null || return 0
  kill -TERM -- "-$group_id" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 -- "-$group_id" 2>/dev/null || return 0
    sleep 0.1
  done
  # ros2 launch can exit before Ignition or move_group.  Those orphaned
  # children keep FastDDS ports alive and prevent the next controller_manager
  # from starting, so cleanup must escalate for this exact run's process group.
  kill -KILL -- "-$group_id" 2>/dev/null || true
}
cleanup() {
  trap - EXIT INT TERM
  # Let OpenCV write the MP4 trailer before Gazebo and its camera topics are
  # torn down.  Sending SIGTERM without waiting can leave a large but
  # undecodable file with no `moov` atom.
  if [[ -n "$VIDEO_PID" ]] && kill -0 "$VIDEO_PID" 2>/dev/null; then
    kill -TERM "$VIDEO_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$VIDEO_PID" 2>/dev/null || break
      sleep 0.1
    done
  fi
  for group_id in "${CHILD_PROCESS_GROUPS[@]:-}"; do
    terminate_process_group "$group_id"
  done
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
# A signal must terminate the launcher after cleanup. Using cleanup itself as
# the signal handler returns to an interrupted `wait` on some bash versions,
# leaving a completed recovery case alive indefinitely.
trap 'exit 130' INT TERM

echo "Pi0.5 v9 absolute-action Gazebo reproduction"
echo "  checkpoint: $CHECKPOINT"
echo "  state gripper: $STATE_GRIPPER_MODE"
echo "  inference:  $INFERENCE_SCRIPT"
echo "  episode:    $EPISODE"
echo "  peg:        ($PEG_X, $PEG_Y)"
echo "  hole:       ($HOLE_X, $HOLE_Y)"
echo "  GUI:        $GUI"
echo "  seed:       $INFERENCE_SEED"
echo "  fixed noise:$FIXED_NOISE_PER_REPLAN"
echo "  ensemble:   $ENSEMBLE_SIZE"
echo "  physical gain: $PHYSICAL_ARM_RESIDUAL_SCALE"
echo "  arm limit:  $MAX_ARM_STEP_RAD rad/step"
echo "  trace chunks: $TRACE_FIRST_CHUNKS"
echo "  multimodal shadow trace: $TRACE_MULTIMODAL"
echo "  requested execution prefix: $ACTION_CHUNK_SIZE"
echo "  effective execution prefix: $INFERENCE_CONSUMED_STEPS"
echo "  hold gripper/replan: $HOLD_GRIPPER_PER_REPLAN"
echo "  RTC:        $RTC_ENABLED (horizon=$RTC_EXECUTION_HORIZON, weight=$RTC_MAX_GUIDANCE_WEIGHT, schedule=$RTC_PREFIX_ATTENTION_SCHEDULE)"
echo "  task:       $TASK_PROMPT (dynamic=$DYNAMIC_TASK_PROMPT)"
echo "  recovery:   ${ROLLOUT_RECOVERY_SESSION_DIR:-disabled}"
echo "  recovery IK:${LAUNCH_MOVEIT_FOR_RECOVERY}"
echo "  record video:$RECORD_VIDEO"
echo "  logs:       $LOG_DIR"

setsid ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py \
  gazebo_gui:="$GUI" launch_rviz:=false sim_position_gain:="$SIM_POSITION_GAIN" \
  >"$LOG_DIR/gazebo.log" 2>&1 &
GAZEBO_PID=$!
GAZEBO_PROCESS_GROUP=$GAZEBO_PID
CHILD_PIDS+=("$GAZEBO_PID")
CHILD_PROCESS_GROUPS+=("$GAZEBO_PROCESS_GROUP")

echo "Waiting for joint states, force sensor and both cameras..."
TOPIC_READY_STREAK=0
for _ in $(seq 1 80); do
  TOPICS="$(ros2 topic list 2>/dev/null || true)"
  if grep -Fxq /joint_states <<<"$TOPICS" \
    && grep -Fxq /wrist_camera/color/image_raw <<<"$TOPICS" \
    && grep -Fxq /global_camera/color/image_raw <<<"$TOPICS" \
    && grep -Fxq /force_torque_sensor_broadcaster/wrench <<<"$TOPICS"; then
    TOPIC_READY_STREAK=$((TOPIC_READY_STREAK + 1))
  else
    TOPIC_READY_STREAK=0
  fi
  (( TOPIC_READY_STREAK >= 3 )) && break
  sleep 0.5
done
TOPICS="$(ros2 topic list 2>/dev/null || true)"
# gz_ros2_control occasionally services two parallel spawners while the third
# times out.  Retry only the missing broadcaster against the still-running
# controller manager instead of tearing down Gazebo and reloading Pi0.5.
if ! grep -Fxq /joint_states <<<"$TOPICS"; then
  echo "Retrying missing joint_state_broadcaster activation..."
  timeout 70 ros2 run controller_manager spawner joint_state_broadcaster \
    -c /controller_manager --controller-manager-timeout 60 \
    >>"$LOG_DIR/gazebo_controller_retry.log" 2>&1 || true
fi
if ! grep -Fxq /force_torque_sensor_broadcaster/wrench <<<"$TOPICS"; then
  echo "Retrying missing force_torque_sensor_broadcaster activation..."
  timeout 70 ros2 run controller_manager spawner force_torque_sensor_broadcaster \
    -c /controller_manager --controller-manager-timeout 60 \
    >>"$LOG_DIR/gazebo_controller_retry.log" 2>&1 || true
fi
for _ in $(seq 1 40); do
  TOPICS="$(ros2 topic list 2>/dev/null || true)"
  grep -Fxq /joint_states <<<"$TOPICS" \
    && grep -Fxq /force_torque_sensor_broadcaster/wrench <<<"$TOPICS" \
    && break
  sleep 0.5
done
TOPICS="$(ros2 topic list 2>/dev/null || true)"
for topic in \
  /joint_states \
  /wrist_camera/color/image_raw \
  /global_camera/color/image_raw \
  /force_torque_sensor_broadcaster/wrench; do
  if ! grep -Fxq "$topic" <<<"$TOPICS"; then
    echo "ERROR: timed out waiting for $topic; see $LOG_DIR/gazebo.log" >&2
    exit 3
  fi
done
# Topic discovery can briefly retain publishers from the previous Gazebo
# process. Require live messages from the new controller before model loading;
# otherwise a stale name can pass the gate and disappear immediately after.
for topic in /joint_states /force_torque_sensor_broadcaster/wrench; do
  # A controller creates its topic name before activation.  Under CPU load
  # the spawner may need all three 10-second service attempts, so require a
  # message (not merely discovery) and allow that retry window to finish.
  if ! timeout 45 ros2 topic echo --once "$topic" >/dev/null 2>&1; then
    if [[ "$topic" == "/joint_states" ]]; then
      controller="joint_state_broadcaster"
    else
      controller="force_torque_sensor_broadcaster"
    fi
    echo "Retrying non-publishing controller $controller..."
    timeout 20 ros2 control set_controller_state "$controller" active \
      >>"$LOG_DIR/gazebo_controller_retry.log" 2>&1 \
      || timeout 70 ros2 run controller_manager spawner "$controller" \
        -c /controller_manager --controller-manager-timeout 60 \
        >>"$LOG_DIR/gazebo_controller_retry.log" 2>&1 \
      || true
    if ! timeout 45 ros2 topic echo --once "$topic" >/dev/null 2>&1; then
      echo "ERROR: $topic exists but produced no live message after controller retry; see $LOG_DIR/gazebo.log" >&2
      exit 3
    fi
  fi
done

# The FollowJointTrajectory action name is advertised as soon as the
# controller is configured, even when its initial activation timed out.  An
# action-server-only readiness check therefore lets the first policy chunk be
# rejected with "Controller is not running".  Require the command controller
# itself to be active before spending several minutes loading Pi0.5.
trajectory_controller_active() {
  # ros2controlcli colorizes the state column when stdout is attached to some
  # environments; strip ANSI sequences before matching the final state.
  ros2 control list_controllers 2>/dev/null \
    | sed -E $'s/\x1B\[[0-9;]*[[:alpha:]]//g' \
    | awk '$1 == "joint_trajectory_controller" && $NF == "active" { found=1 } END { exit !found }'
}
if ! trajectory_controller_active; then
  echo "Retrying inactive joint_trajectory_controller..."
  timeout 20 ros2 control set_controller_state joint_trajectory_controller active \
    >>"$LOG_DIR/gazebo_controller_retry.log" 2>&1 || true
fi
if ! trajectory_controller_active; then
  echo "ERROR: joint_trajectory_controller is not active; see $LOG_DIR/gazebo.log" >&2
  exit 3
fi

if [[ "$LAUNCH_MOVEIT_FOR_RECOVERY" == "true" ]]; then
  setsid ros2 launch ur3_ft300_moveit_config move_group.launch.py \
    >"$LOG_DIR/moveit.log" 2>&1 &
  MOVEIT_PID=$!
  CHILD_PIDS+=("$MOVEIT_PID")
  CHILD_PROCESS_GROUPS+=("$MOVEIT_PID")
  echo "Waiting for MoveIt /compute_ik service..."
  for _ in $(seq 1 120); do
    ros2 service list 2>/dev/null | grep -Fxq /compute_ik && break
    sleep 0.5
  done
  if ! ros2 service list 2>/dev/null | grep -Fxq /compute_ik; then
    echo "ERROR: /compute_ik is unavailable; see $LOG_DIR/moveit.log" >&2
    exit 3
  fi
fi

INFERENCE_ARGS=(
  --checkpoint "$CHECKPOINT"
  --state-gripper-mode "$STATE_GRIPPER_MODE"
  --seed "$INFERENCE_SEED"
  --ensemble-size "$ENSEMBLE_SIZE"
  --physical-arm-residual-scale "$PHYSICAL_ARM_RESIDUAL_SCALE"
  --max-arm-step-rad "$MAX_ARM_STEP_RAD"
  --trace-dir "$LOG_DIR/online_trace"
  --trace-first-chunks "$TRACE_FIRST_CHUNKS"
  --rtc-consumed-steps "$INFERENCE_CONSUMED_STEPS"
  --task-prompt "$TASK_PROMPT"
)
if [[ "$TRACE_MULTIMODAL" == "true" ]]; then
  INFERENCE_ARGS+=(--trace-multimodal)
fi
if [[ "$DYNAMIC_TASK_PROMPT" == "true" ]]; then
  INFERENCE_ARGS+=(--enable-dynamic-task-prompt)
fi
if [[ -n "$ACTION_OUT_ADAPTER" ]]; then
  INFERENCE_ARGS+=(--action-out-adapter "$ACTION_OUT_ADAPTER")
fi
if [[ -n "$LORA_ADAPTER" ]]; then
  INFERENCE_ARGS+=(--lora-adapter "$LORA_ADAPTER")
fi
if [[ -n "$STAGE_LORA_ADAPTER" ]]; then
  INFERENCE_ARGS+=(
    --stage-lora-adapter "$STAGE_LORA_ADAPTER"
    --stage-lora-reference "${STAGE_LORA_REFERENCE_VALUES[@]}"
    --stage-lora-threshold "$STAGE_LORA_THRESHOLD"
  )
fi
if [[ -n "$FINAL_STAGE_LORA_ADAPTER" ]]; then
  INFERENCE_ARGS+=(
    --final-stage-lora-adapter "$FINAL_STAGE_LORA_ADAPTER"
    --final-stage-lora-reference "${FINAL_STAGE_LORA_REFERENCE_VALUES[@]}"
    --final-stage-lora-threshold "$FINAL_STAGE_LORA_THRESHOLD"
  )
fi
if [[ -n "$GRASP_STAGE_LORA_ADAPTER" ]]; then
  INFERENCE_ARGS+=(
    --grasp-stage-lora-adapter "$GRASP_STAGE_LORA_ADAPTER"
    --grasp-stage-lora-reference "${GRASP_STAGE_LORA_REFERENCE_VALUES[@]}"
    --grasp-stage-lora-threshold "$GRASP_STAGE_LORA_THRESHOLD"
  )
fi
if [[ -n "$POST_GRASP_LORA_ADAPTER" ]]; then
  INFERENCE_ARGS+=(
    --post-grasp-lora-adapter "$POST_GRASP_LORA_ADAPTER"
    --post-grasp-lora-reference "${POST_GRASP_LORA_REFERENCE_VALUES[@]}"
    --post-grasp-lora-threshold "$POST_GRASP_LORA_THRESHOLD"
  )
fi
if [[ -n "$INSERTION_LORA_ADAPTER" ]]; then
  INFERENCE_ARGS+=(
    --insertion-lora-adapter "$INSERTION_LORA_ADAPTER"
    --insertion-lora-reference "${INSERTION_LORA_REFERENCE_VALUES[@]}"
    --insertion-lora-threshold "$INSERTION_LORA_THRESHOLD"
  )
fi
if [[ -n "$CONTACT_LORA_ADAPTER" ]]; then
  INFERENCE_ARGS+=(
    --contact-lora-adapter "$CONTACT_LORA_ADAPTER"
    --contact-lora-reference "${CONTACT_LORA_REFERENCE_VALUES[@]}"
    --contact-lora-threshold "$CONTACT_LORA_THRESHOLD"
  )
fi
if [[ -n "$INSERTION_FEEDBACK_ADAPTER" ]]; then
  INFERENCE_ARGS+=(--insertion-feedback-adapter "$INSERTION_FEEDBACK_ADAPTER")
fi
if [[ -n "$GRIPPER_CLOSE_REFERENCE" ]]; then
  INFERENCE_ARGS+=(
    --gripper-close-reference "${GRIPPER_CLOSE_REFERENCE_VALUES[@]}"
    --gripper-close-threshold "$GRIPPER_CLOSE_THRESHOLD"
  )
fi
if [[ -n "$GRIPPER_OPEN_REFERENCE" ]]; then
  INFERENCE_ARGS+=(
    --gripper-open-reference "${GRIPPER_OPEN_REFERENCE_VALUES[@]}"
    --gripper-open-threshold "$GRIPPER_OPEN_THRESHOLD"
  )
fi
if [[ "$HOLD_ARM_ON_CLOSE" == "true" ]]; then
  INFERENCE_ARGS+=(--hold-arm-on-close-transition)
fi
if [[ "$FIXED_NOISE_PER_REPLAN" == "true" ]]; then
  INFERENCE_ARGS+=(--fixed-noise-per-replan)
fi
if [[ "$HOLD_GRIPPER_PER_REPLAN" == "true" ]]; then
  INFERENCE_ARGS+=(--hold-gripper-per-replan)
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

echo "Loading inference model..."
for _ in $(seq 1 "$INFERENCE_READY_POLLS"); do
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

"$PI_ENV_PYTHON" - "$ENSEMBLE_SIZE" "$MAX_ARM_STEP_RAD" "$PHYSICAL_ARM_RESIDUAL_SCALE" "$CHECKPOINT" "$INFERENCE_CONSUMED_STEPS" "$HOLD_GRIPPER_PER_REPLAN" "$RTC_ENABLED" "$RTC_EXECUTION_HORIZON" "$RTC_MAX_GUIDANCE_WEIGHT" "$RTC_PREFIX_ATTENTION_SCHEDULE" "$TASK_PROMPT" "$DYNAMIC_TASK_PROMPT" "$ACTION_OUT_ADAPTER" "$LORA_ADAPTER" "$STAGE_LORA_ADAPTER" "$STAGE_LORA_REFERENCE" "$STAGE_LORA_THRESHOLD" "$FINAL_STAGE_LORA_ADAPTER" "$FINAL_STAGE_LORA_REFERENCE" "$FINAL_STAGE_LORA_THRESHOLD" "$GRASP_STAGE_LORA_ADAPTER" "$GRASP_STAGE_LORA_REFERENCE" "$GRASP_STAGE_LORA_THRESHOLD" "$POST_GRASP_LORA_ADAPTER" "$POST_GRASP_LORA_REFERENCE" "$POST_GRASP_LORA_THRESHOLD" "$INSERTION_LORA_ADAPTER" "$INSERTION_LORA_REFERENCE" "$INSERTION_LORA_THRESHOLD" "$CONTACT_LORA_ADAPTER" "$CONTACT_LORA_REFERENCE" "$CONTACT_LORA_THRESHOLD" "$INSERTION_FEEDBACK_ADAPTER" "$STATE_GRIPPER_MODE" <<'PY'
import json
import math
import sys
metadata=json.load(open('/tmp/ur3_inference_ready.txt'))
checkpoint_config=json.load(open(f'{sys.argv[4]}/config.json'))
expected=(50,int(sys.argv[5]),0.1)
actual=(metadata['predicted_action_steps'],metadata['executed_action_steps'],metadata['action_dt_s'])
if actual != expected:
    raise SystemExit(f'ERROR: online horizon mismatch: expected={expected}, actual={actual}')
expected_gripper_mode = (
    'continuous_radians_0_0.8'
    if sys.argv[-1] == 'continuous_radians_0_0.8'
    else (
        'binary_executed_prefix_hold_per_replan'
        if sys.argv[6] == 'true'
        else 'binary_threshold_0.5'
    )
)
if metadata.get('gripper_action_mode') != expected_gripper_mode:
    raise SystemExit(
        'ERROR: online gripper contract mismatch: '
        f"{metadata.get('gripper_action_mode')!r} != {expected_gripper_mode!r}"
    )
if metadata.get('state_gripper_mode') != sys.argv[-1]:
    raise SystemExit(
        'ERROR: online state-gripper contract mismatch: '
        f"{metadata.get('state_gripper_mode')!r} != {sys.argv[-1]!r}"
    )
if metadata.get('task_prompt') != sys.argv[11]:
    raise SystemExit('ERROR: task-prompt metadata mismatch')
if metadata.get('dynamic_task_prompt_enabled') is not (sys.argv[12] == 'true'):
    raise SystemExit('ERROR: dynamic-task-prompt metadata mismatch')
expected_adapter = str(__import__('pathlib').Path(sys.argv[13]).resolve()) if sys.argv[13] else None
if metadata.get('action_out_adapter') != expected_adapter:
    raise SystemExit('ERROR: action-output-adapter metadata mismatch')
expected_lora = str(__import__('pathlib').Path(sys.argv[14]).resolve()) if sys.argv[14] else None
if metadata.get('lora_adapter') != expected_lora:
    raise SystemExit('ERROR: LoRA-adapter metadata mismatch')
expected_stage_lora = str(__import__('pathlib').Path(sys.argv[15]).resolve()) if sys.argv[15] else None
if metadata.get('stage_lora_adapter') != expected_stage_lora:
    raise SystemExit('ERROR: stage-LoRA-adapter metadata mismatch')
expected_stage_reference = [float(value) for value in sys.argv[16].split()] if sys.argv[16] else None
if metadata.get('stage_lora_reference') != expected_stage_reference:
    raise SystemExit('ERROR: stage-LoRA-reference metadata mismatch')
if not math.isclose(metadata.get('stage_lora_threshold', -1.0), float(sys.argv[17]), abs_tol=1e-9):
    raise SystemExit('ERROR: stage-LoRA-threshold metadata mismatch')
def expected_path(raw):
    return str(__import__('pathlib').Path(raw).resolve()) if raw else None
for name, adapter_index, reference_index, threshold_index in (
    ('final_stage', 18, 19, 20),
    ('grasp_stage', 21, 22, 23),
    ('post_grasp', 24, 25, 26),
    ('insertion', 27, 28, 29),
    ('contact', 30, 31, 32),
):
    expected_adapter = expected_path(sys.argv[adapter_index])
    expected_reference = (
        [float(value) for value in sys.argv[reference_index].split()]
        if sys.argv[reference_index] else None
    )
    if metadata.get(f'{name}_lora_adapter') != expected_adapter:
        raise SystemExit(f'ERROR: {name} LoRA-adapter metadata mismatch')
    if metadata.get(f'{name}_lora_reference') != expected_reference:
        raise SystemExit(f'ERROR: {name} LoRA-reference metadata mismatch')
    if expected_adapter is not None and not math.isclose(
        metadata.get(f'{name}_lora_threshold', -1.0),
        float(sys.argv[threshold_index]),
        abs_tol=1e-9,
    ):
        raise SystemExit(f'ERROR: {name} LoRA-threshold metadata mismatch')
expected_feedback = expected_path(sys.argv[33])
if metadata.get('insertion_feedback_adapter') != expected_feedback:
    raise SystemExit('ERROR: insertion-feedback-adapter metadata mismatch')
if expected_feedback is not None and metadata.get('insertion_feedback_online_object_truth') is not False:
    raise SystemExit('ERROR: insertion feedback adapter must prohibit online object truth')
expected_ensemble = int(sys.argv[1])
expected_limit = float(sys.argv[2])
expected_physical_gain = float(sys.argv[3])
expected_ros_prefix = int(sys.argv[5])
expected_rtc = sys.argv[7] == 'true'
if not 1 <= expected_ros_prefix <= actual[1]:
    raise SystemExit(
        f'ERROR: ROS execution prefix must be in [1, {actual[1]}], '
        f'got {expected_ros_prefix}'
    )
if not 0.0 < expected_physical_gain <= 1.0:
    raise SystemExit(
        'ERROR: PI05_PHYSICAL_ARM_RESIDUAL_SCALE must be in (0, 1], '
        f'got {expected_physical_gain}'
    )
if metadata.get('ensemble_size') != expected_ensemble:
    raise SystemExit(
        f"ERROR: ensemble mismatch: {metadata.get('ensemble_size')} != {expected_ensemble}"
    )
if metadata.get('rtc_enabled') is not expected_rtc:
    raise SystemExit(
        f"ERROR: RTC mismatch: {metadata.get('rtc_enabled')} != {expected_rtc}"
    )
if expected_rtc:
    expected_horizon = int(sys.argv[8])
    expected_weight = float(sys.argv[9])
    expected_schedule = sys.argv[10]
    if metadata.get('rtc_execution_horizon') != expected_horizon:
        raise SystemExit('ERROR: RTC execution-horizon mismatch')
    if not math.isclose(
        metadata.get('rtc_max_guidance_weight', -1.0), expected_weight, abs_tol=1e-9
    ):
        raise SystemExit('ERROR: RTC guidance-weight mismatch')
    if metadata.get('rtc_prefix_attention_schedule') != expected_schedule:
        raise SystemExit('ERROR: RTC attention-schedule mismatch')
    if metadata.get('rtc_consumed_steps') != expected_ros_prefix:
        raise SystemExit('ERROR: RTC consumed-step mismatch')
    if metadata.get('rtc_action_dimensions') != 'arm_only':
        raise SystemExit('ERROR: RTC must exclude the gripper dimension')
if not math.isclose(metadata.get('max_arm_step_rad', -1.0), expected_limit, abs_tol=1e-9):
    raise SystemExit(
        f"ERROR: arm-limit mismatch: {metadata.get('max_arm_step_rad')} != {expected_limit}"
    )
if not math.isclose(
    metadata.get('physical_arm_residual_scale', -1.0),
    expected_physical_gain,
    abs_tol=1e-9,
):
    raise SystemExit(
        'ERROR: physical-arm gain mismatch: '
        f"{metadata.get('physical_arm_residual_scale')} != {expected_physical_gain}"
    )
expected_release_override = bool(
    checkpoint_config.get('use_release_gripper_override', False)
)
if metadata.get('release_gripper_override') is not expected_release_override:
    raise SystemExit(
        'ERROR: release-head online contract mismatch: '
        f"{metadata.get('release_gripper_override')} != {expected_release_override}"
    )
if expected_release_override:
    expected_threshold = checkpoint_config['release_head_probability_threshold']
    if not math.isclose(
        metadata.get('release_head_probability_threshold', -1.0),
        expected_threshold,
        abs_tol=1e-9,
    ):
        raise SystemExit(
            'ERROR: release-head threshold mismatch: '
            f"{metadata.get('release_head_probability_threshold')} != {expected_threshold}"
        )
print(
    f"Inference ready: predict {actual[0]}, model max execute {actual[1]}, "
    f"ROS prefix {expected_ros_prefix}, dt={actual[2]:.1f}s"
)
PY

if [[ "$RECORD_VIDEO" == "true" ]]; then
  "$ROS_PYTHON" "$VIDEO_RECORDER" --output "$LOG_DIR/pi05_pure_rollout.mp4" \
    --title "Pure Pi0.5 closed-loop rollout" \
    >"$LOG_DIR/video.log" 2>&1 &
  VIDEO_PID=$!
  CHILD_PIDS+=("$VIDEO_PID")
fi

echo "Starting ROS-side controller. Press Ctrl+C to stop."
ROS_SIDE_ARGS=(
  --spawn --ep "$EPISODE"
  --peg-x "$PEG_X" --peg-y "$PEG_Y" --hole-x "$HOLE_X" --hole-y "$HOLE_Y"
  --action-chunk-size "$INFERENCE_CONSUMED_STEPS"
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
  --ros-args -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/ros_side.log"
CONTROLLER_STATUS=${PIPESTATUS[0]}
set -e
set +e
"$PI_ENV_PYTHON" "$RESULT_WRITER" \
  --log "$LOG_DIR/ros_side.log" \
  --output "$LOG_DIR/result.json" \
  --policy pi05_v9_absolute \
  --checkpoint "$CHECKPOINT" \
  --episode "$EPISODE" \
  --seed "$INFERENCE_SEED" \
  --execution-prefix "$INFERENCE_CONSUMED_STEPS" \
  --physical-arm-residual-scale "$PHYSICAL_ARM_RESIDUAL_SCALE" \
  --max-arm-step-rad "$MAX_ARM_STEP_RAD"
RESULT_STATUS=$?
set -e
if (( CONTROLLER_STATUS != 0 )); then
  echo "ERROR: ROS-side controller exited with status $CONTROLLER_STATUS" >&2
  exit "$CONTROLLER_STATUS"
fi
exit "$RESULT_STATUS"

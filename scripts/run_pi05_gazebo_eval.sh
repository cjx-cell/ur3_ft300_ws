#!/usr/bin/env bash
# Guarded single-episode Gazebo evaluation for the corrected Pi0.5 baseline.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
SETUP_FILE="$WS_DIR/install/setup.bash"
PI_ENV_PYTHON="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
ROS_PYTHON="/usr/bin/python3"
PEG_SCRIPT_DIR="$WS_DIR/src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole"
INFERENCE_SCRIPT="$PEG_SCRIPT_DIR/ur3_baseline_peg_in_hole_inference.py"
ROS_SIDE_SCRIPT="$PEG_SCRIPT_DIR/ur3_pi05_peg_in_hole_ros_side.py"
CHECKPOINT="$WS_DIR/outputs/train/pi05_relative_h50_global_task_rebalanced_10k/checkpoints/010000/pretrained_model"

GUI="${1:-false}"
EPISODE="${2:-301}"

if [[ "$GUI" != "true" && "$GUI" != "false" ]]; then
  echo "ERROR: GUI must be 'true' or 'false', got: $GUI" >&2
  exit 2
fi
if [[ ! "$EPISODE" =~ ^(30[1-5]|[1-5])$ ]]; then
  echo "ERROR: episode must be 301-305 (or alias 1-5), got: $EPISODE" >&2
  exit 2
fi
if [[ ! -f "$SETUP_FILE" ]]; then
  echo "ERROR: workspace is not built: $SETUP_FILE is missing" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/model.safetensors" ]]; then
  echo "ERROR: checkpoint is missing: $CHECKPOINT" >&2
  exit 2
fi

# ROS/colcon setup hooks legitimately probe optional variables such as
# COLCON_TRACE. Temporarily disable nounset only while sourcing them.
set +u
source "$SETUP_FILE"
set -u

topic_in_list() {
  local wanted_topic="$1"
  local topic_list="$2"
  local topic
  while IFS= read -r topic; do
    [[ "$topic" == "$wanted_topic" ]] && return 0
  done <<<"$topic_list"
  return 1
}

topic_has_publishers() {
  local wanted_topic="$1"
  local info line
  info="$(ros2 topic info "$wanted_topic" 2>/dev/null || true)"
  while IFS= read -r line; do
    if [[ "$line" =~ ^Publisher[[:space:]]count:[[:space:]]([1-9][0-9]*)$ ]]; then
      return 0
    fi
  done <<<"$info"
  return 1
}

if topic_has_publishers "/clock"; then
  echo "ERROR: an existing ROS/Gazebo simulation appears to be running (/clock exists)." >&2
  echo "Stop it before starting an isolated evaluation." >&2
  exit 2
fi

export PYTHONPATH="/home/ubuntu/lerobot/src:$PEG_SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_TAG="$(date +%Y%m%d_%H%M%S)_ep${EPISODE}"
LOG_DIR="$WS_DIR/artifacts/gazebo_pi05_$RUN_TAG"
mkdir -p "$LOG_DIR" "/tmp/pap_moe_roslog"
export ROS_LOG_DIR="/tmp/pap_moe_roslog"

rm -f \
  /tmp/ur3_inference_ready.txt \
  /tmp/ur3_joint_state.txt \
  /tmp/ur3_action.txt \
  /tmp/ur3_action_chunk.npy \
  /tmp/ur3_action_chunk_tmp.npy \
  /tmp/ur3_camera0.npy \
  /tmp/ur3_camera1.npy \
  /tmp/ur3_force.npy

CHILD_PIDS=()
GAZEBO_PROCESS_GROUP=""
cleanup() {
  local pid
  trap - EXIT INT TERM
  if [[ -n "$GAZEBO_PROCESS_GROUP" ]] \
    && kill -0 -- "-$GAZEBO_PROCESS_GROUP" 2>/dev/null; then
    # Gazebo launch starts several ROS bridge processes. Signal the complete
    # isolated process group so those children cannot survive as orphans.
    kill -TERM -- "-$GAZEBO_PROCESS_GROUP" 2>/dev/null || true
  fi
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo "Pi0.5 guarded Gazebo evaluation"
echo "  checkpoint: $CHECKPOINT"
echo "  episode:    $EPISODE"
echo "  GUI:        $GUI"
echo "  logs:       $LOG_DIR"

setsid ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py \
  gazebo_gui:="$GUI" launch_rviz:=false \
  >"$LOG_DIR/gazebo.log" 2>&1 &
GAZEBO_PID="$!"
GAZEBO_PROCESS_GROUP="$GAZEBO_PID"
CHILD_PIDS+=("$GAZEBO_PID")

echo "Waiting for joint states and both cameras..."
for _ in $(seq 1 180); do
  TOPICS="$(ros2 topic list 2>/dev/null || true)"
  if topic_in_list "/joint_states" "$TOPICS" \
    && topic_in_list "/wrist_camera/color/image_raw" "$TOPICS" \
    && topic_in_list "/global_camera/color/image_raw" "$TOPICS"; then
    break
  fi
  sleep 0.5
done

TOPICS="$(ros2 topic list 2>/dev/null || true)"
for required_topic in \
  /joint_states \
  /wrist_camera/color/image_raw \
  /global_camera/color/image_raw; do
  if ! topic_in_list "$required_topic" "$TOPICS"; then
    echo "ERROR: timed out waiting for $required_topic; see $LOG_DIR/gazebo.log" >&2
    exit 3
  fi
done

"$PI_ENV_PYTHON" "$INFERENCE_SCRIPT" --checkpoint "$CHECKPOINT" \
  >"$LOG_DIR/inference.log" 2>&1 &
INFERENCE_PID="$!"
CHILD_PIDS+=("$INFERENCE_PID")

echo "Loading inference model (this machine currently needs about 2-3 minutes)..."
for _ in $(seq 1 420); do
  if [[ -f /tmp/ur3_inference_ready.txt ]]; then
    break
  fi
  if ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
    echo "ERROR: inference process exited; see $LOG_DIR/inference.log" >&2
    exit 4
  fi
  sleep 0.5
done
if [[ ! -f /tmp/ur3_inference_ready.txt ]]; then
  echo "ERROR: inference readiness timed out; see $LOG_DIR/inference.log" >&2
  exit 4
fi

echo "Inference server ready. Starting ROS-side controller."
echo "Press Ctrl+C to stop the episode."
"$ROS_PYTHON" "$ROS_SIDE_SCRIPT" --spawn --ep "$EPISODE" \
  2>&1 | tee "$LOG_DIR/ros_side.log"

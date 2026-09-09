#!/usr/bin/env bash
# Isolated multi-episode PAP-MoE v6 collection and schema validation.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
SETUP_FILE="$WS_DIR/install/setup.bash"
RECORDER="$WS_DIR/pap_moe_framework/scripts/pap_moe_peg_in_hole_record.py"
CONVERTER="$WS_DIR/pap_moe_framework/scripts/pap_moe_peg_in_hole_convert_to_lerobot.py"
DATA_ROOT="${PAP_MOE_DATA_ROOT:-$WS_DIR/pap_moe_framework/datasets/raw_v6_admittance}"
ROS_PYTHON="/usr/bin/python3"
PI_PYTHON="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"

GUI="${1:-true}"
START_EPISODE="${2:-10000}"
EPISODES="${3:-20}"
SEARCH_MODE="${4:-mixed}"
MAX_PEG_TILT="${5:-0.0}"
CONTACT_KP="${6:-}"
RANDOM_SEED="${7:-20260729}"
RECOVERY_OFFSET_MAX="${8:-0.006}"
FINE_INSERTION_STEP="${9:-0.000010}"
CAMERA_DEGRADATION_MODE="${10:-random}"
GRASP_RECOVERY_OFFSET_MIN="${11:-0.0}"
GRASP_RECOVERY_OFFSET_MAX="${12:-0.0}"
GRASP_RECOVERY_ONLY="${13:-false}"

if [[ "$GUI" != "true" && "$GUI" != "false" ]]; then
  echo "ERROR: GUI must be true or false, got: $GUI" >&2
  exit 2
fi
if [[ ! "$START_EPISODE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: start episode must be a non-negative integer, got: $START_EPISODE" >&2
  exit 2
fi
if [[ ! "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: episode count must be a positive integer, got: $EPISODES" >&2
  exit 2
fi
case "$SEARCH_MODE" in
  mixed|direct|force_gradient|admittance|spiral) ;;
  *)
    echo "ERROR: search mode must be mixed, direct, force_gradient, admittance, or spiral" >&2
    exit 2
    ;;
esac
case "$CAMERA_DEGRADATION_MODE" in
  random|normal|dropout|glare) ;;
  *)
    echo "ERROR: camera mode must be random, normal, dropout, or glare" >&2
    exit 2
    ;;
esac

set +u
source "$SETUP_FILE"
set -u

topic_has_publishers() {
  local topic="$1"
  local info line
  info="$(ros2 topic info "$topic" 2>/dev/null || true)"
  while IFS= read -r line; do
    if [[ "$line" =~ ^Publisher[[:space:]]count:[[:space:]]([1-9][0-9]*)$ ]]; then
      return 0
    fi
  done <<<"$info"
  return 1
}

action_has_servers() {
  local action="$1"
  local info line
  info="$(ros2 action info "$action" 2>/dev/null || true)"
  while IFS= read -r line; do
    if [[ "$line" =~ ^Action[[:space:]]servers:[[:space:]]([1-9][0-9]*)$ ]]; then
      return 0
    fi
  done <<<"$info"
  return 1
}

if topic_has_publishers "/clock"; then
  echo "ERROR: an existing ROS/Gazebo simulation is running." >&2
  exit 2
fi
if action_has_servers "/move_action"; then
  echo "ERROR: an existing MoveIt /move_action server is running." >&2
  echo "Stop the stale move_group before starting an isolated collection." >&2
  exit 2
fi

TASK_PREFIX="pick_up_the_peg_and_insert_it_into_the_hole"
LAST_EPISODE=$((START_EPISODE + EPISODES - 1))
for episode in $(seq "$START_EPISODE" "$LAST_EPISODE"); do
  for suffix in success failed; do
    target="$DATA_ROOT/${TASK_PREFIX}_episode_$(printf '%04d' "$episode")_${suffix}"
    if [[ -e "$target" ]]; then
      echo "ERROR: target already exists: $target" >&2
      exit 2
    fi
  done
done

RUN_TAG="$(date +%Y%m%d_%H%M%S)_ep${START_EPISODE}-${LAST_EPISODE}"
LOG_DIR="$WS_DIR/artifacts/pap_moe_v6_collection_$RUN_TAG"
mkdir -p "$LOG_DIR" "$DATA_ROOT" "/tmp/pap_moe_roslog"
export ROS_LOG_DIR="/tmp/pap_moe_roslog"

GAZEBO_PROCESS_GROUP=""
MOVEIT_PROCESS_GROUP=""
terminate_process_tree() {
  local parent_pid="$1"
  local child_pid
  while IFS= read -r child_pid; do
    [[ -n "$child_pid" ]] && terminate_process_tree "$child_pid"
  done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
  kill -TERM "$parent_pid" 2>/dev/null || true
}
cleanup() {
  trap - EXIT INT TERM
  [[ -n "$MOVEIT_PROCESS_GROUP" ]] \
    && terminate_process_tree "$MOVEIT_PROCESS_GROUP"
  [[ -n "$GAZEBO_PROCESS_GROUP" ]] \
    && terminate_process_tree "$GAZEBO_PROCESS_GROUP"
  if [[ -n "$MOVEIT_PROCESS_GROUP" ]] \
    && kill -0 -- "-$MOVEIT_PROCESS_GROUP" 2>/dev/null; then
    kill -TERM -- "-$MOVEIT_PROCESS_GROUP" 2>/dev/null || true
  fi
  if [[ -n "$GAZEBO_PROCESS_GROUP" ]] \
    && kill -0 -- "-$GAZEBO_PROCESS_GROUP" 2>/dev/null; then
    kill -TERM -- "-$GAZEBO_PROCESS_GROUP" 2>/dev/null || true
  fi
  wait "$MOVEIT_PROCESS_GROUP" "$GAZEBO_PROCESS_GROUP" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "PAP-MoE v6 isolated batch collection"
echo "  episodes:   $START_EPISODE..$LAST_EPISODE ($EPISODES)"
echo "  search:     $SEARCH_MODE"
echo "  max tilt:   $MAX_PEG_TILT rad"
echo "  contact kp: ${CONTACT_KP:-randomized}"
echo "  seed:       $RANDOM_SEED"
echo "  offset max: $RECOVERY_OFFSET_MAX m"
echo "  insert step:$FINE_INSERTION_STEP m/cycle at 100 Hz"
echo "  camera mode:$CAMERA_DEGRADATION_MODE"
echo "  grasp recovery offset: $GRASP_RECOVERY_OFFSET_MIN..$GRASP_RECOVERY_OFFSET_MAX m"
echo "  grasp recovery only: $GRASP_RECOVERY_ONLY"
echo "  GUI:        $GUI"
echo "  output:     $DATA_ROOT"
echo "  logs:       $LOG_DIR"

setsid ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py \
  gazebo_gui:="$GUI" launch_rviz:=false \
  >"$LOG_DIR/gazebo.log" 2>&1 &
GAZEBO_PROCESS_GROUP="$!"

echo "Waiting for joint states, both cameras, and native force..."
for _ in $(seq 1 240); do
  if topic_has_publishers "/joint_states" \
    && topic_has_publishers "/wrist_camera/color/image_raw" \
    && topic_has_publishers "/global_camera/color/image_raw" \
    && topic_has_publishers "/force_torque_sensor_broadcaster/wrench"; then
    break
  fi
  sleep 0.5
done
for topic in \
  /joint_states \
  /wrist_camera/color/image_raw \
  /global_camera/color/image_raw \
  /force_torque_sensor_broadcaster/wrench; do
  if ! topic_has_publishers "$topic"; then
    echo "ERROR: missing publisher for $topic; see $LOG_DIR/gazebo.log" >&2
    exit 3
  fi
done

setsid ros2 launch ur3_ft300_moveit_config move_group.launch.py \
  >"$LOG_DIR/moveit.log" 2>&1 &
MOVEIT_PROCESS_GROUP="$!"

echo "Waiting for MoveIt /move_action server..."
for _ in $(seq 1 120); do
  if action_has_servers "/move_action"; then
    break
  fi
  sleep 0.5
done
if ! action_has_servers "/move_action"; then
  echo "ERROR: /move_action has no server; see $LOG_DIR/moveit.log" >&2
  exit 4
fi

RECORDER_ARGS=(
  --episodes "$EPISODES"
  --start_episode "$START_EPISODE"
  --hz 10
  --seed "$RANDOM_SEED"
  --search-mode "$SEARCH_MODE"
  --max-peg-tilt "$MAX_PEG_TILT"
  --recovery-offset-max "$RECOVERY_OFFSET_MAX"
  --fine-insertion-step "$FINE_INSERTION_STEP"
  --camera-degradation-mode "$CAMERA_DEGRADATION_MODE"
  --grasp-recovery-offset-min "$GRASP_RECOVERY_OFFSET_MIN"
  --grasp-recovery-offset-max "$GRASP_RECOVERY_OFFSET_MAX"
  --output "$DATA_ROOT"
)
if [[ -n "$CONTACT_KP" ]]; then
  RECORDER_ARGS+=(--contact-kp "$CONTACT_KP")
fi
if [[ "$GRASP_RECOVERY_ONLY" == "true" ]]; then
  RECORDER_ARGS+=(--grasp-recovery-only)
fi

"$ROS_PYTHON" -u "$RECORDER" \
  "${RECORDER_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/recorder.log"

EPISODE_FILTERS=()
for episode in $(seq "$START_EPISODE" "$LAST_EPISODE"); do
  EPISODE_FILTERS+=("$(printf '%04d' "$episode")")
done

PYTHONPATH="/home/ubuntu/lerobot/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PI_PYTHON" "$CONVERTER" \
  --input "$DATA_ROOT" \
  --validate_only \
  --min_force_hz 50 \
  --episodes_filter "${EPISODE_FILTERS[@]}" \
  2>&1 | tee "$LOG_DIR/validation.log"

SUCCESS_COUNT=0
FAILED_COUNT=0
for episode in $(seq "$START_EPISODE" "$LAST_EPISODE"); do
  success_target="$DATA_ROOT/${TASK_PREFIX}_episode_$(printf '%04d' "$episode")_success"
  failed_target="$DATA_ROOT/${TASK_PREFIX}_episode_$(printf '%04d' "$episode")_failed"
  [[ -d "$success_target" ]] && SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  [[ -d "$failed_target" ]] && FAILED_COUNT=$((FAILED_COUNT + 1))
done

echo "PAP-MoE v6 batch schema passed."
echo "  successful trajectories: $SUCCESS_COUNT"
echo "  safe recovery failures:  $FAILED_COUNT"
echo "  logs: $LOG_DIR"

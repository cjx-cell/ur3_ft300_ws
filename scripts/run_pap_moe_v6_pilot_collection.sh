#!/usr/bin/env bash
# Isolated one-episode PAP-MoE v6 collection and schema validation.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
ROS_SETUP_FILE="/opt/ros/humble/setup.bash"
SETUP_FILE="$WS_DIR/install/setup.bash"
RECORDER="$WS_DIR/pap_moe_framework/scripts/pap_moe_peg_in_hole_record.py"
CONVERTER="$WS_DIR/pap_moe_framework/scripts/pap_moe_peg_in_hole_convert_to_lerobot.py"
PILOT_ROOT="${PAP_MOE_PILOT_ROOT:-$WS_DIR/pap_moe_framework/datasets/raw_v6_pilot}"
ROS_PYTHON="/usr/bin/python3"
PI_PYTHON="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"

GUI="${1:-true}"
EPISODE="${2:-9001}"
SEARCH_MODE="${3:-force_gradient}"
MAX_PEG_TILT="${4:-0.0}"
CONTACT_KP="${5:-}"
RANDOM_SEED="${6:-20260728}"
RECOVERY_OFFSET_MAX="${7:-0.006}"
PHYSICS_ENGINE="${8:-ignition-physics-dartsim-plugin}"
SIM_POSITION_GAIN="${9:-0.5}"
DIAGNOSTIC_MODE="${10:-false}"
FINE_INSERTION_STEP="${11:-0.000010}"
CAMERA_DEGRADATION_MODE="${12:-random}"
GRASP_RECOVERY_OFFSET_MIN="${13:-0.0}"
GRASP_RECOVERY_OFFSET_MAX="${14:-0.0}"
GRASP_RECOVERY_ONLY="${15:-false}"
TRAJECTORY_STYLE="${16:--1}"
POSITION_REPEAT_COUNT="${17:-1}"
TRAJECTORY_STYLE_OFFSET="${18:-0}"
EPISODE_COUNT="${19:-1}"
FIXED_PEG_X="${PAP_MOE_FIXED_PEG_X:-}"
FIXED_PEG_Y="${PAP_MOE_FIXED_PEG_Y:-}"
FIXED_HOLE_X="${PAP_MOE_FIXED_HOLE_X:-}"
FIXED_HOLE_Y="${PAP_MOE_FIXED_HOLE_Y:-}"

fixed_scene_count=0
for fixed_value in \
  "$FIXED_PEG_X" "$FIXED_PEG_Y" "$FIXED_HOLE_X" "$FIXED_HOLE_Y"; do
  [[ -n "$fixed_value" ]] && fixed_scene_count=$((fixed_scene_count + 1))
done
if [[ "$fixed_scene_count" -ne 0 && "$fixed_scene_count" -ne 4 ]]; then
  echo "ERROR: PAP_MOE_FIXED_PEG_X/Y and PAP_MOE_FIXED_HOLE_X/Y must all be set together" >&2
  exit 2
fi

if [[ "$GUI" != "true" && "$GUI" != "false" ]]; then
  echo "ERROR: GUI must be true or false, got: $GUI" >&2
  exit 2
fi
if [[ ! "$EPISODE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: episode must be a non-negative integer, got: $EPISODE" >&2
  exit 2
fi
if [[ ! "$EPISODE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: episode count must be a positive integer, got: $EPISODE_COUNT" >&2
  exit 2
fi
case "$SEARCH_MODE" in
  direct|force_gradient|admittance|spiral) ;;
  *)
    echo "ERROR: pilot search mode must be direct, force_gradient, admittance, or spiral" >&2
    exit 2
    ;;
esac
case "$CAMERA_DEGRADATION_MODE" in
  random|normal|dropout|glare|balanced) ;;
  *)
    echo "ERROR: camera mode must be random, normal, dropout, glare, or balanced" >&2
    exit 2
    ;;
esac
if [[ "$DIAGNOSTIC_MODE" != "true" \
  && "$PHYSICS_ENGINE" != "ignition-physics-dartsim-plugin" ]]; then
  echo "ERROR: v6 training pilots require ignition-physics-dartsim-plugin; got: $PHYSICS_ENGINE" >&2
  echo "Use diagnostic mode only for an isolated physics-engine comparison." >&2
  exit 2
fi
if [[ "$DIAGNOSTIC_MODE" != "true" && "$DIAGNOSTIC_MODE" != "false" ]]; then
  echo "ERROR: diagnostic mode must be true or false, got: $DIAGNOSTIC_MODE" >&2
  exit 2
fi
if [[ "$DIAGNOSTIC_MODE" != "true" \
  && "$SIM_POSITION_GAIN" != "0.5" && "$SIM_POSITION_GAIN" != ".5" ]]; then
  echo "ERROR: v6 training pilots require the validated sim position gain 0.5; got: $SIM_POSITION_GAIN" >&2
  echo "Use the final argument 'true' only for an isolated controller diagnostic." >&2
  exit 2
fi

set +u
source "$ROS_SETUP_FILE"
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
  echo "Stop the stale move_group before starting an isolated pilot." >&2
  exit 2
fi

TASK_PREFIX="pick_up_the_peg_and_insert_it_into_the_hole"
EPISODE_FILTERS=()
for ((episode_index=EPISODE; episode_index<EPISODE+EPISODE_COUNT; episode_index++)); do
  EPISODE_FILTERS+=("$(printf '%04d' "$episode_index")")
  for suffix in success failed; do
    target="$PILOT_ROOT/${TASK_PREFIX}_episode_$(printf '%04d' "$episode_index")_${suffix}"
    if [[ -e "$target" ]]; then
      echo "ERROR: pilot target already exists: $target" >&2
      exit 2
    fi
  done
done

RUN_TAG="$(date +%Y%m%d_%H%M%S)_ep${EPISODE}"
LOG_DIR="$WS_DIR/artifacts/pap_moe_v6_pilot_$RUN_TAG"
mkdir -p "$LOG_DIR" "$PILOT_ROOT" "/tmp/pap_moe_roslog"
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
  # Gazebo launch actions may create children in their own process groups.
  # Terminate descendants before the launch parent so they cannot be
  # re-parented and leave a stale /clock publisher behind.
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

echo "PAP-MoE v6 isolated pilot"
echo "  episode:    $EPISODE"
echo "  search:     $SEARCH_MODE"
echo "  max tilt:   $MAX_PEG_TILT rad"
echo "  contact kp: ${CONTACT_KP:-randomized}"
echo "  seed:       $RANDOM_SEED"
echo "  offset max: $RECOVERY_OFFSET_MAX m"
echo "  physics:    $PHYSICS_ENGINE"
echo "  sim gain:   $SIM_POSITION_GAIN"
echo "  diagnostic: $DIAGNOSTIC_MODE"
echo "  insert step:$FINE_INSERTION_STEP m/cycle at 100 Hz"
echo "  camera mode:$CAMERA_DEGRADATION_MODE"
echo "  grasp recovery offset: $GRASP_RECOVERY_OFFSET_MIN..$GRASP_RECOVERY_OFFSET_MAX m"
echo "  grasp recovery only: $GRASP_RECOVERY_ONLY"
echo "  trajectory style: $TRAJECTORY_STYLE"
echo "  position repeat count: $POSITION_REPEAT_COUNT"
echo "  trajectory style offset: $TRAJECTORY_STYLE_OFFSET"
echo "  episode count: $EPISODE_COUNT"
echo "  GUI:        $GUI"
echo "  output:     $PILOT_ROOT"
echo "  logs:       $LOG_DIR"

setsid ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py \
  gazebo_gui:="$GUI" launch_rviz:=false \
  physics_engine:="$PHYSICS_ENGINE" \
  sim_position_gain:="$SIM_POSITION_GAIN" \
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

# A publisher can be registered before DART has advanced far enough to emit
# its first sample.  Starting the recorder in that gap intermittently creates
# an empty episode, especially under headless camera load.  Require one actual
# message from every synchronized observation stream.
echo "Waiting for first actual observation messages..."
for topic_and_type in \
  "/joint_states|sensor_msgs/msg/JointState" \
  "/wrist_camera/color/image_raw|sensor_msgs/msg/Image" \
  "/global_camera/color/image_raw|sensor_msgs/msg/Image" \
  "/force_torque_sensor_broadcaster/wrench|geometry_msgs/msg/WrenchStamped"; do
  topic="${topic_and_type%%|*}"
  message_type="${topic_and_type#*|}"
  if ! timeout 30 ros2 topic echo "$topic" "$message_type" --once \
      >/dev/null 2>&1; then
    echo "ERROR: no actual message received from $topic" >&2
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
  --episodes "$EPISODE_COUNT"
  --start_episode "$EPISODE"
  --hz 10
  --seed "$RANDOM_SEED"
  --search-mode "$SEARCH_MODE"
  --max-peg-tilt "$MAX_PEG_TILT"
  --recovery-offset-max "$RECOVERY_OFFSET_MAX"
  --physics-engine "$PHYSICS_ENGINE"
  --sim-position-gain "$SIM_POSITION_GAIN"
  --fine-insertion-step "$FINE_INSERTION_STEP"
  --camera-degradation-mode "$CAMERA_DEGRADATION_MODE"
  --grasp-recovery-offset-min "$GRASP_RECOVERY_OFFSET_MIN"
  --grasp-recovery-offset-max "$GRASP_RECOVERY_OFFSET_MAX"
  --trajectory-style "$TRAJECTORY_STYLE"
  --trajectory-style-offset "$TRAJECTORY_STYLE_OFFSET"
  --position-repeat-count "$POSITION_REPEAT_COUNT"
  --output "$PILOT_ROOT"
)
if [[ "$fixed_scene_count" -eq 4 ]]; then
  RECORDER_ARGS+=(
    --fixed-peg-x "$FIXED_PEG_X"
    --fixed-peg-y "$FIXED_PEG_Y"
    --fixed-hole-x "$FIXED_HOLE_X"
    --fixed-hole-y "$FIXED_HOLE_Y"
  )
fi
if [[ -n "$CONTACT_KP" ]]; then
  RECORDER_ARGS+=(--contact-kp "$CONTACT_KP")
fi
if [[ "$GRASP_RECOVERY_ONLY" == "true" ]]; then
  RECORDER_ARGS+=(--grasp-recovery-only)
fi

"$ROS_PYTHON" -u "$RECORDER" \
  "${RECORDER_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/recorder.log"

if [[ "$DIAGNOSTIC_MODE" == "true" ]]; then
  echo "Controller diagnostic complete; deliberately skipping training-data validation."
  exit 0
fi

set +e
PYTHONPATH="/home/ubuntu/lerobot/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PI_PYTHON" "$CONVERTER" \
  --input "$PILOT_ROOT" \
  --episodes_filter "${EPISODE_FILTERS[@]}" \
  --validate_only \
  --min_force_hz 50 \
  2>&1 | tee "$LOG_DIR/validation.log"
VALIDATION_STATUS="${PIPESTATUS[0]}"
set -e

FAILED_TARGET=""
for episode_filter in "${EPISODE_FILTERS[@]}"; do
  candidate="$PILOT_ROOT/${TASK_PREFIX}_episode_${episode_filter}_failed"
  if [[ -d "$candidate" ]]; then
    FAILED_TARGET="$candidate"
    break
  fi
done
if [[ -n "$FAILED_TARGET" ]]; then
  if [[ "$VALIDATION_STATUS" -eq 0 ]]; then
    echo "PAP-MoE v6 schema passed, but the pilot task failed." >&2
  else
    echo "Failed pilot was rejected by the training-quality validator as expected." >&2
  fi
  echo "Keep it as a diagnostic/recovery trajectory; it is excluded from training." >&2
  exit 5
fi
if [[ "$VALIDATION_STATUS" -ne 0 ]]; then
  echo "ERROR: successful pilot failed PAP-MoE v6 validation." >&2
  exit 6
fi

echo "PAP-MoE v6 pilot passed. Inspect images and subtask timing before FROZEN."

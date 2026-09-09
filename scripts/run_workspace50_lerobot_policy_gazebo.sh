#!/usr/bin/env bash
# Evaluate LeRobot baselines or PAP-MoE on an exact Workspace50 scene.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
ROS_SETUP_FILE="/opt/ros/humble/setup.bash"
SETUP_FILE="$WS_DIR/install/setup.bash"
PI_PYTHON="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
ROS_PYTHON="/usr/bin/python3"
PEG_DIR="$WS_DIR/src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole"
MANIFEST="$WS_DIR/pap_moe_framework/datasets/workspace_50_v10_canonical/manifest.json"
ROS_SIDE="$PEG_DIR/ur3_workspace50_peg_in_hole_ros_side.py"
VIDEO_RECORDER="$WS_DIR/scripts/record_gazebo_camera_video.py"
RESULT_WRITER="$WS_DIR/scripts/write_gazebo_eval_result.py"

POLICY="${1:-}"
CHECKPOINT="${2:-}"
FORMAL_EPISODE="${3:-1}"
GUI="${4:-false}"
RECORD_VIDEO="${WORKSPACE50_RECORD_VIDEO:-true}"
PI05_ENSEMBLE_SIZE="${WORKSPACE50_PI05_ENSEMBLE_SIZE:-1}"
PI05_MAX_ARM_STEP_RAD="${WORKSPACE50_PI05_MAX_ARM_STEP_RAD:-0.0}"
PAP_MOE_BASELINE_BACKBONE_CHECKPOINT="${PAP_MOE_BASELINE_BACKBONE_CHECKPOINT:-}"
PAP_MOE_ACTION_CONDITIONING_SCALE="${PAP_MOE_ACTION_CONDITIONING_SCALE:-}"
PAP_MOE_ROUTING_SOURCE="${PAP_MOE_ROUTING_SOURCE:-physicsgate}"
PAP_MOE_EXPERT_MASK="${PAP_MOE_EXPERT_MASK:-1,1,1,1}"
POLICY_SEED="${WORKSPACE50_POLICY_SEED:-0}"
export POLICY_ACTION_CHUNK_MAX_STEP_RAD="${POLICY_ACTION_CHUNK_MAX_STEP_RAD:-0.13}"

case "$POLICY" in
  demonstration)
    INFERENCE="$PEG_DIR/ur3_workspace50_demonstration_replay.py"
    ACTION_CHUNK_SIZE=10
    INFERENCE_READY_POLLS=240
    export WORKSPACE50_EVALUATION_KIND=engineering_demonstration_replay
    ;;
  pi05)
    INFERENCE="$PEG_DIR/ur3_baseline_peg_in_hole_inference.py"
    ACTION_CHUNK_SIZE=10
    INFERENCE_READY_POLLS=1800
    ;;
  pap_moe)
    INFERENCE="$PEG_DIR/ur3_pap_moe_peg_in_hole_inference.py"
    ACTION_CHUNK_SIZE=10
    INFERENCE_READY_POLLS=1800
    ;;
  act)
    INFERENCE="$PEG_DIR/ur3_act_peg_in_hole_inference.py"
    ACTION_CHUNK_SIZE=10
    INFERENCE_READY_POLLS=240
    ;;
  diffusion)
    INFERENCE="$PEG_DIR/ur3_diffusion_peg_in_hole_inference.py"
    ACTION_CHUNK_SIZE=1
    INFERENCE_READY_POLLS=240
    ;;
  *)
    echo "Usage: $0 {pi05|pap_moe|act|diffusion} CHECKPOINT [formal_episode_1_to_50] [gui_true_or_false]" >&2
    exit 2
    ;;
esac
[[ "$FORMAL_EPISODE" =~ ^([1-9]|[1-4][0-9]|50)$ ]] || {
  echo "ERROR: formal episode must be 1..50" >&2
  exit 2
}
[[ "$GUI" == "true" || "$GUI" == "false" ]] || {
  echo "ERROR: GUI must be true or false" >&2
  exit 2
}
[[ "$RECORD_VIDEO" == "true" || "$RECORD_VIDEO" == "false" ]] || {
  echo "ERROR: WORKSPACE50_RECORD_VIDEO must be true or false" >&2
  exit 2
}
[[ "$POLICY_SEED" =~ ^[0-9]+$ ]] || {
  echo "ERROR: WORKSPACE50_POLICY_SEED must be a non-negative integer" >&2
  exit 2
}
for required in "$ROS_SETUP_FILE" "$SETUP_FILE" "$MANIFEST" \
  "$INFERENCE" "$ROS_SIDE" "$RESULT_WRITER"; do
  [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 2; }
done
if [[ ! -f "$CHECKPOINT/model.safetensors" && ! -f "$CHECKPOINT/adapter_model.safetensors" ]]; then
  echo "ERROR: checkpoint has neither model.safetensors nor adapter_model.safetensors: $CHECKPOINT" >&2
  exit 2
fi

read -r PEG_X PEG_Y HOLE_X HOLE_Y STYLE_ID GROUP_ID < <(
  jq -r --argjson episode "$FORMAL_EPISODE" '
    .episodes[] | select(.formal_episode == $episode)
    | [.peg_x,.peg_y,.hole_x,.hole_y,.trajectory_style_id,.position_group_id]
    | @tsv
  ' "$MANIFEST"
)
[[ -n "${PEG_X:-}" ]] || { echo "ERROR: episode absent from manifest" >&2; exit 2; }

set +u
source "$ROS_SETUP_FILE"
source "$SETUP_FILE"
set -u
# Isolate transport discovery between trials without changing scene/physics.
export IGN_PARTITION="workspace50_eval_${$}"
export GZ_PARTITION="$IGN_PARTITION"
export ROS_DOMAIN_ID="${WORKSPACE50_ROS_DOMAIN_ID:-77}"
exec 9>/tmp/workspace50_evaluation.lock
flock -n 9 || { echo "ERROR: another Workspace50 evaluation owns the transport files" >&2; exit 3; }
command -v ros2 >/dev/null || {
  echo "ERROR: ros2 is unavailable after sourcing $ROS_SETUP_FILE" >&2
  exit 2
}
if ros2 topic info /clock 2>/dev/null | grep -Eq '^Publisher count: [1-9]'; then
  echo "ERROR: an existing Gazebo instance is publishing /clock" >&2
  exit 2
fi

export PYTHONPATH="/home/ubuntu/lerobot/src:$PEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN_TAG="$(date +%Y%m%d_%H%M%S)_ep$(printf '%04d' "$FORMAL_EPISODE")_seed${POLICY_SEED}"
LOG_DIR="$WS_DIR/artifacts/gazebo_${POLICY}_workspace50_${RUN_TAG}"
mkdir -p "$LOG_DIR" /tmp/pap_moe_roslog
export ROS_LOG_DIR=/tmp/pap_moe_roslog

rm -f /tmp/ur3_inference_ready.txt /tmp/ur3_joint_state.txt \
  /tmp/ur3_action.txt /tmp/ur3_action_chunk.npy \
  /tmp/ur3_action_chunk_tmp.npy /tmp/ur3_camera0.npy \
  /tmp/ur3_camera1.npy /tmp/ur3_force.npy \
  /tmp/ur3_force_fast.npy /tmp/ur3_force_slow.npy \
  /tmp/ur3_state_history.npy /tmp/ur3_visual_quality.npy \
  /tmp/ur3_pap_moe_observation_meta.json

CHILD_PIDS=()
GAZEBO_GROUP=""
VIDEO_PID=""
cleanup() {
  trap - EXIT INT TERM
  # Finalize MP4 before tearing down camera topics.  OpenCV must release the
  # writer to emit the MP4 moov atom; merely signalling and exiting the parent
  # produced an unplayable video on short successful rollouts.
  if [[ -n "$VIDEO_PID" ]] && kill -0 "$VIDEO_PID" 2>/dev/null; then
    kill -TERM "$VIDEO_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$VIDEO_PID" 2>/dev/null || break
      sleep 0.1
    done
  fi
  if [[ -n "$GAZEBO_GROUP" ]] && kill -0 -- "-$GAZEBO_GROUP" 2>/dev/null; then
    kill -TERM -- "-$GAZEBO_GROUP" 2>/dev/null || true
  fi
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    [[ -z "$GAZEBO_GROUP" ]] && break
    kill -0 -- "-$GAZEBO_GROUP" 2>/dev/null || break
    sleep 0.1
  done
  if [[ -n "$GAZEBO_GROUP" ]]; then
    kill -KILL -- "-$GAZEBO_GROUP" 2>/dev/null || true
  fi
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Workspace50 $POLICY Gazebo evaluation"
echo "  checkpoint: $CHECKPOINT"
echo "  formal ep:  $FORMAL_EPISODE (group=$GROUP_ID style=$STYLE_ID)"
echo "  peg:        ($PEG_X, $PEG_Y)"
echo "  hole:       ($HOLE_X, $HOLE_Y)"
echo "  ROS prefix: $ACTION_CHUNK_SIZE"
echo "  controller arm limit: $POLICY_ACTION_CHUNK_MAX_STEP_RAD rad / 0.1s"
echo "  policy seed:$POLICY_SEED"
echo "  artifacts:  $LOG_DIR"
if [[ "$POLICY" == "pi05" || "$POLICY" == "pap_moe" || "$POLICY" == "demonstration" ]]; then
  export POLICY_PAIRED_ACTION_FILE="$LOG_DIR/paired_action_reply.npz"
  echo "  action exchange: paired-v1 (timeout abort, no legacy fallback)"
else
  unset POLICY_PAIRED_ACTION_FILE
fi

setsid ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py \
  gazebo_gui:="$GUI" launch_rviz:=false >"$LOG_DIR/gazebo.log" 2>&1 &
GAZEBO_PID=$!
GAZEBO_GROUP=$GAZEBO_PID
CHILD_PIDS+=("$GAZEBO_PID")

"$ROS_PYTHON" "$WS_DIR/scripts/check_workspace50_ros_ready.py" >"$LOG_DIR/readiness.json" 2>&1 || {
  echo "ERROR: live controller/observation readiness failed; infrastructure-invalid trial" >&2
  exit 3
}
if [[ "${WORKSPACE50_STARTUP_ONLY:-false}" == "true" ]]; then
  cat "$LOG_DIR/readiness.json"
  exit 0
fi

if [[ "${WORKSPACE50_DIAGNOSTIC_TRACE:-false}" == "true" ]]; then
  export POLICY_DIAGNOSTIC_TRACE_DIR="$LOG_DIR/diagnostic"
fi
INFERENCE_ARGS=(--checkpoint "$CHECKPOINT")
if [[ "$POLICY" == "pi05" ]]; then
  INFERENCE_ARGS+=(
    --state-gripper-mode continuous_radians_0_0.8
    --seed "$POLICY_SEED"
    --ensemble-size "$PI05_ENSEMBLE_SIZE"
    --max-arm-step-rad "$PI05_MAX_ARM_STEP_RAD"
    --physical-arm-residual-scale 1.0
    --rtc
    --rtc-execution-horizon 10
    --rtc-consumed-steps 10
    --rtc-max-guidance-weight 10.0
    --rtc-prefix-attention-schedule EXP
    --task-prompt "pick up the peg and insert it into the hole"
  )
elif [[ "$POLICY" == "pap_moe" ]]; then
  INFERENCE_ARGS+=(
    --seed "$POLICY_SEED"
    --execute-steps 10
    --expert-mask "$PAP_MOE_EXPERT_MASK"
    --routing-source "$PAP_MOE_ROUTING_SOURCE"
    --rtc
    --rtc-execution-horizon 10
    --rtc-max-guidance-weight 10.0
    --rtc-prefix-attention-schedule EXP
  )
  if [[ -n "$PAP_MOE_BASELINE_BACKBONE_CHECKPOINT" ]]; then
    INFERENCE_ARGS+=(--pi05-backbone-checkpoint "$PAP_MOE_BASELINE_BACKBONE_CHECKPOINT")
  fi
  if [[ -n "$PAP_MOE_ACTION_CONDITIONING_SCALE" ]]; then
    INFERENCE_ARGS+=(--action-conditioning-scale "$PAP_MOE_ACTION_CONDITIONING_SCALE")
  fi
elif [[ "$POLICY" == "act" ]]; then
  INFERENCE_ARGS+=(--execute-steps 10)
elif [[ "$POLICY" == "demonstration" ]]; then
  REPLAY_EPISODE="$WS_DIR/pap_moe_framework/datasets/workspace_50_v10_canonical/pick_up_the_peg_and_insert_it_into_the_hole_episode_$(printf '%04d' "$FORMAL_EPISODE")_success/data.npz"
  INFERENCE_ARGS+=(--episode "$REPLAY_EPISODE")
fi
"$PI_PYTHON" "$INFERENCE" "${INFERENCE_ARGS[@]}" >"$LOG_DIR/inference.log" 2>&1 &
INFERENCE_PID=$!
CHILD_PIDS+=("$INFERENCE_PID")
for _ in $(seq 1 "$INFERENCE_READY_POLLS"); do
  [[ -f /tmp/ur3_inference_ready.txt ]] && break
  kill -0 "$INFERENCE_PID" 2>/dev/null || {
    echo "ERROR: inference exited; see $LOG_DIR/inference.log" >&2
    exit 4
  }
  sleep 0.5
done
[[ -f /tmp/ur3_inference_ready.txt ]] || { echo "ERROR: inference timeout" >&2; exit 4; }

if [[ "$RECORD_VIDEO" == "true" ]]; then
  VIDEO_TITLE="Workspace50 $POLICY pure-model rollout"
  VIDEO_OUTPUT="$LOG_DIR/${POLICY}_pure_rollout.mp4"
  if [[ "$POLICY" == "demonstration" ]]; then
    VIDEO_TITLE="Workspace50 demonstration replay (NOT model inference)"
    VIDEO_OUTPUT="$LOG_DIR/demonstration_replay.mp4"
  fi
  "$ROS_PYTHON" "$VIDEO_RECORDER" \
    --output "$VIDEO_OUTPUT" \
    --title "$VIDEO_TITLE" \
    >"$LOG_DIR/video.log" 2>&1 &
  VIDEO_PID=$!
  CHILD_PIDS+=("$VIDEO_PID")
fi

if [[ "${WORKSPACE50_GEOMETRY_SHADOW:-false}" == "true" ]]; then
  "$ROS_PYTHON" "$WS_DIR/scripts/observe_fixture_geometry.py" --samples 600 --period 0.5 \
    >"$LOG_DIR/geometry_shadow.jsonl" 2>"$LOG_DIR/geometry_shadow.stderr" &
  CHILD_PIDS+=("$!")
fi

set +e
"$ROS_PYTHON" "$ROS_SIDE" --spawn --ep "$FORMAL_EPISODE" \
  --peg-x "$PEG_X" --peg-y "$PEG_Y" --hole-x "$HOLE_X" --hole-y "$HOLE_Y" \
  --action-chunk-size "$ACTION_CHUNK_SIZE" --ros-args -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/ros_side.log"
CONTROLLER_STATUS=${PIPESTATUS[0]}
set -e

set +e
"$PI_PYTHON" "$RESULT_WRITER" --log "$LOG_DIR/ros_side.log" \
  --output "$LOG_DIR/result.json" --policy "$POLICY" \
  --checkpoint "$CHECKPOINT" --episode "$FORMAL_EPISODE" --seed "$POLICY_SEED" \
  --execution-prefix "$ACTION_CHUNK_SIZE"
RESULT_STATUS=$?
set -e
(( CONTROLLER_STATUS == 0 )) || exit "$CONTROLLER_STATUS"
exit "$RESULT_STATUS"

#!/usr/bin/env bash
set -eo pipefail

WS=/home/ubuntu/ur3_ft300_ws
OUTPUT=${1:-$WS/pap_moe_framework/datasets/teleop_workspace_50}
EPISODES=${EPISODES:-50}
START_EPISODE=${START_EPISODE:-1}
SEED_START=${SEED_START:-1}

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
set -u

if ros2 node list 2>/dev/null | grep -Eq '/(controller_manager|servo_node|move_group|rviz2)$'; then
  echo "检测到已运行的 Gazebo/MoveIt/RViz/Servo。请先在旧终端 Ctrl+C 关闭，再重新执行本脚本。"
  exit 2
fi

cleanup() {
  trap - INT TERM EXIT
  local pids=()
  for pid in "${GUI_PID:-}" "${SERVO_PID:-}" "${RVIZ_PID:-}" \
             "${MOVE_GROUP_PID:-}" "${SIM_PID:-}"; do
    if [[ -n "$pid" ]]; then
      pids+=("$pid")
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for _ in $(seq 1 30); do
    local alive=false
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=true; fi
    done
    if [[ "$alive" == false ]]; then break; fi
    sleep 0.1
  done
  for pid in "${pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

echo "[1/4] 启动 Gazebo 与独立机械臂/夹爪控制器..."
ros2 launch ur_simulation_gz ur3_ft300_robotiq_teleop_v2.launch.py &
SIM_PID=$!

echo "等待控制器就绪..."
for _ in $(seq 1 120); do
  if ros2 control list_controllers 2>/dev/null | grep -q 'gripper_trajectory_controller.*active'; then
    break
  fi
  sleep 1
done
if ! ros2 control list_controllers 2>/dev/null | grep -q 'arm_servo_controller.*active'; then
  echo "机械臂控制器未就绪，停止。"
  exit 3
fi
if ! ros2 control list_controllers 2>/dev/null | grep -q 'gripper_trajectory_controller.*active'; then
  echo "夹爪控制器未就绪，停止。"
  exit 3
fi

echo "[2/4] 启动与原流程相同的 MoveIt move_group 和相机 RViz..."
ros2 launch ur3_ft300_moveit_config move_group.launch.py &
MOVE_GROUP_PID=$!
for _ in $(seq 1 60); do
  if ros2 node list 2>/dev/null | grep -q '^/move_group$'; then
    break
  fi
  sleep 1
done
if ! ros2 node list 2>/dev/null | grep -q '^/move_group$'; then
  echo "move_group 未就绪，停止。"
  exit 4
fi
ros2 launch ur3_ft300_moveit_config moveit_rviz.launch.py &
RVIZ_PID=$!

echo "[3/4] 启动 MoveIt Servo 与顺滑键鼠窗口..."
ros2 launch ur_simulation_gz pap_moe_keyboard_mouse_teleop_v2.launch.py launch_gui:=false &
SERVO_PID=$!
for _ in $(seq 1 60); do
  if ros2 service list 2>/dev/null | grep -q '^/servo_node/start_servo$'; then
    break
  fi
  sleep 1
done
# Start the Tk process only after Servo's service exists.  Keeping it as a
# separate child also makes startup failure and shutdown unambiguous.
ros2 run ur_simulation_gz pap_moe_keyboard_mouse_teleop_v2.py &
GUI_PID=$!
GUI_READY=false
for _ in $(seq 1 30); do
  if timeout 2 ros2 topic echo --once \
      /pap_moe/teleop_diagnostics std_msgs/msg/String >/dev/null 2>&1; then
    GUI_READY=true
    break
  fi
done
if [[ "$GUI_READY" != true ]]; then
  echo "键鼠窗口未产生诊断心跳，停止采集，避免无控制界面进入录制。"
  exit 5
fi

echo "[4/4] 开始连续 $EPISODES 条成功轨迹采集。"
/usr/bin/python3 "$WS/pap_moe_framework/scripts/pap_moe_keyboard_teleop_batch.py" \
  --episodes "$EPISODES" \
  --start-episode "$START_EPISODE" \
  --seed-start "$SEED_START" \
  --output "$OUTPUT"

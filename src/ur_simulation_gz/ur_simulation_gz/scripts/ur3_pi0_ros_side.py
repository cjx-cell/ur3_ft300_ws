#!/usr/bin/env python3
"""
UR3 Pi0 ROS2 通信端 — 系统 Python 3.10（非 conda）

功能:
  - 订阅腕部+全局相机 RGB → 写入 /tmp/ur3_camera{0,1}.npy
  - 订阅 /joint_states → 写入 /tmp/ur3_joint_state.txt
  - 读取 /tmp/ur3_action.txt → 通过 FollowJointTrajectory 发送给机械臂
  - --spawn: 在 Gazebo 中生成红色方块 + 蓝色碗

用法:
  /usr/bin/python3.10 ur3_pi0_ros_side.py
  /usr/bin/python3.10 ur3_pi0_ros_side.py --spawn --block-x 0.20 --block-y 0.35 --bowl-x -0.15 --bowl-y 0.35
"""

import argparse, threading, os, sys, subprocess, random, atexit
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

# ── 文件路径 ──
JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
ACTION_FILE = "/tmp/ur3_action.txt"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"
TARGET_SIZE = (224, 224)

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "robotiq_85_left_knuckle_joint",
]

# ── 方块/碗 (与录制脚本一致) ──
BLOCK_Z = 0.795
BOWL_Z  = 0.775
BLOCK_X_RANGE = (-0.25, 0.25)
BLOCK_Y_RANGE = (0.18, 0.42)
BOWL_X_RANGE  = (-0.25, 0.25)
BOWL_Y_RANGE  = (0.18, 0.42)
MIN_BLOCK_BOWL_DIST = 0.15
MAX_XY_SQ = 0.23  # x² + y² ≤ 0.23

BLOCK_SDF = """<sdf version='1.9'><model name='{name}'><link name='link'>
<collision name='c'><geometry><box><size>0.04 0.04 0.04</size></box></geometry></collision>
<visual name='v'><geometry><box><size>0.04 0.04 0.04</size></box></geometry>
<material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material></visual>
<inertial><mass>5</mass><inertia><ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.01</iyy><iyz>0</iyz><izz>0.01</izz></inertia></inertial></link></model></sdf>"""

BOWL_SDF = """<sdf version='1.9'><model name='{name}'><link name='link'>
<collision name='c'><geometry><cylinder><radius>0.08</radius><length>0.03</length></cylinder></geometry></collision>
<visual name='v'><geometry><cylinder><radius>0.08</radius><length>0.03</length></cylinder></geometry>
<material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<inertial><mass>1</mass><inertia><ixx>0.005</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.005</iyy><iyz>0</iyz><izz>0.005</izz></inertia></inertial></link></model></sdf>"""


def spawn_model(name, x, y, z, sdf):
    r = subprocess.run(["/opt/ros/humble/bin/ros2", "run", "ros_gz_sim", "create",
                        "-world", "simulation_world", "-name", name,
                        "-x", str(x), "-y", str(y), "-z", str(z),
                        "-string", sdf],
                       capture_output=True, timeout=10)
    return r.returncode == 0


def delete_model(name):
    subprocess.run(["ign", "service", "-s", "/world/simulation_world/remove",
                    "--reqtype", "ignition.msgs.Entity", "--reptype", "ignition.msgs.Boolean",
                    "--timeout", "1000", "-r", f'name: "{name}" type: MODEL'],
                   capture_output=True, timeout=5)


class UR3Pi0ROSSide(Node):
    def __init__(self, controller_name="joint_trajectory_controller"):
        super().__init__("ur3_pi0_ros_side")
        self.get_logger().info(f"UR3 Pi0 ROS2 端启动 (控制器: {controller_name})")

        self.bridge = CvBridge()
        self._init_files()

        self.joint_positions = np.zeros(7, dtype=np.float32)
        self.latest_wrist = None
        self.latest_global = None
        cb_group = ReentrantCallbackGroup()

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10, callback_group=cb_group)

        self.wrist_rgb_sub = self.create_subscription(
            Image, "/wrist_camera/color/image_raw", self._wrist_cb, 10, callback_group=cb_group)

        self.global_rgb_sub = self.create_subscription(
            Image, "/global_camera/color/image_raw", self._global_cb, 10, callback_group=cb_group)

        action_topic = f"/{controller_name}/follow_joint_trajectory"
        self._action_client = ActionClient(self, FollowJointTrajectory, action_topic)
        self.get_logger().info(f"Action 客户端: {action_topic}")

    def _init_files(self):
        with open(JOINT_STATE_FILE, "w") as f:
            f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")
        with open(ACTION_FILE, "w") as f:
            f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")

    def _joint_cb(self, msg: JointState):
        positions = []
        for name in ARM_JOINTS:
            if name in msg.name:
                positions.append(msg.position[msg.name.index(name)])
            else:
                positions.append(0.0)
        if len(positions) == 7:
            self.joint_positions = np.array(positions, dtype=np.float32)
            with open(JOINT_STATE_FILE, "w") as f:
                f.write(" ".join(f"{p:.6f}" for p in self.joint_positions) + "\n")

    def _preprocess(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb, TARGET_SIZE).astype(np.float32) / 255.0
        except Exception as e:
            self.get_logger().error(f"图像预处理异常: {e}")
            return None

    def _wrist_cb(self, msg: Image):
        img = self._preprocess(msg)
        if img is not None:
            self.latest_wrist = img

    def _global_cb(self, msg: Image):
        img = self._preprocess(msg)
        if img is not None:
            self.latest_global = img

    def _save_images(self):
        if self.latest_wrist is not None:
            np.save(CAMERA0_FILE, self.latest_wrist, allow_pickle=False)
        if self.latest_global is not None:
            np.save(CAMERA1_FILE, self.latest_global, allow_pickle=False)

    def _send_action(self, action):
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            return
        try:
            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory.joint_names = list(ARM_JOINTS)
            target = np.clip(action, -np.pi, np.pi)
            point = JointTrajectoryPoint()
            point.positions = target.tolist()
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(1e9 / 10)
            goal_msg.trajectory.points.append(point)
            self._action_client.send_goal_async(goal_msg)
        except Exception as e:
            self.get_logger().error(f"动作发送失败: {e}")

    def run_loop(self):
        self.get_logger().info("主循环启动 (10 Hz)...")
        rate = self.create_rate(10)
        last_action = None
        while rclpy.ok():
            try:
                self._save_images()
                if os.path.exists(ACTION_FILE):
                    mtime = os.path.getmtime(ACTION_FILE)
                    with open(ACTION_FILE, "r") as f:
                        line = f.readline().strip()
                        if line:
                            action = np.array([float(x) for x in line.split()], dtype=np.float32)
                            if mtime != last_action and np.abs(action).max() > 0.01:
                                self._send_action(action)
                                last_action = mtime
                rate.sleep()
            except Exception as e:
                self.get_logger().error(f"主循环异常: {e}")
                rate.sleep()


def main(args=None):
    parser = argparse.ArgumentParser(description="UR3 Pi0 ROS2 通信端")
    parser.add_argument("--controller", type=str, default="joint_trajectory_controller")
    parser.add_argument("--spawn", action="store_true", help="生成方块和碗")
    parser.add_argument("--block-x", type=float, default=0.20)
    parser.add_argument("--block-y", type=float, default=0.35)
    parser.add_argument("--bowl-x", type=float, default=-0.15)
    parser.add_argument("--bowl-y", type=float, default=0.35)
    parsed_args, _ = parser.parse_known_args()

    # ── Spawn objects (与录制脚本相同的位置约束) ──
    spawned = []
    if parsed_args.spawn:
        block_name, bowl_name = "test_block", "test_bowl"

        # 如果指定了位置就用指定的，否则随机（与录制脚本一致）
        bx, by = parsed_args.block_x, parsed_args.block_y
        wx, wy = parsed_args.bowl_x, parsed_args.bowl_y
        if not any(f"--{a}" in sys.argv for a in ["block-x", "block-y", "bowl-x", "bowl-y"]):
            for _ in range(100):
                bx = random.uniform(*BLOCK_X_RANGE)
                by = random.uniform(*BLOCK_Y_RANGE)
                wx = random.uniform(*BOWL_X_RANGE)
                wy = random.uniform(*BOWL_Y_RANGE)
                if (bx**2 + by**2 <= MAX_XY_SQ and wx**2 + wy**2 <= MAX_XY_SQ and
                    ((bx-wx)**2 + (by-wy)**2)**0.5 >= MIN_BLOCK_BOWL_DIST):
                    break

        print(f"Spawning block at ({bx:.2f}, {by:.2f})")
        if spawn_model(block_name, bx, by, BLOCK_Z, BLOCK_SDF.format(name=block_name)):
            print("  Block OK")
            spawned.append(block_name)
        else:
            print("  Block FAILED")

        print(f"Spawning bowl at ({wx:.2f}, {wy:.2f})")
        if spawn_model(bowl_name, wx, wy, BOWL_Z, BOWL_SDF.format(name=bowl_name)):
            print("  Bowl OK")
            spawned.append(bowl_name)
        else:
            print("  Bowl FAILED")

    if spawned:
        def cleanup():
            for name in spawned:
                print(f"Deleting {name}...")
                delete_model(name)
        atexit.register(cleanup)

    rclpy.init(args=args)
    node = UR3Pi0ROSSide(controller_name=parsed_args.controller)

    control_thread = threading.Thread(target=node.run_loop, daemon=True)
    control_thread.start()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("停止...")
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

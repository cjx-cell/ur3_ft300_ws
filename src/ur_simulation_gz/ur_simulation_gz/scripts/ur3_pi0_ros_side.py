#!/usr/bin/env python3
"""
UR3 Pi0 ROS2 通信端 — 系统 Python 3.10（非 conda）

功能:
  - 订阅腕部+全局相机 RGB → 写入 /tmp/ur3_camera{0,1}.npy
  - 订阅 /joint_states → 写入 /tmp/ur3_joint_state.txt
  - 读取 /tmp/ur3_action.txt → 通过 FollowJointTrajectory 发送给机械臂

用法:
  /usr/bin/python3.10 ur3_pi0_ros_side.py
  /usr/bin/python3.10 ur3_pi0_ros_side.py --controller joint_trajectory_controller
"""

import argparse, threading, os
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
CAMERA0_FILE = "/tmp/ur3_camera0.npy"   # 腕部相机
CAMERA1_FILE = "/tmp/ur3_camera1.npy"   # 全局相机
TARGET_SIZE = (224, 224)

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "robotiq_85_left_knuckle_joint",
]


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
            target = np.clip(action, -np.pi, np.pi)  # 动作是绝对位置，非增量
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
                            # 只在文件有更新且非零时才发送
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
    parsed_args, _ = parser.parse_known_args()

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

#!/usr/bin/env python3
"""
UR3 Pi0 数据采集 — 运行 pick_and_place 同时录制观测+动作，存为 LeRobot 格式。

用法（系统 Python 3.10，需 Gazebo + ROS 端已运行）：
  /usr/bin/python3.10 ur3_collect_dataset.py --episodes 50 --output ~/ur3_dataset

每 episode 运行一次 pick-and-place，同时以 20Hz 录制：
  - 腕部相机 (224x224 RGB) → observation.images.camera0
  - 全局相机 (224x224 RGB) → observation.images.camera1
  - 6维关节位置 → observation.state
  - 6维关节目标 → action
  - 任务文本 → task
"""

import argparse, os, sys, time, threading
import numpy as np
import cv2
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

# ── 配置 ──
ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
TARGET_SIZE = (224, 224)
RECORD_FPS = 20

# pick_and_place 关节轨迹（6维）
HOME     = [-1.834, -1.883, -1.128, -1.646,  1.572,  0.297]
ABOVE    = [-1.803, -1.942, -1.579, -1.136,  1.570, -0.202]
LIFT     = [-1.803, -1.856, -1.293, -1.508,  1.570, -0.202]
ABOVE_B  = [-0.761, -1.910, -1.230, -1.545,  1.523,  0.839]
PLACE    = [-0.765, -1.925, -1.358, -1.402,  1.523,  0.835]
GRASP_CLOSE = 0.44
GRASP_OPEN  = 0.0

TASK_TEXT = "pick up the red cube and place it into the bowl"


class PickAndPlaceRecorder(Node):
    def __init__(self, output_dir, episode_idx, record_fps=20):
        super().__init__(f"ur3_recorder_{episode_idx}")
        self.output_dir = output_dir
        self.episode_idx = episode_idx
        self.record_fps = record_fps
        self.bridge = CvBridge()

        # 最新数据缓存
        self.joint_positions = np.zeros(6, dtype=np.float32)
        self.last_wrist_img = np.zeros((*TARGET_SIZE, 3), dtype=np.float32)
        self.last_global_img = np.zeros((*TARGET_SIZE, 3), dtype=np.float32)
        self.current_target = np.zeros(6, dtype=np.float32)

        # 录制缓冲区
        self.frames = []
        self.recording = False
        self.lock = threading.Lock()

        # 订阅
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10)
        self.wrist_sub = self.create_subscription(
            Image, "/wrist_camera/color/image_raw", self._wrist_cb, 10)
        self.global_sub = self.create_subscription(
            Image, "/global_camera/color/image_raw", self._global_cb, 10)

        self.get_logger().info(f"Recorder 初始化 (episode {episode_idx})")

    def _joint_cb(self, msg):
        positions = []
        for name in ARM_JOINTS:
            if name in msg.name:
                positions.append(msg.position[msg.name.index(name)])
            else:
                positions.append(0.0)
        if len(positions) == 6:
            self.joint_positions = np.array(positions, dtype=np.float32)

    def _process_image(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb, TARGET_SIZE).astype(np.float32) / 255.0
        except Exception:
            return None

    def _wrist_cb(self, msg):
        img = self._process_image(msg)
        if img is not None:
            self.last_wrist_img = img

    def _global_cb(self, msg):
        img = self._process_image(msg)
        if img is not None:
            self.last_global_img = img

    def record_frame(self, target_joints=None):
        """录制一帧：当前关节状态 + 相机图像 + 目标动作"""
        if target_joints is not None:
            self.current_target = np.array(target_joints, dtype=np.float32)

        frame = {
            "observation.state": self.joint_positions.copy(),
            "action": self.current_target.copy(),
            "observation.images.camera0": self.last_wrist_img.copy(),
            "observation.images.camera1": self.last_global_img.copy(),
        }
        with self.lock:
            self.frames.append(frame)

    def save_episode(self):
        """保存当前 episode 到磁盘"""
        os.makedirs(self.output_dir, exist_ok=True)
        ep_dir = os.path.join(self.output_dir, f"episode_{self.episode_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        with self.lock:
            frames = list(self.frames)
            self.frames = []

        if not frames:
            self.get_logger().warn("无帧可保存")
            return

        # 保存为 npz
        states = np.stack([f["observation.state"] for f in frames])
        actions = np.stack([f["action"] for f in frames])
        cam0 = np.stack([f["observation.images.camera0"] for f in frames])
        cam1 = np.stack([f["observation.images.camera1"] for f in frames])

        np.savez_compressed(
            os.path.join(ep_dir, "data.npz"),
            state=states, action=actions,
            camera0=cam0, camera1=cam1,
            task=TASK_TEXT,
        )
        self.get_logger().info(
            f"Episode {self.episode_idx} 已保存: {len(frames)} 帧, "
            f"state={states.shape}, action={actions.shape}"
        )
        return len(frames)


def run_pick_and_place(recorder, executor, gripper_control_fn):
    """执行 pick-and-place 序列，同时录制"""
    rate = recorder.create_rate(RECORD_FPS)
    steps_per_move = int(RECORD_FPS * 2)  # 每步动作约2秒

    def move_and_record(target_joints, label, duration=2.0):
        recorder.get_logger().info(f"  {label} → {np.array2string(np.array(target_joints), precision=2)}")
        n_frames = int(RECORD_FPS * duration)
        for _ in range(n_frames):
            recorder.record_frame(target_joints)
            rate.sleep()

    # ── pick-and-place 序列 ──
    recorder.get_logger().info("开始 pick-and-place + 录制")

    # 1. Home
    move_and_record(HOME, "Home", 2.0)

    # 2. Above block
    move_and_record(ABOVE, "Above block", 2.0)

    # 3. Close gripper (arm stays at ABOVE)
    gripper_control_fn(GRASP_CLOSE)
    move_and_record(ABOVE, "Grasp (close)", 1.0)

    # 4. Lift
    move_and_record(LIFT, "Lift", 2.0)

    # 5. Above bowl
    move_and_record(ABOVE_B, "Above bowl", 2.0)

    # 6. Place
    move_and_record(PLACE, "Place", 2.0)

    # 7. Open gripper
    gripper_control_fn(GRASP_OPEN)
    move_and_record(PLACE, "Release", 1.0)

    # 8. Retract
    move_and_record(ABOVE_B, "Retract", 1.5)

    # 9. Home
    move_and_record(HOME, "Return home", 2.0)

    recorder.get_logger().info("Episode 完成")


def main():
    parser = argparse.ArgumentParser(description="UR3 Pi0 数据采集")
    parser.add_argument("--episodes", type=int, default=10, help="采集 episode 数")
    parser.add_argument("--output", type=str, default=os.path.expanduser("~/ur3_dataset"),
                        help="输出目录")
    parser.add_argument("--fps", type=int, default=20, help="录制帧率")
    parser.add_argument("--gripper_on_ros", action="store_true",
                        help="通过 ROS action 控制夹爪（需要 MoveIt 运行）")
    args = parser.parse_args()

    rclpy.init()

    # 导入 pymoveit2 用于机械臂控制
    from pymoveit2 import MoveIt2
    from rclpy.callback_groups import ReentrantCallbackGroup

    arm = MoveIt2(
        node=rclpy.create_node("pick_and_place_arm"),
        joint_names=list(ARM_JOINTS),
        base_link_name="base_link", end_effector_name="robotiq_85_base_link",
        group_name="ur_manipulator", callback_group=ReentrantCallbackGroup(),
    )
    arm.max_velocity = 0.3
    arm.max_acceleration = 0.3

    gripper = MoveIt2(
        node=rclpy.create_node("pick_and_place_gripper"),
        joint_names=["robotiq_85_left_knuckle_joint"],
        base_link_name="base_link", end_effector_name="robotiq_85_base_link",
        group_name="gripper", callback_group=ReentrantCallbackGroup(),
    )
    gripper.max_velocity = 0.3

    def move_arm(target, label):
        arm.move_to_configuration(target)
        arm.wait_until_executed()

    def control_gripper(position):
        gripper.move_to_configuration([position])
        gripper.wait_until_executed()

    # 执行多个 episodes
    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{args.episodes}")
        print(f"{'='*60}")

        recorder = PickAndPlaceRecorder(args.output, ep, args.fps)
        executor = MultiThreadedExecutor()
        executor.add_node(recorder)

        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        time.sleep(0.5)  # 等订阅稳定

        # 执行 pick-and-place
        move_arm(HOME, "Home")
        control_gripper(GRASP_OPEN)
        time.sleep(0.5)

        # 开始录制并执行
        record_thread = threading.Thread(
            target=run_pick_and_place, args=(recorder, executor, control_gripper),
            daemon=True,
        )
        record_thread.start()
        record_thread.join()

        # 保存
        n_frames = recorder.save_episode()
        print(f"Episode {ep+1}: {n_frames} 帧 已保存")

        executor.shutdown()
        spin_thread.join(timeout=1.0)

    rclpy.shutdown()
    print(f"\n数据集已保存到: {args.output}")
    print(f"用 pi0-env 运行 convert_to_lerobot.py 转换为 LeRobot 格式")


if __name__ == "__main__":
    main()

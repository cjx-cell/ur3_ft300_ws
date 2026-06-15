#!/usr/bin/env python3
"""
运行 pick_and_place 并逐帧录制数据。

改进版：
- 关节状态从 MoveIt2 内部 /joint_states 缓存实时读取
- 相机从 ROS topic 直接订阅
- 运动中逐帧录制，state 为实际关节值（全轨迹收集）

用法（需 Gazebo + MoveIt + ROS 端已运行）：
  /usr/bin/python3.10 ur3_record_pick_place.py --episodes 1
"""

import argparse, os, time, threading, subprocess
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pymoveit2 import MoveIt2
from pymoveit2.moveit2 import MoveIt2State
from sensor_msgs.msg import JointState

ALL_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "robotiq_85_left_knuckle_joint",
]
ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
GRIPPER_JOINT = ["robotiq_85_left_knuckle_joint"]
GRIPPER_JOINT_NAME = "robotiq_85_left_knuckle_joint"
IMG_SIZE = (224, 224)

HOME     = [ 0.0,   -1.57,    0.0,  -1.57,    0.0,   0.0  ]
ABOVE    = [-1.834, -1.883, -1.128, -1.646,  1.572,  0.297]
GRASP    = [-1.803, -1.942, -1.579, -1.136,  1.570, -0.202]
LIFT     = [-1.803, -1.856, -1.293, -1.508,  1.570, -0.202]
ABOVE_B  = [-0.761, -1.910, -1.230, -1.545,  1.523,  0.839]
PLACE    = [-0.765, -1.925, -1.358, -1.402,  1.523,  0.835]
GRASP_CLOSE = 0.75
GRASP_OPEN  = 0.0

TASK = "pick up the red cube and place it into the bowl"


def get_current_joints(arm):
    """从 MoveIt2 内部 /joint_states 缓存读取 7 维关节状态 [arm_6, gripper]"""
    js = arm.joint_state
    if js is None:
        return np.zeros(7, dtype=np.float32)
    pos = []
    for name in ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]:
        try:
            pos.append(js.position[js.name.index(name)])
        except ValueError:
            pos.append(0.0)
    return np.array(pos, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=str, default=os.path.expanduser("~/ur3_ft300_ws/ai-models/ur3_pick_place_raw"))
    args = parser.parse_args()

    rclpy.init()
    os.makedirs(args.output, exist_ok=True)

    # ── MoveIt2 ──
    node = rclpy.create_node("record_pp")
    cb = ReentrantCallbackGroup()
    arm = MoveIt2(node=node, joint_names=list(ALL_JOINTS[:6]),
                  base_link_name="base_link", end_effector_name="robotiq_85_base_link",
                  group_name="ur_manipulator", callback_group=cb,
                  use_move_group_action=True)
    arm.max_velocity = 1.0
    arm.max_acceleration = 1.0

    gripper = MoveIt2(node=node, joint_names=list(GRIPPER_JOINT),
                      base_link_name="base_link", end_effector_name="robotiq_85_base_link",
                      group_name="gripper", callback_group=cb,
                      use_move_group_action=True)
    gripper.max_velocity = 1.0

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    def wait_for_idle(moveit, timeout=30.0):
        """轮询等待 MoveIt2 变为 IDLE（不调用 spin_once，避免与后台 executor 冲突）"""
        start = time.time()
        while moveit.query_state() != MoveIt2State.IDLE:
            if time.time() - start > timeout:
                print("  ⚠ Timeout waiting for motion to complete")
                break
            time.sleep(0.05)

    # ── 相机订阅 ──
    class CameraCache:
        wrist = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        global_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        lock = threading.Lock()

    cam_cache = CameraCache()

    class CamNode(Node):
        def __init__(self):
            super().__init__("record_cams")
            self.bridge = CvBridge()
            cbg = ReentrantCallbackGroup()
            self.create_subscription(Image, "/wrist_camera/color/image_raw",
                                      self._wrist, 10, callback_group=cbg)
            self.create_subscription(Image, "/global_camera/color/image_raw",
                                      self._global, 10, callback_group=cbg)

        def _decode(self, msg):
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                  IMG_SIZE).astype(np.float32) / 255.0
            except Exception:
                return None

        def _wrist(self, msg):
            img = self._decode(msg)
            if img is not None:
                with cam_cache.lock:
                    cam_cache.wrist = img

        def _global(self, msg):
            img = self._decode(msg)
            if img is not None:
                with cam_cache.lock:
                    cam_cache.global_img = img

    cam_node = CamNode()
    cam_exec = MultiThreadedExecutor(2)
    cam_exec.add_node(cam_node)
    cam_thread = threading.Thread(target=cam_exec.spin, daemon=True)
    cam_thread.start()
    time.sleep(1.0)
    print("相机订阅就绪")

    # ── 录制辅助 ──
    def record_while_moving(target_6d, gripper_val, rate):
        """在 arm 或 gripper 运动期间逐帧录制，读取实际关节状态 + 相机图像"""
        action_7d = np.array(list(target_6d) + [gripper_val], dtype=np.float32)
        frames = []

        while (arm.query_state() != MoveIt2State.IDLE or
               gripper.query_state() != MoveIt2State.IDLE):
            state_7d = get_current_joints(arm)
            with cam_cache.lock:
                w = cam_cache.wrist.copy()
                g = cam_cache.global_img.copy()
            frames.append({"state": state_7d, "action": action_7d, "camera0": w, "camera1": g})
            rate.sleep()

        # 到达后补录几帧稳定状态
        for _ in range(3):
            state_7d = get_current_joints(arm)
            with cam_cache.lock:
                w = cam_cache.wrist.copy()
                g = cam_cache.global_img.copy()
            frames.append({"state": state_7d, "action": action_7d, "camera0": w, "camera1": g})
            rate.sleep()

        return frames

    total_frames = 0

    # 等待首个有效关节状态
    while arm.joint_state is None:
        rclpy.spin_once(node, timeout_sec=0.5)

    for ep in range(args.episodes):
        all_frames = []
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{args.episodes}")
        print(f"{'='*60}")

        rate = node.create_rate(20.0)

        # Step 1: Open → Home
        print("  Open → Home")
        gripper.move_to_configuration([GRASP_OPEN])
        wait_for_idle(gripper)
        arm.move_to_configuration(HOME)
        all_frames += record_while_moving(HOME, GRASP_OPEN, rate)

        # Step 2: Above block
        print("  Above block")
        arm.move_to_configuration(ABOVE)
        all_frames += record_while_moving(ABOVE, GRASP_OPEN, rate)

        # Step 3: Grasp block
        print("  Grasp block")
        arm.move_to_configuration(GRASP)
        all_frames += record_while_moving(GRASP, GRASP_OPEN, rate)

        # Step 4: Close gripper
        print("  Close gripper")
        gripper.move_to_configuration([GRASP_CLOSE])
        all_frames += record_while_moving(GRASP, GRASP_CLOSE, rate)

        # Step 5: Lift
        print("  Lift")
        arm.move_to_configuration(LIFT)
        all_frames += record_while_moving(LIFT, GRASP_CLOSE, rate)

        # Step 6: Above bowl
        print("  Above bowl")
        arm.move_to_configuration(ABOVE_B)
        all_frames += record_while_moving(ABOVE_B, GRASP_CLOSE, rate)

        # Step 7: Place
        print("  Place")
        arm.move_to_configuration(PLACE)
        all_frames += record_while_moving(PLACE, GRASP_CLOSE, rate)

        # Step 8: Open gripper
        print("  Open gripper")
        gripper.move_to_configuration([GRASP_OPEN])
        all_frames += record_while_moving(PLACE, GRASP_OPEN, rate)

        # Step 9: Retract
        print("  Retract")
        arm.move_to_configuration(ABOVE_B)
        all_frames += record_while_moving(ABOVE_B, GRASP_OPEN, rate)

        # Step 10: Home
        print("  Home")
        arm.move_to_configuration(HOME)
        all_frames += record_while_moving(HOME, GRASP_OPEN, rate)

        # 保存
        if all_frames:
            states = np.stack([f["state"] for f in all_frames])
            actions = np.stack([f["action"] for f in all_frames])
            cam0 = np.stack([f["camera0"] for f in all_frames])
            cam1 = np.stack([f["camera1"] for f in all_frames])
            ep_dir = os.path.join(args.output, f"episode_{ep:04d}")
            os.makedirs(ep_dir, exist_ok=True)
            np.savez_compressed(os.path.join(ep_dir, "data.npz"),
                                state=states, action=actions,
                                camera0=cam0, camera1=cam1, task=TASK)
            total_frames += len(all_frames)
            print(f"  Saved: {len(all_frames)} frames | "
                  f"state range=[{states.min():.1f},{states.max():.1f}]")

            # 重置方块
            try:
                subprocess.run(["/opt/ros/humble/bin/ros2", "run", "ros_gz_sim", "create",
                                "-world", "simulation_world", "-name", "pick_block",
                                "-x", "0.2", "-y", "0.35", "-z", "0.795",
                                "-string",
                                "<sdf version='1.9'><model name='pick_block'>"
                                "<link name='link'>"
                                "<collision name='collision'><geometry><box><size>0.04 0.04 0.04</size></box></geometry></collision>"
                                "<visual name='visual'><geometry><box><size>0.04 0.04 0.04</size></box></geometry>"
                                "<material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material></visual>"
                                "<inertial><mass>5</mass>"
                                "<inertia><ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.01</iyy><iyz>0</iyz><izz>0.01</izz></inertia></inertial>"
                                "</link></model></sdf>",
                                "-allow_renaming", "true"],
                               capture_output=True, timeout=5)
            except Exception:
                pass

    print(f"\nDone: {total_frames} frames, {args.episodes} episodes → {args.output}")
    print(f"Next: conda activate pi0-env && python3 ur3_convert_to_lerobot.py --input {args.output}")

    executor.shutdown()
    cam_exec.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
运行 pick_and_place 并逐帧录制数据。

改进版：
- 关节状态从 /tmp/ur3_joint_state.txt 读取（需 ROS 端先启动）
- 相机从 ROS topic 直接订阅
- 运动中逐帧录制，state 为实际关节值 + 插值混合

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

ALL_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "robotiq_85_left_knuckle_joint",
]
GRIPPER_JOINT = ["robotiq_85_left_knuckle_joint"]
IMG_SIZE = (224, 224)
JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"

HOME     = [ 0.0,   -1.57,    0.0,  -1.57,    0.0,   0.0  ]
ABOVE    = [-1.834, -1.883, -1.128, -1.646,  1.572,  0.297]
GRASP    = [-1.803, -1.942, -1.579, -1.136,  1.570, -0.202]
LIFT     = [-1.803, -1.856, -1.293, -1.508,  1.570, -0.202]
ABOVE_B  = [-0.761, -1.910, -1.230, -1.545,  1.523,  0.839]
PLACE    = [-0.765, -1.925, -1.358, -1.402,  1.523,  0.835]
GRASP_CLOSE = 0.75
GRASP_OPEN  = 0.0

TASK = "pick up the red cube and place it into the bowl"


def read_joints():
    """从 /tmp/ 文件读取关节状态（ROS 端写入，格式: 7个空格分隔浮点数）"""
    try:
        with open(JOINT_STATE_FILE, "r") as f:
            line = f.readline().strip()
            if line:
                vals = [float(x) for x in line.split()]
                if len(vals) >= 6:
                    if len(vals) == 6:
                        vals.append(0.0)
                    return np.array(vals[:7], dtype=np.float32)
    except Exception:
        pass
    return np.zeros(7, dtype=np.float32)


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
                  group_name="ur_manipulator", callback_group=cb)
    arm.max_velocity = 1.0
    arm.max_acceleration = 1.0

    gripper = MoveIt2(node=node, joint_names=list(GRIPPER_JOINT),
                      base_link_name="base_link", end_effector_name="robotiq_85_base_link",
                      group_name="gripper", callback_group=cb)
    gripper.max_velocity = 1.0

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

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
    def record_frames(arm_from_6d, arm_to_6d, grip_val, duration):
        """线性插值 state + 真实相机图像录制"""
        frames = []
        n = max(int(duration * 20), 5)
        for i in range(n):
            t = i / n
            arm_interp = np.array(arm_from_6d) * (1 - t) + np.array(arm_to_6d) * t
            state_7d = np.array(list(arm_interp) + [grip_val], dtype=np.float32)
            action_7d = np.array(list(arm_to_6d) + [grip_val], dtype=np.float32)
            with cam_cache.lock:
                w = cam_cache.wrist.copy()
                g = cam_cache.global_img.copy()
            frames.append({"state": state_7d, "action": action_7d, "camera0": w, "camera1": g})
            time.sleep(0.05)
        # 到达后补录实际关节值
        js = read_joints()
        if np.abs(js).max() > 0.01:
            frames[-1]["state"] = js
        return frames

    last_arm = np.array(HOME, dtype=np.float32)[:6]
    current_gripper = GRASP_OPEN
    total_frames = 0

    for ep in range(args.episodes):
        all_frames = []
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{args.episodes}")
        print(f"{'='*60}")

        def move_arm(arm_6d, duration=1.5):
            nonlocal last_arm, current_gripper
            from_6d = last_arm.copy()
            arm.move_to_configuration(list(arm_6d))
            frames = record_frames(from_6d, arm_6d, current_gripper, duration)
            arm.wait_until_executed()
            last_arm = np.array(arm_6d, dtype=np.float32)
            return frames

        def move_gripper(val):
            nonlocal current_gripper
            gripper.move_to_configuration([val])
            gripper.wait_until_executed()
            current_gripper = val

        print("  Open → Home")
        move_gripper(GRASP_OPEN)
        all_frames += move_arm(HOME)

        print("  Above block")
        all_frames += move_arm(ABOVE)

        print("  Grasp block")
        all_frames += move_arm(GRASP)

        print("  Close gripper")
        move_gripper(GRASP_CLOSE)
        all_frames += record_frames(GRASP, GRASP, current_gripper, 0.3)

        print("  Lift")
        all_frames += move_arm(LIFT)

        print("  Above bowl")
        all_frames += move_arm(ABOVE_B)

        print("  Place")
        all_frames += move_arm(PLACE)

        print("  Open gripper")
        move_gripper(GRASP_OPEN)
        all_frames += record_frames(PLACE, PLACE, current_gripper, 0.3)

        print("  Retract")
        all_frames += move_arm(ABOVE_B)

        print("  Home")
        all_frames += move_arm(HOME)

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

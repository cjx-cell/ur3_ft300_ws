#!/usr/bin/env python3
"""
UR3 pick-and-place data recorder.

- Spawns block AND bowl at randomized positions in Gazebo
- Runs C++ pick_and_place (passes block + bowl positions as ROS params)
- Records joint_states + 2 cameras in background at 20 Hz
- Deletes block and bowl after each episode
- Saves LeRobot-compatible .npz per episode

Run (Gazebo + MoveIt must be running):
  /usr/bin/python3 ur3_record_pick_place.py --episodes 1
"""

import argparse, os, time, threading, subprocess, random
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
IMG_SIZE = (224, 224)
STATE_DIM = 7  # 6 arm joints + 1 gripper; Pi0 auto-pads to max_state_dim=32
TASK = "pick up the red cube and place it into the bowl"

# ── World constants (match C++ pick_and_place.cpp) ──
BLOCK_Z = 0.795
BOWL_Z  = 0.775  # bowl sits on table

# Both within UR3 workspace: x² + y² ≤ 0.23 (reach radius ~0.5m from shoulder at 0,0,0.91)
# Block and bowl at least 0.15m apart.
BLOCK_X_RANGE = (-0.25, 0.25)
BLOCK_Y_RANGE = (0.18, 0.42)
BOWL_X_RANGE  = (-0.25, 0.25)
BOWL_Y_RANGE  = (0.18, 0.42)
MIN_BLOCK_BOWL_DIST = 0.15
MAX_XY_SQ = 0.23  # x² + y² ≤ 0.23

RECORD_HZ = 50  # Pi0 pretrained at 50Hz (chunk_size=50 = 1.0s action horizon)


def spawn_model(name, x, y, z, sdf_string):
    """Spawn a model in Gazebo via ros_gz_sim create. Returns True on success."""
    try:
        r = subprocess.run(
            ["/opt/ros/humble/bin/ros2", "run", "ros_gz_sim", "create",
             "-world", "simulation_world", "-name", name,
             "-x", str(x), "-y", str(y), "-z", str(z),
             "-string", sdf_string],
            capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception as e:
        print(f"  ⚠ spawn failed: {e}")
        return False


def get_model_pose(name):
    """Query a model's world pose via ign topic (Ignition Transport)."""
    import re
    try:
        r = subprocess.run(
            ["ign", "topic", "-t", "/world/simulation_world/pose/info",
             "-e", "-n", "1"],
            capture_output=True, timeout=5)
        if r.returncode == 0 and r.stdout:
            text = r.stdout.decode()
            idx = text.find(f'name: "{name}"')
            if idx >= 0:
                snippet = text[idx:idx+200]
                m = re.search(
                    r'position\s*\{\s*x:\s*([\d.e-]+)\s*y:\s*([\d.e-]+)\s*z:\s*([\d.e-]+)',
                    snippet)
                if m:
                    return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    except Exception:
        pass
    return None



def delete_model(name):
    """Delete a model from Gazebo via ign service (Ignition Transport)."""
    try:
        r = subprocess.run(
            ["ign", "service", "-s", "/world/simulation_world/remove",
             "--reqtype", "ignition.msgs.Entity",
             "--reptype", "ignition.msgs.Boolean",
             "--timeout", "1000",
             "-r", f'name: "{name}" type: MODEL'],
            capture_output=True, timeout=5)
        if r.returncode == 0 and b"data: true" in r.stdout:
            return True
        else:
            print(f"  ⚠ delete failed for '{name}': "
                  f"stdout={r.stdout.decode()[:100]} "
                  f"stderr={r.stderr.decode()[:100]}")
            return False
    except Exception as e:
        print(f"  ⚠ delete exception: {e}")
        return False


# ── SDF templates ──
BLOCK_SDF = """<sdf version='1.9'>
<model name='{name}'>
<link name='link'>
<collision name='collision'>
<geometry><box><size>0.04 0.04 0.04</size></box></geometry></collision>
<visual name='visual'>
<geometry><box><size>0.04 0.04 0.04</size></box></geometry>
<material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material></visual>
<inertial><mass>5</mass><inertia>
<ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.01</iyy><iyz>0</iyz><izz>0.01</izz>
</inertia></inertial></link></model></sdf>"""

# Bowl: flat cylinder, 0.08m radius, 0.03m tall, blue
BOWL_SDF = """<sdf version='1.9'>
<model name='{name}'>
<link name='link'>
<collision name='collision'>
<geometry><cylinder><radius>0.08</radius><length>0.03</length></cylinder></geometry></collision>
<visual name='visual'>
<geometry><cylinder><radius>0.08</radius><length>0.03</length></cylinder></geometry>
<material><ambient>0 0 0.8 1</ambient><diffuse>0 0 0.8 1</diffuse></material></visual>
<inertial><mass>1</mass><inertia>
<ixx>0.005</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.005</iyy><iyz>0</iyz><izz>0.005</izz>
</inertia></inertial></link></model></sdf>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=str,
                        default=os.path.expanduser(
                            "~/ur3_ft300_ws/ai-models/ur3_pick_place_raw"))
    args = parser.parse_args()

    rclpy.init()
    os.makedirs(args.output, exist_ok=True)

    # ── Shared buffer (thread-safe) ──
    class Buffer:
        def __init__(self):
            self.lock = threading.Lock()
            self.joint_positions = None
            self.wrist_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
            self.global_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)

    buf = Buffer()

    # ── Subscriber node ──
    class RecorderNode(Node):
        def __init__(self):
            super().__init__("record_pp")
            self.bridge = CvBridge()
            cbg = ReentrantCallbackGroup()
            self.create_subscription(JointState, "/joint_states",
                                      self._js, 10, callback_group=cbg)
            self.create_subscription(Image, "/wrist_camera/color/image_raw",
                                      self._wrist, 10, callback_group=cbg)
            self.create_subscription(Image, "/global_camera/color/image_raw",
                                      self._global, 10, callback_group=cbg)

        def _js(self, msg):
            try:
                pos = [msg.position[msg.name.index(n)] for n in ALL_JOINTS]
                with buf.lock:
                    buf.joint_positions = pos
            except ValueError:
                pass

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
                with buf.lock:
                    buf.wrist_img = img

        def _global(self, msg):
            img = self._decode(msg)
            if img is not None:
                with buf.lock:
                    buf.global_img = img

    recorder = RecorderNode()
    executor = MultiThreadedExecutor(3)
    executor.add_node(recorder)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Wait for joint states
    print("Waiting for joint states...")
    for _ in range(50):
        with buf.lock:
            if buf.joint_positions is not None:
                print("Joint states available")
                break
        time.sleep(0.1)
    else:
        print("ERROR: no joint states received")
        return

    # ── Background recording ──
    recording = threading.Event()
    recording.set()
    frames = []
    episode_active = threading.Event()  # set when C++ reaches HOME

    def recorder_thread():
        rate = recorder.create_rate(RECORD_HZ)
        while recording.is_set():
            with buf.lock:
                js = (list(buf.joint_positions) if buf.joint_positions
                      else [0.0] * 7)
                w = buf.wrist_img.copy()
                g = buf.global_img.copy()
            if episode_active.is_set():
                frames.append({
                    "state": np.array(js, dtype=np.float32),
                    "camera0": w,
                    "camera1": g,
                    "timestamp": time.time(),
                })
            rate.sleep()

    rec_thread = threading.Thread(target=recorder_thread, daemon=True)
    rec_thread.start()
    print(f"Recording started ({RECORD_HZ} Hz)")

    total_frames = 0

    try:
        for ep in range(args.episodes):
            print(f"\n{'='*60}")
            print(f"Episode {ep+1}/{args.episodes}")

            # ── Randomized positions (ensure block and bowl far enough apart) ──
            for _ in range(100):  # retry if constraints not met
                block_x = round(random.uniform(*BLOCK_X_RANGE), 3)
                block_y = round(random.uniform(*BLOCK_Y_RANGE), 3)
                bowl_x  = round(random.uniform(*BOWL_X_RANGE), 3)
                bowl_y  = round(random.uniform(*BOWL_Y_RANGE), 3)
                # Workspace: x² + y² ≤ MAX_XY_SQ (~0.48m from shoulder)
                b_ok = block_x**2 + block_y**2 <= MAX_XY_SQ
                w_ok = bowl_x**2  + bowl_y**2  <= MAX_XY_SQ
                dist = ((block_x - bowl_x)**2 + (block_y - bowl_y)**2)**0.5
                if b_ok and w_ok and dist >= MIN_BLOCK_BOWL_DIST:
                    break
            print(f"  Block: x={block_x:.3f}, y={block_y:.3f}")
            print(f"  Bowl:  x={bowl_x:.3f}, y={bowl_y:.3f}")

            block_name = f"pick_block_{ep:04d}"
            bowl_name  = f"pick_bowl_{ep:04d}"

            # ── Spawn block and bowl ──
            ok_block = spawn_model(block_name, block_x, block_y, BLOCK_Z,
                                   BLOCK_SDF.format(name=block_name))
            ok_bowl  = spawn_model(bowl_name, bowl_x, bowl_y, BOWL_Z,
                                   BOWL_SDF.format(name=bowl_name))
            if not ok_block or not ok_bowl:
                print("  ⚠ spawn failed, skipping episode")
                delete_model(block_name)
                delete_model(bowl_name)
                continue

            time.sleep(0.5)

            # ── Run C++ pick_and_place (start recording after RECORD_START) ──
            print(f"  Running C++ pick_and_place...")
            episode_active.clear()
            frames.clear()
            proc = subprocess.Popen(
                ["/opt/ros/humble/bin/ros2", "run", "ur_simulation_gz",
                 "pick_and_place",
                 "--ros-args",
                 "-p", f"block_x:={block_x}",
                 "-p", f"block_y:={block_y}",
                 "-p", f"bowl_x:={bowl_x}",
                 "-p", f"bowl_y:={bowl_y}",
                 "-p", "skip_home:=true"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True)

            for line in proc.stdout:
                if "RECORD_START" in line:
                    episode_active.set()
                    frames.clear()
                    print("  RECORD_START → recording task trajectory")
            proc.wait()
            episode_active.clear()
            result_rc = proc.returncode
            print(f"  C++ rc={result_rc}, frames recorded: {len(frames)}")

            # ── Determine success (pose query if available, else trust C++) ──
            time.sleep(1.0)  # let physics settle
            block_pose = get_model_pose(block_name)
            if block_pose is not None:
                dist = ((block_pose[0] - bowl_x)**2 + (block_pose[1] - bowl_y)**2)**0.5
                ok = dist <= 0.08
                success_str = "success" if ok else "failed"
                status_icon = "✓" if ok else "✗"
                print(f"  {status_icon} {success_str} — block {dist:.4f}m from bowl center")
            else:
                success_str = "success" if result_rc == 0 else "failed"
                print(f"  C++ rc={result_rc} → {success_str}")

            # ── Delete block and bowl ──
            delete_model(block_name)
            delete_model(bowl_name)

            # ── Save (Pi0 format: 7D absolute, action=next_state) ──
            if len(frames) > 1:
                raw_states = np.stack([f["state"] for f in frames])
                cam0 = np.stack([f["camera0"] for f in frames])
                cam1 = np.stack([f["camera1"] for f in frames])

                # 7D absolute joint positions → Pi0 auto-pads to max_state_dim=32
                states_out  = raw_states[:-1]
                actions_out = raw_states[1:]   # next_state as absolute action
                cam0_out    = cam0[:-1]
                cam1_out    = cam1[:-1]

                task_prefix = TASK.replace(" ", "_")
                ep_name = f"{task_prefix}_episode_{ep:04d}_{success_str}"
                ep_dir  = os.path.join(args.output, ep_name)
                os.makedirs(ep_dir, exist_ok=True)
                timestamps = np.array([f["timestamp"] for f in frames[:-1]],
                                      dtype=np.float64)
                np.savez_compressed(
                    os.path.join(ep_dir, "data.npz"),
                    state=states_out, action=actions_out,
                    camera0=cam0_out, camera1=cam1_out,
                    timestamp=timestamps, task=TASK)
                total_frames += len(states_out)
                print(f"  Saved: {ep_name} — {len(states_out)} frames")

    finally:
        recording.clear()
        rec_thread.join(timeout=2.0)
        print(f"\nDone: {total_frames} frames, {args.episodes} episodes → "
              f"{args.output}")
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

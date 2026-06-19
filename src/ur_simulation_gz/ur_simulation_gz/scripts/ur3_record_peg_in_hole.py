#!/usr/bin/env python3
"""
UR3 peg-in-hole data recorder (SA-MOE / Force modality dataset).

- Spawns peg AND hole plate at randomized positions in Gazebo
- Runs C++ peg_in_hole (passes peg + hole positions as ROS params)
- Records joint_states + 2 cameras + force/torque at 50 Hz
- Auto-generates 4-stage labels (approach/align/insert/tighten)
- Saves LeRobot-compatible .npz per episode with force + stage keys

Run (Gazebo + MoveIt must be running):
  /usr/bin/python3 ur3_record_peg_in_hole.py --episodes 50
"""

import argparse, os, time, threading, subprocess, random, re
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped
from cv_bridge import CvBridge

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
IMG_SIZE = (224, 224)
STATE_DIM = 7      # 6 arm joints + 1 gripper; Pi0 auto-pads to max_state_dim=32
FORCE_DIM = 6      # [Fx, Fy, Fz, Tx, Ty, Tz]
TASK = "pick up the peg and insert it into the hole"

# ── World constants ──
PEG_Z       = 0.825   # peg centre (0.10m tall, bottom on table at 0.775)
HOLE_Z      = 0.775   # hole plate sits on table
TABLE_Z     = 0.775

# Workspace constraints: x² + y² ≤ MAX_XY_SQ (UR3 reach ~0.5m from shoulder at 0,0,0.91)
BLOCK_X_RANGE = (-0.25, 0.25)
BLOCK_Y_RANGE = (0.18, 0.42)
HOLE_X_RANGE  = (-0.25, 0.25)
HOLE_Y_RANGE  = (0.18, 0.42)
MIN_PEG_HOLE_DIST = 0.10
MAX_XY_SQ = 0.23

RECORD_HZ = 50  # Pi0 pretrained at 50Hz


def spawn_model(name, x, y, z, sdf_string=None, file_path=None):
    """Spawn a model in Gazebo via ros_gz_sim create. Returns True on success."""
    try:
        cmd = ["/opt/ros/humble/bin/ros2", "run", "ros_gz_sim", "create",
               "-world", "simulation_world", "-name", name,
               "-x", str(x), "-y", str(y), "-z", str(z)]
        if file_path is not None:
            cmd += ["-file", file_path]
        elif sdf_string is not None:
            cmd += ["-string", sdf_string]
        else:
            raise ValueError("Must provide sdf_string or file_path")
        r = subprocess.run(cmd, capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception as e:
        print(f"  ⚠ spawn failed: {e}")
        return False


def get_model_pose(name):
    """Query a model's world pose via ign topic (Ignition Transport)."""
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
    """Delete a model from Gazebo via ign service."""
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

# Peg: metal-like cylinder, 0.025m radius × 0.10m height, 1 kg
# Spawns at PEG_Z=0.825 (centre at 0.825, bottom at 0.775 on table)
PEG_SDF = """<sdf version='1.9'>
<model name='{name}'>
<link name='link'>
<collision name='collision'>
<geometry><cylinder><radius>0.025</radius><length>0.10</length></cylinder></geometry>
<surface>
<contact><ode><kp>100000</kp><kd>100</kd><max_vel>100.0</max_vel><min_depth>0.001</min_depth></ode></contact>
<friction><torsional><coefficient>1.0</coefficient><use_patch_radius>0</use_patch_radius><surface_radius>0.01</surface_radius></torsional></friction>
</surface>
</collision>
<visual name='visual'>
<geometry><cylinder><radius>0.025</radius><length>0.10</length></cylinder></geometry>
<material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.6 0.6 0.6 1</diffuse></material>
</visual>
<inertial><mass>1</mass><inertia>
<ixx>0.0009</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.0009</iyy><iyz>0</iyz><izz>0.0003</izz>
</inertia></inertial>
</link>
</model></sdf>"""


def make_hole_sdf(name):
    """
    Generate hole fixture SDF with 8 pillars forming an octagonal socket.
    Inner inscribed circle radius = 0.027m (peg r=0.025 + 0.002 clearance).
    8 pillars spaced at 45° around the circle, each oriented tangentially.
    All in ONE link for physics stability.

    Geometry:
      Pillar box: radial=0.015m × tangential=0.024m × height=0.05m
      Center at distance r=0.0345m from origin (0.027 + 0.015/2)
      z=0.045 above model origin → world z=0.820, extends 0.795→0.845
    """
    import math
    HOLE_RADIUS = 0.027      # inner radius (peg r=0.025 + 2mm clearance)
    PILLAR_THICK = 0.015     # radial thickness
    PILLAR_WIDTH = 0.024     # tangential width (octagon side fill)
    PILLAR_HEIGHT = 0.05     # wall height
    PILLAR_D = HOLE_RADIUS + PILLAR_THICK / 2.0  # centre distance from origin
    PILLAR_Z = 0.045         # above link origin

    pillar_xml = ""
    for i in range(8):
        theta = i * math.pi / 4.0  # 0, 45°, 90°, ...
        x = round(PILLAR_D * math.cos(theta), 5)
        y = round(PILLAR_D * math.sin(theta), 5)
        yaw = round(theta, 5)
        pid = f"p{i}"
        pillar_xml += f"""
  <collision name='{pid}_col'>
    <pose>{x} {y} {PILLAR_Z} 0 0 {yaw}</pose>
    <geometry><box><size>{PILLAR_THICK} {PILLAR_WIDTH} {PILLAR_HEIGHT}</size></box></geometry>
    <surface>
      <contact><ode><kp>100000</kp><kd>100</kd><max_vel>100.0</max_vel><min_depth>0.001</min_depth></ode></contact>
      <friction><ode><mu>0.5</mu><mu2>0.5</mu2></ode></friction>
    </surface>
  </collision>
  <visual name='{pid}_vis'>
    <pose>{x} {y} {PILLAR_Z} 0 0 {yaw}</pose>
    <geometry><box><size>{PILLAR_THICK} {PILLAR_WIDTH} {PILLAR_HEIGHT}</size></box></geometry>
    <material><ambient>0.3 0.3 0.35 1</ambient><diffuse>0.45 0.45 0.5 1</diffuse></material>
  </visual>"""

    return f"""<sdf version='1.9'>
<model name='{name}'>
<link name='socket'>

<!-- Base plate: 0.15×0.15×0.02m -->
<collision name='plate_col'>
  <pose>0 0 0.01 0 0 0</pose>
  <geometry><box><size>0.15 0.15 0.02</size></box></geometry>
  <surface>
    <contact><ode><kp>100000</kp><kd>100</kd><max_vel>100.0</max_vel><min_depth>0.001</min_depth></ode></contact>
    <friction><ode><mu>0.5</mu><mu2>0.5</mu2></ode></friction>
  </surface>
</collision>
<visual name='plate_vis'>
  <pose>0 0 0.01 0 0 0</pose>
  <geometry><box><size>0.15 0.15 0.02</size></box></geometry>
  <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
</visual>

<!-- 8 pillars forming octagonal socket (inner inscribed r=0.027m) -->{pillar_xml}

<inertial>
  <mass>1.5</mass>
  <inertia>
    <ixx>0.003</ixx><ixy>0</ixy><ixz>0</ixz>
    <iyy>0.003</iyy><iyz>0</iyz>
    <izz>0.006</izz>
  </inertia>
</inertial>

</link>
</model></sdf>"""


# ── Stage auto-labeling (matches SA-MOE generate_stage_labels) ──
def auto_label_stage(gripper_pos, force_6d, force_bias=None):
    """
    Classify frame into one of 4 stages based on gripper state + force/torque.

    force_bias: idle force reading (6,) subtracted to compensate FT300 tool weight (~12N Fz).

    Returns: 0=approach, 1=align, 2=insert, 3=tighten
    """
    if force_bias is not None:
        f_comp = force_6d - force_bias  # compensated force
    else:
        f_comp = force_6d

    gripper_open = gripper_pos > 0.05
    f_mag = float(np.linalg.norm(f_comp[:3]))   # ||F_comp||
    t_mag = float(np.linalg.norm(f_comp[3:]))    # ||T_comp||

    # Thresholds tuned for UR3 + FT300 + Robotiq 2F85 (~12N tool weight compensated)
    if gripper_open and f_mag < 5.0:
        return 0  # approach: gripper open, no significant contact
    elif gripper_open and 5.0 <= f_mag < 20.0:
        return 1  # align: gripper open, making contact (searching for hole)
    elif not gripper_open and t_mag < 3.0:
        return 2  # insert: gripper closed, axial force dominant, low torque
    else:
        return 3  # tighten: high torque (lateral seating / twisting)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=str,
                        default=os.path.expanduser(
                            "~/ur3_ft300_ws/ai-models/ur3_peg_in_hole_raw"))
    args = parser.parse_args()

    rclpy.init()
    os.makedirs(args.output, exist_ok=True)

    # ── Shared buffer (thread-safe) ──
    class Buffer:
        def __init__(self):
            self.lock = threading.Lock()
            self.joint_positions = None
            self.wrench = np.zeros(FORCE_DIM, dtype=np.float32)
            self.wrist_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
            self.global_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)

    buf = Buffer()

    # ── Subscriber node ──
    class RecorderNode(Node):
        def __init__(self):
            super().__init__("record_pih")
            self.bridge = CvBridge()
            cbg = ReentrantCallbackGroup()
            self.create_subscription(JointState, "/joint_states",
                                      self._js, 10, callback_group=cbg)
            self.create_subscription(Image, "/wrist_camera/color/image_raw",
                                      self._wrist, 10, callback_group=cbg)
            self.create_subscription(Image, "/global_camera/color/image_raw",
                                      self._global, 10, callback_group=cbg)
            self.create_subscription(WrenchStamped,
                                      "/force_torque_sensor_broadcaster/wrench",
                                      self._wrench, 10, callback_group=cbg)

        def _js(self, msg):
            try:
                pos = [msg.position[msg.name.index(n)] for n in ALL_JOINTS]
                with buf.lock:
                    buf.joint_positions = pos
            except ValueError:
                pass

        def _wrench(self, msg):
            w = msg.wrench
            with buf.lock:
                buf.wrench = np.array(
                    [w.force.x, w.force.y, w.force.z,
                     w.torque.x, w.torque.y, w.torque.z],
                    dtype=np.float32)

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
    executor = MultiThreadedExecutor(4)
    executor.add_node(recorder)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Wait for joint states
    print("Waiting for joint states and FT sensor...")
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
    episode_active = threading.Event()

    def recorder_thread():
        rate = recorder.create_rate(RECORD_HZ)
        while recording.is_set():
            with buf.lock:
                js = (list(buf.joint_positions) if buf.joint_positions
                      else [0.0] * STATE_DIM)
                w = buf.wrist_img.copy()
                g = buf.global_img.copy()
                ft = buf.wrench.copy()
            if episode_active.is_set():
                # Auto-label stage from gripper position + force/torque (bias compensated)
                gripper_pos = js[-1] if len(js) == STATE_DIM else 0.0
                stage = auto_label_stage(gripper_pos, ft, force_bias)
                frames.append({
                    "state": np.array(js, dtype=np.float32),
                    "force": ft,
                    "stage": stage,
                    "camera0": w,
                    "camera1": g,
                    "timestamp": time.time(),
                })
            rate.sleep()

    rec_thread = threading.Thread(target=recorder_thread, daemon=True)
    rec_thread.start()
    print(f"Recording started ({RECORD_HZ} Hz) — stages: "
          "0=approach 1=align 2=insert 3=tighten")

    # ── Capture force bias at idle (FT300 tool weight ~12N in Fz) ──
    print("Capturing force bias (2s idle)...")
    time.sleep(2.0)
    with buf.lock:
        force_bias = buf.wrench.copy()
    print(f"  Force bias: F=[{force_bias[0]:.2f} {force_bias[1]:.2f} {force_bias[2]:.2f}] "
          f"T=[{force_bias[3]:.3f} {force_bias[4]:.3f} {force_bias[5]:.3f}]")

    total_frames = 0

    try:
        for ep in range(args.episodes):
            print(f"\n{'='*60}")
            print(f"Episode {ep+1}/{args.episodes}")

            # ── Randomized positions ──
            for _ in range(100):
                peg_x  = round(random.uniform(*BLOCK_X_RANGE), 3)
                peg_y  = round(random.uniform(*BLOCK_Y_RANGE), 3)
                hole_x = round(random.uniform(*HOLE_X_RANGE), 3)
                hole_y = round(random.uniform(*HOLE_Y_RANGE), 3)
                p_ok = peg_x**2 + peg_y**2 <= MAX_XY_SQ
                h_ok = hole_x**2 + hole_y**2 <= MAX_XY_SQ
                dist = ((peg_x - hole_x)**2 + (peg_y - hole_y)**2)**0.5
                if p_ok and h_ok and dist >= MIN_PEG_HOLE_DIST:
                    break
            print(f"  Peg:  x={peg_x:.3f}, y={peg_y:.3f}")
            print(f"  Hole: x={hole_x:.3f}, y={hole_y:.3f}")

            peg_name  = f"peg_{ep:04d}"
            hole_name = f"hole_plate_{ep:04d}"

            # ── Spawn peg and hole plate ──
            ok_peg  = spawn_model(peg_name, peg_x, peg_y, PEG_Z,
                                   PEG_SDF.format(name=peg_name))
            ok_hole = spawn_model(hole_name, hole_x, hole_y, HOLE_Z,
                                   file_path=os.path.expanduser(
                                       "~/.gazebo/models/ring_hole/model.sdf"))
            if not ok_peg or not ok_hole:
                print("  ⚠ spawn failed, skipping episode")
                delete_model(peg_name)
                delete_model(hole_name)
                continue

            time.sleep(1.0)  # let physics settle

            # ── Run C++ peg_in_hole ──
            print(f"  Running C++ peg_in_hole...")
            episode_active.clear()
            frames.clear()
            proc = subprocess.Popen(
                ["/opt/ros/humble/bin/ros2", "run", "ur_simulation_gz",
                 "peg_in_hole",
                 "--ros-args",
                 "-p", f"peg_x:={peg_x}",
                 "-p", f"peg_y:={peg_y}",
                 "-p", f"hole_x:={hole_x}",
                 "-p", f"hole_y:={hole_y}",
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

            # ── Determine success — peg should be in/near the hole ──
            time.sleep(1.0)
            peg_pose = get_model_pose(peg_name)
            if peg_pose is not None:
                # After release, peg center should be inside ring (r_inner=0.027m)
                dist = ((peg_pose[0] - hole_x)**2 + (peg_pose[1] - hole_y)**2)**0.5
                # Peg center in hole: bottom ~0.80 (on ring bottom) + half height 0.05 ≈ 0.85
                peg_low = peg_pose[2] < 0.88  # well below peg resting height (0.825 on table)
                ok = dist <= 0.06 and peg_low   # within hole ring + margin
                success_str = "success" if ok else "failed"
                status_icon = "✓" if ok else "✗"
                print(f"  {status_icon} {success_str} — peg dist from hole={dist:.4f}m, "
                      f"z={peg_pose[2]:.3f}")
            else:
                success_str = "success" if result_rc == 0 else "failed"
                print(f"  C++ rc={result_rc} → {success_str}")

            # ── Delete models ──
            delete_model(peg_name)
            delete_model(hole_name)

            # ── Save npz ──
            if len(frames) > 1:
                raw_states = np.stack([f["state"] for f in frames])
                forces     = np.stack([f["force"] for f in frames])
                stages     = np.array([f["stage"] for f in frames], dtype=np.int64)
                cam0       = np.stack([f["camera0"] for f in frames])
                cam1       = np.stack([f["camera1"] for f in frames])

                # Action = next_state (absolute joint positions)
                states_out  = raw_states[:-1]
                actions_out = raw_states[1:]
                forces_out  = forces[:-1]
                stages_out  = stages[:-1]
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
                    force=forces_out, stage=stages_out,
                    camera0=cam0_out, camera1=cam1_out,
                    timestamp=timestamps, task=TASK)

                # Stage distribution summary
                unique_stages, counts = np.unique(stages_out, return_counts=True)
                stage_names = {0: "approach", 1: "align", 2: "insert", 3: "tighten"}
                stage_str = ", ".join(
                    f"{stage_names.get(s, '?')}:{c}" for s, c in zip(unique_stages, counts))
                print(f"  Saved: {ep_name} — {len(states_out)} frames")
                print(f"  Stages: {stage_str}")
                total_frames += len(states_out)

    finally:
        recording.clear()
        rec_thread.join(timeout=2.0)
        print(f"\nDone: {total_frames} frames, {args.episodes} episodes → "
              f"{args.output}")
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

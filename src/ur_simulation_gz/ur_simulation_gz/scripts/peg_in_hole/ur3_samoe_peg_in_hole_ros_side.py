#!/usr/bin/env python3
"""
UR3 SA-MOE ROS side — 系统 Python 3.10
Streams: cameras, joint_states, force/torque → /tmp/ files
Reads: action from /tmp/ur3_action.txt → sends to robot
"""
import argparse, os, sys, threading, time, subprocess, random, re, atexit
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
from geometry_msgs.msg import WrenchStamped
from cv_bridge import CvBridge

# ── Spawn helpers ──
PEG_Z = 0.825
HOLE_Z = 0.775
BLOCK_X_RANGE = (-0.25, 0.25)
BLOCK_Y_RANGE = (0.18, 0.42)
HOLE_X_RANGE = (-0.25, 0.25)
HOLE_Y_RANGE = (0.18, 0.42)
MIN_PEG_HOLE_DIST = 0.10
MAX_XY_SQ = 0.23

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


def spawn_model(name, x, y, z, sdf_string=None, file_path=None):
    cmd = ["/opt/ros/humble/bin/ros2", "run", "ros_gz_sim", "create",
           "-world", "simulation_world", "-name", name,
           "-x", str(x), "-y", str(y), "-z", str(z)]
    if file_path is not None:
        cmd += ["-file", file_path]
    elif sdf_string is not None:
        cmd += ["-string", sdf_string]
    else:
        raise ValueError("Need sdf_string or file_path")
    r = subprocess.run(cmd, capture_output=True, timeout=10)
    return r.returncode == 0


def delete_model(name):
    try:
        r = subprocess.run(
            ["ign", "service", "-s", "/world/simulation_world/remove",
             "--reqtype", "ignition.msgs.Entity", "--reptype", "ignition.msgs.Boolean",
             "--timeout", "1000", "-r", f'name: "{name}" type: MODEL'],
            capture_output=True, timeout=5)
        return r.returncode == 0 and b"data: true" in r.stdout
    except Exception:
        return False

JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
ACTION_FILE = "/tmp/ur3_action.txt"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"
FORCE_FILE = "/tmp/ur3_force.npy"

ARM_JOINTS = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
              "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
IMG_SIZE = (224, 224)


class SAMoeROSSide(Node):
    def __init__(self):
        super().__init__("samoe_ros_side")
        self.bridge = CvBridge()
        cbg = ReentrantCallbackGroup()

        self.create_subscription(JointState, "/joint_states", self._js, 10, callback_group=cbg)
        self.create_subscription(Image, "/wrist_camera/color/image_raw", self._wrist, 10, callback_group=cbg)
        self.create_subscription(Image, "/global_camera/color/image_raw", self._global, 10, callback_group=cbg)
        self.create_subscription(WrenchStamped, "/force_torque_sensor_broadcaster/wrench",
                                 self._wrench, 10, callback_group=cbg)

        self._action_client = ActionClient(self, FollowJointTrajectory,
                                           "/joint_trajectory_controller/follow_joint_trajectory")
        self.lock = threading.Lock()
        self.wrist_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        self.global_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        self.wrench = np.zeros(6, dtype=np.float32)

    def _js(self, msg):
        try:
            pos = [msg.position[msg.name.index(n)] for n in ALL_JOINTS]
            with open(JOINT_STATE_FILE, "w") as f:
                f.write(" ".join(f"{p:.6f}" for p in pos))
        except ValueError:
            pass

    def _wrist(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), IMG_SIZE)
            with self.lock:
                self.wrist_img = rgb.astype(np.float32) / 255.0
        except Exception:
            pass

    def _global(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), IMG_SIZE)
            with self.lock:
                self.global_img = rgb.astype(np.float32) / 255.0
        except Exception:
            pass

    def _wrench(self, msg):
        w = msg.wrench
        self.wrench = np.array([w.force.x, w.force.y, w.force.z,
                                w.torque.x, w.torque.y, w.torque.z], dtype=np.float32)

    def _save_images(self):
        with self.lock:
            np.save(CAMERA0_FILE, self.wrist_img)
            np.save(CAMERA1_FILE, self.global_img)
        np.save(FORCE_FILE, self.wrench)

    def _send_action(self, action):
        try:
            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory.joint_names = ALL_JOINTS
            point = JointTrajectoryPoint()
            point.positions = [float(a) for a in action]
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(1e9 / 10)
            goal_msg.trajectory.points.append(point)
            self._action_client.send_goal_async(goal_msg)
        except Exception as e:
            self.get_logger().error(f"Action send failed: {e}")

    def run_loop(self):
        self.get_logger().info("SA-MOE ROS side running (10 Hz)...")
        # Clear stale action file from previous run
        if os.path.exists(ACTION_FILE):
            os.remove(ACTION_FILE)
        rate = self.create_rate(10)
        last_mtime = None
        while rclpy.ok():
            try:
                self._save_images()
                if os.path.exists(ACTION_FILE):
                    mtime = os.path.getmtime(ACTION_FILE)
                    if mtime != last_mtime:
                        with open(ACTION_FILE, "r") as f:
                            line = f.readline().strip()
                            if line:
                                action = np.array([float(x) for x in line.split()], dtype=np.float32)
                                if np.abs(action).max() > 0.01:
                                    self._send_action(action)
                                    last_mtime = mtime
                rate.sleep()
            except Exception as e:
                self.get_logger().error(f"Loop error: {e}")
                rate.sleep()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", action="store_true", help="Spawn peg and hole in Gazebo")
    parser.add_argument("--peg-x", type=float, default=None)
    parser.add_argument("--peg-y", type=float, default=None)
    parser.add_argument("--hole-x", type=float, default=None)
    parser.add_argument("--hole-y", type=float, default=None)
    parsed_args, _ = parser.parse_known_args()

    spawned = []
    if parsed_args.spawn:
        # Randomize or use specified positions
        px = parsed_args.peg_x if parsed_args.peg_x is not None else random.uniform(*BLOCK_X_RANGE)
        py = parsed_args.peg_y if parsed_args.peg_y is not None else random.uniform(*BLOCK_Y_RANGE)
        hx = parsed_args.hole_x if parsed_args.hole_x is not None else random.uniform(*HOLE_X_RANGE)
        hy = parsed_args.hole_y if parsed_args.hole_y is not None else random.uniform(*HOLE_Y_RANGE)

        # Validate
        for _ in range(100):
            p_ok = px**2 + py**2 <= MAX_XY_SQ
            h_ok = hx**2 + hy**2 <= MAX_XY_SQ
            dist = ((px-hx)**2 + (py-hy)**2)**0.5
            if p_ok and h_ok and dist >= MIN_PEG_HOLE_DIST:
                break
            px = random.uniform(*BLOCK_X_RANGE)
            py = random.uniform(*BLOCK_Y_RANGE)
            hx = random.uniform(*HOLE_X_RANGE)
            hy = random.uniform(*HOLE_Y_RANGE)

        peg_name = "inference_peg"
        hole_name = "inference_hole_plate"

        print(f"Spawning peg at ({px:.2f}, {py:.2f})")
        if spawn_model(peg_name, px, py, PEG_Z, sdf_string=PEG_SDF.format(name=peg_name)):
            print("  Peg OK")
            spawned.append(peg_name)
        else:
            print("  Peg FAILED")

        hole_path = os.path.expanduser("~/.gazebo/models/ring_hole/model.sdf")
        print(f"Spawning hole at ({hx:.2f}, {hy:.2f})")
        if spawn_model(hole_name, hx, hy, HOLE_Z, file_path=hole_path):
            print("  Hole OK")
            spawned.append(hole_name)
        else:
            print("  Hole FAILED")

    if spawned:
        def cleanup():
            for name in spawned:
                print(f"Deleting {name}...")
                delete_model(name)
        atexit.register(cleanup)

    rclpy.init(args=args)
    node = SAMoeROSSide()
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

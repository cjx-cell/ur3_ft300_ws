#!/usr/bin/env python3
"""Shared UR3 peg-in-hole ROS-side implementation.

Policy-specific entry points live beside this module:

- ``ur3_samoe_peg_in_hole_ros_side.py``
- ``ur3_pap_moe_peg_in_hole_ros_side.py``

This module owns the common Gazebo spawning, observation bridge, trajectory
execution, and safety logic. It is not intended to be launched directly.
"""

import argparse, os, sys, threading, time, subprocess, random, re, atexit
from pathlib import Path
import numpy as np
import policy_diagnostic_trace as diagnostic_trace
import policy_action_exchange as action_exchange
import cv2
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import WrenchStamped
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener

_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
from pap_moe_framework.rollout_recovery.protocol import (  # noqa: E402
    EXPERT_MODE,
    POLICY_MODE,
    clamp_continuous_for_controller,
    clamp_for_controller,
    initialize_session,
    read_control_mode,
    read_expert_chunk,
    request_takeover,
    write_action_completion,
    write_action_event,
    write_expert_execution_ack,
)

# ── Spawn helpers ──
PEG_Z = 0.865
HOLE_Z = 0.775
PEG_SPAWN_CENTER = (0.110, 0.255)
HOLE_SPAWN_CENTER = (-0.120, 0.245)
SPAWN_HALF_RANGE_XY = (0.015, 0.015)
BLOCK_X_RANGE = (
    PEG_SPAWN_CENTER[0] - SPAWN_HALF_RANGE_XY[0],
    PEG_SPAWN_CENTER[0] + SPAWN_HALF_RANGE_XY[0],
)
BLOCK_Y_RANGE = (
    PEG_SPAWN_CENTER[1] - SPAWN_HALF_RANGE_XY[1],
    PEG_SPAWN_CENTER[1] + SPAWN_HALF_RANGE_XY[1],
)
HOLE_X_RANGE = (
    HOLE_SPAWN_CENTER[0] - SPAWN_HALF_RANGE_XY[0],
    HOLE_SPAWN_CENTER[0] + SPAWN_HALF_RANGE_XY[0],
)
HOLE_Y_RANGE = (
    HOLE_SPAWN_CENTER[1] - SPAWN_HALF_RANGE_XY[1],
    HOLE_SPAWN_CENTER[1] + SPAWN_HALF_RANGE_XY[1],
)
MIN_PEG_HOLE_DIST = 0.10
MAX_XY_SQ = 0.0900

# Same rigid, real-size fixture contract used by the PAP-MoE recorder.
PEG_SDF = """<sdf version='1.9'>
<model name='{name}'>
<link name='link'>
<collision name='collision'>
<geometry><mesh><uri>model://pap_moe_real_peg/meshes/peg_body_collision.stl</uri></mesh></geometry>
<surface>
<contact><poissons_ratio>0.36</poissons_ratio><elastic_modulus>1.5e9</elastic_modulus><ode><kp>{kp}</kp><kd>{kd}</kd><max_vel>0.05</max_vel><min_depth>0.0001</min_depth></ode></contact>
<friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode><torsional><coefficient>0.8</coefficient><use_patch_radius>0</use_patch_radius><surface_radius>0.01</surface_radius></torsional></friction>
</surface>
</collision>
<sensor name='peg_body_contact_sensor' type='contact'>
  <always_on>true</always_on><update_rate>100</update_rate>
  <contact>
    <collision>collision</collision>
    <topic>/pap_moe/peg_body_contacts</topic>
  </contact>
</sensor>
<collision name='analytic_grasp_handle'>
<pose>0 0 0.050 0 0 0</pose>
<geometry><cylinder><radius>0.010</radius><length>0.080</length></cylinder></geometry>
<surface>
<contact><poissons_ratio>0.36</poissons_ratio><elastic_modulus>1.5e9</elastic_modulus><ode><kp>{kp}</kp><kd>{kd}</kd><max_vel>0.05</max_vel><min_depth>0.0001</min_depth></ode></contact>
<friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode><torsional><coefficient>0.8</coefficient><use_patch_radius>0</use_patch_radius><surface_radius>0.01</surface_radius></torsional></friction>
</surface>
</collision>
<collision name='rigid_tip_cap'>
<pose>0 0 -0.089 0 0 0</pose>
<geometry><cylinder><radius>0.010</radius><length>0.002</length></cylinder></geometry>
<surface>
<contact><poissons_ratio>0.36</poissons_ratio><elastic_modulus>1.5e9</elastic_modulus><ode><kp>{kp}</kp><kd>{kd}</kd><max_vel>0.05</max_vel><min_depth>0.0001</min_depth></ode></contact>
<friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
</surface>
</collision>
<sensor name='peg_handle_contact_sensor' type='contact'>
  <always_on>true</always_on><update_rate>100</update_rate>
  <contact>
    <collision>analytic_grasp_handle</collision>
    <topic>/pap_moe/peg_handle_contacts</topic>
  </contact>
</sensor>
<visual name='visual'>
<geometry><mesh><uri>model://pap_moe_real_peg/meshes/peg.stl</uri></mesh></geometry>
<material><ambient>0.72 0.62 0.42 1</ambient><diffuse>0.82 0.72 0.52 1</diffuse></material>
</visual>
<inertial><pose>0 0 -0.01077 0 0 0</pose><mass>{mass}</mass><inertia>
<ixx>{ixx}</ixx><ixy>0</ixy><ixz>0</ixz><iyy>{iyy}</iyy><iyz>0</iyz><izz>{izz}</izz>
</inertia></inertial>
</link>
</model></sdf>"""


# The laboratory socket is rigidly mounted.  Only the 80 mm frustum is hollow;
# the lower 20 mm is solid and there is no cylindrical cavity segment.
def make_hole_sdf(name, kp=5000, kd=40, mu=0.35, mu2=0.35):
    return f"""<sdf version='1.9'>
<model name='{name}'>
<static>true</static>
<link name='socket'>
  <collision name='socket_col'>
    <geometry><mesh><uri>model://pap_moe_real_hole/meshes/hole_side_collision.stl</uri></mesh></geometry>
    <surface>
      <contact><poissons_ratio>0.36</poissons_ratio><elastic_modulus>1.5e9</elastic_modulus><ode><kp>{kp}</kp><kd>{kd}</kd><max_vel>0.05</max_vel><min_depth>0.0001</min_depth></ode></contact>
      <friction><ode><mu>{mu}</mu><mu2>{mu2}</mu2></ode></friction>
    </surface>
  </collision>
  <sensor name='hole_side_contact_sensor' type='contact'>
    <always_on>true</always_on><update_rate>100</update_rate>
    <contact>
      <collision>socket_col</collision>
      <topic>/pap_moe/hole_side_contacts</topic>
    </contact>
  </sensor>
  <collision name='rigid_floor_col'>
    <pose>0 0 0.010 0 0 0</pose>
    <geometry><cylinder><radius>0.01075</radius><length>0.020</length></cylinder></geometry>
    <surface>
      <contact><poissons_ratio>0.36</poissons_ratio><elastic_modulus>1.5e9</elastic_modulus><ode><kp>{kp}</kp><kd>{kd}</kd><max_vel>0.05</max_vel><min_depth>0.0001</min_depth></ode></contact>
      <friction><ode><mu>{mu}</mu><mu2>{mu2}</mu2></ode></friction>
    </surface>
  </collision>
  <sensor name='hole_floor_contact_sensor' type='contact'>
    <always_on>true</always_on><update_rate>100</update_rate>
    <contact>
      <collision>rigid_floor_col</collision>
      <topic>/pap_moe/hole_floor_contacts</topic>
    </contact>
  </sensor>
  <visual name='socket_vis'>
    <geometry><mesh><uri>model://pap_moe_real_hole/meshes/hole.stl</uri></mesh></geometry>
    <material><ambient>0.72 0.62 0.42 1</ambient><diffuse>0.82 0.72 0.52 1</diffuse></material>
  </visual>
</link>
</model></sdf>"""


def spawn_model(name, x, y, z, sdf_string=None, file_path=None):
    cmd = [
        "/opt/ros/humble/bin/ros2",
        "run",
        "ros_gz_sim",
        "create",
        "-world",
        "simulation_world",
        "-name",
        name,
        "-x",
        str(x),
        "-y",
        str(y),
        "-z",
        str(z),
    ]
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
            [
                "ign",
                "service",
                "-s",
                "/world/simulation_world/remove",
                "--reqtype",
                "ignition.msgs.Entity",
                "--reptype",
                "ignition.msgs.Boolean",
                "--timeout",
                "1000",
                "-r",
                f'name: "{name}" type: MODEL',
            ],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0 and b"data: true" in r.stdout
    except Exception:
        return False


def get_model_pose(name):
    """Return a Gazebo model world position from the pose-info stream."""
    try:
        result = subprocess.run(
            [
                "ign",
                "topic",
                "-t",
                "/world/simulation_world/pose/info",
                "-e",
                "-n",
                "1",
            ],
            capture_output=True,
            timeout=3,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        text = result.stdout.decode(errors="replace")
        index = text.find(f'name: "{name}"')
        if index < 0:
            return None
        match = re.search(
            r"position\s*\{\s*x:\s*([-\d.e+]+)\s*"
            r"y:\s*([-\d.e+]+)\s*z:\s*([-\d.e+]+)",
            text[index : index + 300],
        )
        if match is None:
            return None
        return np.array([float(value) for value in match.groups()], dtype=np.float64)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
ACTION_FILE = "/tmp/ur3_action.txt"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"
FORCE_FILE = "/tmp/ur3_force.npy"

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
IMG_SIZE = (224, 224)


def ur3_fk(q):
    d1 = 0.1519
    a2 = -0.24365
    a3 = -0.21325
    d4 = 0.11235
    d5 = 0.08535
    d6 = 0.2619  # tool0 + FT300 + Robotiq gripper compound length

    q1, q2, q3, q4, q5, q6 = q[:6]
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23 = np.sin(q2 + q3)
    c23 = np.cos(q2 + q3)
    s234 = np.sin(q2 + q3 + q4)
    c234 = np.cos(q2 + q3 + q4)

    x = c1 * (a2 * c2 + a3 * c23 - d4 * s234 + d6 * c234) - d5 * s1
    y = s1 * (a2 * c2 + a3 * c23 - d4 * s234 + d6 * c234) + d5 * c1
    z = d1 + a2 * s2 + a3 * s23 + d4 * c234 + d6 * s234
    return np.array([x, y, z])


class PegInHoleROSSide(Node):
    # Zero inherits the controller's historical grace period. Positive values
    # are an explicit per-goal experiment; do not modify global controllers.
    GOAL_TIME_TOLERANCE_S = float(os.environ.get("POLICY_GOAL_TIME_TOLERANCE_S", "0"))
    """Policy-agnostic ROS bridge for the UR3 peg-in-hole environment."""

    CONTROL_HZ = 10
    ACTION_DT_S = 0.165
    REPLAN_INTERVAL_S = 0.8
    ACTION_CHUNK_SIZE = 10
    ACTION_CHUNK_MAX_STEP_RAD = 0.12
    # Legacy scripted scenes used a 0.30 m radial workspace.  Dataset-specific
    # controllers may override this, but explicit requested poses must never be
    # silently replaced with random poses.
    SPAWN_MAX_XY_SQ = MAX_XY_SQ
    # Universal Robotiq 2F-85 command contract.  The policy commands only the
    # mechanism endpoints; an intermediate measured angle while holding an
    # object is a contact outcome, never an object-specific target.
    GRIPPER_OPEN_POSITION_RAD = 0.0
    GRIPPER_CLOSED_POSITION_RAD = 0.8
    GRIPPER_ACTION_MODE = "binary_semantic"
    ATTACH_MAX_XY_M = 0.025
    # The finger-tip midpoint is not necessarily coplanar with the grasped
    # object's centre.  Subclasses with a known tool/object geometry may use
    # a non-zero admissible vertical-offset window.
    ATTACH_MIN_Z_M = 0.0
    ATTACH_MAX_Z_M = 0.020
    ENABLE_DETACHABLE_JOINT = False
    MAX_EPISODE_DURATION_S = 60.0
    FIRST_ACTION_TIMEOUT_S = 30.0
    OBSERVATION_READY_TIMEOUT_S = 20.0
    RESET_AFTER_EPISODE = True
    ENABLE_GAZEBO_SUCCESS_CHECK = False
    SUCCESS_MAX_XY_M = 0.05
    SUCCESS_MAX_PEG_Z_M = 0.888
    SUCCESS_REQUIRED_CHECKS = 2
    START_POSE_TOLERANCE_RAD = 0.03
    START_POSE_TIMEOUT_S = 40.0
    ACTION_RESULT_TIMEOUT_S = 30.0
    TRAINING_STATE_MIN = None
    TRAINING_STATE_MAX = None
    START_POSE = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0, 0.1)

    def __init__(self, peg_pos=None, node_name="peg_in_hole_ros_side"):
        super().__init__(node_name)
        self.bridge = CvBridge()
        cbg = ReentrantCallbackGroup()

        # Serialize only this subscription. Image/force callbacks remain
        # independent; a newer JointState must not overtake an older callback.
        self._joint_callback_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(JointState, "/joint_states", self._js, 10, callback_group=self._joint_callback_group)
        self.create_subscription(Image, "/wrist_camera/color/image_raw", self._wrist, 10, callback_group=cbg)
        self.create_subscription(
            Image, "/global_camera/color/image_raw", self._global, 10, callback_group=cbg
        )
        self.create_subscription(
            WrenchStamped, "/force_torque_sensor_broadcaster/wrench", self._wrench, 10, callback_group=cbg
        )

        self._action_client = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.lock = threading.Lock()
        self._source_stamps = {}
        self.wrist_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        self.global_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        self.wrench = np.zeros(6, dtype=np.float32)
        self.latest_pos = None
        if peg_pos is not None:
            self.peg_pos_world = np.array(peg_pos, dtype=np.float32)
            p_xyz = list(peg_pos)
            p_xyz[2] = p_xyz[2] - 0.775  # Convert Z_world (0.825m) to Z_base_link (0.050m)
            self.peg_pos = np.array(p_xyz, dtype=np.float32)
        else:
            self.peg_pos_world = None
            self.peg_pos = None
        self.is_attached = False
        self.rollout_recovery_session_dir = None
        self.rollout_recovery_takeover_active = False
        self.rollout_recovery_trigger = None
        self.rollout_recovery_control_sequence = -1
        self.rollout_recovery_event_sequence = 0
        self.rollout_recovery_policy_sequence = 0
        self.rollout_recovery_expert_sequence = -1
        # A contact-rich task can satisfy its geometric success condition
        # while FollowJointTrajectory is still waiting for an exact terminal
        # joint tolerance that is physically unreachable against the socket.
        # Pure evaluation latches that result inside the wait loop so a valid
        # insertion is not later reported as a controller timeout.
        self._success_during_action_metrics = None
        self.recovery_guard_premature_release = (
            os.environ.get("ROLLOUT_RECOVERY_GUARD_PREMATURE_RELEASE", "false").lower()
            == "true"
        )
        self.recovery_guard_closed_rad = float(
            os.environ.get("ROLLOUT_RECOVERY_GUARD_CLOSED_RAD", "0.58")
        )
        self.recovery_guard_min_lift_m = float(
            os.environ.get("ROLLOUT_RECOVERY_GUARD_MIN_LIFT_M", "0.05")
        )
        self.recovery_guard_min_hole_distance_m = float(
            os.environ.get("ROLLOUT_RECOVERY_GUARD_MIN_HOLE_DISTANCE_M", "0.08")
        )

    def configure_rollout_recovery(self, session_dir, metadata):
        """Enable the independent, auditable multi-handoff recovery path."""
        session_path = Path(session_dir).resolve()
        initialize_session(session_path, metadata)
        self.rollout_recovery_session_dir = session_path
        self.get_logger().info(
            f"Rollout recovery IPC enabled for new session: {session_path}"
        )

    def _refresh_rollout_recovery_takeover(self):
        if self.rollout_recovery_session_dir is None:
            return False
        request = read_control_mode(self.rollout_recovery_session_dir)
        if request is not None and int(request["sequence"]) > self.rollout_recovery_control_sequence:
            self.rollout_recovery_control_sequence = int(request["sequence"])
            self.rollout_recovery_takeover_active = int(request["mode"]) == EXPERT_MODE
            self.rollout_recovery_trigger = str(request["reason"])
            owner = "EXPERT" if self.rollout_recovery_takeover_active else "POLICY"
            self.get_logger().warn(
                f"Rollout recovery control switched to {owner} "
                f"(handoff={self.rollout_recovery_control_sequence}): "
                f"{self.rollout_recovery_trigger}"
            )
        return self.rollout_recovery_takeover_active

    def _guard_policy_chunk_for_recovery(self, chunk):
        """Intercept a predicted premature release before it can drop the peg.

        This privileged guard exists only in an explicitly configured recovery
        collection session. It never modifies deployment or normal evaluation.
        """
        if (
            self.rollout_recovery_session_dir is None
            or not self.recovery_guard_premature_release
            or self.rollout_recovery_takeover_active
        ):
            return False
        with self.lock:
            current = None if self.latest_pos is None else np.asarray(self.latest_pos).copy()
        if current is None or float(current[6]) < self.recovery_guard_closed_rad:
            return False
        proposed = np.asarray(chunk, dtype=np.float32)
        if proposed.ndim != 2:
            return False
        premature_release = bool(
            np.any(proposed[:, 6] < self.recovery_guard_closed_rad)
        )
        predicted_ood = False
        if self.TRAINING_STATE_MIN is not None and self.TRAINING_STATE_MAX is not None:
            lower = np.asarray(self.TRAINING_STATE_MIN, dtype=np.float32)[:6]
            upper = np.asarray(self.TRAINING_STATE_MAX, dtype=np.float32)[:6]
            predicted_ood = bool(
                np.any((proposed[:, :6] < lower) | (proposed[:, :6] > upper))
            )
        if not premature_release and not predicted_ood:
            return False
        peg = get_model_pose("peg")
        hole = get_model_pose("hole_plate")
        if peg is None or hole is None or self.peg_pos_world is None:
            return False
        peg_lift = float(peg[2] - self.peg_pos_world[2])
        peg_hole_xy = float(np.linalg.norm(peg[:2] - hole[:2]))
        if (
            peg_lift < self.recovery_guard_min_lift_m
            or peg_hole_xy < self.recovery_guard_min_hole_distance_m
        ):
            return False
        trigger = (
            "predicted transport deviation before dispatch: "
            f"premature_release={premature_release}, predicted_ood={predicted_ood}, "
            f"current_gripper={current[6]:.4f}, "
            f"proposed_min_gripper={np.min(proposed[:, 6]):.4f}, "
            f"peg_lift={peg_lift:.4f}m, peg_hole_xy={peg_hole_xy:.4f}m"
        )
        request_takeover(
            self.rollout_recovery_session_dir,
            trigger=trigger,
            requester="ros_predispatch_transport_deviation_guard_v1",
            sequence=self.rollout_recovery_control_sequence + 1,
        )
        self.get_logger().warn(f"Recovery guard blocked policy chunk; {trigger}")
        return True

    def _gripper_center_world(self):
        """Return the measured midpoint of both finger tips in world coordinates."""
        positions = []
        for frame in (
            "robotiq_85_left_finger_tip_link",
            "robotiq_85_right_finger_tip_link",
        ):
            try:
                transform = self.tf_buffer.lookup_transform("world", frame, rclpy.time.Time())
            except Exception:
                return None
            translation = transform.transform.translation
            positions.append([translation.x, translation.y, translation.z])
        return np.mean(np.asarray(positions, dtype=np.float64), axis=0)

    def _gripper_to_peg_distance(self):
        if self.peg_pos_world is None:
            return None
        gripper_center = self._gripper_center_world()
        if gripper_center is None:
            return None
        # The free peg can move after contact.  Attachment safeguards must use
        # its measured Gazebo pose, not the episode's spawn coordinates;
        # otherwise a knocked-over peg can be rigidly attached from afar.
        peg_position = get_model_pose("peg")
        if peg_position is None:
            return None
        distance_xy = float(np.linalg.norm(gripper_center[:2] - peg_position[:2]))
        distance_z = float(abs(gripper_center[2] - peg_position[2]))
        return distance_xy, distance_z

    def _attachment_pose_is_safe(self, distance_xy, distance_z):
        return (
            distance_xy <= self.ATTACH_MAX_XY_M
            and self.ATTACH_MIN_Z_M <= distance_z <= self.ATTACH_MAX_Z_M
        )

    @staticmethod
    def _publish_detachable_joint(topic, attempts=5):
        for _ in range(attempts):
            subprocess.run(
                ["ign", "topic", "-t", topic, "-m", "ignition.msgs.Empty", "-p", ""],
                capture_output=True,
                timeout=1,
            )
            time.sleep(0.1)

    def _js(self, msg):
        try:
            if action_exchange.enabled() and any(name not in msg.name for name in ALL_JOINTS):
                return False  # Never synthesize missing measured joints as 0.1.
            pos = []
            for n in ALL_JOINTS:
                if n in msg.name:
                    pos.append(msg.position[msg.name.index(n)])
                else:
                    pos.append(0.1)
            if action_exchange.enabled() and not np.isfinite(pos).all():
                return False
            with self.lock:
                self.latest_pos = pos
                self._source_stamps["joint"] = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, time.monotonic())
                history_hook = getattr(self, "_record_joint_history", None)
                if history_hook is not None:
                    history_hook(self._source_stamps["joint"][0], pos)
            return True
        except Exception:
            return False

    def _wrist(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), IMG_SIZE, interpolation=cv2.INTER_AREA)
            with self.lock:
                self.wrist_img = rgb.astype(np.float32) / 255.0
                self._source_stamps["camera0"] = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, time.monotonic())
        except Exception:
            pass

    def _global(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), IMG_SIZE, interpolation=cv2.INTER_AREA)
            with self.lock:
                self.global_img = rgb.astype(np.float32) / 255.0
                self._source_stamps["camera1"] = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, time.monotonic())
        except Exception:
            pass

    def _wrench(self, msg):
        w = msg.wrench
        self.wrench = np.array(
            [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z], dtype=np.float32
        )

    def _save_images(self):
        with self.lock:
            self._snapshot_joint_pos = None if self.latest_pos is None else list(self.latest_pos)
            np.save(CAMERA0_FILE, self.wrist_img)
            np.save(CAMERA1_FILE, self.global_img)
        np.save(FORCE_FILE, self.wrench)

    def _send_action(self, action):
        try:
            with self.lock:
                curr = self.latest_pos

            if curr is None:
                return

            # ── 1. Absolute Action Execution & Delta Clamping (Max 0.15 rad per 100ms step) ──
            clamped_action = []
            max_step_delta = 0.15  # rad (max ~8.5 deg per 100ms step)

            raw_arm_action = np.array(action[:6], dtype=np.float32)
            # Auto-detect degrees vs radians (joint limits in rad are [-pi, pi])
            if self.GRIPPER_ACTION_MODE != "continuous_radians" and np.any(np.abs(raw_arm_action) > 3.14159):
                raw_arm_action = np.radians(raw_arm_action)

            for i in range(6):
                c_val = curr[i]
                a_val = float(raw_arm_action[i])  # Policy outputs ABSOLUTE target joint position (radians)
                delta = a_val - c_val
                if abs(delta) > max_step_delta:
                    a_val = c_val + float(np.sign(delta) * max_step_delta)
                clamped_action.append(float(a_val))

            if len(action) > 6:
                if self.GRIPPER_ACTION_MODE == "continuous_radians":
                    gripper_cmd = float(np.clip(
                        action[6], self.GRIPPER_OPEN_POSITION_RAD,
                        self.GRIPPER_CLOSED_POSITION_RAD,
                    ))
                else:
                    gripper_cmd = (
                        self.GRIPPER_CLOSED_POSITION_RAD
                        if float(action[6]) >= 0.5
                        else self.GRIPPER_OPEN_POSITION_RAD
                    )
            else:
                gripper_cmd = self.GRIPPER_OPEN_POSITION_RAD
            clamped_action.append(gripper_cmd)

            # A trajectory point's time scale is a policy-level parameter.
            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory.joint_names = ALL_JOINTS

            point = JointTrajectoryPoint()
            point.positions = clamped_action
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(self.ACTION_DT_S * 1e9)
            point.velocities = [0.0] * 7
            goal_msg.trajectory.points = [point]

            self._action_client.wait_for_server(timeout_sec=1.0)
            self._action_client.send_goal_async(goal_msg)

            # Legacy DetachableJoint support is opt-in only.  The current
            # baseline/PAP-MoE evaluation uses physical fingertip contact.
            gripper_distance = self._gripper_to_peg_distance()
            if self.ENABLE_DETACHABLE_JOINT and gripper_distance is not None:
                dist_xy, dist_z = gripper_distance

                if gripper_cmd > 0.4 and not self.is_attached:
                    if self._attachment_pose_is_safe(dist_xy, dist_z):
                        self.get_logger().info(
                            f"✓ Safeguard PASSED (xy={dist_xy * 100:.1f}cm, z={dist_z * 100:.1f}cm) -> Triggering /peg/attach"
                        )
                        subprocess.run(
                            ["ign", "topic", "-t", "/peg/attach", "-m", "ignition.msgs.Empty", "-p", ""],
                            capture_output=True,
                            timeout=2,
                        )
                        self.is_attached = True
                    else:
                        self.get_logger().warn(
                            f"⚠ Safeguard REJECTED distant attach! (xy={dist_xy * 100:.1f}cm, z={dist_z * 100:.1f}cm). Enforcing /peg/detach!"
                        )
                        subprocess.run(
                            ["ign", "topic", "-t", "/peg/detach", "-m", "ignition.msgs.Empty", "-p", ""],
                            capture_output=True,
                            timeout=1,
                        )
                elif gripper_cmd < 0.2 and self.is_attached:
                    self.get_logger().info("Triggering /peg/detach")
                    subprocess.run(
                        ["ign", "topic", "-t", "/peg/detach", "-m", "ignition.msgs.Empty", "-p", ""],
                        capture_output=True,
                        timeout=2,
                    )
                    self.is_attached = False
        except Exception as e:
            self.get_logger().error(f"Action send failed: {e}")

    def _send_action_chunk(self, chunk, *, recovery_mode=POLICY_MODE, source_sequence=-1):
        """Sends a multi-point Action Chunk (shape K x 7) to ROS 2 joint_trajectory_controller."""
        try:
            with self.lock:
                curr = self.latest_pos
            if curr is None:
                return

            requested_chunk = np.asarray(chunk, dtype=np.float32).copy()
            if requested_chunk.ndim != 2 or requested_chunk.shape[1] != len(ALL_JOINTS):
                raise ValueError(f"invalid requested action chunk shape {requested_chunk.shape}")
            # Modern continuous policies already output physical radians.
            # A valid joint angle > pi is not evidence of degrees. Retain
            # auto-detection only for explicitly legacy gripper contracts.
            if self.GRIPPER_ACTION_MODE != "continuous_radians":
                for row in requested_chunk:
                    if np.any(np.abs(row[:6]) > np.pi):
                        row[:6] = np.radians(row[:6])
            if self.GRIPPER_ACTION_MODE == "continuous_radians":
                executed_chunk, controller_chunk = clamp_continuous_for_controller(
                    requested_chunk,
                    np.asarray(curr, dtype=np.float32),
                    max_arm_step_rad=self.ACTION_CHUNK_MAX_STEP_RAD,
                    gripper_open_rad=self.GRIPPER_OPEN_POSITION_RAD,
                    gripper_closed_rad=self.GRIPPER_CLOSED_POSITION_RAD,
                )
            else:
                requested_chunk[:, 6] = (requested_chunk[:, 6] >= 0.5).astype(np.float32)
                executed_chunk, controller_chunk = clamp_for_controller(
                    requested_chunk,
                    np.asarray(curr, dtype=np.float32),
                    max_arm_step_rad=self.ACTION_CHUNK_MAX_STEP_RAD,
                    gripper_open_rad=self.GRIPPER_OPEN_POSITION_RAD,
                    gripper_closed_rad=self.GRIPPER_CLOSED_POSITION_RAD,
                )

            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory.joint_names = ALL_JOINTS

            goal_grace = getattr(self, "GOAL_TIME_TOLERANCE_S", 0.0)
            if not np.isfinite(goal_grace) or goal_grace < 0:
                raise ValueError("goal time tolerance must be finite and nonnegative")
            if goal_grace > 0:
                grace_ns = round(goal_grace * 1e9)
                goal_msg.goal_time_tolerance.sec = grace_ns // 1_000_000_000
                goal_msg.goal_time_tolerance.nanosec = grace_ns % 1_000_000_000

            # Attachment decisions use the measured world-frame midpoint of
            # both finger tips, never a future point from the predicted chunk.
            current_gripper = float(curr[6])
            requested_gripper_close = bool(
                np.any(executed_chunk[:, 6] >= 0.5)
            )
            gripper_distance = self._gripper_to_peg_distance()
            if gripper_distance is not None:
                dist_xy, dist_z = gripper_distance
                if (
                    self.ENABLE_DETACHABLE_JOINT
                    and (current_gripper >= 0.5 or requested_gripper_close)
                    and not self.is_attached
                ):
                    if self._attachment_pose_is_safe(dist_xy, dist_z):
                        self.get_logger().info(
                            f"✓ Model-close/actual-pose attach safeguard passed "
                            f"(xy={dist_xy * 100:.1f}cm, z={dist_z * 100:.1f}cm)"
                        )
                        # Publish once before executing the closing trajectory,
                        # otherwise contact can displace the free peg during the
                        # one-second chunk before the next measured-state check.
                        self._publish_detachable_joint("/peg/attach", attempts=1)
                        threading.Thread(
                            target=self._publish_detachable_joint,
                            args=("/peg/attach", 4),
                            daemon=True,
                        ).start()
                        self.is_attached = True
                elif self.ENABLE_DETACHABLE_JOINT and current_gripper < 0.2 and self.is_attached:
                    self.get_logger().info("✓ Actual gripper opened -> detaching peg")
                    threading.Thread(
                        target=self._publish_detachable_joint,
                        args=("/peg/detach",),
                        daemon=True,
                    ).start()
                    self.is_attached = False

            for k, point_pos in enumerate(controller_chunk):
                pt = JointTrajectoryPoint()
                pt.positions = point_pos.tolist()
                t_sec = (k + 1) * self.ACTION_DT_S
                pt.time_from_start.sec = int(t_sec)
                pt.time_from_start.nanosec = int((t_sec - int(t_sec)) * 1e9)
                goal_msg.trajectory.points.append(pt)

            dispatch_timestamp = time.time()
            trace_id = getattr(self, "_diagnostic_chunk_id", 0)
            if diagnostic_trace.enabled():
                self._diagnostic_chunk_id = trace_id + 1
                diagnostic_trace.event(
                    "dispatch", chunk_id=trace_id, state=list(curr),
                    requested=requested_chunk.tolist(), executed=executed_chunk.tolist(),
                    controller=controller_chunk.tolist(), action_dt=self.ACTION_DT_S,
                    gripper_distance=gripper_distance,
                )

                def trace_feedback(message, chunk_id=trace_id):
                    feedback = message.feedback
                    diagnostic_trace.event(
                        "feedback", chunk_id=chunk_id, joints=list(feedback.joint_names),
                        sim_stamp=feedback.header.stamp.sec + feedback.header.stamp.nanosec * 1e-9,
                        desired=list(feedback.desired.positions),
                        actual=list(feedback.actual.positions), error=list(feedback.error.positions),
                        trajectory_time=feedback.desired.time_from_start.sec
                        + feedback.desired.time_from_start.nanosec * 1e-9,
                    )

                future = self._action_client.send_goal_async(goal_msg, feedback_callback=trace_feedback)
            else:
                future = self._action_client.send_goal_async(goal_msg)
            # Bind feedback's reference to this goal, not a mutable last-goal
            # field or the model's pre-clamp request. Controller units are rad
            # even when the legacy policy encodes the gripper as binary.
            controller_target = np.asarray(controller_chunk).copy()
            controller_target.setflags(write=False)
            future._pap_controller_target = controller_target
            if self.rollout_recovery_session_dir is not None:
                write_action_event(
                    self.rollout_recovery_session_dir,
                    sequence=self.rollout_recovery_event_sequence,
                    mode=recovery_mode,
                    source_sequence=source_sequence,
                    requested_action=requested_chunk,
                    executed_action=executed_chunk,
                    controller_action=controller_chunk,
                    action_dt_s=self.ACTION_DT_S,
                    dispatch_timestamp=dispatch_timestamp,
                )
                self.rollout_recovery_event_sequence += 1
            return future
        except Exception as e:
            self.get_logger().error(f"Failed to send action chunk: {e}")
            return None

    def _wait_for_action_chunk(self, send_goal_future):
        """Wait for a trajectory in wall time, but let ROS/Gazebo own its duration."""
        if send_goal_future is None:
            return False
        deadline = time.monotonic() + self.ACTION_RESULT_TIMEOUT_S
        while rclpy.ok() and not send_goal_future.done():
            if time.monotonic() >= deadline:
                self.get_logger().error("Timed out waiting for trajectory goal acceptance.")
                return False
            time.sleep(0.01)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Action-chunk trajectory goal was rejected.")
            return False

        result_future = goal_handle.get_result_async()
        success_checks = 0
        last_success_check = None
        while rclpy.ok() and not result_future.done():
            now = time.monotonic()
            sim_now = self.get_clock().now().nanoseconds * 1e-9
            if last_success_check is not None and sim_now < last_success_check:
                success_checks = 0
                last_success_check = None
            # Recovery collection owns its own terminal/ack protocol.  This
            # early geometric-success path is deliberately limited to pure
            # evaluation so it cannot truncate an expert-labelled chunk.
            if (
                self.rollout_recovery_session_dir is None
                and (last_success_check is None or sim_now - last_success_check >= self.ACTION_DT_S)
            ):
                last_success_check = sim_now
                success, metrics = self._gazebo_task_success()
                success_checks = success_checks + 1 if success else 0
                if success_checks >= self.SUCCESS_REQUIRED_CHECKS:
                    self._success_during_action_metrics = metrics
                    goal_handle.cancel_goal_async()
                    return True
            if now >= deadline:
                self.get_logger().error("Timed out waiting for action-chunk trajectory completion.")
                return False
            time.sleep(0.01)
        wrapped_result = result_future.result()
        if wrapped_result is None or wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            status = None if wrapped_result is None else wrapped_result.status
            result = None if wrapped_result is None else wrapped_result.result
            error_code = None if result is None else result.error_code
            error_string = None if result is None else result.error_string
            # A position-controlled gripper commanded to its universal
            # full-close endpoint is expected to stop early when it contacts
            # an object.  joint_trajectory_controller reports that healthy
            # force-closure outcome as GOAL_TOLERANCE_VIOLATED.  Accept only
            # this precise case, and only when every arm joint reached its
            # target while the measured knuckle actually closed.  This keeps
            # the command object-independent (0=open, 0.8=close) without
            # hiding arm tracking failures or a broken/non-moving gripper.
            expected = getattr(send_goal_future, "_pap_controller_target", None)
            with self.lock:
                measured = None if self.latest_pos is None else np.asarray(self.latest_pos).copy()
            contact_limited_close = False
            if (
                error_code == -5
                and expected is not None
                and expected.ndim == 2
                and expected.shape[0] > 0
                and expected.shape[1] == len(ALL_JOINTS)
                and np.isfinite(expected).all()
                and measured is not None
                and measured.shape == (len(ALL_JOINTS),)
                and np.isfinite(measured).all()
                and float(expected[-1, 6]) >= 0.5
            ):
                arm_error = float(np.max(np.abs(measured[:6] - expected[-1, :6])))
                # This is an acknowledgement that the close *attempt* was
                # physically executed, not a declaration of grasp success.
                # The recovery expert separately requires measured closure
                # and verified peg lift before saving a successful episode.
                # Require meaningful closure, not merely controller noise or
                # a short-lived partial command.  The command remains the
                # object-independent 0.8-rad endpoint; this lower bound only
                # decides whether a GOAL_TOLERANCE_VIOLATED result can be
                # interpreted as physical contact instead of a stalled or
                # oscillating gripper.
                # Only forgive the gripper's contact-limited endpoint.  A
                # large arm residual is a collision/tracking failure, not a
                # successful close.  The previous 0.20-rad allowance hid an
                # expert arm that was effectively stationary at the socket.
                # During recovery collection, a low-angle unilateral contact
                # is itself the failure state the expert must observe and
                # correct. Keep the strict 0.20-rad acceptance in pure eval,
                # but allow the auditable recovery loop to continue from the
                # physically executed contact instead of terminating first.
                minimum_closure = (
                    0.05 if self.rollout_recovery_session_dir is not None else 0.20
                )
                contact_limited_close = (
                    arm_error <= 0.15 and float(measured[6]) >= minimum_closure
                )
                if contact_limited_close:
                    self.get_logger().info(
                        "Accepted contact-limited full-close command "
                        f"(measured_gripper={measured[6]:.3f}, arm_error={arm_error:.4f}rad)."
                    )
                    return True
                self.get_logger().warn(
                    "Rejected contact-limited close exception "
                    f"(measured_gripper={measured[6]:.3f}, arm_error={arm_error:.4f}rad)."
                )
            self.get_logger().error(
                "Action-chunk trajectory did not succeed "
                f"(status={status}, error_code={error_code}, error_string={error_string!r})."
            )
            return False
        return True

    def _finish_success_reached_during_action(self):
        metrics = self._success_during_action_metrics
        if metrics is None:
            return False
        self._success_during_action_metrics = None
        distance_xy, peg_z = metrics
        self.get_logger().info("==================================================")
        self.get_logger().info(
            "SUCCESS: peg is inserted "
            f"(xy={distance_xy:.4f}m, z={peg_z:.4f}m); "
            "accepted while the contact trajectory was active."
        )
        self.get_logger().info("==================================================")
        return True

    def move_to_start_pose(self):
        self.get_logger().info("Waiting for FollowJointTrajectory action server...")
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("FollowJointTrajectory action server not available!")
            return False

        target = np.asarray(self.START_POSE, dtype=np.float64)
        self.get_logger().info(f"Moving robot to starting posture {target.tolist()}...")

        # The simulator normally starts at this pose. Trust measured joint
        # feedback instead of requiring an unnecessary action round trip.
        feedback_deadline = time.monotonic() + 5.0
        while time.monotonic() < feedback_deadline:
            with self.lock:
                current = None if self.latest_pos is None else np.asarray(self.latest_pos)
            if current is not None and np.max(np.abs(current - target)) <= self.START_POSE_TOLERANCE_RAD:
                self.get_logger().info("Robot is already at the starting posture.")
                return True
            time.sleep(0.05)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ALL_JOINTS
        point = JointTrajectoryPoint()
        point.positions = target.tolist()
        point.time_from_start.sec = 5
        point.time_from_start.nanosec = 0
        goal_msg.trajectory.points.append(point)

        send_goal_future = self._action_client.send_goal_async(goal_msg)

        # Gazebo can accept the trajectory just as the action-response future
        # times out under load. The measured state is the authoritative test.
        pose_deadline = time.monotonic() + self.START_POSE_TIMEOUT_S
        while time.monotonic() < pose_deadline:
            if send_goal_future.done():
                goal_handle = send_goal_future.result()
                if goal_handle is not None and not goal_handle.accepted:
                    self.get_logger().error("Starting-posture goal was rejected.")
                    return False
            with self.lock:
                current = None if self.latest_pos is None else np.asarray(self.latest_pos)
            if current is not None and np.max(np.abs(current - target)) <= self.START_POSE_TOLERANCE_RAD:
                self.get_logger().info("Robot starting posture confirmed from joint feedback.")
                return True
            time.sleep(0.05)

        with self.lock:
            current = None if self.latest_pos is None else np.asarray(self.latest_pos).copy()
        residual = None if current is None else np.abs(current - target)
        self.get_logger().error(
            f"Starting posture was not reached within {self.START_POSE_TIMEOUT_S:.1f}s; "
            f"measured={None if current is None else current.tolist()}, "
            f"abs_error={None if residual is None else residual.tolist()}. Policy will not run."
        )
        return False

    def _outside_training_state_envelope(self, current):
        if self.TRAINING_STATE_MIN is None or self.TRAINING_STATE_MAX is None:
            return None
        lower = np.asarray(self.TRAINING_STATE_MIN, dtype=np.float64)
        upper = np.asarray(self.TRAINING_STATE_MAX, dtype=np.float64)
        values = np.asarray(current, dtype=np.float64)
        bad = np.flatnonzero((values < lower) | (values > upper))
        return bad.tolist() if bad.size else None

    def _gazebo_task_success(self):
        if not self.ENABLE_GAZEBO_SUCCESS_CHECK:
            return False, None
        peg_pose = get_model_pose("peg")
        hole_pose = get_model_pose("hole_plate")
        if peg_pose is None or hole_pose is None:
            return False, None
        distance_xy = float(np.linalg.norm(peg_pose[:2] - hole_pose[:2]))
        success = distance_xy <= self.SUCCESS_MAX_XY_M and peg_pose[2] < self.SUCCESS_MAX_PEG_Z_M
        return success, (distance_xy, float(peg_pose[2]))

    def _observation_ready(self):
        """Return whether policy-specific temporal observations are usable."""
        return True

    def _reset_policy_observation_context(self):
        """Discard temporal samples collected while moving to the start pose."""

    def _status_force_norm(self):
        """Return the force norm shown in the periodic evaluation status."""
        with self.lock:
            force = np.asarray(self.wrench[:3], dtype=np.float32).copy()
        return float(np.linalg.norm(force))

    def run_loop(self):
        if not self.move_to_start_pose():
            self.get_logger().error("ABORT: initial posture validation failed.")
            return
        self._reset_policy_observation_context()

        self.get_logger().info("Waiting for inference server to finish loading model weights...")
        while not os.path.exists("/tmp/ur3_inference_ready.txt") and rclpy.ok():
            time.sleep(0.2)

        observation_deadline = time.monotonic() + self.OBSERVATION_READY_TIMEOUT_S
        observation_wait_logged = False
        while rclpy.ok() and not self._observation_ready():
            if not observation_wait_logged:
                self.get_logger().info("Waiting for policy-specific observation history to fill...")
                observation_wait_logged = True
            # Publish the current validity metadata while temporal windows
            # fill. Inference can then distinguish warmup from malformed data.
            self._save_images()
            if time.monotonic() >= observation_deadline:
                self.get_logger().error(
                    "ABORT: policy observation history did not become ready "
                    f"within {self.OBSERVATION_READY_TIMEOUT_S:.1f}s."
                )
                return
            time.sleep(0.1)
        if observation_wait_logged:
            self.get_logger().info("Policy observation history is ready.")

        self.get_logger().info(
            "✓ Inference server READY! Starting "
            f"{self.CONTROL_HZ} Hz observation/action loop "
            f"(action_dt={self.ACTION_DT_S:.3f}s, "
            f"replan_interval={self.REPLAN_INTERVAL_S:.3f}s)..."
        )
        # Clear stale action and joint state files
        if os.path.exists(ACTION_FILE):
            os.remove(ACTION_FILE)
        if os.path.exists(JOINT_STATE_FILE):
            os.remove(JOINT_STATE_FILE)

        rate = self.create_rate(self.CONTROL_HZ)
        last_action_mtime = None
        episode_start = None
        first_action_deadline = time.monotonic() + self.FIRST_ACTION_TIMEOUT_S
        success_checks = 0
        last_status_log = -5.0

        while rclpy.ok():
            try:
                if episode_start is None and time.monotonic() >= first_action_deadline:
                    self.get_logger().error(
                        f"ABORT: no valid first action received within {self.FIRST_ACTION_TIMEOUT_S:.1f}s."
                    )
                    return
                elapsed = (
                    0.0
                    if episode_start is None
                    else (self.get_clock().now() - episode_start).nanoseconds * 1e-9
                )
                if episode_start is not None and elapsed >= self.MAX_EPISODE_DURATION_S:
                    self.get_logger().info("==================================================")
                    self.get_logger().info(f"TIMEOUT: episode reached {self.MAX_EPISODE_DURATION_S:.1f}s.")
                    self.get_logger().info("==================================================")
                    if self.is_attached:
                        subprocess.run(
                            ["ign", "topic", "-t", "/peg/detach", "-m", "ignition.msgs.Empty", "-p", ""],
                            capture_output=True,
                            timeout=2,
                        )
                        self.is_attached = False
                    if not self.RESET_AFTER_EPISODE:
                        return
                    if not self.move_to_start_pose():
                        return
                    episode_start = None
                    first_action_deadline = time.monotonic() + self.FIRST_ACTION_TIMEOUT_S
                    success_checks = 0
                    time.sleep(1.0)
                    continue

                # 1. Capture current simulation observations
                if action_exchange.enabled():
                    with self.lock:
                        source_stamps = dict(self._source_stamps)
                    try:
                        action_exchange.validate_source_stamps(
                            source_stamps, self.get_clock().now().nanoseconds * 1e-9, time.monotonic()
                        )
                    except ValueError as error:
                        self.get_logger().error(f"INFRASTRUCTURE INVALID: {error}")
                        return
                self._save_images()
                with self.lock:
                    curr_pos = self._snapshot_joint_pos if action_exchange.enabled() else self.latest_pos

                if curr_pos is None:
                    rate.sleep()
                    continue

                bad_joints = self._outside_training_state_envelope(curr_pos)
                if bad_joints:
                    details = ", ".join(f"{ALL_JOINTS[index]}={curr_pos[index]:.3f}" for index in bad_joints)
                    self.get_logger().error(
                        "OUT-OF-DISTRIBUTION: measured state left the guarded "
                        f"training envelope ({details}). Stopping episode."
                    )
                    return

                success, metrics = self._gazebo_task_success()
                if success:
                    success_checks += 1
                else:
                    success_checks = 0

                if elapsed - last_status_log >= 5.0:
                    last_status_log = elapsed
                    arm_text = np.array2string(
                        np.asarray(curr_pos[:6]),
                        precision=3,
                        suppress_small=True,
                    )
                    force_norm = self._status_force_norm()
                    task_text = "pose unavailable"
                    if metrics is not None:
                        distance_xy, peg_z = metrics
                        task_text = f"peg_hole_xy={distance_xy:.4f}m, peg_z={peg_z:.4f}m"
                    gripper_distance = self._gripper_to_peg_distance()
                    grasp_text = "gripper TF unavailable"
                    if gripper_distance is not None:
                        gripper_xy, gripper_z = gripper_distance
                        grasp_text = f"gripper_peg_xy={gripper_xy:.4f}m, gripper_peg_z={gripper_z:.4f}m"
                    self.get_logger().info(
                        f"EVAL status: t={elapsed:.1f}s, arm={arm_text}, "
                        f"gripper={curr_pos[6]:.3f}, |force|={force_norm:.2f}N, "
                        f"attached={self.is_attached}, {grasp_text}, {task_text}"
                    )

                if success_checks >= self.SUCCESS_REQUIRED_CHECKS:
                    distance_xy, peg_z = metrics
                    self.get_logger().info("==================================================")
                    self.get_logger().info(
                        f"SUCCESS: peg is inserted (xy={distance_xy:.4f}m, z={peg_z:.4f}m)."
                    )
                    self.get_logger().info("==================================================")
                    return

                # 2. Write joint states atomically to trigger inference
                tmp_js = JOINT_STATE_FILE + ".tmp"
                with open(tmp_js, "w") as f:
                    f.write(" ".join(f"{p:.6f}" for p in curr_pos))
                os.rename(tmp_js, JOINT_STATE_FILE)
                request_id = action_exchange.observation_id(JOINT_STATE_FILE)
                if diagnostic_trace.enabled():
                    diagnostic_trace.event("observation_request", request_id=request_id, state=list(curr_pos))

                # 3. Wait for inference loop to write the action chunk file (up to 1.0s)
                t_start = time.time()
                action_received = False
                paired_actions = action_exchange.enabled()
                chunk_file = os.environ["POLICY_PAIRED_ACTION_FILE"] if paired_actions else "/tmp/ur3_action_chunk.npy"

                while time.time() - t_start < 3.0:
                    if paired_actions and os.environ.get("WORKSPACE50_EVALUATION_KIND") == "engineering_demonstration_replay":
                        finished = Path(chunk_file + ".finished")
                        if finished.exists() and finished.read_text() == request_id:
                            self.get_logger().info("DEMONSTRATION COMPLETE: recorded actions exhausted without geometric success")
                            return
                    recovery_mode = POLICY_MODE
                    source_sequence = -1
                    if self._refresh_rollout_recovery_takeover():
                        expert_chunk = read_expert_chunk(
                            self.rollout_recovery_session_dir,
                            after_sequence=self.rollout_recovery_expert_sequence,
                            max_age_s=3.0,
                        )
                        if expert_chunk is not None:
                            # ACTION_CHUNK_SIZE is the policy execution horizon.
                            # A recovery expert chunk is already an explicitly
                            # acknowledged, self-contained trajectory and must
                            # not be truncated to the policy horizon.  Truncating
                            # a 10-point expert motion to one point made the
                            # expert believe the whole Cartesian step had run
                            # and caused raise/descend oscillations.
                            chunk = expert_chunk.action_chunk
                            self.rollout_recovery_expert_sequence = expert_chunk.sequence
                            recovery_mode = EXPERT_MODE
                            source_sequence = expert_chunk.sequence
                            action_event_sequence = self.rollout_recovery_event_sequence
                            if episode_start is None:
                                episode_start = self.get_clock().now()
                            send_goal_future = self._send_action_chunk(
                                chunk,
                                recovery_mode=recovery_mode,
                                source_sequence=source_sequence,
                            )
                            if not self._wait_for_action_chunk(send_goal_future):
                                return
                            if self._finish_success_reached_during_action():
                                return
                            write_action_completion(
                                self.rollout_recovery_session_dir,
                                action_event_sequence=action_event_sequence,
                                mode=recovery_mode,
                                source_sequence=source_sequence,
                            )
                            write_expert_execution_ack(
                                self.rollout_recovery_session_dir,
                                expert_sequence=source_sequence,
                                action_event_sequence=action_event_sequence,
                            )
                            action_received = True
                            break
                        time.sleep(0.001)
                        continue
                    if os.path.exists(chunk_file):
                        mtime = os.path.getmtime(chunk_file)
                        if mtime != last_action_mtime:
                            last_action_mtime = mtime
                            try:
                                chunk = action_exchange.read_reply(request_id) if paired_actions else np.load(chunk_file)
                                if chunk is None:
                                    continue
                                if paired_actions and diagnostic_trace.enabled():
                                    diagnostic_trace.event("paired_action_received", request_id=request_id)
                                if chunk.ndim != 2 or chunk.shape[1] != len(ALL_JOINTS):
                                    raise ValueError(
                                        f"Invalid action chunk shape {chunk.shape}; "
                                        f"expected [K, {len(ALL_JOINTS)}]"
                                    )
                                chunk = chunk[: self.ACTION_CHUNK_SIZE]
                                if self._guard_policy_chunk_for_recovery(chunk):
                                    continue
                                if episode_start is None:
                                    episode_start = self.get_clock().now()
                                source_sequence = self.rollout_recovery_policy_sequence
                                self.rollout_recovery_policy_sequence += 1
                                action_event_sequence = self.rollout_recovery_event_sequence
                                send_goal_future = self._send_action_chunk(
                                    chunk,
                                    recovery_mode=POLICY_MODE,
                                    source_sequence=source_sequence,
                                )
                                if not self._wait_for_action_chunk(send_goal_future):
                                    return
                                if self._finish_success_reached_during_action():
                                    return
                                if self.rollout_recovery_session_dir is not None:
                                    write_action_completion(
                                        self.rollout_recovery_session_dir,
                                        action_event_sequence=action_event_sequence,
                                        mode=POLICY_MODE,
                                        source_sequence=source_sequence,
                                    )
                                action_received = True
                                break
                            except Exception as error:
                                if paired_actions:
                                    self.get_logger().error(f"INFRASTRUCTURE INVALID: paired action failed: {error}")
                                    return
                    elif (
                        self.rollout_recovery_session_dir is None
                        and not paired_actions
                        and os.path.exists(ACTION_FILE)
                    ):
                        mtime = os.path.getmtime(ACTION_FILE)
                        if mtime != last_action_mtime:
                            last_action_mtime = mtime
                            with open(ACTION_FILE, "r") as f:
                                line = f.readline().strip()
                            if line:
                                action = np.array([float(x) for x in line.split()], dtype=np.float32)
                                self._send_action(action)
                                action_received = True
                                break
                    time.sleep(0.001)

                if not action_received:
                    if paired_actions:
                        self.get_logger().error("INFRASTRUCTURE INVALID: paired action timeout; no new observation issued")
                        return
                    self.get_logger().warn("Inference timeout: action not received in 3.0s")
                    rate.sleep()
            except Exception as e:
                self.get_logger().error(f"Loop error: {e}")
                rate.sleep()


# Exact training dataset peg and hole positions (Episodes 301 to 305 and 1 to 5)
DATASET_TRAIN_POSITIONS = {
    301: {"peg": (-0.061, 0.241), "hole": (0.086, 0.259)},
    302: {"peg": (-0.012, 0.344), "hole": (-0.188, 0.341)},
    303: {"peg": (-0.063, 0.344), "hole": (0.156, 0.349)},
    304: {"peg": (0.022, 0.308), "hole": (0.151, 0.315)},
    305: {"peg": (0.010, 0.289), "hole": (-0.188, 0.251)},
    # Direct alias mapping 1->301, 2->302, etc.
    1: {"peg": (-0.061, 0.241), "hole": (0.086, 0.259)},
    2: {"peg": (-0.012, 0.344), "hole": (-0.188, 0.341)},
    3: {"peg": (-0.063, 0.344), "hole": (0.156, 0.349)},
    4: {"peg": (0.022, 0.308), "hole": (0.151, 0.315)},
    5: {"peg": (0.010, 0.289), "hole": (-0.188, 0.251)},
}


def run_ros_side(
    controller_cls=PegInHoleROSSide,
    *,
    node_name="peg_in_hole_ros_side",
    args=None,
):
    """Run a policy-specific ROS-side controller using the shared environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", action="store_true", help="Spawn peg and hole in Gazebo")
    parser.add_argument(
        "--ep",
        type=int,
        default=301,
        help="Dataset episode index (301 to 305 or 1 to 5) for exact training positions",
    )
    parser.add_argument("--peg-x", type=float, default=None)
    parser.add_argument("--peg-y", type=float, default=None)
    parser.add_argument("--hole-x", type=float, default=None)
    parser.add_argument("--hole-y", type=float, default=None)
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=None,
        help="Override the policy-specific execution prefix for diagnostics.",
    )
    parser.add_argument(
        "--rollout-recovery-session-dir",
        type=Path,
        default=None,
        help=(
            "Enable the independent policy-to-expert recovery IPC in a new, "
            "empty per-run session directory. Disabled by default."
        ),
    )
    parser.add_argument(
        "--rollout-recovery-source-checkpoint",
        default="",
        help="Policy checkpoint provenance stored in recovery session metadata.",
    )
    parsed_args, _ = parser.parse_known_args()

    spawned = []
    if parsed_args.spawn:
        # Default to exact training positions for specified episode if available
        ep_idx = parsed_args.ep
        default_pos = DATASET_TRAIN_POSITIONS.get(ep_idx, DATASET_TRAIN_POSITIONS[301])

        default_peg_x, default_peg_y = default_pos["peg"]
        default_hole_x, default_hole_y = default_pos["hole"]

        px = parsed_args.peg_x if parsed_args.peg_x is not None else default_peg_x
        py = parsed_args.peg_y if parsed_args.peg_y is not None else default_peg_y
        hx = parsed_args.hole_x if parsed_args.hole_x is not None else default_hole_x
        hy = parsed_args.hole_y if parsed_args.hole_y is not None else default_hole_y

        # Validate against the selected controller's dataset contract.  An
        # explicit manifest pose is provenance-critical: reject an invalid pose
        # instead of silently evaluating the policy on a random scene.
        explicit_pose = all(
            value is not None
            for value in (
                parsed_args.peg_x,
                parsed_args.peg_y,
                parsed_args.hole_x,
                parsed_args.hole_y,
            )
        )
        for _ in range(100):
            p_ok = px**2 + py**2 <= controller_cls.SPAWN_MAX_XY_SQ
            h_ok = hx**2 + hy**2 <= controller_cls.SPAWN_MAX_XY_SQ
            dist = ((px - hx) ** 2 + (py - hy) ** 2) ** 0.5
            if p_ok and h_ok and dist >= MIN_PEG_HOLE_DIST:
                break
            if explicit_pose:
                radius = controller_cls.SPAWN_MAX_XY_SQ**0.5
                raise ValueError(
                    "Explicit peg/hole pose violates the selected scene contract: "
                    f"peg=({px:.6f}, {py:.6f}), hole=({hx:.6f}, {hy:.6f}), "
                    f"max_radius={radius:.3f}m, min_separation={MIN_PEG_HOLE_DIST:.3f}m"
                )
            px = random.uniform(*BLOCK_X_RANGE)
            py = random.uniform(*BLOCK_Y_RANGE)
            hx = random.uniform(*HOLE_X_RANGE)
            hy = random.uniform(*HOLE_Y_RANGE)

        peg_name = "peg"
        hole_name = "hole_plate"

        peg_mass = 0.135
        ixx_peg = 0.000248
        izz_peg = 0.00001745
        kp_val = 5000
        kd_val = 40

        peg_sdf_str = PEG_SDF.format(
            name=peg_name, mass=peg_mass, ixx=ixx_peg, iyy=ixx_peg, izz=izz_peg, kp=kp_val, kd=kd_val
        )

        hole_sdf_str = make_hole_sdf(name=hole_name, kp=kp_val, kd=kd_val, mu=0.35, mu2=0.35)

        print(f"Spawning real-size solid printed Peg ({peg_name}) at ({px:.2f}, {py:.2f})")
        if spawn_model(peg_name, px, py, PEG_Z, sdf_string=peg_sdf_str):
            print("  Peg OK (model://pap_moe_real_peg/meshes/peg.stl)")
            spawned.append(peg_name)
            if controller_cls.ENABLE_DETACHABLE_JOINT:
                # Compatibility path for legacy experiments only.
                for _ in range(5):
                    time.sleep(0.3)
                    subprocess.run(
                        ["ign", "topic", "-t", "/peg/detach", "-m", "ignition.msgs.Empty", "-p", ""],
                        capture_output=True,
                        timeout=2,
                    )
                print("  --> ✓ Sent 5x post-spawn /peg/detach for legacy detachable-joint mode.")
        else:
            print("  Peg FAILED")

        print(f"Spawning rigid tapered lab socket ({hole_name}) at ({hx:.2f}, {hy:.2f})")
        if spawn_model(hole_name, hx, hy, HOLE_Z, sdf_string=hole_sdf_str):
            print("  Hole OK (rigid model://pap_moe_real_hole/meshes/hole.stl)")
            spawned.append(hole_name)
        else:
            print("  Hole FAILED")

    if spawned:

        def cleanup():
            for name in spawned:
                print(f"Deleting {name}...")
                delete_model(name)

        atexit.register(cleanup)

    peg_pose_tuple = (px, py, PEG_Z) if parsed_args.spawn else None

    rclpy.init(args=args)
    node = controller_cls(peg_pos=peg_pose_tuple, node_name=node_name)
    if parsed_args.action_chunk_size is not None:
        if parsed_args.action_chunk_size < 1:
            raise ValueError("--action-chunk-size must be positive")
        node.ACTION_CHUNK_SIZE = parsed_args.action_chunk_size
        node.REPLAN_INTERVAL_S = node.ACTION_DT_S * parsed_args.action_chunk_size
    if parsed_args.rollout_recovery_session_dir is not None:
        if not parsed_args.rollout_recovery_source_checkpoint.strip():
            raise ValueError(
                "--rollout-recovery-source-checkpoint is required when recovery IPC is enabled"
            )
        node.configure_rollout_recovery(
            parsed_args.rollout_recovery_session_dir,
            {
                "source_policy_checkpoint": parsed_args.rollout_recovery_source_checkpoint,
                "episode": parsed_args.ep,
                "action_chunk_size": node.ACTION_CHUNK_SIZE,
                "action_dt_s": node.ACTION_DT_S,
                "controller_max_arm_step_rad": node.ACTION_CHUNK_MAX_STEP_RAD,
                "predispatch_premature_release_guard": (
                    node.recovery_guard_premature_release
                ),
                "peg_spawn_xy": [px, py] if parsed_args.spawn else None,
                "hole_spawn_xy": [hx, hy] if parsed_args.spawn else None,
            },
        )
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

#!/usr/bin/env python3
"""
UR3 Peg-in-Hole active data recorder (PAP-MoE specific version).
Runs 'pap_moe_peg_in_hole' C++ node and records multi-modal dataset.
Includes controlled variations for the four PAP-MoE experts:
- E1: Free-space motion with the fixed, measured lab-fixture dynamics.
- E2: Wrist camera occlusion/dropout & extreme overexposure glare simulation.
- E3: Variable contact stiffness (kp) and friction coefficient (mu).
- E4: Observable axial/lateral force-displacement response during controlled
  insertion; the ground-validation fixture itself remains rigid.
"""

import argparse, os, sys, time, threading, subprocess, random, re
from collections import deque
import numpy as np
import cv2
from pap_moe_routing_prior import (
    compute_physical_state_vector,
    reset_label_buffers,
)

import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import WrenchStamped
from ros_gz_interfaces.msg import Contacts
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
GRIPPER_KINEMATIC_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
GRIPPER_MIMIC_MULTIPLIERS = np.asarray(
    [1.0, -1.0, 1.0, -1.0, -1.0, 1.0], dtype=np.float32
)
GRIPPER_MIMIC_MAX_ERROR_RAD = 0.01
IMG_SIZE = (224, 224)
STATE_DIM = 7
FORCE_DIM = 6
TASK = "pick up the peg and insert it into the hole"
SKILL_PROGRESS_PHASES = (
    "enter", "approach", "align", "interact",
    "stabilize", "verify", "exit", "recover",
)
GRIPPER_ACTION_OPEN_RAD = 0.0
GRIPPER_ACTION_CLOSED_RAD = 0.8
GRIPPER_ACTION_ENDPOINT_EPS = 1.0e-3
CONTACT_TRUTH_TIMEOUT_S = 0.25
CONTACT_TRUTH_TOPICS = (
    "/pap_moe/peg_body_contacts",
    "/pap_moe/peg_handle_contacts",
    "/pap_moe/hole_side_contacts",
    "/pap_moe/hole_floor_contacts",
)


def canonicalize_gripper_actions(actions):
    """Keep continuous transitions while making universal endpoints exact."""
    result = np.asarray(actions, dtype=np.float32).copy()
    gripper = np.clip(
        result[..., 6], GRIPPER_ACTION_OPEN_RAD, GRIPPER_ACTION_CLOSED_RAD
    )
    gripper = np.where(
        np.abs(gripper - GRIPPER_ACTION_OPEN_RAD) <=
        GRIPPER_ACTION_ENDPOINT_EPS,
        GRIPPER_ACTION_OPEN_RAD,
        gripper,
    )
    gripper = np.where(
        np.abs(gripper - GRIPPER_ACTION_CLOSED_RAD) <=
        GRIPPER_ACTION_ENDPOINT_EPS,
        GRIPPER_ACTION_CLOSED_RAD,
        gripper,
    )
    result[..., 6] = gripper
    return result

# ── UR3 Workspace Constraints ──
# Reference: ur3_samoe_peg_in_hole_record.py, narrowed for safety margin.
# UR3 base at world origin, reach ≈0.5m at table height (z≈0.78).
# x² + y² ∈ [MIN_XY_SQ, MAX_XY_SQ] keeps objects reachable & clear of base.
TABLE_Z     = 0.775
PEG_HEIGHT  = 0.180
PEG_Z       = TABLE_Z + PEG_HEIGHT / 2.0
HOLE_Z      = TABLE_Z
HOLE_TOP_Z  = TABLE_Z + 0.100
PEG_MASS    = 0.135
PEG_IXX     = 0.000248
PEG_IZZ     = 0.00001745

# Canonical task domain in the UR3 base/world XY frame.  The robot base is at
# world (0, 0, 0.760), the tabletop is z=0.775 and now extends along +X with
# bounds x=[-0.1065, 0.8065], y=[-0.4565, 0.4565].  The upright far-side
# oblique camera maps image down/up to +X/-X and image right/left to +Y/-Y.
# Peg is image-left (negative Y); hole is image-right (positive Y).  Their
# sampling boxes mirror each other across the tabletop y=0 centreline: the X
# ranges are identical and the Y ranges have opposite signs.  Centres are
# selected from the validated tabletop task region, not episode 13001,
# and the two +/-15 mm random boxes are sampled independently.
# HOME is also on the image-left side, but is separated from the peg box in
# XY and remains high above the table.  Including the +/-15 mm corners, both
# task boxes occupy a 0.324--0.366 m radial band. Reachability is still
# verified by the real MoveIt plan for every scripted episode.
PEG_SPAWN_CENTER = (0.300, -0.170)
HOLE_SPAWN_CENTER = (0.300, 0.170)
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
MIN_PEG_HOLE_DIST = 0.12        # between ref (0.10) and original PAP (0.15)

# Episode-level smooth path styles.  Values are quadratic-Bezier midpoint
# bulges in world metres; all styles share identical task endpoints.  This
# creates low-frequency path diversity without frame-wise action noise.
TRAJECTORY_STYLES = (
    {"name": "direct_standard", "peg": (0.0, 0.0, 0.0), "transport": (0.0, 0.0, 0.0), "hole": (0.0, 0.0), "speed": 0.35},
    {"name": "peg_arc_pos_x", "peg": (0.025, 0.0, 0.005), "transport": (0.0, 0.020, 0.0), "hole": (0.0, 0.0), "speed": 0.32},
    {"name": "peg_arc_neg_x", "peg": (-0.025, 0.0, 0.005), "transport": (0.0, -0.020, 0.0), "hole": (0.0, 0.0), "speed": 0.38},
    {"name": "peg_arc_pos_y", "peg": (0.0, 0.025, 0.0), "transport": (0.020, 0.0, 0.005), "hole": (0.0, 0.0), "speed": 0.30},
    {"name": "peg_arc_neg_y", "peg": (0.0, -0.025, 0.0), "transport": (-0.020, 0.0, 0.005), "hole": (0.0, 0.0), "speed": 0.40},
    {"name": "transport_wide_pos_y", "peg": (0.012, 0.0, 0.0), "transport": (0.0, 0.040, 0.005), "hole": (0.0, 0.0), "speed": 0.32},
    {"name": "transport_wide_neg_y", "peg": (-0.012, 0.0, 0.0), "transport": (0.0, -0.040, 0.005), "hole": (0.0, 0.0), "speed": 0.32},
    {"name": "hole_arc_pos_x", "peg": (0.0, 0.015, 0.0), "transport": (0.020, 0.0, 0.0), "hole": (0.006, 0.0), "speed": 0.35},
    {"name": "hole_arc_pos_y", "peg": (0.0, -0.015, 0.0), "transport": (-0.020, 0.0, 0.0), "hole": (0.0, 0.006), "speed": 0.35},
    {"name": "compound_arc", "peg": (0.018, -0.012, 0.004), "transport": (-0.025, 0.025, 0.005), "hole": (-0.005, -0.003), "speed": 0.30},
)
# At the 1.185 m collision-clear transport plane, targets beyond roughly
# 0.38 m repeatedly reached the UR3 high-z IK boundary. Keep the first
# ground-validation domain inside 0.361 m; expand only with a validated
# lower/curved transport planner.
# The 180 mm lab peg must be transported with a vertical wrist.  The former
# 0.361 m radial limit admitted poses that planned successfully but stopped
# about 10 mm short at the UR3 workspace boundary.
MAX_XY_SQ = 0.1444              # maximum radius 0.380 m for new task boxes
MIN_XY_SQ = 0.04                # min 0.20m from origin → avoid base self-collision

RECORD_HZ = 10
_HOLE_ZONE_Z_MAX = 0.88
FAST_FORCE_SAMPLES = 64
SLOW_FORCE_SAMPLES = 50
STATE_HISTORY_SAMPLES = 10
RESAMPLE_PERIOD_S = 0.1
FAST_FORCE_CLIP_N = 29.9

SUBTASKS = (
    "grasp the peg",
    "transport to the hole",
    "approach and align with the hole",
    "recover contact and relocate the hole",
    "insert the peg into the hole",
    "verify insertion success",
    "release the peg after verification",
    "retract and go back to home",
)


def _message_timestamp(message, fallback_clock):
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is not None:
        timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if timestamp > 0.0:
            return timestamp
    return fallback_clock()


def _should_resample(timestamp, previous):
    return previous is None or timestamp < previous or timestamp - previous >= 0.095


def _left_pad_window(values, size, width):
    if not values:
        return np.zeros((size, width), dtype=np.float32)
    selected = [np.asarray(value, dtype=np.float32) for value in values[-size:]]
    if len(selected) < size:
        selected = [selected[0]] * (size - len(selected)) + selected
    return np.stack(selected, axis=0)


def _clip_calibrated_fast_force(window):
    """Direction-preserving clip applied only after payload calibration."""
    clipped = np.asarray(window, dtype=np.float32).copy()
    force_norm = np.linalg.norm(clipped[..., :3], axis=-1)
    mask = force_norm > FAST_FORCE_CLIP_N
    if np.any(mask):
        scale = FAST_FORCE_CLIP_N / force_norm[mask]
        clipped[mask] *= scale[..., None]
    return clipped


def _left_pad_flags(values, size):
    if not values:
        return np.zeros(size, dtype=bool)
    selected = [bool(value) for value in values[-size:]]
    if len(selected) < size:
        selected = [selected[0]] * (size - len(selected)) + selected
    return np.asarray(selected, dtype=bool)


def _visual_quality(camera0, camera1):
    cameras = np.stack([camera0, camera1], axis=0).astype(np.float32, copy=False)
    gray = cameras.mean(axis=-1)
    finite = bool(np.isfinite(cameras).all())
    black = float(np.mean(gray <= 0.02)) if finite else 1.0
    saturated = float(np.mean(gray >= 0.98)) if finite else 1.0
    contrast = float(np.mean(np.std(gray, axis=(1, 2)))) if finite else 0.0
    valid = float(finite and contrast >= 0.01)
    return np.asarray([black, saturated, contrast, valid], dtype=np.float32)


def _derive_skill_progress(semantic_subtasks, states, actions, tool0_z):
    """Create task-agnostic local progress targets from controller evidence."""
    names = np.asarray([str(value) for value in semantic_subtasks], dtype=object)
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    tool0_z = np.asarray(tool0_z, dtype=np.float32)
    phase_for_subtask = {
        "transport to the hole": 1,
        "approach and align with the hole": 2,
        "recover contact and relocate the hole": 7,
        "insert the peg into the hole": 3,
        "verify insertion success": 5,
        "release the peg after verification": 6,
        "retract and go back to home": 6,
    }
    phase = np.asarray(
        [phase_for_subtask.get(name, 0) for name in names], dtype=np.int64
    )
    segment = names.copy()

    # The controller emits one coarse grasp subtask. Split it using measured
    # tool height and the continuous command/state gripper signals so the
    # labels do not depend on a task-specific object width.
    grasp = np.flatnonzero(names == "grasp the peg")
    if len(grasp):
        if not np.array_equal(grasp, np.arange(grasp[0], grasp[-1] + 1)):
            raise ValueError("coarse grasp segment must be contiguous")
        start, end = int(grasp[0]), int(grasp[-1] + 1)
        local_z = tool0_z[start:end]
        local_action = actions[start:end, 6]
        local_state = states[start:end, 6]
        close_candidates = np.flatnonzero(local_action > 0.12)
        if not len(close_candidates):
            raise ValueError("successful grasp has no continuous close command")
        close = int(close_candidates[0])
        peak = int(np.argmax(local_z[: max(close, 1)]))
        bottom = peak + int(np.argmin(local_z[peak : close + 1]))
        closed = np.flatnonzero(local_state[close:] > 0.60)
        if not len(closed):
            raise ValueError("successful grasp never reaches measured closure")
        lift = close + int(closed[0])
        boundaries = np.maximum.accumulate(
            np.asarray([0, peak + 1, bottom, close, lift, end - start])
        )
        boundaries[-1] = end - start
        grasp_parts = (
            ("grasp_approach", 1),
            ("grasp_align", 2),
            ("grasp_stabilize", 4),
            ("grasp_interact", 3),
            ("grasp_exit", 6),
        )
        for (part_name, phase_id), lower, upper in zip(
            grasp_parts, boundaries[:-1], boundaries[1:], strict=True
        ):
            lower, upper = int(lower), int(upper)
            phase[start + lower : start + upper] = phase_id
            segment[start + lower : start + upper] = part_name

    progress = np.zeros(len(names), dtype=np.float32)
    readiness = np.zeros(len(names), dtype=np.float32)
    begin = 0
    while begin < len(names):
        finish = begin + 1
        while finish < len(names) and segment[finish] == segment[begin]:
            finish += 1
        local = np.linspace(0.0, 1.0, finish - begin, dtype=np.float32)
        progress[begin:finish] = local
        readiness[begin:finish] = np.clip(
            (local - 0.75) / 0.25, 0.0, 1.0
        )
        begin = finish
    return {
        "phase": phase,
        "phase_name": np.asarray(
            [SKILL_PROGRESS_PHASES[index] for index in phase], dtype=np.str_
        ),
        "progress": progress,
        "readiness": readiness,
        "valid": np.ones(len(names), dtype=bool),
        "source": np.full(
            len(names), "controller_semantic_kinematic_v1", dtype="U40"
        ),
        "confidence": np.full(len(names), 0.85, dtype=np.float32),
    }

def spawn_model(name, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, sdf_string=None, file_path=None):
    try:
        cmd = ["/opt/ros/humble/bin/ros2", "run", "ros_gz_sim", "create",
               "-world", "simulation_world", "-name", name,
               "-x", str(x), "-y", str(y), "-z", str(z),
               "-R", str(roll), "-P", str(pitch), "-Y", str(yaw)]
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
    try:
        r = subprocess.run(
            ["ign", "topic", "-t", "/world/simulation_world/pose/info",
             "-e", "-n", "1"],
            capture_output=True, timeout=5)
        if r.returncode == 0 and r.stdout:
            text = r.stdout.decode()
            idx = text.find(f'name: "{name}"')
            if idx >= 0:
                snippet = text[idx:idx+700]
                m = re.search(
                    r'position\s*\{\s*x:\s*([\d.e-]+)\s*y:\s*([\d.e-]+)\s*z:\s*([\d.e-]+)',
                    snippet)
                if m:
                    position = tuple(float(m.group(i)) for i in range(1, 4))
                    q = re.search(
                        r'orientation\s*\{\s*x:\s*([\d.e-]+)\s*'
                        r'y:\s*([\d.e-]+)\s*z:\s*([\d.e-]+)\s*'
                        r'w:\s*([\d.e-]+)',
                        snippet,
                    )
                    if q:
                        return position + tuple(
                            float(q.group(i)) for i in range(1, 5)
                        )
                    return position
    except Exception:
        pass
    return None


def delete_model(name):
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
            return False
    except Exception as e:
        print(f"  ⚠ delete exception: {e}")
        return False


def wait_model_absent(name, timeout_s=5.0):
    """Wait for Gazebo's asynchronous entity removal to become observable."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if get_model_pose(name) is None:
            return True
        time.sleep(0.1)
    return get_model_pose(name) is None

# ── Dynamic SDF Templates ──

# Lab peg: solid printed part, 180 mm total height.  The mesh contains an
# 80x20 mm handle, 20x40 mm collar and 80 mm tapered insertion section.
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

# Rigid lab socket: outer diameter 60 mm, height 100 mm.  Its 80 mm tapered
# cavity has the same nominal frustum profile as the peg (zero designed radial
# clearance); the lower 20 mm is solid and there is no cylindrical segment.
def make_hole_sdf(name, kp=100000, kd=100, mu=0.5, mu2=0.5):
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

  <inertial>
    <mass>0.25</mass>
    <inertia>
      <ixx>0.003</ixx><iyy>0.003</iyy><izz>0.006</izz>
      <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
    </inertia>
  </inertial>
</link>
</model></sdf>"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for reproducible episode randomization")
    parser.add_argument(
        "--physics-engine",
        choices=[
            "ignition-physics-dartsim-plugin",
            "ignition-physics-bullet-plugin",
        ],
        default="ignition-physics-dartsim-plugin",
        help="Gazebo physics engine recorded in episode metadata",
    )
    parser.add_argument(
        "--sim-position-gain",
        type=float,
        default=0.5,
        help="gz_ros2_control global position proportional gain metadata",
    )
    parser.add_argument(
        "--max-peg-tilt",
        type=float,
        default=0.0,
        help=(
            "Experimental maximum spawn roll/pitch in radians. Defaults to 0; "
            "enable only after validating actual-pose orientation compensation"
        ),
    )
    parser.add_argument(
        "--search-mode",
        choices=["mixed", "direct", "force_gradient", "admittance", "spiral"],
        default="mixed",
        help=(
            "Use mixed direct/gradient/admittance/spiral scheduling or force "
            "one mode for validation"
        ))
    parser.add_argument(
        "--contact-kp",
        type=int,
        default=None,
        help=(
            "Optional Gazebo normal-contact stiffness override for controlled "
            "physics sweeps; default uses the fixed lab-fixture baseline"
        ),
    )
    parser.add_argument(
        "--recovery-offset-max",
        type=float,
        default=0.006,
        help=(
            "Maximum initial XY offset in metres for force-gradient and "
            "admittance recovery. Use 0.006 for the initial curriculum and "
            "increase toward 0.012 only after recovery success is stable."
        ),
    )
    parser.add_argument(
        "--grasp-recovery-offset-min",
        type=float,
        default=0.0,
        help=(
            "Minimum high-hover XY roll-in offset in metres. Zero together "
            "with --grasp-recovery-offset-max keeps standard collection."
        ),
    )
    parser.add_argument(
        "--grasp-recovery-offset-max",
        type=float,
        default=0.0,
        help=(
            "Maximum high-hover XY roll-in offset in metres. Recording starts "
            "after roll-in so saved actions only supervise recovery."
        ),
    )
    parser.add_argument(
        "--grasp-recovery-only",
        action="store_true",
        help="Stop after pose-verified grasp and save a grasp recovery prefix.",
    )
    parser.add_argument(
        "--fine-insertion-step",
        type=float,
        default=0.000010,
        help=(
            "Guarded insertion cruise advance per 100 Hz force-control cycle "
            "in metres; the final 1 mm automatically returns to 0.000005 m "
            "per cycle for safe seating"
        ),
    )
    parser.add_argument(
        "--camera-degradation-mode",
        choices=["random", "normal", "dropout", "glare", "balanced"],
        default="random",
        help=(
            "Control E2 visual degradation per run. 'random' keeps the 35%% "
            "curriculum probability; 'balanced' assigns the first half of "
            "each repeated-position group to dropout and the second half to "
            "glare; the other modes make pilot coverage deterministic."
        ),
    )
    parser.add_argument(
        "--trajectory-style",
        type=int,
        default=-1,
        help=(
            "Smooth trajectory style id 0-9. -1 cycles styles by episode "
            "within each repeated-position group."
        ),
    )
    parser.add_argument(
        "--trajectory-style-offset",
        type=int,
        default=0,
        help=(
            "Starting style id when --trajectory-style=-1. This lets a "
            "continued batch reuse a known scene position without repeating "
            "style 0."
        ),
    )
    parser.add_argument(
        "--position-repeat-count",
        type=int,
        default=1,
        help=(
            "Reuse exactly the same peg/hole pose for this many consecutive "
            "episodes while varying trajectory style."
        ),
    )
    parser.add_argument("--fixed-peg-x", type=float, default=None)
    parser.add_argument("--fixed-peg-y", type=float, default=None)
    parser.add_argument("--fixed-hole-x", type=float, default=None)
    parser.add_argument("--fixed-hole-y", type=float, default=None)
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help=(
            "Continue a multi-episode process after a failed task. Disabled "
            "by default because a failure-hold robot state can contaminate "
            "the next spawned zero-clearance scene."
        ),
    )
    parser.add_argument("--output", type=str,
                        default=os.path.expanduser(
                            "~/ur3_ft300_ws/pap_moe_framework/datasets/raw"))
    args = parser.parse_args()
    fixed_scene_values = (
        args.fixed_peg_x,
        args.fixed_peg_y,
        args.fixed_hole_x,
        args.fixed_hole_y,
    )
    if any(value is not None for value in fixed_scene_values) and not all(
        value is not None for value in fixed_scene_values
    ):
        parser.error(
            "--fixed-peg-x/--fixed-peg-y/--fixed-hole-x/--fixed-hole-y "
            "must be supplied together"
        )
    if not 0.0 <= args.max_peg_tilt <= 0.08:
        parser.error("--max-peg-tilt must be in [0.0, 0.08] radians")
    if args.contact_kp is not None and not 500 <= args.contact_kp <= 20000:
        parser.error("--contact-kp must be in [500, 20000]")
    if not 0.003 <= args.recovery_offset_max <= 0.012:
        parser.error("--recovery-offset-max must be in [0.003, 0.012] metres")
    if not (
        0.0 <= args.grasp_recovery_offset_min
        <= args.grasp_recovery_offset_max
        <= 0.12
    ):
        parser.error(
            "grasp recovery offsets must satisfy 0 <= min <= max <= 0.12 metres"
        )
    if args.grasp_recovery_only and args.grasp_recovery_offset_max <= 0.0:
        parser.error("--grasp-recovery-only requires a non-zero recovery offset")
    if not 0.000005 <= args.fine_insertion_step <= 0.000025:
        parser.error(
            "--fine-insertion-step must be in [0.000005, 0.000025] metres"
        )
    if not 0.01 <= args.sim_position_gain <= 100.0:
        parser.error("--sim-position-gain must be in [0.01, 100.0]")
    if args.trajectory_style not in range(-1, len(TRAJECTORY_STYLES)):
        parser.error("--trajectory-style must be -1 or an id in [0, 9]")
    if args.trajectory_style_offset not in range(len(TRAJECTORY_STYLES)):
        parser.error("--trajectory-style-offset must be in [0, 9]")
    if not 1 <= args.position_repeat_count <= len(TRAJECTORY_STYLES):
        parser.error("--position-repeat-count must be in [1, 10]")
    record_hz = args.hz
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    rclpy.init()
    os.makedirs(args.output, exist_ok=True)

    class Buffer:
        def __init__(self):
            self.lock = threading.Lock()
            self.joint_positions = None
            self.gripper_joint_positions = None
            # Commanded joint trajectory is the behavior-cloning action.
            # Measured JointState remains the observation.  Keeping these
            # distinct is essential for a gripper: a generic 0.8-rad close
            # command can physically stop near 0.63 rad after object contact.
            self.joint_position_commands = None
            self.wrench = np.zeros(FORCE_DIM, dtype=np.float32)
            self.wrist_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
            self.global_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
            self.tool0_z = None
            self.current_stage = -1
            self.force_bias = np.zeros(FORCE_DIM, dtype=np.float32)
            self.force_bias_valid = False
            # False: empty-tool reference; True: physically grasped payload
            # reference. This is sampled together with every native wrench so
            # multi-rate windows crossing grasp/release can be calibrated per
            # sample instead of applying one frame-level bias to the past.
            self.force_reference_payload = False
            self.force_sample_count = 0
            self.search_mode = "unknown"
            self.recovery_active = False
            self.recovery_attempt = 0
            self.recovery_direction = np.zeros(2, dtype=np.float32)
            self.recovery_result = -1
            self.recovery_event_id = 0
            self.recovery_safety = False
            self.semantic_subtask = SUBTASKS[0]
            self.force_native = deque(maxlen=512)
            # The wrench topic is 100 Hz. The median preserves the sustained
            # seating plateau. DART-only rigid-constraint impulses above 30 N
            # are median-filtered from slow/frame streams and directionally
            # capped in the fast stream, while remaining verbatim in the raw
            # native stream for duration-based validation.
            self.force_filter = deque(maxlen=11)
            self.force_slow = deque(maxlen=SLOW_FORCE_SAMPLES)
            self.state_history = deque(maxlen=STATE_HISTORY_SAMPLES)
            self.raw_force_timestamps = []
            self.raw_force_wall_timestamps = []
            self.raw_force_values = []
            self.capture_force_stream = False
            self.last_force_timestamp = None
            self.last_force_slow_timestamp = None
            self.last_state_timestamp = None
            # Simulator contact truth is teacher/audit metadata only.  It is
            # never exposed as a policy observation.  Each topic publishes
            # an empty Contacts message after separation; the timeout also
            # prevents a stale positive from leaking across scene resets.
            self.contact_truth_by_topic = {
                topic: {
                    "timestamp": float("-inf"),
                    "any": False,
                    "gripper": False,
                    "hole": False,
                    "table": False,
                    "max_depth": 0.0,
                }
                for topic in CONTACT_TRUTH_TOPICS
            }
            
            # Episode visual degradation settings (E2)
            self.cam_degradation = False
            self.cam_degradation_type = "normal"
            self.cam_glare_gain = 1.5        # default: minimal glare (1.5 = just noticeable)
            self.degrade_schedule = None      # pre-computed per-frame bool array for block degradation
            self.frame_idx = 0                # frame counter within episode

    buf = Buffer()

    class RecorderNode(Node):
        def __init__(self):
            super().__init__(
                "pap_moe_record_pih",
                parameter_overrides=[
                    Parameter("use_sim_time", Parameter.Type.BOOL, True)
                ],
            )
            self.bridge = CvBridge()
            self.attached = False
            self.attached_confirmed = False
            self.grasp_initial_pose = None
            self.release_baseline = None
            self.release_hole_name = ""
            cbg = ReentrantCallbackGroup()
            
            # TF Listener for precise world coordinate tracking
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            
            self.create_subscription(JointState, "/joint_states",
                                      self._js, 10, callback_group=cbg)
            self.create_subscription(
                JointTrajectoryControllerState,
                "/joint_trajectory_controller/controller_state",
                self._controller_state,
                10,
                callback_group=cbg,
            )
            self.create_subscription(Image, "/wrist_camera/color/image_raw",
                                      self._wrist, 10, callback_group=cbg)
            self.create_subscription(Image, "/global_camera/color/image_raw",
                                      self._global, 10, callback_group=cbg)
            self.create_subscription(WrenchStamped,
                                      "/force_torque_sensor_broadcaster/wrench",
                                      self._wrench, 10, callback_group=cbg)
            for contact_topic in CONTACT_TRUTH_TOPICS:
                self.create_subscription(
                    Contacts,
                    contact_topic,
                    lambda msg, topic=contact_topic: self._contacts(msg, topic),
                    10,
                    callback_group=cbg,
                )
            self.create_service(
                Trigger,
                "/pap_moe/confirm_grasp_attachment",
                self._confirm_grasp_attachment,
                callback_group=cbg,
            )
            self.create_service(
                Trigger,
                "/pap_moe/detach_peg",
                self._detach_peg,
                callback_group=cbg,
            )
            self.create_service(
                Trigger,
                "/pap_moe/verify_peg_release",
                self._verify_peg_release,
                callback_group=cbg,
            )
            self.create_service(
                Trigger,
                "/pap_moe/confirm_insertion_geometry",
                self._confirm_insertion_geometry,
                callback_group=cbg,
            )
            self.create_service(
                Trigger,
                "/pap_moe/query_peg_hole_alignment",
                self._query_peg_hole_alignment,
                callback_group=cbg,
            )

        def _confirm_grasp_attachment(self, request, response):
            del request
            peg_pose = get_model_pose("peg")
            initial = self.grasp_initial_pose
            xy_drift = float("inf")
            z_lift = float("-inf")
            if peg_pose is not None and initial is not None:
                xy_drift = float(np.hypot(
                    peg_pose[0] - initial[0],
                    peg_pose[1] - initial[1],
                ))
                z_lift = float(peg_pose[2] - initial[2])
            self.attached_confirmed = (
                peg_pose is not None
                and initial is not None
                and xy_drift <= 0.010
                and z_lift >= 0.050
            )
            self.attached = self.attached_confirmed
            if self.attached_confirmed:
                with buf.lock:
                    buf.force_reference_payload = True
            response.success = self.attached_confirmed
            response.message = (
                f"physical_grasp xy_drift={xy_drift:.4f}, z_lift={z_lift:.4f}"
            )
            return response

        def _confirm_insertion_geometry(self, request, response):
            del request
            peg_pose = get_model_pose("peg")
            hole_pose = get_model_pose(self.release_hole_name)
            if peg_pose is None or hole_pose is None:
                response.success = False
                response.message = "missing peg/hole pose"
                return response

            hole_dist = float(np.hypot(
                peg_pose[0] - hole_pose[0],
                peg_pose[1] - hole_pose[1],
            ))
            # The peg tip seats on the 20 mm solid bottom:
            # TABLE_Z + 20 mm + half the 180 mm peg = 0.885 m centre height.
            # The tapered pair has the same nominal profile, so a loose XY
            # tolerance would classify a wall-overlap pose as seated.
            target_seat_z = 0.885
            response.success = (
                hole_dist <= 0.0003
                and abs(peg_pose[2] - target_seat_z) <= 0.00035
            )
            response.message = (
                f"hole_dist={hole_dist:.6f}, z={peg_pose[2]:.6f}"
            )
            return response

        def _query_peg_hole_alignment(self, request, response):
            """Expose simulator truth to the scripted demonstrator only.

            These values are never stored as policy observations. They make
            the expert compensate the episode-specific transform created by
            a purely physical (non-attached) grasp before exact-fit descent.
            """
            del request
            peg_pose = get_model_pose("peg")
            hole_pose = get_model_pose(self.release_hole_name)
            if peg_pose is None or hole_pose is None:
                response.success = False
                response.message = "missing peg/hole pose"
                return response
            dx = float(peg_pose[0] - hole_pose[0])
            dy = float(peg_pose[1] - hole_pose[1])
            tilt_rad = float("nan")
            if len(peg_pose) >= 7:
                qx, qy = peg_pose[3], peg_pose[4]
                local_z_world_z = np.clip(
                    1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0
                )
                tilt_rad = float(np.arccos(local_z_world_z))
            response.success = True
            response.message = (
                f"dx={dx:.9f},dy={dy:.9f},"
                f"hole_dist={np.hypot(dx, dy):.9f},"
                f"peg_z={peg_pose[2]:.9f},tilt_rad={tilt_rad:.9f}"
            )
            return response

        def _detach_peg(self, request, response):
            del request
            peg_pose = get_model_pose("peg")
            # The current episode always has exactly one hole_plate_* model;
            # the main loop updates this name before requesting release.
            hole_pose = get_model_pose(self.release_hole_name)
            self.release_baseline = peg_pose
            response.success = (
                peg_pose is not None and hole_pose is not None
            )
            response.message = (
                f"physical_release_baseline={peg_pose}"
            )
            if response.success:
                self.attached = False
                self.attached_confirmed = False
            return response

        def _verify_peg_release(self, request, response):
            del request
            time.sleep(0.5)
            peg_pose = get_model_pose("peg")
            hole_pose = get_model_pose(self.release_hole_name)
            baseline = self.release_baseline
            if peg_pose is None or hole_pose is None or baseline is None:
                response.success = False
                response.message = "missing peg/hole/baseline pose"
                return response

            hole_dist = float(np.hypot(
                peg_pose[0] - hole_pose[0],
                peg_pose[1] - hole_pose[1],
            ))
            baseline_xy_shift = float(np.hypot(
                peg_pose[0] - baseline[0],
                peg_pose[1] - baseline[1],
            ))
            upward_follow = float(peg_pose[2] - baseline[2])
            response.success = (
                hole_dist <= 0.0003
                and baseline_xy_shift <= 0.0003
                # The complete assembly pose must be achieved by the
                # closed-gripper policy action. Opening may cause only a
                # sub-millimetre solver relaxation, never gravity completion.
                and abs(upward_follow) <= 0.00035
                and abs(peg_pose[2] - 0.885) <= 0.00035
            )
            response.message = (
                f"hole_dist={hole_dist:.4f}, "
                f"xy_shift={baseline_xy_shift:.4f}, "
                f"z_rise={upward_follow:.4f}, z={peg_pose[2]:.4f}"
            )
            return response

        def _js(self, msg):
            try:
                pos = [msg.position[msg.name.index(n)] for n in ALL_JOINTS]
                gripper_pos = [
                    msg.position[msg.name.index(name)]
                    for name in GRIPPER_KINEMATIC_JOINTS
                ]
                timestamp = _message_timestamp(
                    msg, lambda: self.get_clock().now().nanoseconds * 1e-9
                )
                with buf.lock:
                    buf.joint_positions = pos
                    buf.gripper_joint_positions = gripper_pos
                    if (
                        buf.last_state_timestamp is not None
                        and timestamp < buf.last_state_timestamp
                    ):
                        buf.state_history.clear()
                        buf.last_state_timestamp = None
                    if _should_resample(timestamp, buf.last_state_timestamp):
                        buf.state_history.append(
                            (timestamp, np.asarray(pos, dtype=np.float32))
                        )
                        buf.last_state_timestamp = timestamp
            except (ValueError, IndexError):
                pass
                
            try:
                # Lookup exact transform of tool0 relative to world frame
                trans = self.tf_buffer.lookup_transform("world", "tool0", rclpy.time.Time())
                tz = trans.transform.translation.z
                with buf.lock:
                    buf.tool0_z = float(tz)
            except Exception:
                pass

        def _controller_state(self, msg):
            try:
                desired = [
                    msg.desired.positions[msg.joint_names.index(name)]
                    for name in ALL_JOINTS
                ]
                if not np.isfinite(desired).all():
                    return
                with buf.lock:
                    buf.joint_position_commands = desired
            except (ValueError, IndexError):
                pass

        def _wrench(self, msg):
            w = msg.wrench
            timestamp = _message_timestamp(
                msg, lambda: self.get_clock().now().nanoseconds * 1e-9
            )
            wall_timestamp = time.time()
            value = np.array(
                [
                    w.force.x,
                    w.force.y,
                    w.force.z,
                    w.torque.x,
                    w.torque.y,
                    w.torque.z,
                ],
                dtype=np.float32,
            )
            with buf.lock:
                if (
                    buf.last_force_timestamp is not None
                    and timestamp < buf.last_force_timestamp
                ):
                    buf.force_native.clear()
                    buf.force_slow.clear()
                    buf.force_filter.clear()
                    buf.last_force_slow_timestamp = None
                # Median filtering suppresses an isolated DART constraint
                # impulse without hiding a sustained contact load.  The old
                # implementation refused to append every sample above 30 N;
                # once a rigid contact stayed above that value, the filter was
                # permanently frozen at its pre-contact baseline and PAP-MoE's
                # force windows/routing labels falsely looked contact-free.
                # Always advance the 11-sample window: a short impulse remains
                # below the median, while a real load lasting > half a window
                # must become visible.  The untouched 100 Hz stream and its
                # duration gate remain the independent overload audit.
                buf.force_filter.append(value.copy())
                filtered_value = np.median(
                    np.stack(buf.force_filter), axis=0
                ).astype(np.float32)
                buf.wrench = filtered_value
                buf.force_sample_count += 1
                reference_payload = bool(buf.force_reference_payload)
                # Keep the full native contact event for calibration and
                # subsequent direction-preserving clipping. Frame-level and
                # slow streams remain median-filtered.
                # Preserve the untouched fast sample here. Directional
                # clipping must happen only after the per-sample empty/payload
                # reference is subtracted; clipping in raw sensor coordinates
                # can increase the calibrated norm.
                fast_value = value.copy().astype(np.float32)
                buf.force_native.append(
                    (timestamp, fast_value.copy(), reference_payload)
                )
                buf.last_force_timestamp = timestamp
                if _should_resample(timestamp, buf.last_force_slow_timestamp):
                    buf.force_slow.append(
                        (timestamp, filtered_value.copy(), reference_payload)
                    )
                    buf.last_force_slow_timestamp = timestamp
                if buf.capture_force_stream:
                    buf.raw_force_timestamps.append(timestamp)
                    buf.raw_force_wall_timestamps.append(wall_timestamp)
                    buf.raw_force_values.append(value.copy())

        def _contacts(self, msg, topic):
            """Cache Gazebo collision truth without adding it to observations."""
            now = self.get_clock().now().nanoseconds * 1e-9
            names = []
            max_depth = 0.0
            for contact in msg.contacts:
                names.extend(
                    [contact.collision1.name.lower(), contact.collision2.name.lower()]
                )
                if contact.depths:
                    max_depth = max(max_depth, max(contact.depths))
            joined = " ".join(names)
            state = {
                "timestamp": now,
                "any": bool(msg.contacts),
                "gripper": any(
                    token in joined
                    for token in ("robotiq", "finger", "knuckle")
                ),
                "hole": any(
                    token in joined
                    for token in ("hole_plate", "socket_col", "rigid_floor_col")
                ),
                "table": any(
                    token in joined for token in ("cafe_table", "table", "ground_plane")
                ),
                "max_depth": float(max_depth),
            }
            with buf.lock:
                buf.contact_truth_by_topic[topic] = state

        def _decode(self, msg):
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                  IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
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

    print("Waiting for joint states and FT sensor...")
    for _ in range(50):
        with buf.lock:
            if (
                buf.joint_positions is not None
                and buf.gripper_joint_positions is not None
                and buf.joint_position_commands is not None
                and buf.force_sample_count > 0
            ):
                print("Joint states, controller commands, and FT samples available")
                break
        time.sleep(0.1)
    else:
        print("ERROR: no joint states received")
        executor.shutdown()
        recorder.destroy_node()
        rclpy.shutdown()
        return

    # Do not begin an episode with padded multi-rate histories.  At a 500 Hz
    # physics step the headless renderer may run below real time, so warm the
    # windows by sample count rather than a fixed wall-clock sleep.
    print("Warming 64-sample force, 50-sample slow-force, and state windows...")
    for _ in range(300):
        with buf.lock:
            histories_ready = (
                len(buf.force_native) >= FAST_FORCE_SAMPLES
                and len(buf.force_slow) >= SLOW_FORCE_SAMPLES
                and len(buf.state_history) >= STATE_HISTORY_SAMPLES
                and bool(np.any(buf.wrist_img))
                and bool(np.any(buf.global_img))
            )
        if histories_ready:
            print("Multi-rate histories are warm")
            break
        time.sleep(0.1)
    else:
        print("ERROR: multi-rate observation windows did not become ready")
        executor.shutdown()
        recorder.destroy_node()
        rclpy.shutdown()
        return

    recording = threading.Event()
    recording.set()
    frames = []
    episode_active = threading.Event()

    def recorder_thread():
        rate = recorder.create_rate(record_hz)
        local_frame_cam_degraded = False
        ep_force_bias = None
        prev_js = None
        while recording.is_set():
            with buf.lock:
                js = (list(buf.joint_positions) if buf.joint_positions
                      else [0.0] * STATE_DIM)
                gripper_js = (
                    list(buf.gripper_joint_positions)
                    if buf.gripper_joint_positions is not None
                    else [0.0] * len(GRIPPER_KINEMATIC_JOINTS)
                )
                command = (
                    list(buf.joint_position_commands)
                    if buf.joint_position_commands is not None
                    else list(js)
                )
                w = buf.wrist_img.copy()
                g = buf.global_img.copy()
                ft = buf.wrench.copy()
                stage = buf.current_stage
                tool0_z = buf.tool0_z
                cam_deg = buf.cam_degradation
                cam_deg_type = buf.cam_degradation_type
                cam_gain = buf.cam_glare_gain
                search_mode = buf.search_mode
                recovery_active = buf.recovery_active
                recovery_attempt = buf.recovery_attempt
                recovery_direction = buf.recovery_direction.copy()
                recovery_result = buf.recovery_result
                recovery_event_id = buf.recovery_event_id
                recovery_safety = buf.recovery_safety
                semantic_subtask = buf.semantic_subtask
                controller_force_bias = buf.force_bias.copy()
                controller_force_bias_valid = bool(buf.force_bias_valid)
                frame_force_reference_payload = bool(
                    buf.force_reference_payload
                )
                fast_pairs = list(buf.force_native)[-FAST_FORCE_SAMPLES:]
                slow_pairs = list(buf.force_slow)
                state_pairs = list(buf.state_history)
                contact_now = recorder.get_clock().now().nanoseconds * 1e-9
                fresh_contacts = [
                    state
                    for state in buf.contact_truth_by_topic.values()
                    if contact_now - state["timestamp"] <= CONTACT_TRUTH_TIMEOUT_S
                ]
                contact_truth = {
                    # Gazebo contact sensors are event-driven: they publish
                    # while colliding and are silent in free space.  The four
                    # publishers are a launch/schema contract, so silence is
                    # a valid negative after the short persistence timeout.
                    "valid": True,
                    "any": any(state["any"] for state in fresh_contacts),
                    "gripper": any(state["gripper"] for state in fresh_contacts),
                    "hole": any(state["hole"] for state in fresh_contacts),
                    "table": any(state["table"] for state in fresh_contacts),
                    "max_depth": max(
                        (state["max_depth"] for state in fresh_contacts),
                        default=0.0,
                    ),
                }

            if episode_active.is_set():
                # Preserve the same-time clean views before applying a
                # synthetic blackout/glare.  These images are training-only
                # teacher targets for the E2 visual-memory objective; they
                # are never exposed as policy observations.  Keeping both
                # views in the raw episode also makes the degradation recipe
                # auditable and exactly reproducible.
                w_clean_teacher = w.copy()
                g_clean_teacher = g.copy()
                if ep_force_bias is None:
                    ep_force_bias = ft.copy()
                # Before approach, reference wrench inputs to the empty-tool
                # baseline captured at RECORD_START. From approach onward,
                # use the controller's payload-aware bias measured after the
                # peg is attached. The untouched native stream is retained as
                # ``raw_force`` for auditing and sensor-rate validation.
                active_force_bias = (
                    controller_force_bias
                    if stage >= 1 and controller_force_bias_valid
                    else ep_force_bias
                )
                ft_calibrated = ft - active_force_bias
                ft_for_label = ft_calibrated

                # E2: Block-based visual degradation (1-3s continuous blocks, 2-5s gaps).
                # Replaces per-frame flicker — realistic occlusion/glare lasts seconds, not 0.1s.
                if stage >= 1 and cam_deg and buf.degrade_schedule is not None:
                    idx = buf.frame_idx
                    buf.frame_idx += 1
                    local_frame_cam_degraded = (idx < len(buf.degrade_schedule) and buf.degrade_schedule[idx])
                else:
                    local_frame_cam_degraded = False

                # Apply visual mask if this specific frame is degraded
                if local_frame_cam_degraded:
                    if cam_deg_type == "dropout":
                        # E2's q=0 target means the policy has no reliable
                        # camera.  Degrade both policy views; labeling only a
                        # black wrist view as total visual failure while the
                        # global view stays clean would be contradictory.
                        w = np.zeros_like(w)
                        g = np.zeros_like(g)
                    elif cam_deg_type == "glare":
                        # Synchronous exposure failure on both policy cameras.
                        w = np.clip(w * cam_gain, 0.0, 1.0)
                        g = np.clip(g * cam_gain, 0.0, 1.0)

                # Compute joint velocity online using backward difference
                if prev_js is not None:
                    q_dot = (np.array(js[:6]) - np.array(prev_js[:6])) * record_hz
                    v_norm = float(np.linalg.norm(q_dot))
                else:
                    v_norm = 0.0
                prev_js = js.copy()

                force_fast = _left_pad_window(
                    [value for _, value, _ in fast_pairs],
                    FAST_FORCE_SAMPLES,
                    FORCE_DIM,
                )
                force_slow = _left_pad_window(
                    [value for _, value, _ in slow_pairs],
                    SLOW_FORCE_SAMPLES,
                    FORCE_DIM,
                )
                force_fast_payload = _left_pad_flags(
                    [payload for _, _, payload in fast_pairs],
                    FAST_FORCE_SAMPLES,
                )
                force_slow_payload = _left_pad_flags(
                    [payload for _, _, payload in slow_pairs],
                    SLOW_FORCE_SAMPLES,
                )
                force_fast_calibrated = _clip_calibrated_fast_force(
                    force_fast - active_force_bias[None, :]
                )
                force_slow_calibrated = (
                    force_slow - active_force_bias[None, :]
                ).astype(np.float32)
                state_history = _left_pad_window(
                    [value for _, value in state_pairs],
                    STATE_HISTORY_SAMPLES,
                    STATE_DIM,
                )

                # Compute the ground-truth 4D soft physical state vector using frame-level state
                stage_vector = compute_physical_state_vector(
                    ft_for_label,
                    force_fast_calibrated,
                    tool0_z,
                    local_frame_cam_degraded,
                    cam_degradation_type=cam_deg_type, cam_glare_gain=cam_gain,
                    gripper_joint_val=float(js[6]),
                    joint_vel_norm=v_norm,
                    current_stage=stage,
                    semantic_subtask=semantic_subtask)

                visual_quality = _visual_quality(w, g)
                timestamp_ros = recorder.get_clock().now().nanoseconds * 1e-9

                frames.append({
                    "state": np.array(js, dtype=np.float32),
                    "gripper_kinematic_state": np.array(
                        gripper_js, dtype=np.float32
                    ),
                    "action_command": np.array(command, dtype=np.float32),
                    "force_raw": ft.astype(np.float32),
                    "force_fast_raw": force_fast.astype(np.float32),
                    "force_slow_raw": force_slow.astype(np.float32),
                    "force_reference_payload": bool(
                        frame_force_reference_payload
                    ),
                    "force_fast_reference_payload": force_fast_payload,
                    "force_slow_reference_payload": force_slow_payload,
                    "empty_force_bias": ep_force_bias.astype(np.float32),
                    "force": ft_calibrated.astype(np.float32),
                    "force_fast": force_fast_calibrated,
                    "force_slow": force_slow_calibrated,
                    "state_history": state_history,
                    "visual_quality": visual_quality,
                    "force_fast_valid": len(fast_pairs),
                    "force_slow_valid": len(slow_pairs),
                    "state_history_valid": len(state_pairs),
                    "stage": stage_vector,
                    "tool0_z": float(tool0_z) if tool0_z is not None else 0.0,
                    "camera0": w,
                    "camera1": g,
                    "camera0_clean_teacher": w_clean_teacher,
                    "camera1_clean_teacher": g_clean_teacher,
                    "timestamp": timestamp_ros,
                    "timestamp_ros": timestamp_ros,
                    # Keep the language instruction identical for baseline and
                    # PAP-MoE. Per-frame semantics are auxiliary supervision,
                    # never a privileged prompt exposed to either policy.
                    "task": TASK,
                    "semantic_subtask": semantic_subtask,
                    "search_mode": search_mode,
                    "recovery_active": recovery_active,
                    "recovery_attempt": recovery_attempt,
                    "recovery_direction": recovery_direction,
                    "recovery_result": recovery_result,
                    "recovery_event_id": recovery_event_id,
                    "recovery_safety": recovery_safety,
                    "current_stage": stage,
                    "joint_vel_norm": v_norm,
                    "cam_degraded_frame": local_frame_cam_degraded,
                    "cam_degradation_type": cam_deg_type,
                    "cam_glare_gain": cam_gain,
                    "contact_truth_valid": contact_truth["valid"],
                    "contact_truth_any": contact_truth["any"],
                    "contact_truth_task": (
                        contact_truth["gripper"] or contact_truth["hole"]
                    ),
                    "contact_truth_gripper": contact_truth["gripper"],
                    "contact_truth_hole": contact_truth["hole"],
                    "contact_truth_table": contact_truth["table"],
                    "contact_truth_max_depth": contact_truth["max_depth"],
                })
            else:
                ep_force_bias = None
            rate.sleep()

    rec_thread = threading.Thread(target=recorder_thread, daemon=True)
    rec_thread.start()
    print(f"Recording started ({record_hz} Hz) — PAP-MoE multi-expert scheme")



    total_frames = 0
    if args.search_mode == "mixed":
        direct_count = round(args.episodes * 0.25)
        gradient_count = round(args.episodes * 0.30)
        admittance_count = round(args.episodes * 0.35)
        spiral_count = (
            args.episodes - direct_count - gradient_count - admittance_count
        )
        search_mode_schedule = (
            ["direct"] * direct_count
            + ["force_gradient"] * gradient_count
            + ["admittance"] * admittance_count
            + ["spiral"] * spiral_count
        )
        random.shuffle(search_mode_schedule)
        print("Mixed search schedule: "
              f"direct={direct_count}, force_gradient={gradient_count}, "
              f"admittance={admittance_count}, spiral={spiral_count}")
    else:
        search_mode_schedule = [args.search_mode] * args.episodes
        print(f"Forced search mode: {args.search_mode} "
              f"({args.episodes} episodes)")

    cached_scene_position = None
    try:
        for ep in range(args.start_episode, args.start_episode + args.episodes):
            episode_offset = ep - args.start_episode
            episode_search_mode = search_mode_schedule[episode_offset]
            position_group_id = episode_offset // args.position_repeat_count
            position_repeat_index = episode_offset % args.position_repeat_count
            trajectory_style_id = (
                args.trajectory_style
                if args.trajectory_style >= 0
                else (
                    args.trajectory_style_offset + position_repeat_index
                ) % len(TRAJECTORY_STYLES)
            )
            trajectory_style = TRAJECTORY_STYLES[trajectory_style_id]
            final_episode = args.start_episode + args.episodes - 1
            print(f"\nEpisode {ep:04d}/{final_episode:04d}")
            print(f"  Search mode: {episode_search_mode}")
            print(
                f"  Trajectory style: {trajectory_style_id} "
                f"({trajectory_style['name']}), position group "
                f"{position_group_id}, repeat {position_repeat_index}"
            )
            
            # Safety check: avoid overwriting existing data
            task_prefix = TASK.replace(" ", "_")
            success_name = f"{task_prefix}_episode_{ep:04d}_success"
            failed_name  = f"{task_prefix}_episode_{ep:04d}_failed"
            success_dir = os.path.join(args.output, success_name)
            failed_dir = os.path.join(args.output, failed_name)
            
            if os.path.exists(success_dir) or os.path.exists(failed_dir):
                print(f"  ❌ Error: Episode {ep:04d} already exists at {success_dir} or {failed_dir}!")
                print("  Aborting to prevent accidental data overwrite. Please increase --start_episode.")
                sys.exit(1)
            
            # Reset attachment flag for new episode
            recorder.attached = False
            recorder.attached_confirmed = False
            
            # Randomize camera degradation settings (E2)
            with buf.lock:
                buf.frame_idx = 0
                buf.search_mode = episode_search_mode
                buf.recovery_active = False
                buf.recovery_attempt = 0
                buf.recovery_direction[:] = 0.0
                buf.recovery_result = -1
                buf.recovery_event_id = 0
                buf.recovery_safety = False
                buf.semantic_subtask = SUBTASKS[0]
                buf.force_bias_valid = False
                buf.force_reference_payload = False
                buf.capture_force_stream = False
                buf.raw_force_timestamps.clear()
                buf.raw_force_wall_timestamps.clear()
                buf.raw_force_values.clear()
                buf.force_filter.clear()
                for contact_topic in CONTACT_TRUTH_TOPICS:
                    buf.contact_truth_by_topic[contact_topic].update(
                        timestamp=float("-inf"),
                        any=False,
                        gripper=False,
                        hole=False,
                        table=False,
                        max_depth=0.0,
                    )
                degradation_requested = (
                    args.camera_degradation_mode
                    in {"dropout", "glare", "balanced"}
                    or (
                        args.camera_degradation_mode == "random"
                        and random.random() < 0.35
                    )
                )
                if degradation_requested:
                    buf.cam_degradation = True
                    if args.camera_degradation_mode == "random":
                        buf.cam_degradation_type = random.choice(["dropout", "glare"])
                    elif args.camera_degradation_mode == "balanced":
                        buf.cam_degradation_type = (
                            "dropout"
                            if position_repeat_index
                            < args.position_repeat_count / 2
                            else "glare"
                        )
                    else:
                        buf.cam_degradation_type = args.camera_degradation_mode
                    if buf.cam_degradation_type == "glare":
                        buf.cam_glare_gain = round(random.uniform(1.5, 6.0), 1)
                    else:
                        buf.cam_glare_gain = 1.5
                    # Pre-compute block degradation schedule (~3000 frames max)
                    buf.degrade_schedule = np.zeros(3000, dtype=bool)
                    pos = 20  # start after ~2s of approach
                    while pos < 3000:
                        block_len = random.randint(10, 30)  # 1-3s continuous block
                        buf.degrade_schedule[pos:pos+block_len] = True
                        gap = random.randint(20, 50)  # 2-5s gap between blocks
                        pos += block_len + gap
                else:
                    buf.cam_degradation = False
                    buf.cam_degradation_type = "normal"
                    buf.cam_glare_gain = 1.5
                    buf.degrade_schedule = None
            print(f"  Camera degradation active: {buf.cam_degradation} ({buf.cam_degradation_type}"
                  + (f", gain={buf.cam_glare_gain:.1f}" if buf.cam_degradation_type == "glare" else "") + ")")

            if all(value is not None for value in fixed_scene_values):
                peg_x, peg_y, hole_x, hole_y = fixed_scene_values
                p_r2 = peg_x**2 + peg_y**2
                h_r2 = hole_x**2 + hole_y**2
                peg_hole_dist = np.hypot(peg_x - hole_x, peg_y - hole_y)
                if not (
                    MIN_XY_SQ <= p_r2 <= MAX_XY_SQ
                    and MIN_XY_SQ <= h_r2 <= MAX_XY_SQ
                    and peg_hole_dist >= MIN_PEG_HOLE_DIST
                ):
                    parser.error(
                        "fixed peg/hole scene violates the current workspace "
                        "or separation contract"
                    )
                cached_scene_position = (peg_x, peg_y, hole_x, hole_y)
                print(
                    "  Fixed replay scene: "
                    f"peg=({peg_x:.6f}, {peg_y:.6f}), "
                    f"hole=({hole_x:.6f}, {hole_y:.6f})"
                )
            elif position_repeat_index == 0 or cached_scene_position is None:
                for _ in range(200):
                    peg_x  = round(random.uniform(*BLOCK_X_RANGE), 3)
                    peg_y  = round(random.uniform(*BLOCK_Y_RANGE), 3)
                    hole_x = round(random.uniform(*HOLE_X_RANGE), 3)
                    hole_y = round(random.uniform(*HOLE_Y_RANGE), 3)

                    # Origin-centred annulus: MIN_XY_SQ ≤ x²+y² ≤ MAX_XY_SQ
                    p_r2 = peg_x**2 + peg_y**2
                    h_r2 = hole_x**2 + hole_y**2
                    p_ok = MIN_XY_SQ <= p_r2 <= MAX_XY_SQ
                    h_ok = MIN_XY_SQ <= h_r2 <= MAX_XY_SQ

                    peg_hole_dist = ((peg_x - hole_x)**2 + (peg_y - hole_y)**2)**0.5
                    if p_ok and h_ok and peg_hole_dist >= MIN_PEG_HOLE_DIST:
                        cached_scene_position = (peg_x, peg_y, hole_x, hole_y)
                        break
                else:
                    print("  ⚠ Could not find valid positions after 200 attempts, skipping episode")
                    continue
            else:
                peg_x, peg_y, hole_x, hole_y = cached_scene_position

            grasp_recovery_offset = np.zeros(2, dtype=np.float64)
            if args.grasp_recovery_offset_max > 0.0:
                for _ in range(200):
                    radius = random.uniform(
                        args.grasp_recovery_offset_min,
                        args.grasp_recovery_offset_max,
                    )
                    angle = random.uniform(-np.pi, np.pi)
                    candidate = radius * np.array(
                        [np.cos(angle), np.sin(angle)], dtype=np.float64
                    )
                    rollout_xy = np.array([peg_x, peg_y]) + candidate
                    rollout_r2 = float(np.dot(rollout_xy, rollout_xy))
                    if MIN_XY_SQ <= rollout_r2 <= MAX_XY_SQ:
                        grasp_recovery_offset = candidate
                        break
                else:
                    print("  ⚠ Could not sample reachable grasp recovery roll-in, skipping")
                    continue
            print(
                "  Grasp recovery roll-in XY: "
                f"({grasp_recovery_offset[0]:+.3f}, "
                f"{grasp_recovery_offset[1]:+.3f}) m"
            )

            peg_name  = "peg"
            hole_name = f"hole_plate_{ep:04d}"
            recorder.release_hole_name = hole_name
            recorder.release_baseline = None
            # Solid printed lab peg.  PLA density (1.24 g/cm^3) times the
            # measured 109 cm^3 composite volume gives 0.135 kg.  Keep the
            # dynamics fixed until a scale measurement supplies a replacement.
            peg_mass = PEG_MASS
            ixx_peg = PEG_IXX
            izz_peg = PEG_IZZ
            
            # Fixed ground-validation contact domain. Parameter sweeps remain
            # available through --contact-kp but do not contaminate the first
            # training baseline.
            kp_val = (
                args.contact_kp
                if args.contact_kp is not None
                else 5000
            )
            kd_val = 40
            # The nominally zero-clearance taper already supplies geometric
            # guidance.  A 0.35 mesh friction coefficient made DART's exact
            # coincident side contacts self-lock and left 40--55 N of lateral
            # constraint load after seating.  Use a low-friction printed/
            # finished guide surface; geometry, contact stiffness and the
            # closed-gripper/no-gravity success gates remain unchanged.
            mu_val = 0.35
            
            peg_sdf_str = PEG_SDF.format(name=peg_name, mass=peg_mass, ixx=ixx_peg, iyy=ixx_peg, izz=izz_peg, kp=kp_val, kd=kd_val)
            print(f"  Peg Mass: {peg_mass:.2f}kg")

            # Ground-validation baseline: rigidly mounted socket. E4 labels
            # come from the measured force/displacement response, not from a
            # privileged or moving fixture.
            spring_k_val = 0.0
            spring_d_val = 0.0

            hole_sdf_str = make_hole_sdf(
                hole_name, kp=kp_val, kd=kd_val,
                mu=mu_val, mu2=mu_val)
            print(
                f"  Hole Contact: kp={kp_val}, mu={mu_val} | "
                "Fixture: rigid 60x100 mm tapered lab socket")

            # Gazebo entity removal is asynchronous.  A failed episode leaves
            # the robot in failure-hold and used to let the next episode race
            # a stale entity named ``peg``.  Never spawn a new formal scene
            # until the previous entity is observably absent.
            if get_model_pose(peg_name) is not None:
                delete_model(peg_name)
                if not wait_model_absent(peg_name):
                    raise RuntimeError(
                        "Stale peg entity remained after reset; aborting batch"
                    )

            # Spawn randomized models with configurable 3D pose tilt.
            peg_roll = round(
                random.uniform(-args.max_peg_tilt, args.max_peg_tilt), 4
            )
            peg_pitch = round(
                random.uniform(-args.max_peg_tilt, args.max_peg_tilt), 4
            )
            print(f"  Peg 3D Tilt: roll={peg_roll:.3f} rad, pitch={peg_pitch:.3f} rad")
            ok_peg  = spawn_model(peg_name, peg_x, peg_y, PEG_Z, roll=peg_roll, pitch=peg_pitch, sdf_string=peg_sdf_str)
            ok_hole = spawn_model(hole_name, hole_x, hole_y, HOLE_Z, sdf_string=hole_sdf_str)
            
            if not ok_peg or not ok_hole:
                print("  ⚠ spawn failed, skipping")
                delete_model(peg_name)
                delete_model(hole_name)
                continue

            # Verify spawned models are actually present in the world
            time.sleep(0.5)
            peg_check = get_model_pose(peg_name)
            hole_check = get_model_pose(hole_name)
            if peg_check is None:
                print(f"  ✗ Peg model '{peg_name}' not found in world after spawn! Retrying...")
                delete_model(peg_name)
                delete_model(hole_name)
                time.sleep(1.0)
                continue
            spawn_xy_error = float(np.hypot(
                peg_check[0] - peg_x,
                peg_check[1] - peg_y,
            ))
            if spawn_xy_error > 0.002:
                print(
                    "  ✗ Spawned peg pose does not match the requested scene "
                    f"(xy error={spawn_xy_error:.4f} m); aborting this batch."
                )
                delete_model(peg_name)
                delete_model(hole_name)
                wait_model_absent(peg_name)
                break
            if hole_check is None:
                print(f"  ✗ Hole model '{hole_name}' not found in world after spawn! Retrying...")
                delete_model(peg_name)
                delete_model(hole_name)
                time.sleep(1.0)
                continue

            # Longer stabilization: let Gazebo fully register models and settle physics
            time.sleep(2.0)

            recorder.grasp_initial_pose = get_model_pose(peg_name)
            if recorder.grasp_initial_pose is None:
                print("  ✗ Cannot establish initial peg pose for grasp verification")
                delete_model(peg_name)
                delete_model(hole_name)
                continue
            print("  --> ✓ Physical grasp mode; initial peg pose captured.")

            # Keep contact motion conservative but make free-space transport
            # continuous and representative of the real arm.
            vel_scale = 0.05
            # The long solid peg is held only by fingertip friction.  A 0.25
            # scale produced about 10 mm of transport slip; 0.10 preserves a
            # physical grasp without reintroducing a detachable joint.
            # Validate the physical grasp at the normal collection speed.
            # Stability must come from the centred rubber-pad contact, not a
            # task-specific close angle, rigid attach, or artificially slow
            # transport.
            trans_vel_scale = float(trajectory_style["speed"])
            print(f"  Controller speeds: search_vel={vel_scale:.3f}, transport_vel={trans_vel_scale:.2f}")

            print("  Running C++ pap_moe_peg_in_hole...")
            episode_active.clear()
            frames.clear()
            _record_started = False  # Reset per-episode guard for C++ ATTACH PEG
            controller_result = "unknown"
            # A successful task has already passed release stability before
            # TASK_RESULT is emitted.  Cache that verified terminal scene and
            # remove its contact bodies while the controller performs the
            # unrecorded reset.  Otherwise the zero-clearance seated peg keeps
            # DART's mesh contact solver busy throughout retract and HOME.
            terminal_scene_pose_cache = None
            reset_label_buffers()    # Clear rolling torque/Fz buffers for new episode

            # Run C++ active alignment search controller with workspace sourcing
            ros2_cmd = (
                "source /home/ubuntu/ur3_ft300_ws/install/setup.bash && "
                "ros2 run ur_simulation_gz pap_moe_peg_in_hole "
                f"--ros-args -p peg_x:={peg_x} -p peg_y:={peg_y} "
                f"-p peg_roll:={peg_roll} -p peg_pitch:={peg_pitch} "
                f"-p hole_x:={hole_x} -p hole_y:={hole_y} "
                f"-p velocity_scaling:={vel_scale} "
                f"-p transport_velocity_scaling:={trans_vel_scale} "
                f"-p fine_insertion_step:={args.fine_insertion_step} "
                f"-p search_mode:={episode_search_mode} "
                f"-p recovery_offset_max:={args.recovery_offset_max} "
                f"-p grasp_recovery_offset_x:={grasp_recovery_offset[0]} "
                f"-p grasp_recovery_offset_y:={grasp_recovery_offset[1]} "
                f"-p grasp_recovery_only:={'true' if args.grasp_recovery_only else 'false'} "
                f"-p trajectory_style_id:={trajectory_style_id} "
                f"-p peg_arc_x:={trajectory_style['peg'][0]} "
                f"-p peg_arc_y:={trajectory_style['peg'][1]} "
                f"-p peg_arc_z:={trajectory_style['peg'][2]} "
                f"-p transport_arc_x:={trajectory_style['transport'][0]} "
                f"-p transport_arc_y:={trajectory_style['transport'][1]} "
                f"-p transport_arc_z:={trajectory_style['transport'][2]} "
                f"-p hole_arc_x:={trajectory_style['hole'][0]} "
                f"-p hole_arc_y:={trajectory_style['hole'][1]} "
                "-p skip_home:=false -p use_sim_time:=true"
            )
            proc = subprocess.Popen(
                ["/bin/bash", "-c", ros2_cmd],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True)

            for line in proc.stdout:
                line_str = line.strip()
                print(f"    [C++] {line_str}")
                if "RECORD_START" in line_str:
                    episode_active.set()
                    frames.clear()
                    _record_started = True  # Guard: only allow attach after task has begun
                    with buf.lock:
                        buf.semantic_subtask = SUBTASKS[0]
                        buf.raw_force_timestamps.clear()
                        buf.raw_force_wall_timestamps.clear()
                        buf.raw_force_values.clear()
                        buf.capture_force_stream = True
                elif "SUBTASK:" in line_str:
                    label = line_str.split("SUBTASK:", 1)[1].strip()
                    if label not in SUBTASKS:
                        raise RuntimeError(f"Unknown controller subtask label: {label}")
                    with buf.lock:
                        buf.semantic_subtask = label
                elif "=== 8. APPROACH" in line_str:
                    with buf.lock:
                        buf.semantic_subtask = SUBTASKS[2]
                elif "=== 10. INSERT" in line_str:
                    with buf.lock:
                        buf.semantic_subtask = SUBTASKS[4]
                elif "=== 11. SEAT" in line_str:
                    with buf.lock:
                        buf.semantic_subtask = SUBTASKS[5]
                elif "=== 12. Open gripper" in line_str:
                    with buf.lock:
                        buf.semantic_subtask = SUBTASKS[6]
                elif "=== 13. RETRACT" in line_str or "=== 14. Return HOME" in line_str:
                    with buf.lock:
                        buf.semantic_subtask = SUBTASKS[7]
                elif "SEARCH_MODE:" in line_str:
                    mode = line_str.split("SEARCH_MODE:", 1)[1].strip().split()[0]
                    with buf.lock:
                        buf.search_mode = mode
                elif "RECOVERY_EVENT:" in line_str:
                    with buf.lock:
                        buf.semantic_subtask = SUBTASKS[3]
                        buf.recovery_active = True
                        buf.recovery_attempt = 0
                        buf.recovery_direction[:] = 0.0
                        buf.recovery_result = -1
                        buf.recovery_event_id += 1
                elif "RECOVERY_TRIAL:" in line_str:
                    try:
                        vals = line_str.split("RECOVERY_TRIAL:", 1)[1].strip().split(",")
                        with buf.lock:
                            buf.recovery_active = True
                            buf.recovery_attempt = int(vals[0])
                            buf.recovery_direction = np.array(
                                [float(vals[1]), float(vals[2])],
                                dtype=np.float32)
                    except (ValueError, IndexError) as exc:
                        print(f"  Error parsing recovery trial: {exc}")
                elif "RECOVERY_SAFETY:" in line_str:
                    with buf.lock:
                        buf.recovery_safety = True
                elif "RECOVERY_RESULT:" in line_str:
                    try:
                        vals = line_str.split("RECOVERY_RESULT:", 1)[1].strip().split(",")
                        with buf.lock:
                            buf.recovery_active = False
                            buf.recovery_result = 1 if vals[0] == "success" else 0
                            buf.recovery_attempt = int(vals[1])
                            buf.semantic_subtask = SUBTASKS[2]
                    except (ValueError, IndexError) as exc:
                        print(f"  Error parsing recovery result: {exc}")
                elif "TASK_RESULT:" in line_str:
                    result = line_str.split("TASK_RESULT:", 1)[1].strip().split()[0]
                    controller_result = result
                    print(f"  --> Controller task result: {controller_result}")
                    # The policy episode ends when the verified task result is
                    # known.  Retraction and return-home still run below as a
                    # safe environment reset, but are not demonstrations of
                    # the global insertion instruction.  This matches common
                    # benchmark collection, where success terminates the
                    # episode instead of teaching a post-success reversal.
                    episode_active.clear()
                    with buf.lock:
                        buf.capture_force_stream = False
                    if controller_result == "success":
                        terminal_peg_pose = get_model_pose(peg_name)
                        terminal_hole_pose = get_model_pose(hole_name)
                        if (
                            terminal_peg_pose is not None
                            and terminal_hole_pose is not None
                        ):
                            terminal_scene_pose_cache = (
                                terminal_peg_pose,
                                terminal_hole_pose,
                            )
                            # This is reset-only acceleration: all policy
                            # samples and physical success checks have ended.
                            peg_deleted_early = delete_model(peg_name)
                            hole_deleted_early = delete_model(hole_name)
                            if peg_deleted_early and hole_deleted_early:
                                print(
                                    "  --> Verified terminal scene cached; "
                                    "contact bodies removed before reset."
                                )
                elif "STAGE:" in line_str:
                    try:
                        stage_id = int(line_str.split("STAGE:")[1].split("]")[0].strip()[:1])
                        with buf.lock:
                            buf.current_stage = stage_id
                    except (ValueError, IndexError):
                        pass
                elif "C++ PHYSICAL GRASP PEG" in line_str:
                    # Guard: ignore any attach signal before RECORD_START
                    # (prevents spurious attach during simulation/C++ initialization on first episode)
                    if not _record_started:
                        print("  --> ⚠ Ignoring C++ ATTACH PEG before RECORD_START (initialization noise)")
                        continue
                    if not recorder.attached:
                        print(
                            "  --> C++ signal: closing fingers for physical grasp; "
                            "lift will be pose-verified."
                        )
                        pre_attach_pose = get_model_pose(peg_name)
                        pre_attach_drift = (
                            float(np.hypot(
                                pre_attach_pose[0] - peg_x,
                                pre_attach_pose[1] - peg_y,
                            ))
                            if pre_attach_pose is not None
                            else float("inf")
                        )
                        if pre_attach_drift > 0.003:
                            print(
                                "  --> ✗ Peg moved before attachment "
                                f"(xy drift={pre_attach_drift:.4f} m)."
                            )
                            continue
                    else:
                        print("  --> C++ ATTACH PEG signal received but peg already attached (skipping)")
                elif "C++ DETACH PEG" in line_str:
                    print(
                        "  --> Release requested; synchronous detach service "
                        "will publish and capture the peg baseline."
                    )
                    with buf.lock:
                        buf.semantic_subtask = SUBTASKS[6]
                elif "GRIPPER_OPEN_VERIFY:success" in line_str:
                    # The physical payload leaves the fingertips only after
                    # opening is measured. Samples before and after this
                    # boundary retain their own reference phase.
                    with buf.lock:
                        buf.force_reference_payload = False
                elif "Force bias vector:" in line_str:
                    try:
                        vals = line_str.split("Force bias vector:")[1].strip().split(",")
                        bias_arr = np.array([float(v) for v in vals], dtype=np.float32)
                        with buf.lock:
                            buf.force_bias = bias_arr
                            buf.force_bias_valid = True
                        print(f"  Parsed 6D force bias: {bias_arr}")
                    except Exception as e:
                        print(f"  Error parsing force bias: {e}")
            proc.wait()
            episode_active.clear()
            with buf.lock:
                buf.capture_force_stream = False

            # Wait for C++ process ROS 2 node to fully deregister
            # Prevents node name collision on next episode
            time.sleep(1.0)

            time.sleep(1.0)
            if terminal_scene_pose_cache is not None:
                peg_pose, hole_pose = terminal_scene_pose_cache
            else:
                peg_pose = get_model_pose(peg_name)
                hole_pose = get_model_pose(hole_name)
            terminal_pre_release_peg_pose = np.full(7, np.nan, dtype=np.float32)
            terminal_post_release_peg_pose = np.full(7, np.nan, dtype=np.float32)
            terminal_release_delta_z = float("nan")
            ok = False
            if peg_pose is not None and hole_pose is not None:
                # Compare peg center to actual hole center
                dist = ((peg_pose[0] - hole_pose[0])**2 + (peg_pose[1] - hole_pose[1])**2)**0.5
                pre_release = recorder.release_baseline
                pre_release_ok = (
                    pre_release is not None
                    and np.hypot(
                        pre_release[0] - hole_pose[0],
                        pre_release[1] - hole_pose[1],
                    ) <= 0.0003
                    and abs(pre_release[2] - 0.885) <= 0.00035
                )
                release_delta_z = (
                    float(peg_pose[2] - pre_release[2])
                    if pre_release is not None
                    else float("nan")
                )
                if pre_release is not None:
                    terminal_pre_release_peg_pose = np.asarray(
                        pre_release, dtype=np.float32
                    )
                terminal_post_release_peg_pose = np.asarray(
                    peg_pose, dtype=np.float32
                )
                terminal_release_delta_z = release_delta_z
                geometry_ok = (
                    dist <= 0.0003
                    and abs(peg_pose[2] - 0.885) <= 0.00035
                    and pre_release_ok
                    and abs(release_delta_z) <= 0.00035
                )
                ok = (
                    controller_result == "success"
                    and (
                        recorder.attached_confirmed
                        if args.grasp_recovery_only
                        else geometry_ok
                    )
                )
                success_str = "success" if ok else "failed"
                print(
                    f"  {'✓' if ok else '✗'} {success_str} — "
                    f"controller={controller_result}, "
                    f"peg dist={dist:.4f}m, z={peg_pose[2]:.6f}, "
                    f"release_dz={release_delta_z:+.6f}m")
                if (
                    controller_result == "success"
                    and not args.grasp_recovery_only
                    and not geometry_ok
                ):
                    print("  ⚠ Controller reported success but final geometry failed validation.")

            # Clean up models with error checking
            peg_deleted = (
                get_model_pose(peg_name) is None or delete_model(peg_name)
            )
            hole_deleted = (
                get_model_pose(hole_name) is None or delete_model(hole_name)
            )
            if not peg_deleted:
                print(f"  ⚠ Failed to delete peg model '{peg_name}'")
            if not hole_deleted:
                print(f"  ⚠ Failed to delete hole model '{hole_name}'")
            peg_absent = wait_model_absent(peg_name)
            hole_absent = wait_model_absent(hole_name)
            if not peg_absent or not hole_absent:
                raise RuntimeError(
                    "Gazebo scene reset did not remove all episode entities; "
                    "refusing to contaminate the next episode"
                )

            # Stabilization delay between episodes: let Gazebo physics settle
            # and ensure DetachableJoint fully releases previous attachment
            time.sleep(1.5)

            if len(frames) > 1:
                raw_states = np.stack([f["state"] for f in frames])
                raw_gripper_kinematic_states = np.stack(
                    [f["gripper_kinematic_state"] for f in frames]
                )
                expected_gripper_states = (
                    raw_gripper_kinematic_states[:, :1]
                    * GRIPPER_MIMIC_MULTIPLIERS[None, :]
                )
                gripper_mimic_error = np.abs(
                    raw_gripper_kinematic_states - expected_gripper_states
                )
                max_gripper_mimic_error = float(np.max(gripper_mimic_error))
                if (
                    not np.isfinite(raw_gripper_kinematic_states).all()
                    or max_gripper_mimic_error > GRIPPER_MIMIC_MAX_ERROR_RAD
                ):
                    ok = False
                    print(
                        "  ✗ Gripper linkage symmetry validation failed: "
                        f"max mimic error={max_gripper_mimic_error:.6f} rad "
                        f"(limit={GRIPPER_MIMIC_MAX_ERROR_RAD:.3f})"
                    )
                raw_action_commands = np.stack(
                    [f["action_command"] for f in frames]
                )
                # Re-reference the complete episode after the controller has
                # reported its payload-aware bias. This also calibrates the
                # earlier transport frames that were recorded before that
                # bias became available online.
                raw_forces = np.stack([f["force_raw"] for f in frames])
                raw_force_fast = np.stack(
                    [f["force_fast_raw"] for f in frames]
                )
                raw_force_slow = np.stack(
                    [f["force_slow_raw"] for f in frames]
                )
                empty_force_bias = frames[0]["empty_force_bias"]
                with buf.lock:
                    payload_force_bias = buf.force_bias.copy()
                    payload_force_bias_valid = bool(buf.force_bias_valid)
                frame_payload_phase = np.asarray(
                    [f["force_reference_payload"] for f in frames],
                    dtype=bool,
                )
                frame_force_bias = np.stack([
                    payload_force_bias
                    if payload_force_bias_valid and payload_phase
                    else empty_force_bias
                    for payload_phase in frame_payload_phase
                ]).astype(np.float32)
                forces = raw_forces - frame_force_bias
                fast_payload_phase = np.stack([
                    f["force_fast_reference_payload"] for f in frames
                ])
                slow_payload_phase = np.stack([
                    f["force_slow_reference_payload"] for f in frames
                ])
                fast_bias = np.where(
                    fast_payload_phase[..., None],
                    payload_force_bias[None, None, :]
                    if payload_force_bias_valid
                    else empty_force_bias[None, None, :],
                    empty_force_bias[None, None, :],
                )
                slow_bias = np.where(
                    slow_payload_phase[..., None],
                    payload_force_bias[None, None, :]
                    if payload_force_bias_valid
                    else empty_force_bias[None, None, :],
                    empty_force_bias[None, None, :],
                )
                force_fast = _clip_calibrated_fast_force(
                    raw_force_fast - fast_bias
                )
                force_slow = (raw_force_slow - slow_bias).astype(np.float32)
                state_history = np.stack([f["state_history"] for f in frames])
                visual_quality = np.stack([f["visual_quality"] for f in frames])
                # Recompute formal targets only after the final payload-aware
                # force reference is known. Online targets recorded before
                # the C++ force-bias message would interpret the carried
                # 20--25 N payload as contact throughout transport.
                reset_label_buffers()
                stages = np.stack([
                    compute_physical_state_vector(
                        forces[i],
                        force_fast[i],
                        frames[i]["tool0_z"],
                        frames[i]["cam_degraded_frame"],
                        cam_degradation_type=frames[i]["cam_degradation_type"],
                        cam_glare_gain=frames[i]["cam_glare_gain"],
                        gripper_joint_val=float(raw_states[i, 6]),
                        joint_vel_norm=frames[i]["joint_vel_norm"],
                        current_stage=frames[i]["current_stage"],
                        semantic_subtask=frames[i]["semantic_subtask"],
                    )
                    for i in range(len(frames))
                ]).astype(np.float32)
                cam0       = np.stack([f["camera0"] for f in frames])
                cam1       = np.stack([f["camera1"] for f in frames])
                cam0_clean_teacher = np.stack(
                    [f["camera0_clean_teacher"] for f in frames]
                )
                cam1_clean_teacher = np.stack(
                    [f["camera1_clean_teacher"] for f in frames]
                )
                visual_degradation_active = np.asarray(
                    [f["cam_degraded_frame"] for f in frames], dtype=bool
                )
                with buf.lock:
                    raw_force_timestamps = np.asarray(
                        buf.raw_force_timestamps, dtype=np.float64
                    )
                    raw_force_wall_timestamps = np.asarray(
                        buf.raw_force_wall_timestamps, dtype=np.float64
                    )
                    raw_force_values = np.asarray(
                        buf.raw_force_values, dtype=np.float32
                    ).reshape(-1, FORCE_DIM)
                    force_bias = buf.force_bias.copy()
                    force_bias_valid = bool(buf.force_bias_valid)

                states_out = raw_states[:-1]
                gripper_kinematic_states_out = (
                    raw_gripper_kinematic_states[:-1]
                )
                gripper_mimic_error_out = gripper_mimic_error[:-1]
                # One-step-ahead desired controller positions are the action
                # contract.  Never substitute measured next state: contact
                # makes the achieved gripper angle object-dependent even
                # though the close command is universally 0.8 rad.
                actions_out = canonicalize_gripper_actions(
                    raw_action_commands[1:]
                )
                forces_out  = forces[:-1]
                force_fast_out = force_fast[:-1]
                force_slow_out = force_slow[:-1]
                state_history_out = state_history[:-1]
                visual_quality_out = visual_quality[:-1]
                stages_out  = stages[:-1]
                cam0_out    = cam0[:-1]
                cam1_out    = cam1[:-1]
                cam0_clean_teacher_out = cam0_clean_teacher[:-1]
                cam1_clean_teacher_out = cam1_clean_teacher[:-1]
                visual_degradation_active_out = visual_degradation_active[:-1]

                task_prefix = TASK.replace(" ", "_")
                success_name = f"{task_prefix}_episode_{ep:04d}_success"
                failed_name  = f"{task_prefix}_episode_{ep:04d}_failed"

                if ok:
                    target_name = success_name
                    opposite_name = failed_name
                else:
                    target_name = failed_name
                    opposite_name = success_name

                ep_dir = os.path.join(args.output, target_name)
                opposite_dir = os.path.join(args.output, opposite_name)

                # Clean up stale opposite directory if exists
                if os.path.exists(opposite_dir):
                    import shutil
                    shutil.rmtree(opposite_dir)
                    print(f"  Cleaned up stale opposite episode directory: {opposite_name}")

                os.makedirs(ep_dir, exist_ok=True)
                tool0_z_out = np.array([f["tool0_z"] for f in frames[:-1]], dtype=np.float32)
                timestamps = np.array([f["timestamp"] for f in frames[:-1]],
                                      dtype=np.float64)
                semantic_subtasks_out = np.array(
                    [f["semantic_subtask"] for f in frames[:-1]], dtype=object
                )
                skill_progress = _derive_skill_progress(
                    semantic_subtasks_out,
                    states_out,
                    actions_out,
                    tool0_z_out,
                )
                recovery_directions = np.stack(
                    [f["recovery_direction"] for f in frames[:-1]])

                np.savez_compressed(
                    os.path.join(ep_dir, "data.npz"),
                    state=states_out, action=actions_out,
                    gripper_kinematic_state=gripper_kinematic_states_out,
                    gripper_mimic_error=gripper_mimic_error_out,
                    gripper_kinematic_joint_names=np.asarray(
                        GRIPPER_KINEMATIC_JOINTS, dtype=object
                    ),
                    gripper_mimic_max_error_rad=np.float32(
                        GRIPPER_MIMIC_MAX_ERROR_RAD
                    ),
                    force=forces_out,
                    force_fast=force_fast_out,
                    force_slow=force_slow_out,
                    state_history=state_history_out,
                    visual_quality=visual_quality_out,
                    force_fast_valid=np.asarray(
                        [f["force_fast_valid"] for f in frames[:-1]],
                        dtype=np.int16,
                    ),
                    force_slow_valid=np.asarray(
                        [f["force_slow_valid"] for f in frames[:-1]],
                        dtype=np.int16,
                    ),
                    state_history_valid=np.asarray(
                        [f["state_history_valid"] for f in frames[:-1]],
                        dtype=np.int16,
                    ),
                    raw_force_timestamp=raw_force_timestamps,
                    raw_force_wall_timestamp=raw_force_wall_timestamps,
                    raw_force=raw_force_values,
                    force_bias=force_bias,
                    force_bias_valid=np.bool_(force_bias_valid),
                    force_reference_payload=frame_payload_phase[:-1],
                    force_fast_reference_payload=fast_payload_phase[:-1],
                    force_slow_reference_payload=slow_payload_phase[:-1],
                    stage=stages_out,
                    tool0_z=tool0_z_out,
                    camera0=cam0_out, camera1=cam1_out,
                    camera0_clean_teacher=cam0_clean_teacher_out,
                    camera1_clean_teacher=cam1_clean_teacher_out,
                    visual_degradation_active=visual_degradation_active_out,
                    contact_truth_valid=np.asarray(
                        [f["contact_truth_valid"] for f in frames[:-1]],
                        dtype=bool,
                    ),
                    contact_truth_any=np.asarray(
                        [f["contact_truth_any"] for f in frames[:-1]],
                        dtype=bool,
                    ),
                    contact_truth_task=np.asarray(
                        [f["contact_truth_task"] for f in frames[:-1]],
                        dtype=bool,
                    ),
                    contact_truth_gripper=np.asarray(
                        [f["contact_truth_gripper"] for f in frames[:-1]],
                        dtype=bool,
                    ),
                    contact_truth_hole=np.asarray(
                        [f["contact_truth_hole"] for f in frames[:-1]],
                        dtype=bool,
                    ),
                    contact_truth_table=np.asarray(
                        [f["contact_truth_table"] for f in frames[:-1]],
                        dtype=bool,
                    ),
                    contact_truth_max_depth=np.asarray(
                        [f["contact_truth_max_depth"] for f in frames[:-1]],
                        dtype=np.float32,
                    ),
                    contact_truth_contract=np.str_(
                        "gazebo_teacher_audit_only_gripper_hole_table_v1"
                    ),
                    timestamp=timestamps,
                    timestamp_ros=np.array(
                        [f["timestamp_ros"] for f in frames[:-1]],
                        dtype=np.float64,
                    ),
                    task=np.array([f["task"] for f in frames[:-1]], dtype=object),
                    semantic_subtask=semantic_subtasks_out,
                    skill_progress_phase=skill_progress["phase"],
                    skill_progress_phase_name=skill_progress["phase_name"],
                    skill_progress=skill_progress["progress"],
                    transition_readiness=skill_progress["readiness"],
                    skill_progress_valid=skill_progress["valid"],
                    skill_progress_label_source=skill_progress["source"],
                    skill_progress_confidence=skill_progress["confidence"],
                    search_mode=np.str_(episode_search_mode),
                    control_strategy=np.str_(
                        "cartesian_outer_admittance"
                        if episode_search_mode == "admittance"
                        else episode_search_mode
                    ),
                    admittance_mass_xy=np.float32(
                        1.5 if episode_search_mode == "admittance" else np.nan
                    ),
                    admittance_damping_xy=np.float32(
                        60.0 if episode_search_mode == "admittance" else np.nan
                    ),
                    admittance_stiffness_xy=np.float32(
                        600.0 if episode_search_mode == "admittance" else np.nan
                    ),
                    admittance_target_fz=np.float32(
                        3.0 if episode_search_mode == "admittance" else np.nan
                    ),
                    search_mode_frame=np.array(
                        [f["search_mode"] for f in frames[:-1]], dtype=object),
                    recovery_active=np.array(
                        [f["recovery_active"] for f in frames[:-1]], dtype=bool),
                    recovery_attempt=np.array(
                        [f["recovery_attempt"] for f in frames[:-1]],
                        dtype=np.int16),
                    recovery_direction=recovery_directions,
                    recovery_result=np.array(
                        [f["recovery_result"] for f in frames[:-1]],
                        dtype=np.int8),
                    recovery_event_id=np.array(
                        [f["recovery_event_id"] for f in frames[:-1]],
                        dtype=np.int16),
                    recovery_safety=np.array(
                        [f["recovery_safety"] for f in frames[:-1]], dtype=bool),
                    controller_result=np.str_(controller_result),
                    trajectory_scope=np.str_(
                        "grasp_recovery_prefix_v1"
                        if args.grasp_recovery_only
                        else "full_task"
                    ),
                    cam_degraded=np.bool_(buf.cam_degradation),
                    cam_deg_type=np.str_(buf.cam_degradation_type),
                    cam_glare_gain=np.float32(buf.cam_glare_gain),
                    visual_supervision_contract=np.str_(
                        "shared_policy_view_clean_teacher_v1"
                    ),
                    visual_degradation_scope=np.str_(
                        "both_policy_cameras_v1"
                    ),
                    schema_version=np.str_("pap_moe_v9_contact_truth"),
                    action_source=np.str_(
                        "controller_desired_position_one_step_ahead"
                    ),
                    gripper_command_contract=np.str_(
                        "robotiq_endpoint_0.0_open_0.8_closed_v1"
                    ),
                    gripper_open_command_rad=np.float32(0.0),
                    gripper_close_command_rad=np.float32(0.8),
                    controller_profile_version=np.str_(
                        "active_seating_no_gravity_v11_fast_free_motion"
                    ),
                    terminal_seating_contract=np.str_(
                        "active_pre_release_seat_no_gravity_v1"
                    ),
                    terminal_target_peg_center_z=np.float32(0.885),
                    terminal_pre_release_peg_pose=(
                        terminal_pre_release_peg_pose
                    ),
                    terminal_post_release_peg_pose=(
                        terminal_post_release_peg_pose
                    ),
                    terminal_release_delta_z=np.float32(
                        terminal_release_delta_z
                    ),
                    trajectory_style_id=np.int16(trajectory_style_id),
                    trajectory_style_name=np.str_(trajectory_style["name"]),
                    trajectory_style_vector=np.asarray(
                        (*trajectory_style["peg"],
                         *trajectory_style["transport"],
                         *trajectory_style["hole"],
                         trajectory_style["speed"]),
                        dtype=np.float32,
                    ),
                    position_group_id=np.int32(position_group_id),
                    position_repeat_index=np.int16(position_repeat_index),
                    position_repeat_count=np.int16(args.position_repeat_count),
                    subtask_semantics_version=np.str_(
                        "post_motion_force_verify_v2"
                    ),
                    force_reference_mode=np.str_(
                        "per_sample_phase_payload_bias_v2"
                    ),
                    force_filter_mode=np.str_(
                        "median11_sustained_visible_fast_clip30n_native100hz_v6"
                    ),
                    grasp_mode=np.str_(
                        "physical_analytic_handle_endpoint_pose_verified_v3"
                    ),
                    grasp_pickup_duration_s=np.float32(3.0),
                    grasp_pickup_profile=np.str_("ease_out_sqrt_v1"),
                    gripper_contact_stall_velocity_rad_s=np.float32(0.10),
                    gripper_contact_stall_cycles=np.int16(5),
                    gripper_fingertip_contact_model=np.str_(
                        "analytic_cylinder_rubber_pad_v1"
                    ),
                    routing_prior_version=np.str_(
                        "factorized_general_qch_v11_deadband"
                    ),
                    policy_hz=np.float32(record_hz),
                    policy_timebase=np.str_(
                        "gazebo_sim_clock_10hz_v1"
                    ),
                    policy_episode_endpoint=np.str_(
                        "task_result_before_reset_v1"
                    ),
                    peg_mass=np.float32(peg_mass),
                    peg_ixx=np.float32(ixx_peg),
                    peg_izz=np.float32(izz_peg),
                    peg_roll=np.float32(peg_roll),
                    peg_pitch=np.float32(peg_pitch),
                    contact_kp=np.float32(kp_val),
                    contact_kd=np.float32(kd_val),
                    friction_mu=np.float32(mu_val),
                    spring_k=np.float32(spring_k_val),
                    spring_d=np.float32(spring_d_val),
                    fixture_mode=np.str_(
                        "rigid_real_zero_clearance_split_collision_v8"
                    ),
                    verification_seat_mass=np.float32(0.0),
                    verification_seat_k=np.float32(0.0),
                    verification_seat_d=np.float32(0.0),
                    verification_seat_travel=np.float32(0.0),
                    physics_engine=np.str_(args.physics_engine),
                    sim_position_gain=np.float32(args.sim_position_gain),
                    velocity_scaling=np.float32(vel_scale),
                    transport_velocity_scaling=np.float32(trans_vel_scale),
                    gripper_velocity_scaling=np.float32(1.0),
                    fine_insertion_step=np.float32(
                        args.fine_insertion_step
                    ),
                    recovery_offset_max=np.float32(args.recovery_offset_max),
                    grasp_recovery_offset=grasp_recovery_offset.astype(
                        np.float32
                    ),
                    peg_x=np.float32(peg_x), peg_y=np.float32(peg_y),
                    hole_x=np.float32(hole_x), hole_y=np.float32(hole_y))

                print(f"  Saved {target_name} — {len(states_out)} frames")
                if ok:
                    total_frames += len(states_out)
                elif not args.continue_after_failure:
                    print(
                        "  Aborting remaining episodes after failure; restart "
                        "from a clean simulation with a new seed/style."
                    )
                    break

    finally:
        recording.clear()
        rec_thread.join(timeout=2.0)
        executor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

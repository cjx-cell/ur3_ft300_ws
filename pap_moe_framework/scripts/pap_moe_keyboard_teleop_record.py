#!/usr/bin/env python3
"""Independent multimodal recorder for keyboard-guided PAP-MoE demos.

This file deliberately does not call or modify the scripted trajectory
collector.  It records the same baseline/PAP-MoE training modalities while a
human drives MoveIt Servo through ``pap_moe_keyboard_teleop.py``.
"""

import argparse
from collections import deque
import os
import random
import shutil
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import WrenchStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

from pap_moe_peg_in_hole_record import (
    ALL_JOINTS,
    FORCE_DIM,
    HOLE_X_RANGE,
    HOLE_Y_RANGE,
    IMG_SIZE,
    PEG_IXX,
    PEG_IZZ,
    PEG_MASS,
    PEG_SDF,
    BLOCK_X_RANGE,
    BLOCK_Y_RANGE,
    TASK,
    delete_model,
    get_model_pose,
    make_hole_sdf,
    spawn_model,
)
from pap_moe_routing_prior import PhysicsRoutingPrior


# Give the scripted collector's legacy range name its semantic name locally.
PEG_X_RANGE = BLOCK_X_RANGE
RECORD_HZ = 10.0
FAST_FORCE_SAMPLES = 64
SLOW_FORCE_SAMPLES = 50
STATE_HISTORY_SAMPLES = 10
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
STAGES = (
    "main",
    "descent",
    "lower",
    "grasp",
    "post_grasp",
    "align",
    "contact",
)


def left_pad(values, size, width):
    if not values:
        return np.zeros((size, width), dtype=np.float32)
    selected = [np.asarray(item, dtype=np.float32) for item in values[-size:]]
    selected = [selected[0]] * (size - len(selected)) + selected
    return np.stack(selected, axis=0)


def visual_quality(camera0, camera1):
    cameras = np.stack([camera0, camera1], axis=0)
    gray = cameras.mean(axis=-1)
    finite = np.isfinite(cameras).all()
    return np.asarray(
        [
            np.mean(gray <= 0.02) if finite else 1.0,
            np.mean(gray >= 0.98) if finite else 1.0,
            np.mean(np.std(gray, axis=(1, 2))) if finite else 0.0,
            float(finite and np.mean(np.std(gray, axis=(1, 2))) >= 0.01),
        ],
        dtype=np.float32,
    )


class TeleopRecorder(Node):
    def __init__(self):
        super().__init__(
            "pap_moe_keyboard_teleop_recorder",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.state = None
        self.action = None
        self.arm_action = None
        self.gripper_action = 0.0
        self.force = np.zeros(6, dtype=np.float32)
        self.camera0 = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        self.camera1 = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        self.force_fast = deque(maxlen=FAST_FORCE_SAMPLES)
        self.force_fast_bias = deque(maxlen=FAST_FORCE_SAMPLES)
        self.force_fast_payload = deque(maxlen=FAST_FORCE_SAMPLES)
        self.force_slow = deque(maxlen=SLOW_FORCE_SAMPLES)
        self.force_slow_bias = deque(maxlen=SLOW_FORCE_SAMPLES)
        self.force_slow_payload = deque(maxlen=SLOW_FORCE_SAMPLES)
        self.state_history = deque(maxlen=STATE_HISTORY_SAMPLES)
        self.raw_force_timestamps = []
        self.raw_force_wall_timestamps = []
        self.raw_force = []
        self.last_slow_stamp = None
        self.last_state_stamp = None
        self.stage = STAGES[0]
        self.recording = False
        self.finished = threading.Event()
        self.result = None
        self.frames = []
        self.empty_force_bias = np.zeros(6, dtype=np.float32)
        self.payload_force_bias = np.zeros(6, dtype=np.float32)
        self.payload_reference = False
        self.pre_release_pose = None
        self.routing = PhysicsRoutingPrior()

        self.create_subscription(JointState, "/joint_states", self.on_joint, 20)
        self.create_subscription(
            JointTrajectoryControllerState,
            "/joint_trajectory_controller/controller_state",
            self.on_controller,
            20,
        )
        self.create_subscription(
            Float64MultiArray,
            "/arm_servo_controller/commands",
            self.on_arm_command,
            20,
        )
        self.create_subscription(
            JointTrajectory,
            "/gripper_trajectory_controller/joint_trajectory",
            self.on_gripper_command,
            20,
        )
        self.create_subscription(
            WrenchStamped,
            "/force_torque_sensor_broadcaster/wrench",
            self.on_wrench,
            100,
        )
        self.create_subscription(
            Image, "/wrist_camera/color/image_raw", self.on_camera0, 10
        )
        self.create_subscription(
            Image, "/global_camera/color/image_raw", self.on_camera1, 10
        )
        self.create_subscription(String, "/pap_moe/teleop_stage", self.on_stage, 10)
        self.create_subscription(String, "/pap_moe/teleop_event", self.on_event, 10)
        self.create_timer(1.0 / RECORD_HZ, self.capture)

    def stamp(self, msg):
        value = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        return value if value > 0 else self.get_clock().now().nanoseconds * 1e-9

    def on_joint(self, msg):
        try:
            value = np.asarray(
                [msg.position[msg.name.index(name)] for name in ALL_JOINTS],
                dtype=np.float32,
            )
        except (ValueError, IndexError):
            return
        stamp = self.stamp(msg)
        with self.lock:
            self.state = value
            if self.last_state_stamp is None or stamp - self.last_state_stamp >= 0.095:
                self.state_history.append(value.copy())
                self.last_state_stamp = stamp

    def on_controller(self, msg):
        try:
            value = np.asarray(
                [msg.reference.positions[msg.joint_names.index(name)] for name in ALL_JOINTS],
                dtype=np.float32,
            )
        except (AttributeError, ValueError, IndexError):
            try:
                value = np.asarray(
                    [msg.desired.positions[msg.joint_names.index(name)] for name in ALL_JOINTS],
                    dtype=np.float32,
                )
            except (AttributeError, ValueError, IndexError):
                return
        with self.lock:
            self.action = value

    def on_arm_command(self, msg):
        if len(msg.data) != 6:
            return
        with self.lock:
            self.arm_action = np.asarray(msg.data, dtype=np.float32)
            self.action = np.concatenate(
                [self.arm_action, np.asarray([self.gripper_action], dtype=np.float32)]
            )

    def on_gripper_command(self, msg):
        if not msg.points or not msg.points[-1].positions:
            return
        with self.lock:
            self.gripper_action = float(msg.points[-1].positions[0])
            if self.arm_action is not None:
                self.action = np.concatenate(
                    [self.arm_action, np.asarray([self.gripper_action], dtype=np.float32)]
                )

    def on_wrench(self, msg):
        wrench = msg.wrench
        value = np.asarray(
            [
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                wrench.torque.x,
                wrench.torque.y,
                wrench.torque.z,
            ],
            dtype=np.float32,
        )
        stamp = self.stamp(msg)
        with self.lock:
            self.force = value
            self.force_fast.append(value.copy())
            active_bias = (
                self.payload_force_bias
                if self.payload_reference
                else self.empty_force_bias
            )
            self.force_fast_bias.append(active_bias.copy())
            self.force_fast_payload.append(self.payload_reference)
            if self.last_slow_stamp is None or stamp - self.last_slow_stamp >= 0.095:
                self.force_slow.append(value.copy())
                self.force_slow_bias.append(active_bias.copy())
                self.force_slow_payload.append(self.payload_reference)
                self.last_slow_stamp = stamp
            if self.recording:
                self.raw_force_timestamps.append(stamp)
                self.raw_force_wall_timestamps.append(time.time())
                self.raw_force.append(value.copy())

    def decode(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        except Exception:
            return None

    def on_camera0(self, msg):
        value = self.decode(msg)
        if value is not None:
            with self.lock:
                self.camera0 = value

    def on_camera1(self, msg):
        value = self.decode(msg)
        if value is not None:
            with self.lock:
                self.camera1 = value

    def on_stage(self, msg):
        if msg.data in STAGES:
            with self.lock:
                self.stage = msg.data

    def on_event(self, msg):
        event = msg.data
        if event == "begin":
            with self.lock:
                self.frames.clear()
                self.raw_force.clear()
                self.raw_force_timestamps.clear()
                self.raw_force_wall_timestamps.clear()
                self.empty_force_bias = self.force.copy()
                self.payload_force_bias = self.empty_force_bias.copy()
                self.payload_reference = False
                self.force_fast_bias = deque(
                    [self.empty_force_bias.copy() for _ in self.force_fast],
                    maxlen=FAST_FORCE_SAMPLES,
                )
                self.force_fast_payload = deque(
                    [False for _ in self.force_fast], maxlen=FAST_FORCE_SAMPLES
                )
                self.force_slow_bias = deque(
                    [self.empty_force_bias.copy() for _ in self.force_slow],
                    maxlen=SLOW_FORCE_SAMPLES,
                )
                self.force_slow_payload = deque(
                    [False for _ in self.force_slow], maxlen=SLOW_FORCE_SAMPLES
                )
                self.pre_release_pose = None
                self.routing.reset()
                self.recording = True
            self.get_logger().info("RECORDING STARTED")
        elif event == "payload_reference" and self.recording:
            with self.lock:
                self.payload_force_bias = self.force.copy()
                self.payload_reference = True
            self.get_logger().info("payload FT reference captured")
        elif event == "gripper_open" and self.recording:
            # The keyboard publishes this before executing the opening stroke.
            self.pre_release_pose = get_model_pose("peg")
            with self.lock:
                self.payload_reference = False
        elif event in ("success", "discard", "emergency_stop"):
            with self.lock:
                self.recording = False
                self.result = event
            self.finished.set()

    def ready(self):
        with self.lock:
            return (
                self.state is not None
                and self.action is not None
                and len(self.force_fast) >= FAST_FORCE_SAMPLES
                and len(self.force_slow) >= SLOW_FORCE_SAMPLES
                and len(self.state_history) >= STATE_HISTORY_SAMPLES
                and np.any(self.camera0)
                and np.any(self.camera1)
            )

    def capture(self):
        with self.lock:
            if not self.recording or self.state is None or self.action is None:
                return
            state = self.state.copy()
            action = self.action.copy()
            force_raw = self.force.copy()
            cam0 = self.camera0.copy()
            cam1 = self.camera1.copy()
            fast_raw = left_pad(list(self.force_fast), FAST_FORCE_SAMPLES, FORCE_DIM)
            fast_bias = left_pad(
                list(self.force_fast_bias), FAST_FORCE_SAMPLES, FORCE_DIM
            )
            slow_raw = left_pad(list(self.force_slow), SLOW_FORCE_SAMPLES, FORCE_DIM)
            slow_bias = left_pad(
                list(self.force_slow_bias), SLOW_FORCE_SAMPLES, FORCE_DIM
            )
            state_history = left_pad(
                list(self.state_history), STATE_HISTORY_SAMPLES, len(ALL_JOINTS)
            )
            stage = self.stage
            payload = self.payload_reference
            bias = self.payload_force_bias.copy() if payload else self.empty_force_bias.copy()

        force = force_raw - bias
        fast = fast_raw - fast_bias
        slow = slow_raw - slow_bias
        qdot = np.linalg.norm(state[:6] - state_history[-2, :6]) * RECORD_HZ
        routing = self.routing.compute(
            force,
            fast,
            tool0_z=None,
            cam_degraded=False,
            gripper_joint_val=float(state[-1]),
            joint_vel_norm=float(qdot),
        )
        self.frames.append(
            {
                "state": state,
                "action": action,
                "force": force.astype(np.float32),
                "force_fast": fast.astype(np.float32),
                "force_slow": slow.astype(np.float32),
                "state_history": state_history,
                "camera0": cam0,
                "camera1": cam1,
                "visual_quality": visual_quality(cam0, cam1),
                "routing": routing,
                "semantic_subtask": stage,
                "payload_reference": payload,
                "force_fast_reference_payload": np.asarray(
                    list(self.force_fast_payload), dtype=bool
                ),
                "force_slow_reference_payload": np.asarray(
                    list(self.force_slow_payload), dtype=bool
                ),
                "force_bias": bias,
                "timestamp_ros": self.get_clock().now().nanoseconds * 1e-9,
                "timestamp_wall": time.time(),
            }
        )


def spawn_fixtures(seed):
    random.seed(seed)
    peg_x = random.uniform(*PEG_X_RANGE)
    peg_y = random.uniform(*BLOCK_Y_RANGE)
    hole_x = random.uniform(*HOLE_X_RANGE)
    hole_y = random.uniform(*HOLE_Y_RANGE)
    delete_model("peg")
    delete_model("hole_plate_teleop")
    peg_sdf = PEG_SDF.format(
        name="peg",
        kp=5000,
        kd=20,
        mass=PEG_MASS,
        ixx=PEG_IXX,
        iyy=PEG_IXX,
        izz=PEG_IZZ,
    )
    if not spawn_model("peg", peg_x, peg_y, 0.865, sdf_string=peg_sdf):
        raise RuntimeError("failed to spawn peg")
    if not spawn_model(
        "hole_plate_teleop",
        hole_x,
        hole_y,
        0.775,
        sdf_string=make_hole_sdf("hole_plate_teleop", kp=5000, kd=20),
    ):
        raise RuntimeError("failed to spawn hole")
    return peg_x, peg_y, hole_x, hole_y


def terminal_contract(node):
    post = get_model_pose("peg")
    hole = get_model_pose("hole_plate_teleop")
    pre = node.pre_release_pose
    if pre is None or post is None or hole is None:
        return False, pre, post, "missing pre-release/post-release/hole pose"
    pre_xy = float(np.hypot(pre[0] - hole[0], pre[1] - hole[1]))
    post_xy = float(np.hypot(post[0] - hole[0], post[1] - hole[1]))
    release_xy = float(np.hypot(post[0] - pre[0], post[1] - pre[1]))
    release_dz = float(post[2] - pre[2])
    ok = (
        pre_xy <= 0.0003
        and post_xy <= 0.0003
        and abs(pre[2] - 0.885) <= 0.0003
        and abs(post[2] - 0.885) <= 0.0003
        and release_xy <= 0.0003
        and abs(release_dz) <= 0.0003
    )
    detail = (
        f"pre_xy={pre_xy:.6f},post_xy={post_xy:.6f},"
        f"pre_z={pre[2]:.6f},post_z={post[2]:.6f},"
        f"release_xy={release_xy:.6f},release_dz={release_dz:.6f}"
    )
    return ok, pre, post, detail


def stack(frames, key):
    return np.stack([frame[key] for frame in frames])


def save_episode(node, args, positions):
    frames = node.frames
    if len(frames) < 2:
        raise RuntimeError("episode has fewer than two frames")
    ok, pre, post, detail = terminal_contract(node)
    if node.result != "success" or not ok:
        raise RuntimeError(f"episode rejected: event={node.result}; {detail}")

    episode_name = f"pick_up_the_peg_and_insert_it_into_the_hole_episode_{args.episode:04d}_success"
    episode_dir = os.path.join(args.output, episode_name)
    if os.path.exists(episode_dir):
        raise FileExistsError(f"refusing to overwrite {episode_dir}")
    # Publish the final episode directory only after the compressed archive
    # closes successfully.  Interrupted saves may leave a hidden temporary
    # directory, but can no longer create a false successful episode.
    temp_dir = os.path.join(args.output, f".{episode_name}.tmp-{os.getpid()}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    usable = frames[:-1]
    actions = stack(frames[1:], "action")
    raw_force = np.stack(node.raw_force).astype(np.float32)
    np.savez_compressed(
        os.path.join(temp_dir, "data.npz"),
        state=stack(usable, "state"),
        action=actions,
        force=stack(usable, "force"),
        force_fast=stack(usable, "force_fast"),
        force_slow=stack(usable, "force_slow"),
        state_history=stack(usable, "state_history"),
        visual_quality=stack(usable, "visual_quality"),
        stage=stack(usable, "routing"),
        routing_prior=stack(usable, "routing"),
        semantic_subtask=np.asarray(
            [frame["semantic_subtask"] for frame in usable], dtype=object
        ),
        camera0=stack(usable, "camera0"),
        camera1=stack(usable, "camera1"),
        camera0_clean_teacher=stack(usable, "camera0"),
        camera1_clean_teacher=stack(usable, "camera1"),
        visual_degradation_active=np.zeros(len(usable), dtype=bool),
        force_reference_payload=np.asarray(
            [frame["payload_reference"] for frame in usable], dtype=bool
        ),
        force_fast_reference_payload=stack(
            usable, "force_fast_reference_payload"
        ),
        force_slow_reference_payload=stack(
            usable, "force_slow_reference_payload"
        ),
        force_fast_valid=np.full(len(usable), FAST_FORCE_SAMPLES, dtype=np.int16),
        force_slow_valid=np.full(len(usable), SLOW_FORCE_SAMPLES, dtype=np.int16),
        state_history_valid=np.full(
            len(usable), STATE_HISTORY_SAMPLES, dtype=np.int16
        ),
        force_bias_frame=stack(usable, "force_bias"),
        raw_force_timestamp=np.asarray(node.raw_force_timestamps, dtype=np.float64),
        raw_force_wall_timestamp=np.asarray(
            node.raw_force_wall_timestamps, dtype=np.float64
        ),
        raw_force=raw_force,
        timestamp_ros=np.asarray(
            [frame["timestamp_ros"] for frame in usable], dtype=np.float64
        ),
        timestamp=np.asarray(
            [frame["timestamp_wall"] for frame in usable], dtype=np.float64
        ),
        task=np.asarray([TASK] * len(usable), dtype=object),
        schema_version=np.str_("pap_moe_teleop_v1"),
        collection_mode=np.str_("human_keyboard_moveit_servo_v1"),
        action_source=np.str_("controller_reference_position_one_step_ahead"),
        gripper_command_contract=np.str_("robotiq_endpoint_0.0_open_0.8_closed_v1"),
        gripper_open_command_rad=np.float32(0.0),
        gripper_close_command_rad=np.float32(0.8),
        routing_prior_version=np.str_("factorized_general_qch_v11_deadband"),
        visual_supervision_contract=np.str_("shared_policy_view_clean_teacher_v1"),
        visual_degradation_scope=np.str_("both_policy_cameras_v1"),
        cam_degraded=np.bool_(False),
        terminal_seating_contract=np.str_("active_pre_release_seat_no_gravity_v1"),
        terminal_target_peg_center_z=np.float32(0.885),
        terminal_pre_release_peg_pose=np.asarray(pre, dtype=np.float32),
        terminal_post_release_peg_pose=np.asarray(post, dtype=np.float32),
        terminal_release_delta_z=np.float32(post[2] - pre[2]),
        policy_hz=np.float32(RECORD_HZ),
        policy_timebase=np.str_("gazebo_sim_clock_10hz_v1"),
        policy_episode_endpoint=np.str_("task_result_before_reset_v1"),
        physics_engine=np.str_("ignition-physics-dartsim-plugin"),
        sim_position_gain=np.float32(0.5),
        fixture_mode=np.str_("rigid_real_zero_clearance_split_collision_v8"),
        peg_x=np.float32(positions[0]),
        peg_y=np.float32(positions[1]),
        hole_x=np.float32(positions[2]),
        hole_y=np.float32(positions[3]),
    )
    os.rename(temp_dir, episode_dir)
    return episode_dir, len(usable), detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        default=os.path.expanduser(
            "~/ur3_ft300_ws/pap_moe_framework/datasets/teleop_raw"
        ),
    )
    parser.add_argument("--no-spawn", action="store_true")
    args = parser.parse_args()
    if args.episode < 1:
        parser.error("--episode must be >= 1")
    positions = (np.nan,) * 4 if args.no_spawn else spawn_fixtures(args.seed)

    rclpy.init()
    node = TeleopRecorder()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        print("Warming multimodal windows (two cameras + FT100Hz + state history)...")
        # Readiness is based on simulation-time sample windows. Two rendered
        # cameras can reduce Gazebo to ~0.1 real-time factor, so a 5 s slow-FT
        # window may need close to one wall-clock minute to fill.
        deadline = time.monotonic() + 120.0
        while not node.ready() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not node.ready():
            raise RuntimeError("multimodal inputs did not become ready")
        print("READY: press B in the keyboard terminal, then V only after stable release")
        node.finished.wait()
        if node.result != "success":
            print(f"Episode discarded ({node.result}); no dataset directory written")
            return
        episode_dir, frames, detail = save_episode(node, args, positions)
        print(f"SAVED {episode_dir} ({frames} policy frames)\n{detail}")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

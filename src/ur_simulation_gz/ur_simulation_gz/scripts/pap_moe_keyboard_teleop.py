#!/usr/bin/env python3
"""Safe terminal-keyboard teleoperation for UR3 MoveIt Servo.

This intentionally keeps arm motion continuous and the Robotiq command
binary.  A key press creates only a short command pulse; loss of keyboard
input therefore stops the robot without relying on a key-release event.
"""

import select
import math
import sys
import termios
import threading
import time
import tty

import rclpy
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8, String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
GRIPPER_OPEN_RAD = 0.0
GRIPPER_CLOSED_RAD = 0.8
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class KeyboardTeleop(Node):
    def __init__(self):
        # The keyboard executable is normally started with ``ros2 run`` rather
        # than as a child of the launch file.  Set simulation time here as an
        # invariant; otherwise Twist stamps use Unix wall time while Servo
        # evaluates them against Gazebo time and silently ignores the jog.
        super().__init__(
            "pap_moe_keyboard_teleop",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ],
        )
        self.twist_pub = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10
        )
        self.joint_jog_pub = self.create_publisher(
            JointJog, "/servo_node/delta_joint_cmds", 10
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            "/gripper_trajectory_controller/joint_trajectory",
            10,
        )
        self.event_pub = self.create_publisher(String, "/pap_moe/teleop_event", 10)
        self.stage_pub = self.create_publisher(String, "/pap_moe/teleop_stage", 10)
        self.start_client = self.create_client(Trigger, "/servo_node/start_servo")
        self.stop_client = self.create_client(Trigger, "/servo_node/stop_servo")
        self.pause_client = self.create_client(Trigger, "/servo_node/pause_servo")
        self.unpause_client = self.create_client(Trigger, "/servo_node/unpause_servo")
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 10)
        self.create_subscription(Int8, "/servo_node/status", self.servo_status_cb, 10)

        self.lock = threading.Lock()
        self.command = [0.0] * 6
        self.command_deadline_ros = 0.0
        self.speed_modes = (
            # name, normalized velocity, pulse length in simulation seconds
            ("normal", 0.35, 0.08),
            ("coarse", 1.00, 0.05),
            ("precision", 0.02, 0.12),
        )
        # Start fast for long free-space moves.  Switch to precision near the
        # peg/hole or during contact.
        self.speed_mode_index = 1
        self.running = True
        self.servo_status = -1
        self.control_mode = "cartesian"
        self.continuous_input = False
        self.held_motion_keys = set()
        self.gripper_closed = False
        self.gripper_state_initialized = False
        self.gripper_target = GRIPPER_OPEN_RAD
        self.gripper_deadline = 0.0
        self.gripper_publish_at = 0.0
        self.gripper_trajectory_published = False
        self.measured_gripper = GRIPPER_OPEN_RAD
        self.measured_gripper_velocity = 0.0
        self.gripper_motion_start = GRIPPER_OPEN_RAD
        self.gripper_stall_since = None
        self.servo_restart_pending = False
        self.stage_index = 0
        self.stages = [
            "main",
            "descent",
            "lower",
            "grasp",
            "post_grasp",
            "align",
            "contact",
        ]
        self.create_timer(0.02, self.publish_command)

    def servo_status_cb(self, msg):
        self.servo_status = int(msg.data)

    def start_servo(self):
        if not self.start_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/servo_node/start_servo service is unavailable")
        # move_group, RViz and the post-episode reset client can all finish
        # discovery at nearly the same time.  Servo occasionally rejects the
        # first start request during that transition and accepts the next one.
        # Do not let this short-lived launch race remove the complete Tk GUI.
        for attempt in range(1, 6):
            future = self.start_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if (
                future.done()
                and future.result() is not None
                and future.result().success
            ):
                return
            self.get_logger().warning(
                f"MoveIt Servo start attempt {attempt}/5 failed; retrying"
            )
            time.sleep(0.25)
        raise RuntimeError("failed to start MoveIt Servo after 5 attempts")

    def set_pulse(self, values):
        _, scale, pulse_s = self.speed_modes[self.speed_mode_index]
        with self.lock:
            self.command = [scale * value for value in values]
            # The publisher timer and Servo both run on Gazebo time. Using a
            # wall-clock deadline here makes a pulse expire before the next
            # simulation tick whenever camera rendering lowers real-time
            # factor. Keep the mode-specific pulse in *simulation* seconds.
            self.command_deadline_ros = (
                self.get_clock().now().nanoseconds * 1e-9 + pulse_s
            )

    def set_held_motion_keys(self, keys):
        """Enable GUI-style simultaneous held-key control."""
        with self.lock:
            self.continuous_input = True
            self.held_motion_keys = set(keys)

    def continuous_command(self):
        motions = {
            "w": (1, 0, 0, 0, 0, 0),
            "s": (-1, 0, 0, 0, 0, 0),
            "a": (0, 1, 0, 0, 0, 0),
            "d": (0, -1, 0, 0, 0, 0),
            "r": (0, 0, 1, 0, 0, 0),
            "f": (0, 0, -1, 0, 0, 0),
            "i": (0, 0, 0, 1, 0, 0),
            "k": (0, 0, 0, -1, 0, 0),
            "j": (0, 0, 0, 0, 1, 0),
            "l": (0, 0, 0, 0, -1, 0),
            "q": (0, 0, 0, 0, 0, 1),
            "e": (0, 0, 0, 0, 0, -1),
        }
        values = [0.0] * 6
        for key in self.held_motion_keys:
            for index, value in enumerate(motions.get(key, ())):
                values[index] += value
        # A diagonal command keeps the selected linear/angular speed instead
        # of becoming sqrt(2) or sqrt(3) faster than a single-axis command.
        for start in (0, 3):
            norm = math.sqrt(sum(value * value for value in values[start:start + 3]))
            if norm > 1.0:
                for index in range(start, start + 3):
                    values[index] /= norm
        _, scale, _ = self.speed_modes[self.speed_mode_index]
        return [scale * value for value in values]

    def publish_command(self):
        now = time.monotonic()
        if now <= self.gripper_deadline:
            if (
                now >= self.gripper_publish_at
                and not self.gripper_trajectory_published
            ):
                self.publish_gripper_target()
                self.gripper_trajectory_published = True
            moved = abs(self.measured_gripper - self.gripper_motion_start) > 0.02
            at_target = abs(self.measured_gripper - self.gripper_target) <= 0.02
            stalled_after_motion = (
                moved
                and self.gripper_trajectory_published
                and abs(self.measured_gripper_velocity) < 0.005
            )
            if at_target or stalled_after_motion:
                if self.gripper_stall_since is None:
                    self.gripper_stall_since = now
                elif now - self.gripper_stall_since >= 0.30:
                    self.gripper_deadline = 0.0
            else:
                self.gripper_stall_since = None
        # Arm Servo uses an independent controller, so it continues below even
        # while the gripper trajectory is active.
        with self.lock:
            ros_now = self.get_clock().now().nanoseconds * 1e-9
            if self.continuous_input:
                values = self.continuous_command()
            else:
                values = (
                    self.command.copy()
                    if ros_now <= self.command_deadline_ros
                    else [0.0] * 6
                )
        stamp = self.get_clock().now().to_msg()
        if self.control_mode == "joint":
            msg = JointJog()
            msg.header.stamp = stamp
            msg.header.frame_id = "base_link"
            msg.joint_names = ARM_JOINTS
            msg.velocities = values
            self.joint_jog_pub.publish(msg)
        else:
            msg = TwistStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "base_link"
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = values[:3]
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = values[3:]
            self.twist_pub.publish(msg)

    def publish_event(self, name):
        self.event_pub.publish(String(data=name))

    def joint_state_cb(self, msg):
        try:
            self.measured_gripper = float(
                msg.position[msg.name.index(GRIPPER_JOINT)]
            )
            if not self.gripper_state_initialized:
                self.gripper_closed = self.measured_gripper > 0.20
                self.gripper_state_initialized = True
            if len(msg.velocity) == len(msg.name):
                self.measured_gripper_velocity = float(
                    msg.velocity[msg.name.index(GRIPPER_JOINT)]
                )
        except (ValueError, IndexError):
            pass

    def toggle_gripper(self):
        self.gripper_closed = not self.gripper_closed
        self.gripper_target = (
            GRIPPER_CLOSED_RAD if self.gripper_closed else GRIPPER_OPEN_RAD
        )
        now = time.monotonic()
        self.gripper_publish_at = now + 0.15
        # Gazebo can run below real-time with two cameras. Sixty wall seconds is
        # only a watchdog; normal completion is detected from measured target
        # arrival or the simulated motor-current/contact stall.
        self.gripper_deadline = now + 60.0
        self.gripper_trajectory_published = False
        self.gripper_motion_start = self.measured_gripper
        self.gripper_stall_since = None
        self.servo_restart_pending = False
        self.publish_event("gripper_closed" if self.gripper_closed else "gripper_open")

    def publish_gripper_target(self):
        msg = JointTrajectory()
        msg.joint_names = [GRIPPER_JOINT]
        start = self.measured_gripper
        # One simulated second keeps a smooth continuous 0.0<->0.8 trajectory
        # but avoids blocking arm teleoperation for ~15 wall seconds when
        # camera rendering lowers Gazebo to ~0.2 real-time factor.
        duration_s = 1.0
        point_count = 101
        for index in range(1, point_count + 1):
            alpha = index / point_count
            point = JointTrajectoryPoint()
            point.positions = [start + alpha * (self.gripper_target - start)]
            elapsed_ns = int(alpha * duration_s * 1e9)
            point.time_from_start.sec = elapsed_ns // 1_000_000_000
            point.time_from_start.nanosec = elapsed_ns % 1_000_000_000
            msg.points.append(point)
        self.gripper_pub.publish(msg)

    def advance_stage(self):
        self.stage_index = min(self.stage_index + 1, len(self.stages) - 1)
        stage = self.stages[self.stage_index]
        self.stage_pub.publish(String(data=stage))
        self.get_logger().info(f"stage={stage}")

    def set_stage(self, index):
        self.stage_index = max(0, min(index, len(self.stages) - 1))
        stage = self.stages[self.stage_index]
        self.stage_pub.publish(String(data=stage))
        self.get_logger().info(f"stage={stage}")

    def toggle_control_mode(self):
        with self.lock:
            self.held_motion_keys.clear()
            self.command = [0.0] * 6
            self.command_deadline_ros = 0.0
            self.control_mode = (
                "joint" if self.control_mode == "cartesian" else "cartesian"
            )
        self.get_logger().info(f"control mode={self.control_mode}")

    def handle_key(self, key):
        motions = {
            "w": [1, 0, 0, 0, 0, 0],
            "s": [-1, 0, 0, 0, 0, 0],
            "a": [0, 1, 0, 0, 0, 0],
            "d": [0, -1, 0, 0, 0, 0],
            "r": [0, 0, 1, 0, 0, 0],
            "f": [0, 0, -1, 0, 0, 0],
            "i": [0, 0, 0, 1, 0, 0],
            "k": [0, 0, 0, -1, 0, 0],
            "j": [0, 0, 0, 0, 1, 0],
            "l": [0, 0, 0, 0, -1, 0],
            "q": [0, 0, 0, 0, 0, 1],
            "e": [0, 0, 0, 0, 0, -1],
        }
        if key in motions:
            self.set_pulse(motions[key])
        elif key == "m":
            self.speed_mode_index = (
                self.speed_mode_index + 1
            ) % len(self.speed_modes)
            mode, scale, pulse_s = self.speed_modes[self.speed_mode_index]
            self.get_logger().info(
                f"speed mode={mode}, command scale={scale:.2f}, "
                f"pulse={pulse_s:.2f}s sim"
            )
        elif key == "g":
            self.toggle_control_mode()
        elif key == " ":
            self.toggle_gripper()
        elif key == "t":
            self.advance_stage()
        elif key in "1234567":
            self.set_stage(int(key) - 1)
        elif key == "p":
            self.publish_event("payload_reference")
        elif key == "b":
            self.publish_event("begin")
        elif key == "v":
            self.publish_event("success")
        elif key == "x":
            self.publish_event("discard")
        elif key == "\x1b":
            with self.lock:
                self.command = [0.0] * 6
                self.command_deadline_ros = 0.0
            self.publish_event("emergency_stop")
            self.running = False


def main():
    if not sys.stdin.isatty():
        raise RuntimeError("keyboard teleop requires an interactive terminal (TTY)")
    rclpy.init()
    node = KeyboardTeleop()
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        node.start_servo()
        node.stage_pub.publish(String(data=node.stages[0]))
        print(
            "\nUR3 keyboard teleop\n"
            "  W/S X, A/D Y, R/F Z | I/K roll, J/L pitch, Q/E yaw\n"
            "  M normal/coarse/precision | SPACE open/close | T next stage\n"
            "  1..7 set stage | P capture lifted-payload FT reference\n"
            "  B begin recording | V success | X discard | ESC stop\n"
        )
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.0)
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if ready:
                node.handle_key(sys.stdin.read(1).lower())
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

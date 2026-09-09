#!/usr/bin/env python3
"""Return the split-controller teleop robot safely to its collection home.

This process runs only after the episode recorder has stopped, so none of the
release, retract, or home motion can enter the saved demonstration.
"""

import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
# Image-left tabletop HOME.  Only shoulder pan differs from the canonical
# downward-facing pose, so the gripper keeps pointing vertically downward.
HOME = [-1.254, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]
SAFE_TOOL_Z = 1.14


class ReturnHome(Node):
    def __init__(self):
        super().__init__(
            "pap_moe_teleop_return_home",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.joints = None
        self.reset_pub = self.create_publisher(
            Bool, "/pap_moe/teleop_reset_active", 10
        )
        self.twist_pub = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10
        )
        self.arm_pub = self.create_publisher(
            Float64MultiArray, "/arm_servo_controller/commands", 10
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            "/gripper_trajectory_controller/joint_trajectory",
            10,
        )
        self.start_client = self.create_client(Trigger, "/servo_node/start_servo")
        self.stop_client = self.create_client(Trigger, "/servo_node/stop_servo")
        self.create_subscription(JointState, "/joint_states", self.on_joints, 20)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def on_joints(self, msg):
        try:
            self.joints = [float(msg.position[msg.name.index(j)]) for j in ARM_JOINTS]
        except (ValueError, IndexError):
            return

    def spin_wall(self, duration):
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def sim_time(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def call(self, client, name):
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"{name} service unavailable")
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError(f"{name} failed")

    def set_reset_active(self, active, repeats=10):
        for _ in range(repeats):
            self.reset_pub.publish(Bool(data=active))
            self.spin_wall(0.03)

    def open_gripper(self):
        msg = JointTrajectory()
        msg.joint_names = [GRIPPER_JOINT]
        point = JointTrajectoryPoint()
        point.positions = [0.0]
        point.time_from_start.sec = 1
        msg.points = [point]
        self.gripper_pub.publish(msg)

    def tool_z(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link", "tool0", rclpy.time.Time()
            )
            return float(transform.transform.translation.z)
        except TransformException:
            return None

    def retract(self):
        # Open first so a discarded grasp cannot drag a peg through the scene.
        self.open_gripper()
        self.spin_wall(0.25)
        deadline_wall = time.monotonic() + 30.0
        start_sim = self.sim_time()
        initial_z = self.tool_z()
        target_z = SAFE_TOOL_Z if initial_z is None else max(SAFE_TOOL_Z, initial_z)
        while rclpy.ok() and time.monotonic() < deadline_wall:
            rclpy.spin_once(self, timeout_sec=0.01)
            current_z = self.tool_z()
            if current_z is not None and current_z >= target_z - 0.004:
                break
            if self.sim_time() - start_sim >= 2.0:
                break
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.z = 0.22
            self.twist_pub.publish(msg)
        stop = TwistStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.header.frame_id = "base_link"
        for _ in range(5):
            self.twist_pub.publish(stop)
            self.spin_wall(0.02)

    def interpolate_home(self):
        if self.joints is None:
            raise RuntimeError("joint_states unavailable")
        start = list(self.joints)
        maximum_delta = max(abs(goal - value) for goal, value in zip(HOME, start))
        duration = max(2.0, min(5.0, maximum_delta / 0.8))
        start_sim = self.sim_time()
        deadline_wall = time.monotonic() + 60.0
        while rclpy.ok() and time.monotonic() < deadline_wall:
            rclpy.spin_once(self, timeout_sec=0.01)
            elapsed = max(0.0, self.sim_time() - start_sim)
            alpha = min(1.0, elapsed / duration)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            target = [a + smooth * (b - a) for a, b in zip(start, HOME)]
            self.arm_pub.publish(Float64MultiArray(data=target))
            if alpha >= 1.0:
                break
        for _ in range(20):
            self.arm_pub.publish(Float64MultiArray(data=HOME))
            self.spin_wall(0.02)
        if self.joints is None or max(abs(a - b) for a, b in zip(self.joints, HOME)) > 0.04:
            raise RuntimeError("arm did not reach collection home")

    def run(self):
        deadline = time.monotonic() + 15.0
        while self.joints is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.joints is None:
            raise RuntimeError("joint_states unavailable")
        self.set_reset_active(True)
        try:
            self.retract()
            self.call(self.stop_client, "stop_servo")
            self.interpolate_home()
            self.open_gripper()
            self.spin_wall(0.3)
            self.call(self.start_client, "start_servo")
        finally:
            self.set_reset_active(False)


def main():
    rclpy.init()
    node = ReturnHome()
    try:
        node.run()
        print("HOME_READY: arm at collection home, gripper open")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

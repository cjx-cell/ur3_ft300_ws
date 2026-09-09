#!/usr/bin/env python3
"""Exercise the simulated 2F-85 and reject any broken mimic linkage."""

import argparse
import math
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


MASTER = "robotiq_85_left_knuckle_joint"
JOINTS = (
    MASTER,
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
)
MULTIPLIERS = np.asarray((1.0, -1.0, 1.0, -1.0, -1.0, 1.0))


class SymmetryTest(Node):
    def __init__(self, error_limit: float):
        super().__init__("robotiq_mimic_symmetry_test")
        self.error_limit = error_limit
        self.latest = None
        self.max_error = 0.0
        self.samples = 0
        self.create_subscription(JointState, "/joint_states", self._state, 50)
        self.client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

    def _state(self, msg: JointState) -> None:
        try:
            values = np.asarray(
                [msg.position[msg.name.index(name)] for name in JOINTS],
                dtype=np.float64,
            )
        except (ValueError, IndexError):
            return
        expected = values[0] * MULTIPLIERS
        error = float(np.max(np.abs(values - expected)))
        self.latest = values
        self.max_error = max(self.max_error, error)
        self.samples += 1

    def command(self, position: float, duration_s: float) -> bool:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = [MASTER]
        point = JointTrajectoryPoint()
        point.positions = [position]
        nanoseconds = int(round(duration_s * 1e9))
        point.time_from_start.sec = nanoseconds // 1_000_000_000
        point.time_from_start.nanosec = nanoseconds % 1_000_000_000
        goal.trajectory.points = [point]
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result() is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--duration", type=float, default=1.8)
    parser.add_argument("--error-limit", type=float, default=0.01)
    args = parser.parse_args()
    rclpy.init()
    node = SymmetryTest(args.error_limit)
    try:
        if not node.client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError("joint trajectory action is unavailable")
        deadline = time.monotonic() + 10.0
        while node.latest is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest is None:
            raise RuntimeError("six physical gripper joint states are unavailable")
        for cycle in range(args.cycles):
            for target in (0.8, 0.0):
                if not node.command(target, args.duration):
                    raise RuntimeError(
                        f"trajectory rejected at cycle {cycle}, target {target}"
                    )
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.latest is None or not np.isfinite(node.latest).all():
                    raise RuntimeError("non-finite gripper state")
                if node.max_error > args.error_limit:
                    raise RuntimeError(
                        f"mimic error {node.max_error:.6f} rad exceeds "
                        f"{args.error_limit:.6f} rad"
                    )
        endpoint_error = max(
            abs(float(node.latest[0])),
            float(np.max(np.abs(node.latest[1:]))),
        )
        if not math.isfinite(endpoint_error) or endpoint_error > args.error_limit:
            raise RuntimeError(
                f"open endpoint error {endpoint_error:.6f} rad exceeds limit"
            )
        print(
            f"PASS cycles={args.cycles} samples={node.samples} "
            f"max_mimic_error_rad={node.max_error:.8f} "
            f"open_endpoint_error_rad={endpoint_error:.8f}"
        )
        return 0
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

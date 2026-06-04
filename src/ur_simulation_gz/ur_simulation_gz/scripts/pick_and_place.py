#!/usr/bin/env python3
"""Pick-and-place using joint-space targets (Cartesian planning unreliable via pymoveit2)."""

from threading import Thread
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from pymoveit2 import MoveIt2


ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
GRIPPER_JOINT = ["robotiq_85_left_knuckle_joint"]

# Joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
HOME     = [ 0.0,   -1.57,    0.0,  -1.57,    0.0,   0.0  ]
ABOVE    = [-1.834, -1.883, -1.128, -1.646,  1.572,  0.297]
GRASP    = [-1.803, -1.942, -1.579, -1.136,  1.570, -0.202]
LIFT     = [-1.803, -1.856, -1.293, -1.508,  1.570, -0.202]
ABOVE_B  = [-0.761, -1.910, -1.230, -1.545,  1.523,  0.839]
PLACE    = [-0.765, -1.925, -1.358, -1.402,  1.523,  0.835]

GRASP_CLOSE = [0.65]
GRASP_OPEN  = [0.0]


def main():
    rclpy.init()
    node = Node("pick_and_place")
    cb = ReentrantCallbackGroup()

    arm = MoveIt2(
        node=node, joint_names=ARM_JOINTS,
        base_link_name="base_link", end_effector_name="robotiq_85_base_link",
        group_name="ur_manipulator", callback_group=cb,
    )
    arm.max_velocity = 0.3
    arm.max_acceleration = 0.3

    gripper = MoveIt2(
        node=node, joint_names=GRIPPER_JOINT,
        base_link_name="base_link", end_effector_name="robotiq_85_base_link",
        group_name="gripper", callback_group=cb,
    )
    gripper.max_velocity = 0.3

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    t = Thread(target=executor.spin, daemon=True)
    t.start()
    node.create_rate(1.0).sleep()

    def go(grp, target, label):
        node.get_logger().info(label)
        grp.move_to_configuration(target)
        grp.wait_until_executed()

    go(gripper, GRASP_OPEN, "Open")
    go(arm, HOME, "Home")
    go(arm, ABOVE, "Above block")
    go(arm, GRASP, "Grasp block")
    go(gripper, GRASP_CLOSE, "Close")
    go(arm, LIFT, "Lift")
    go(arm, ABOVE_B, "Above bowl")
    go(arm, PLACE, "Place in bowl")
    go(gripper, GRASP_OPEN, "Release")
    go(arm, ABOVE_B, "Retract")
    go(arm, HOME, "Home")

    node.get_logger().info("Pick-and-place complete!")
    rclpy.shutdown()
    t.join()


if __name__ == "__main__":
    main()

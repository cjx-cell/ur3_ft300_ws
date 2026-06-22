#!/usr/bin/env python3
"""Query move_group /compute_ik to get joint values for pick-and-place."""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.srv import GetPositionIK


def ik(node, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    cli = node.create_client(GetPositionIK, "/compute_ik")
    cli.wait_for_service(timeout_sec=5.0)
    req = GetPositionIK.Request()
    req.ik_request.group_name = "ur_manipulator"
    req.ik_request.pose_stamped.header.frame_id = "base_link"
    req.ik_request.pose_stamped.pose = Pose(
        position=Point(x=float(x), y=float(y), z=float(z)),
        orientation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
    )
    req.ik_request.timeout.sec = 5
    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if future.result() and future.result().error_code.val == 1:
        vals = [round(v, 3) for v in future.result().solution.joint_state.position[:6]]
        return vals
    return None


def main():
    rclpy.init()
    node = Node("compute_ik")
    # Targets relative to base_link (robot at world z=0.76)
    # Block top at world 0.815 → rel 0.055. Bowl surface at world 0.775 → rel 0.015
    targets = {
        "above_block":  (0.20, 0.35, 0.20),
        "grasp_block":  (0.20, 0.35, 0.08),
        "above_bowl":   (-0.20, 0.35, 0.20),
        "place_bowl":   (-0.20, 0.35, 0.08),
    }
    for name, (x, y, z) in targets.items():
        for label, q in [("id", [0,0,0,1]), ("fwd", [0,0.707,0,0.707])]:
            r = ik(node, x, y, z, q[0], q[1], q[2], q[3])
            if r:
                print(f"{name:14s} [{label:3s}] = {r}")
                break
        else:
            print(f"{name:14s} = FAILED")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

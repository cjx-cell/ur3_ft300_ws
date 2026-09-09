"""Require live observations, advancing clock, active controllers and action server."""
import json
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from sensor_msgs.msg import Image, JointState
from rosgraph_msgs.msg import Clock


def main():
    rclpy.init()
    node = rclpy.create_node('workspace50_readiness_probe')
    seen = {}
    clocks = set()
    subs = []
    for topic, cls in [('/joint_states', JointState), ('/wrist_camera/color/image_raw', Image),
                       ('/global_camera/color/image_raw', Image)]:
        subs.append(node.create_subscription(cls, topic, lambda msg, topic=topic: seen.update({topic: time.monotonic()}), qos_profile_sensor_data))
    subs.append(node.create_subscription(Clock, '/clock', lambda msg: clocks.add((msg.clock.sec, msg.clock.nanosec)), qos_profile_sensor_data))
    action = ActionClient(node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
    service = node.create_client(ListControllers, '/controller_manager/list_controllers')
    active, future = set(), None
    required = {'joint_trajectory_controller', 'joint_state_broadcaster', 'force_torque_sensor_broadcaster'}
    next_query = 0.0
    deadline = time.monotonic() + 100
    ready = False
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if (future is None and not required <= active and time.monotonic() >= next_query
                    and service.service_is_ready()):
                future = service.call_async(ListControllers.Request())
                next_query = time.monotonic() + 1.0
            if future is not None and future.done():
                active = {c.name for c in future.result().controller if c.state == 'active'}
                future = None
            ready = (len(seen) == 3 and all(time.monotonic()-t < 3 for t in seen.values())
                     and len(clocks) >= 2 and action.server_is_ready()
                     and required <= active and future is None)
            if ready:
                break
        print(json.dumps({'ready': ready, 'active_controllers': sorted(active), 'topics': sorted(seen),
                          'clock_advanced': len(clocks) >= 2, 'action_server_ready': action.server_is_ready()}), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if ready else 3


if __name__ == '__main__':
    raise SystemExit(main())

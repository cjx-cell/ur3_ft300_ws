"""Print the closest self-collision link pair for the live Gazebo state."""

from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "ur3_ft300_robotiq_2f85",
        package_name="ur3_ft300_moveit_config",
    ).to_moveit_configs()
    return LaunchDescription(
        [
            Node(
                package="ur_simulation_gz",
                executable="pap_moe_self_collision_diagnostic",
                output="screen",
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    {"use_sim_time": True},
                ],
            )
        ]
    )

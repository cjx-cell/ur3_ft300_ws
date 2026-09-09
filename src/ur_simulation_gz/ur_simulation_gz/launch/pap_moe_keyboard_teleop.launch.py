"""Start MoveIt Servo and the PAP-MoE keyboard teleoperator.

Gazebo and robot_state_publisher must already be running.  This launch uses
the same MoveIt model as the scripted demonstrator, but never starts the old
scripted collector.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "ur3_ft300_robotiq_2f85",
        package_name="ur3_ft300_moveit_config",
    ).to_moveit_configs()

    config_path = os.path.join(
        get_package_share_directory("ur_simulation_gz"),
        "config",
        "pap_moe_ur3_servo.yaml",
    )
    with open(config_path, encoding="utf-8") as stream:
        servo_params = {"moveit_servo": yaml.safe_load(stream)}

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            servo_params,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": True},
        ],
    )

    keyboard_node = Node(
        package="ur_simulation_gz",
        executable="pap_moe_keyboard_teleop.py",
        name="pap_moe_keyboard_teleop",
        output="screen",
        prefix=LaunchConfiguration("terminal_prefix"),
        condition=IfCondition(LaunchConfiguration("launch_keyboard")),
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_keyboard",
                default_value="false",
                description=(
                    "Launch the keyboard child process. Normally keep false "
                    "and run it with ros2 run in a separate interactive terminal."
                ),
            ),
            DeclareLaunchArgument(
                "terminal_prefix",
                default_value="",
                description=(
                    "Optional terminal launcher, e.g. 'xterm -e'. Leave empty "
                    "when this launch is run in its own terminal."
                ),
            ),
            servo_node,
            keyboard_node,
        ]
    )

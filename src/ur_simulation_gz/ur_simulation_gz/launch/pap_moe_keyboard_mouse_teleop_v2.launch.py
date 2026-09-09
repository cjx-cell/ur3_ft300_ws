"""Start MoveIt Servo for the independent keyboard/mouse v2 frontend.

Gazebo must already be running.  By default the focused GUI is started as a
separate process by this launch file; set launch_gui:=false when launching it
manually from another terminal.
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

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_gui",
                default_value="true",
                description="Start the focused keyboard/mouse v2 control window.",
            ),
            Node(
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
            ),
            Node(
                package="ur_simulation_gz",
                executable="pap_moe_keyboard_mouse_teleop_v2.py",
                name="pap_moe_keyboard_mouse_teleop_v2",
                output="screen",
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("launch_gui")),
            ),
        ]
    )

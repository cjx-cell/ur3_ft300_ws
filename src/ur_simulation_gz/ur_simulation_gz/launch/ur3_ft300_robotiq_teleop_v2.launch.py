"""Gazebo UR3 setup with independent arm Servo and gripper controllers."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gz"), "launch", "ur3_ft300_robotiq.launch.py"]
            )
        ),
        launch_arguments={
            "gazebo_gui": "true",
            # The collection launcher starts ur3_ft300_moveit_config's RViz
            # separately.  That config contains the wrist/global Image
            # displays; this package's view_robot.rviz does not.
            "launch_rviz": "false",
            "initial_joint_controller": "arm_servo_controller",
        }.items(),
    )
    gripper = TimerAction(
        period=9.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "gripper_trajectory_controller",
                    "--controller-manager",
                    "/controller_manager",
                    "--controller-manager-timeout",
                    "120",
                ],
                output="screen",
            )
        ],
    )
    return LaunchDescription([base, gripper])

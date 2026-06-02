from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "ur3_ft300_robotiq_2f85", package_name="ur3_ft300_moveit_config"
    ).to_moveit_configs()

    # Standard move_group defaults (from moveit_configs_utils)
    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "monitor_dynamics": False,
    }

    move_group_params = [
        moveit_config.to_dict(),
        move_group_configuration,
        {"use_sim_time": True},
    ]

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        parameters=move_group_params,
        arguments=["--ros-args", "--log-level", "tf2_buffer:=error"],
        output="screen",
    )

    return LaunchDescription([move_group_node])

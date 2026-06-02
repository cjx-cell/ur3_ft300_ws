from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "ur3_ft300_robotiq_2f85", package_name="ur3_ft300_moveit_config"
    ).to_moveit_configs()

    rviz_config_file = str(moveit_config.package_path / "config" / "moveit.rviz")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.to_dict(),
            # NO-OP: marker scale is a UI preference, not a launch param
        ],
        output="screen",
    )

    return LaunchDescription([rviz_node])

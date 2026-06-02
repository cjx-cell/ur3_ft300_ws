# Copyright (c) 2024 Integrated UR3 + FT300 + Robotiq + Cameras
# Humble + Gazebo Fortress (Gazebo Sim) launch
#
# Key points:
# - Use gz_ros2_control / GazeboSimSystem
# - Delay controller spawners to avoid startup race
# - Keep /controller_manager as default but allow override
# - Keep RViz optional
#
# Example:
#   ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py
#   ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py launch_rviz:=true

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # Core arguments
    ur_type = LaunchConfiguration("ur_type")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")

    runtime_config_package = LaunchConfiguration("runtime_config_package")
    controllers_file = LaunchConfiguration("controllers_file")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    prefix = LaunchConfiguration("prefix")

    start_joint_controller = LaunchConfiguration("start_joint_controller")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")
    controller_manager_name = LaunchConfiguration("controller_manager_name")
    controller_start_delay = LaunchConfiguration("controller_start_delay")

    launch_rviz = LaunchConfiguration("launch_rviz")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    world_file = LaunchConfiguration("world_file")

    # Gripper / FT options
    gripper_use_fake_hardware = LaunchConfiguration("gripper_use_fake_hardware")
    ft_sensor_use_fake_mode = LaunchConfiguration("ft_sensor_use_fake_mode")

    # MoveIt (external: ros2 launch ur3_ft300_moveit demo.launch.py)
    launch_moveit = LaunchConfiguration("launch_moveit")
    use_sim_time = LaunchConfiguration("use_sim_time")

    initial_joint_controllers = PathJoinSubstitution(
        [FindPackageShare(runtime_config_package), "config", controllers_file]
    )

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(description_package), "rviz", "view_robot.rviz"]
    )

    # Generate robot description
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare(description_package), "urdf", description_file]
            ),
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
            " ",
            "name:=ur",
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "prefix:=",
            prefix,
            " ",
            "sim_ignition:=false",
            " ",
            "use_fake_hardware:=true",
            " ",
            "gripper_use_fake_hardware:=",
            gripper_use_fake_hardware,
            " ",
            "ft_sensor_use_fake_mode:=",
            ft_sensor_use_fake_mode,
            " ",
            "simulation_controllers:=",
            initial_joint_controllers,
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # Robot state publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    # RViz (simple, no MoveIt panel)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(launch_rviz),
        parameters=[{"use_sim_time": True}],
    )

    # MoveIt: launch externally via ros2 launch ur3_ft300_moveit demo.launch.py

    # Gazebo launch
    gz_launch_description_with_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": [" -r -v 2 ", world_file]}.items(),
        condition=IfCondition(gazebo_gui),
    )

    gz_launch_description_without_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": [" -s -v 2 ", world_file]}.items(),
        condition=UnlessCondition(gazebo_gui),
    )

    # Spawn robot in Gazebo
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-world",
            "simulation_world",
            "-string",
            robot_description_content,
            "-name",
            "ur3_ft300_robotiq",
            "-allow_renaming",
            "true",
        ],
    )

    # /clock bridge
    gz_sim_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ],
        output="screen",
    )

    # Camera image bridge
    gz_image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=[
            "/world/simulation_world/model/ur3_ft300_robotiq/link/wrist_3_link/sensor/wrist_camera_sensor/image",
            "/world/simulation_world/model/ur3_ft300_robotiq/link/global_camera_mount/sensor/global_camera_sensor/image",
        ],
        output="screen",
    )

    # Controller spawners
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "-c",
            controller_manager_name,
            "--controller-manager-timeout",
            "60",
        ],
        output="screen",
    )

    initial_joint_controller_spawner_started = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            initial_joint_controller,
            "-c",
            controller_manager_name,
            "--controller-manager-timeout",
            "60",
        ],
        condition=IfCondition(start_joint_controller),
        output="screen",
    )

    initial_joint_controller_spawner_stopped = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            initial_joint_controller,
            "-c",
            controller_manager_name,
            "--controller-manager-timeout",
            "60",
            "--stopped",
        ],
        condition=UnlessCondition(start_joint_controller),
        output="screen",
    )

    ft_sensor_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "force_torque_sensor_broadcaster",
            "-c",
            controller_manager_name,
            "--controller-manager-timeout",
            "60",
        ],
        output="screen",
    )

    # Delay controller loading to avoid race with gz_ros2_control startup
    delayed_controller_spawners = TimerAction(
        period=controller_start_delay,
        actions=[
            joint_state_broadcaster_spawner,
            initial_joint_controller_spawner_stopped,
            initial_joint_controller_spawner_started,
            ft_sensor_spawner,
        ],
    )

    return [
        SetEnvironmentVariable(
            name="GAZEBO_ROS2_CONTROL_USE_PARAM_SERVER",
            value="true",
        ),
        # Force NVIDIA GPU for Gazebo rendering (RTX 5080)
        SetEnvironmentVariable(
            name="__EGL_VENDOR_LIBRARY_FILENAMES",
            value="/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        ),
        # Gazebo model path for pick-and-place objects
        SetEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=os.path.expanduser("~/.gazebo/models"),
        ),
        robot_state_publisher_node,
        gz_launch_description_with_gui,
        gz_launch_description_without_gui,
        gz_spawn_entity,
        gz_sim_bridge,
        gz_image_bridge,
        delayed_controller_spawners,
        rviz_node,
    ]


def generate_launch_description():
    # Add apt-installed package share directories to Gazebo resource path
    apt_share = "/opt/ros/humble/share"
    current_gz_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    if apt_share not in current_gz_path:
        os.environ["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
            [p for p in [apt_share, current_gz_path] if p]
        )

    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            description="Type/series of used UR robot.",
            choices=["ur3", "ur5", "ur10"],
            default_value="ur3",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "runtime_config_package",
            default_value="ur_simulation_gz",
            description='Package with the controller configuration in "config" folder.',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "controllers_file",
            default_value="ur3_ft300_robotiq_controllers.yaml",
            description="YAML file with the controllers configuration.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="ur_simulation_gz",
            description="Description package with robot URDF/XACRO files.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="ur3_ft300_robotiq_2f85.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value='""',
            description="Prefix of the joint names.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "start_joint_controller",
            default_value="true",
            description="Enable headless mode for robot control.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="joint_trajectory_controller",
            description="Robot controller to start.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "controller_manager_name",
            default_value="/controller_manager",
            description="Name of the controller manager ROS node.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "controller_start_delay",
            default_value="30.0",
            description="Delay before starting controller spawners (seconds).",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="false",
            description="Launch RViz?",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "gazebo_gui",
            default_value="true",
            description="Start Gazebo with GUI?",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "world_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("ur_simulation_gz"),
                "worlds",
                "simulation_world.sdf",
            ]),
            description="Gazebo world file path.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "gripper_use_fake_hardware",
            default_value="true",
            description="Use fake hardware interface for gripper in simulation.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "ft_sensor_use_fake_mode",
            default_value="true",
            description="Use fake mode for FT sensor in simulation.",
        )
    )

    # MoveIt-related arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_moveit",
            default_value="false",
            description="Launch MoveIt with move_group and RViz MotionPlanning panel.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_servo",
            default_value="false",
            description="Launch MoveIt Servo for real-time jog control.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_config_package",
            default_value="ur_simulation_gz",
            description="Package with MoveIt SRDF and config files.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_config_file",
            default_value="ur3_ft300_robotiq.srdf.xacro",
            description="MoveIt SRDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock for MoveIt.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "warehouse_sqlite_path",
            default_value=os.path.expanduser("~/.ros/warehouse_ros.sqlite"),
            description="Path to the MoveIt warehouse database.",
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
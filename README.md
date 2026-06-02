# UR3 + FT300 + Robotiq 2F85 — ROS 2 Humble Simulation

Gazebo Fortress simulation of a Universal Robots UR3 collaborative robot equipped with a Robotiq FT300 force-torque sensor and a Robotiq 2F-85 adaptive gripper, with MoveIt2 motion planning and a pick-and-place demo.

## Hardware

| Component | Model |
|-----------|-------|
| Robot arm | Universal Robots UR3 (6-DOF, 3 kg payload, 500 mm reach) |
| Force-torque sensor | Robotiq FT300 (mounted on wrist) |
| Gripper | Robotiq 2F-85 (85 mm stroke, adaptive 2-finger) |
| Vision | Intel RealSense D435i (wrist-mounted) + D415 (global scene camera) |

## Software Stack

| Layer | Technology |
|-------|------------|
| OS | Ubuntu 22.04 |
| ROS 2 | Humble Hawksbill |
| Simulation | Gazebo Fortress (Ignition Gazebo v6) |
| Physics | DART |
| Robot control | `gz_ros2_control` + `joint_trajectory_controller` |
| Motion planning | MoveIt2 with OMPL planner |
| Python API | `pymoveit2` (joint-space targets) |

## Package Structure

```
ur3_ft300_ws/
├── src/
│   ├── ur_simulation_gz/          # Main simulation package (worlds, URDF, launch, scripts)
│   ├── ur3_ft300_moveit_config/   # MoveIt2 configuration (Setup Assistant generated)
│   ├── ur_description/            # UR3 URDF models and meshes
│   ├── ros2_robotiq_gripper/      # Robotiq 2F-85 gripper models and driver
│   ├── rq_fts_ros2_driver/        # Robotiq FT300 force-torque sensor driver
│   ├── serial/                    # Serial communication library (dependency)
│   └── realsense-ros/             # Intel RealSense ROS 2 drivers (optional, COLCON_IGNORE)
├── .gitignore
└── README.md
```

## Prerequisites

```bash
# ROS 2 Humble + Gazebo Fortress
sudo apt install ros-humble-ros-gz ros-humble-moveit ros-humble-ros2-control

# Python dependencies
pip install pymoveit2

# Gazebo model path
export GZ_SIM_RESOURCE_PATH=$HOME/.gazebo/models:$GZ_SIM_RESOURCE_PATH
```

## Build

```bash
cd ~/ur3_ft300_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

> **Note:** `realsense-ros` has a `COLCON_IGNORE` marker and is skipped during build. Remove it if you need camera support.

## Launch Simulation

```bash
# Terminal 1: Start Gazebo with robot + scene
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py

# Terminal 2: Start MoveIt2 motion planning
ros2 launch ur3_ft300_moveit_config move_group.launch.py

# Optional: RViz for visualization and interactive marker control
ros2 launch ur3_ft300_moveit_config moveit_rviz.launch.py
```

## Pick-and-Place Demo

```bash
/usr/bin/python3.10 ~/ur3_ft300_ws/src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place.py
```

The demo sequence:
1. Open gripper → move to HOME
2. Move ABOVE block → close gripper
3. LIFT block → move ABOVE bowl
4. PLACE into bowl → open gripper
5. Retract → return HOME

Joint positions were manually recorded via RViz interactive marker and verified in simulation. Cartesian planning (`move_to_pose`) is unreliable with `pymoveit2` in this setup — joint-space targets via `move_to_configuration()` are used instead.

## Key Configuration Files

| File | Purpose |
|------|---------|
| `src/ur_simulation_gz/ur_simulation_gz/urdf/ur3_ft300_robotiq_2f85.urdf.xacro` | Main robot URDF |
| `src/ur_simulation_gz/ur_simulation_gz/config/ur3_ft300_robotiq_controllers.yaml` | ros2_control JTC + FT sensor config |
| `src/ur_simulation_gz/ur_simulation_gz/worlds/simulation_world.sdf` | Gazebo world (table, block, bowl) |
| `src/ur3_ft300_moveit_config/config/ur3_ft300_robotiq_2f85.srdf` | MoveIt SRDF (groups, chains, poses) |
| `src/ur3_ft300_moveit_config/config/moveit_controllers.yaml` | MoveIt controller mapping |

## Known Issues & Fixes

- **Gripper falling apart in Gazebo**: DART physics does not enforce URDF `<mimic>` joints. Fixed by using `gz_ros2_control` mimic parameters (`<param name="mimic">`) with no command/state interfaces on mimic joints.
- **Friction / slipping**: DART ignores `<mu1>` (non-standard). Use standard `<mu>` in URDF `<ode>` friction blocks. Finger link collision surfaces need explicit friction parameters — not just finger tips.
- **Cartesian planning fails**: `pymoveit2.move_to_pose()` consistently fails. Use joint-space `move_to_configuration()` instead, or CLI `ros2 action send_goal` to JTC.
- **CHOMP planner crash**: CHOMP SEGFAULTs in ROS 2 Humble. Disabled by removing CHOMP planner config from `moveit_config`.
- **Python version**: Must use `/usr/bin/python3.10` — ROS 2 Humble requires Python 3.10.

## License

MIT. See individual `src/` packages for their respective licenses.

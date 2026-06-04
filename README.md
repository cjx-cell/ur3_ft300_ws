# UR3 + FT300 + Robotiq 2F85 — Pi0 Fine-Tuning for Pick-and-Place

Gazebo Fortress simulation of a UR3 collaborative robot with Robotiq FT300 F/T sensor and 2F-85 gripper, with **Pi0 vision-language-action model** fine-tuning for "pick up the red cube and place it into the bowl".

## Hardware

| Component | Model |
|-----------|-------|
| Robot arm | Universal Robots UR3 (6-DOF, 3 kg payload, 500 mm reach) |
| Force-torque sensor | Robotiq FT300 (wrist-mounted) |
| Gripper | Robotiq 2F-85 (85 mm stroke, adaptive 2-finger) |
| Vision | Wrist camera (Realsense D435i) + Global camera (Realsense D415) |

## Software Stack

| Layer | Technology |
|-------|------------|
| OS | Ubuntu 22.04 |
| ROS 2 | Humble Hawksbill |
| Simulation | Gazebo Fortress (Ignition Gazebo v6) |
| Physics | DART |
| Robot control | `gz_ros2_control` + `joint_trajectory_controller` |
| Motion planning | MoveIt2 (data collection only) |
| VLA Model | [Pi0](https://github.com/Physical-Intelligence/openpi) (PaliGemma 2B + Action Expert 300M) |
| Training framework | [LeRobot](https://github.com/huggingface/lerobot) |

## Directory Structure

```
ur3_ft300_ws/
├── src/
│   ├── ur_simulation_gz/          # Main simulation package (worlds, URDF, launch, scripts)
│   ├── ur3_ft300_moveit_config/   # MoveIt2 configuration
│   ├── ur_description/            # UR3 URDF models
│   ├── ros2_robotiq_gripper/      # Robotiq 2F-85 gripper
│   ├── rq_fts_ros2_driver/        # Robotiq FT300 driver
│   ├── serial/                    # Serial library (dependency)
│   └── realsense-ros/             # Realsense drivers (COLCON_IGNORE)
├── ai-models/                     # ★ Pi0 models & training data ★
│   ├── lerobot/pi0/               # Base Pi0 pre-trained weights
│   ├── paligemma_tokenizer/       # PaliGemma tokenizer
│   ├── ur3_pick_place_raw/        # Raw collected trajectory data (.npz)
│   └── train_full_vlm.py          # Full VLM fine-tuning script (A100)
├── outputs/train/                 # Training checkpoints
│   └── pi0_ur3_v3/                # Action-expert-only fine-tuned checkpoint
├── .gitignore
└── README.md
```

## Prerequisites

### ROS 2 + Gazebo
```bash
sudo apt install ros-humble-ros-gz ros-humble-moveit ros-humble-ros2-control ros-humble-cv-bridge
pip install pymoveit2

export GZ_SIM_RESOURCE_PATH=$HOME/.gazebo/models:$GZ_SIM_RESOURCE_PATH
```

### LeRobot + Pi0 (conda environment)
```bash
conda create -n pi0-env python=3.10
conda activate pi0-env
pip install lerobot safetensors torch torchvision transformers accelerate
```

## Build

```bash
cd ~/ur3_ft300_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Data Collection

### 1. Launch Simulation
```bash
# Terminal 1: Gazebo
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py
```

### 2. Collect Trajectories
```bash
# Terminal 2: Record pick-and-place episodes
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_record_pick_place.py --episodes 10
```
Episodes saved to `ai-models/ur3_pick_place_raw/` as `.npz` files.

### 3. Convert to LeRobot Format
```bash
conda activate pi0-env
python3 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_convert_to_lerobot.py \
    --input ai-models/ur3_pick_place_raw
```

### 4. Visualize Data (optional)
```bash
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/visualize_dataset.py --all
# Images saved to ai-models/ur3_pick_place_raw/trajectory_viz/
```

## Training

### Quick: Action-Expert Only (~15GB VRAM)

Fast but model doesn't learn visual grounding — it memorizes the trajectory pattern.

```bash
conda activate pi0-env
# Use LeRobot training CLI with train_expert_only=true
```

### Full VLM Fine-Tuning (2× A100 80GB Recommended)

Model learns to **see the red cube** and servo to its position.

```bash
# On A100 server
conda activate pi0-env
export WANDB_MODE=disabled
cd ~/ur3_ft300_ws
python3 ai-models/train_full_vlm.py
```

Key config differences:
| Setting | Action-Expert Only | Full VLM |
|---------|-------------------|----------|
| `train_expert_only` | `true` | **`false`** |
| `freeze_vision_encoder` | `false` (frozen by expert-only) | **`false`** (trained) |
| `batch_size` | 2 | 1 (per GPU) |
| `lr` | 2.5e-5 | 2e-5 |
| VRAM required | ~15 GB | ~70+ GB |

## Pi0 Inference (3-Terminal Setup)

Data flows through `/tmp/` files — no MoveIt required during inference.

```
Gazebo cameras → ROS topic → ros_side.py → /tmp/camera{0,1}.npy → pi0_inference.py
Gazebo joints  → /joint_states → ros_side.py → /tmp/joint_state.txt → pi0_inference.py
Pi0 output → /tmp/ur3_action.txt → ros_side.py → FollowJointTrajectory → UR3
```

### Terminal 1: Gazebo Simulation
```bash
cd ~/ur3_ft300_ws
source install/setup.bash
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py
```

### Terminal 2: ROS Communication Bridge
```bash
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_ros_side.py
```

### Terminal 3: Pi0 Inference
```bash
conda activate pi0-env
python3 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_inference.py --mode bf16 --hz 5
```

## Scripts Reference

| Script | Python | Purpose |
|--------|--------|---------|
| `ur3_record_pick_place.py` | 3.10 (ROS) | Collect pick-and-place trajectories |
| `ur3_convert_to_lerobot.py` | pi0-env | Convert .npz → LeRobot dataset |
| `ur3_pi0_ros_side.py` | 3.10 (ROS) | ROS ↔ /tmp/ bridge for inference |
| `ur3_pi0_inference.py` | pi0-env | Pi0 inference loop (5 Hz) |
| `ur3_pi0_test.py` | pi0-env | Single-shot inference test |
| `visualize_dataset.py` | 3.10 | Generate trajectory visualization |
| `fix_checkpoint_keys.py` | pi0-env | Fix safetensors key mismatches |
| `train_full_vlm.py` | pi0-env | Full VLM fine-tuning config |

## Key Configuration Files

| File | Purpose |
|------|---------|
| `src/ur_simulation_gz/ur_simulation_gz/urdf/ur3_ft300_robotiq_2f85.urdf.xacro` | Main robot URDF |
| `src/ur_simulation_gz/ur_simulation_gz/config/ur3_ft300_robotiq_controllers.yaml` | ros2_control JTC config |
| `src/ur_simulation_gz/ur_simulation_gz/worlds/simulation_world.sdf` | Gazebo world (table, block, bowl) |
| `src/ur3_ft300_moveit_config/config/ur3_ft300_robotiq_2f85.srdf` | MoveIt SRDF |
| `outputs/train/pi0_ur3_v3/.../config.json` | Fine-tuned model config |
| `ai-models/train_full_vlm.py` | A100 full VLM training config |

## Known Issues

### Simulation
- **Gripper falling apart in Gazebo**: DART ignores URDF `<mimic>`. Fixed via `gz_ros2_control` mimic params.
- **Friction/slipping**: Use `<mu>` in URDF `<ode>` blocks. Finger links need explicit friction.
- **Cartesian planning fails**: Use joint-space `move_to_configuration()` instead.
- **CHOMP planner crash**: CHOMP SEGFAULTs in Humble. Removed from config.

### Inference
- **Gripper closes, arm lifts before grip**: Training data has instant gripper transitions. Fixed with time-based arm freeze in `ur3_pi0_inference.py` (grip_state state machine).
- **Normalization stats mismatch**: Checkpoint has separate state/action stats in preprocessor/postprocessor `.safetensors` files. Must use correct values for each.
- **Gripper velocity too slow**: Increased knuckle joint PID gains (P=500) and trajectory limit (5.0 rad/s) in `ur3_ft300_robotiq_controllers.yaml`.
- **Python version**: ROS scripts use `/usr/bin/python3.10`. Pi0 scripts use `pi0-env` conda environment.

### Training
- **`train_expert_only: true`** (action-expert only): Model memorizes trajectory, doesn't learn visual grounding. Use `train_expert_only: false` for full VLM fine-tuning (requires A100).
- **Tokenization**: Pi0 uses PaliGemma tokenizer from `ai-models/paligemma_tokenizer/`.

## License

MIT. See individual `src/` packages for their respective licenses.

## Transfer to A100 Server

Large model weights (`.safetensors`, ~40GB) are **gitignored**. Transfer them separately:

```bash
# Option 1: rsync the entire workspace
rsync -avz --progress ~/ur3_ft300_ws/ user@a100-server:~/ur3_ft300_ws/

# Option 2: Download base model from HuggingFace on the A100 server
# (requires HuggingFace token + access to Physical Intelligence's pi0 model)
huggingface-cli download physical-intelligence/pi0 --local-dir ai-models/lerobot/pi0/

# Option 3: Copy only safetensors files
rsync -avz ~/ur3_ft300_ws/ai-models/lerobot/pi0/*.safetensors \
    user@a100-server:~/ur3_ft300_ws/ai-models/lerobot/pi0/
```

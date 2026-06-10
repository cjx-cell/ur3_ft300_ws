# UR3 + FT300 + Robotiq 2F85 — Pi0 Pick-and-Place

Gazebo Fortress simulation of a UR3 collaborative robot with Robotiq FT300 F/T sensor and 2F-85 adaptive gripper, with **Pi0 vision-language-action model** fine-tuning for "pick up the red cube and place it into the bowl".

## Hardware

| Component | Model |
|-----------|-------|
| Robot arm | Universal Robots UR3 (6-DOF, 3 kg payload, 500 mm reach) |
| Force-torque sensor | Robotiq FT300 (wrist-mounted) |
| Gripper | Robotiq 2F-85 (85 mm stroke, adaptive 2-finger) |
| Vision | Realsense D435i (wrist) + D415 (global scene) |

## Software Stack

| Layer | Technology |
|-------|------------|
| OS | Ubuntu 22.04 |
| ROS 2 | Humble Hawksbill |
| Simulation | Gazebo Fortress |
| Physics | DART |
| Control | `gz_ros2_control` + `joint_trajectory_controller` |
| Motion planning | MoveIt2 (data collection only, not needed for inference) |
| VLA Model | Pi0 (PaliGemma 2B + Action Expert 300M) |
| Training | LeRobot |

## Directory Structure

```
ur3_ft300_ws/
├── src/
│   └── ur_simulation_gz/          # Simulation + all scripts
├── ai-models/                     # Model weights, tokenizer, training data
│   ├── lerobot/pi0/               # Base Pi0 pre-trained model
│   ├── paligemma_tokenizer/       # PaliGemma tokenizer
│   └── ur3_pick_place_raw/        # Collected trajectory data (.npz)
├── outputs/train/                 # Fine-tuned checkpoints (gitignored)
└── README.md
```

## Prerequisites

```bash
# ROS 2 + Gazebo
sudo apt install ros-humble-ros-gz ros-humble-moveit ros-humble-ros2-control ros-humble-cv-bridge ros-humble-rqt-image-view
pip install pymoveit2
```

### Pi0 Environment (two options)

**Option A: Editable source install (recommended — allows modifying policy code)**

```bash
# Clone lerobot source
git clone https://github.com/huggingface/lerobot.git ~/lerobot

# Create env
conda create -n pi0-env python=3.12
conda activate pi0-env

# Install lerobot in EDITABLE mode + dependencies
cd ~/lerobot
pip install -e .
pip install safetensors torch torchvision transformers accelerate
```

With editable install (`-e`), any changes you make to `~/lerobot/src/lerobot/` take effect immediately — no reinstall needed.

**Option B: Fixed pip install (inference only, no code changes)**

```bash
conda create -n pi0-env python=3.12
conda activate pi0-env
pip install lerobot safetensors torch torchvision transformers accelerate
```

Use this if you only need to run inference and won't modify policy code.

### Where to modify Pi0 policy code

After editable install, the Pi0 source lives at:

| Module | Path | What it does |
|--------|------|-------------|
| Pi0 config | `~/lerobot/src/lerobot/policies/pi0/configuration_pi0.py` | Model hyperparameters, training flags |
| Pi0 model | `~/lerobot/src/lerobot/policies/pi0/modeling_pi0.py` | Forward pass, action selection, diffusion |
| Pi0 processor | `~/lerobot/src/lerobot/policies/pi0/processor_pi0.py` | Normalization, image preprocessing |
| Training script | `~/lerobot/src/lerobot/scripts/lerobot_train.py` | CLI entry point for training |
| Dataset class | `~/lerobot/src/lerobot/datasets/` | Data loading, augmentation |

Your custom inference/ROS scripts are in this repo:
```
~/ur3_ft300_ws/src/ur_simulation_gz/ur_simulation_gz/scripts/
├── ur3_pi0_inference.py    # Inference loop (modify gripper logic here)
├── ur3_pi0_ros_side.py     # ROS bridge (modify control here)
├── ur3_record_pick_place.py # Data collection
└── ...
```

### Full new-server setup

```bash
# 1. ROS 2
sudo apt install ros-humble-ros-gz ros-humble-moveit ros-humble-ros2-control ros-humble-cv-bridge ros-humble-rqt-image-view
pip install pymoveit2

# 2. Clone repos
git clone https://github.com/cjx-cell/ur3_ft300_ws.git ~/ur3_ft300_ws
git clone https://github.com/huggingface/lerobot.git ~/lerobot

# 3. Pi0 env (editable)
conda create -n pi0-env python=3.12
conda activate pi0-env
cd ~/lerobot && pip install -e .
pip install safetensors torch torchvision transformers accelerate

# 4. Transfer model weights (USB or rsync)
# Copy these files into ~/ur3_ft300_ws/:
#   ai-models/lerobot/pi0/*.safetensors       (~40 GB)
#   ai-models/paligemma_tokenizer/*            (tokenizer)
#   outputs/                                    (fine-tuned checkpoints, if any)

# 5. Build workspace
cd ~/ur3_ft300_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Build

```bash
cd ~/ur3_ft300_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 1. Launch Simulation

```bash
cd ~/ur3_ft300_ws
source install/setup.bash

# Basic: Gazebo + robot + controllers (no MoveIt, no RViz)
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py

# Headless (server, no GUI)
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py gazebo_gui:=false

# With RViz (basic robot view, NOT MoveIt panel)
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py launch_rviz:=true
```

**Launch arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `gazebo_gui` | `true` | Show/hide Gazebo window |
| `launch_rviz` | `false` | Launch RViz2 for visualization |
| `world_file` | `simulation_world.sdf` | World file to load |
| `controller_start_delay` | `30.0` | Seconds before spawning controllers |

**Note:** This launch file does **NOT** start MoveIt. MoveIt is only needed for data collection (to plan joint-space trajectories via pymoveit2). During Pi0 inference, MoveIt is not used — the model outputs joint positions directly to the JTC action server.

---

## 2. Terminal Control (Arm + Gripper)

All 7 joints are controlled by one `joint_trajectory_controller` via the `/joint_trajectory_controller/follow_joint_trajectory` action.

### Move arm to a position (absolute joint angles in radians)

```bash
# Go to HOME position (gripper open)
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  '{
    trajectory: {
      joint_names: ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint","robotiq_85_left_knuckle_joint"],
      points: [{
        positions: [0.0, -1.57, 0.0, -1.57, 0.0, 0.0, 0.0],
        time_from_start: {sec:2, nanosec:0}
      }]
    }
  }'
```

### Move to ABOVE block position

```bash
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  '{
    trajectory: {
      joint_names: ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint","robotiq_85_left_knuckle_joint"],
      points: [{
        positions: [-1.834, -1.883, -1.128, -1.646, 1.572, 0.297, 0.0],
        time_from_start: {sec:3, nanosec:0}
      }]
    }
  }'
```

### Close gripper only (arm stays still)

```bash
# Close gripper (robotiq_85_left_knuckle_joint = 0.75)
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  '{
    trajectory: {
      joint_names: ["robotiq_85_left_knuckle_joint"],
      points: [{
        positions: [0.75],
        time_from_start: {sec:1, nanosec:0}
      }]
    }
  }'
```

### Open gripper only

```bash
# Open gripper (robotiq_85_left_knuckle_joint = 0.0)
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  '{
    trajectory: {
      joint_names: ["robotiq_85_left_knuckle_joint"],
      points: [{
        positions: [0.0],
        time_from_start: {sec:1, nanosec:0}
      }]
    }
  }'
```

### Pre-recorded joint positions

| Phase | shoulder_pan | shoulder_lift | elbow | wrist_1 | wrist_2 | wrist_3 | gripper |
|-------|-------------|---------------|-------|---------|---------|---------|---------|
| **HOME** | 0.0 | -1.57 | 0.0 | -1.57 | 0.0 | 0.0 | 0.0 |
| **ABOVE** | -1.834 | -1.883 | -1.128 | -1.646 | 1.572 | 0.297 | 0.0 |
| **GRASP** | -1.803 | -1.942 | -1.579 | -1.136 | 1.570 | -0.202 | **0.75** |
| **LIFT** | -1.803 | -1.856 | -1.293 | -1.508 | 1.570 | -0.202 | 0.75 |
| **ABOVE_B** | -0.761 | -1.910 | -1.230 | -1.545 | 1.523 | 0.839 | 0.75 |
| **PLACE** | -0.765 | -1.925 | -1.358 | -1.402 | 1.523 | 0.835 | **0.0** |

---

## 3. View Sensor Data

### Joint states

```bash
# Stream joint positions and velocities
ros2 topic echo /joint_states

# Filter to just positions
ros2 topic echo /joint_states --field position

# One-shot snapshot
ros2 topic echo /joint_states --once
```

### Camera images

```bash
# View wrist camera in a GUI window
ros2 run rqt_image_view rqt_image_view /wrist_camera/color/image_raw

# View global camera
ros2 run rqt_image_view rqt_image_view /global_camera/color/image_raw

# Check camera info
ros2 topic echo /wrist_camera/color/image_raw --once
# Image size: 640x480 RGB, 30 Hz

# Save a frame to disk (requires image_transport plugins)
ros2 run image_view image_saver --ros-args -r image:=/wrist_camera/color/image_raw
```

### All available topics

```bash
# List all topics
ros2 topic list

# Key topics:
#   /joint_states                         — 7 joint positions + velocities
#   /wrist_camera/color/image_raw          — Wrist camera (640x480 RGB, 30Hz)
#   /global_camera/color/image_raw         — Global camera (640x480 RGB, 30Hz)
#   /joint_trajectory_controller/status    — Controller status
#   /clock                                 — Simulation clock
```

---

## 4. Data Collection

### Record pick-and-place episodes

```bash
# Requires: Gazebo running (terminal 1)
# Requires: MoveIt running (terminal 2, for pymoveit2 IK)
#   ros2 launch ur3_ft300_moveit demo.launch.py

# Terminal 3: Record N episodes
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_record_pick_place.py --episodes 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 1 | Number of episodes to record |
| `--output` | `~/ur3_ft300_ws/ai-models/ur3_pick_place_raw` | Save directory |

Each episode saves as `episode_XXXX/data.npz` containing:
- `state`: (N, 7) joint positions at each frame
- `action`: (N, 7) target joint positions at each frame
- `camera0`: (N, 224, 224, 3) wrist camera images
- `camera1`: (N, 224, 224, 3) global camera images
- `task`: language instruction string

### Visualize collected data (check quality)

```bash
# All episodes with cross-episode comparison
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/visualize_dataset.py --all

# Single episode
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/visualize_dataset.py --episode 0

# Images saved to ai-models/ur3_pick_place_raw/trajectory_viz/
```

### Convert to LeRobot format

```bash
conda activate pi0-env

# Local conversion (default)
python3 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_convert_to_lerobot.py \
    --input ai-models/ur3_pick_place_raw \
    --repo_id cjx-cell/ur3_pick_place

# Also push to HuggingFace Hub
python3 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_convert_to_lerobot.py \
    --input ai-models/ur3_pick_place_raw \
    --repo_id cjx-cell/ur3_pick_place \
    --push_to_hub
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Directory of `episode_XXXX/data.npz` files |
| `--repo_id` | `cjx-cell/ur3_pick_place` | HuggingFace dataset repo ID |
| `--fps` | 20 | Recording framerate |
| `--push_to_hub` | `false` | Upload to HuggingFace Hub |

---

## 5. Training

### Action-Expert Only (~15 GB VRAM, single GPU)

Trains only the Gemma 300M action head. PaliGemma VLM stays frozen. Fast, but model memorizes trajectory instead of learning visual grounding.

```bash
conda activate pi0-env

python -m lerobot.scripts.lerobot_train \
    --policy.path=./ai-models/lerobot/pi0 \
    --dataset.repo_id=cjx-cell/ur3_pick_place \
    --policy.train_expert_only=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.gradient_checkpointing=true \
    --batch_size=2 \
    --steps=5000 \
    --output_dir=./outputs/train/pi0_ur3_v3
```

### Full VLM Fine-Tuning (2× A100 80GB recommended)

Trains PaliGemma 2B + Action Expert together. Model learns to **see the red cube** and servo to visual targets.

```bash
conda activate pi0-env

python -m lerobot.scripts.lerobot_train \
    --policy.path=./ai-models/lerobot/pi0 \
    --dataset.repo_id=cjx-cell/ur3_pick_place \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.gradient_checkpointing=true \
    --policy.optimizer_lr=2e-5 \
    --batch_size=1 \
    --steps=3000 \
    --output_dir=./outputs/train/pi0_ur3_full_vlm
```

### Continue from checkpoint

```bash
# Resume training from a checkpoint
python -m lerobot.scripts.lerobot_train \
    --policy.path=./outputs/train/pi0_ur3_v3/checkpoints/last/pretrained_model \
    --dataset.repo_id=cjx-cell/ur3_pick_place \
    --policy.train_expert_only=true \
    --batch_size=2 \
    --steps=10000 \
    --output_dir=./outputs/train/pi0_ur3_v4
```

### Training Parameter Reference

#### Core Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--policy.path=PATH` | — | Pretrained model dir or HF repo (loads base config + weights) |
| `--dataset.repo_id=ID` | — | LeRobot dataset ID (local or HF Hub) |
| `--batch_size=N` | 8 | Per-GPU batch size |
| `--steps=N` | 100000 | Total training steps |
| `--output_dir=DIR` | auto | Checkpoint save directory |

#### Training Mode (★ critical)

| Flag | Default | Description |
|------|---------|-------------|
| `--policy.train_expert_only=BOOL` | `false` | **`true`**: freeze PaliGemma VLM, only train action expert (15GB VRAM). **`false`**: full VLM fine-tuning (70+GB VRAM) |
| `--policy.freeze_vision_encoder=BOOL` | `false` | Freeze SigLIP vision encoder. Redundant when `train_expert_only=true` |
| `--policy.gradient_checkpointing=BOOL` | `false` | Trade 30% slower for ~40% less VRAM |

#### Precision & Device

| Flag | Default | Description |
|------|---------|-------------|
| `--policy.dtype=STR` | `float32` | `bfloat16` halves VRAM usage |
| `--policy.device=STR` | auto | `cuda`, `cpu`, `mps` |

#### Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--policy.optimizer_lr=FLOAT` | 2.5e-5 | Peak learning rate |
| `--policy.optimizer_weight_decay=FLOAT` | 0.01 | AdamW weight decay |
| `--policy.optimizer_grad_clip_norm=FLOAT` | 1.0 | Max gradient norm |
| `--policy.scheduler_warmup_steps=N` | 1000 | Linear warmup steps |
| `--policy.scheduler_decay_steps=N` | 30000 | Cosine decay steps |
| `--policy.scheduler_decay_lr=FLOAT` | 2.5e-6 | Minimum LR after decay |

#### Model Architecture

| Flag | Default | Description |
|------|---------|-------------|
| `--policy.n_obs_steps=N` | 1 | Number of observation steps (1 = no history) |
| `--policy.chunk_size=N` | 50 | Actions predicted per inference |
| `--policy.num_inference_steps=N` | 10 | Denoising steps at inference |
| `--policy.tokenizer_max_length=N` | 48 | Task prompt token limit |
| `--policy.use_relative_actions=BOOL` | `false` | Predict action deltas instead of absolute positions |

#### Data

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset.image_transforms.enable=BOOL` | `false` | Color jitter + affine augmentation |
| `--num_workers=N` | 4 | DataLoader workers |
| `--save_freq=N` | 20000 | Save checkpoint every N steps |
| `--log_freq=N` | 200 | Print metrics every N steps |

---

## 6. Pi0 Inference (3-Terminal Setup)

Data flows through `/tmp/` files. **No MoveIt needed** — Pi0 outputs go directly to the JTC action server.

```
Gazebo cameras → ROS topic → ros_side.py → /tmp/camera{0,1}.npy → pi0_inference.py
Gazebo joints  → /joint_states → ros_side.py → /tmp/joint_state.txt → pi0_inference.py
Pi0 action → /tmp/ur3_action.txt → ros_side.py → FollowJointTrajectory → UR3
```

### Terminal 1: Gazebo Simulation

```bash
cd ~/ur3_ft300_ws
source install/setup.bash
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py
```

### Terminal 2: ROS Communication Bridge (system Python 3.10)

This subscribes to ROS topics, writes to `/tmp/`, reads actions, and sends them to the arm.

```bash
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_ros_side.py
```

### Terminal 3: Pi0 Inference (conda pi0-env)

Loads the fine-tuned model, reads observations, runs inference, writes actions.

```bash
conda activate pi0-env

# GPU bfloat16 (recommended, ~7.5 GB VRAM)
python3 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_inference.py --mode bf16 --hz 5

# CPU fallback
python3 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_inference.py --mode cpu --hz 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `cpu` | `cpu`, `bf16` (GPU ~7.5GB), `fp32` (GPU ~15GB) |
| `--hz` | 5 | Inference frequency (Hz) |
| `--warmup` | 1 | GPU warmup iterations |

### Single-Shot Test (no loop)

```bash
conda activate pi0-env

# Test with real data from /tmp/
python3 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_test.py

# Test with dummy (zero) data — checks model loading only
python3 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_test.py --dummy

# GPU test
python3 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_test.py --mode bf16
```

---

## 7. Scripts Reference

| Script | Python | Purpose |
|--------|--------|---------|
| `ur3_record_pick_place.py` | 3.10 (ROS) | Collect pick-and-place trajectories via MoveIt |
| `ur3_collect_dataset.py` | 3.10 (ROS) | Alternative data collector (20 Hz recording) |
| `ur3_convert_to_lerobot.py` | pi0-env | Convert .npz episodes → LeRobot dataset |
| `ur3_pi0_ros_side.py` | 3.10 (ROS) | ROS ↔ /tmp/ bridge for inference |
| `ur3_pi0_inference.py` | pi0-env | Pi0 inference loop (5 Hz, gripper state machine) |
| `ur3_pi0_test.py` | pi0-env | Single-shot inference test |
| `visualize_dataset.py` | 3.10 (ROS) | Generate trajectory plots + keyframe images |
| `fix_checkpoint_keys.py` | pi0-env | Fix safetensors key mismatches |
| `pick_and_place.py` | 3.10 (ROS) | Basic MoveIt pick-and-place demo |

---

## 8. Troubleshooting

### Simulation

- **Gripper falls apart**: DART ignores URDF `<mimic>`. Fixed via `gz_ros2_control` mimic parameters.
- **Block slips through fingers**: DART uses `<mu>` (not `<mu1>`). Finger collision surfaces need explicit friction in URDF.
- **Controller not responding**: Wait 30s after launch for controller spawn delay.

### Inference

- **Arm moves while gripper closing**: Training data has instant gripper transitions. Fixed by gripper-arm synchronization state machine in `ur3_pi0_inference.py` — arm freezes for 5s during close/open.
- **Gripper never fully closes**: With block in hand, gripper physically stops at ~0.45. The state machine uses time-based freeze (not position threshold) to handle this.
- **Gripper too slow**: PID gains (P=500) and velocity limit (5.0 rad/s) in `ur3_ft300_robotiq_controllers.yaml`.
- **Action not sent**: ROS side only sends actions when file mtime changes AND `|action| > 0.01`. Stale files are ignored.
- **Model outputs near-zero actions**: Make sure Pi0 terminal sees valid camera images (check `/tmp/ur3_camera0.npy` is not all zeros).

### Training

- **Model doesn't target red cube**: `train_expert_only=true` only trains the action head — model memorizes trajectory, doesn't learn visual grounding. Use `train_expert_only=false` (needs A100).
- **OOM on A100**: Reduce `--batch_size=1`, enable `--policy.gradient_checkpointing=true`, use `--policy.dtype=bfloat16`.
- **Tokenizer not found**: Ensure `ai-models/paligemma_tokenizer/` contains `tokenizer.json` and `tokenizer.model`.

---

## 9. Transfer to A100 Server

Large weight files (`.safetensors`, ~40GB) are gitignored. Transfer them separately:

```bash
# rsync entire workspace (recommended)
rsync -avz --progress ~/ur3_ft300_ws/ user@a100-server:~/ur3_ft300_ws/

# Or: just the model weights + outputs
rsync -avz --progress \
    ~/ur3_ft300_ws/ai-models/lerobot/pi0/ \
    user@a100-server:~/ur3_ft300_ws/ai-models/lerobot/pi0/
rsync -avz --progress \
    ~/ur3_ft300_ws/outputs/ \
    user@a100-server:~/ur3_ft300_ws/outputs/
```

---

## License

MIT. See individual `src/` packages for their respective licenses.

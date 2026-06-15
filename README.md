# UR3 + FT300 + Robotiq 2F85 — Pi0 Pick-and-Place

Gazebo Fortress simulation of a UR3 collaborative robot with Robotiq FT300 F/T sensor and
2F-85 adaptive gripper. **Collects pick-and-place trajectory data and fine-tunes a Pi0
vision-language-action model** for "pick up the red cube and place it into the bowl".

## Hardware (Simulated)

| Component | Model |
|-----------|-------|
| Robot arm | Universal Robots UR3 (6-DOF, 3 kg, 500 mm) |
| Force-torque sensor | Robotiq FT300 (wrist-mounted) |
| Gripper | Robotiq 2F-85 (85 mm stroke) |
| Cameras | Realsense D435i (wrist) + D415 (global) |

## Software Stack

| Layer | Technology |
|-------|------------|
| OS | Ubuntu 22.04 |
| ROS 2 | Humble |
| Simulation | Gazebo Fortress (DART physics) |
| Control | `gz_ros2_control` + `joint_trajectory_controller` |
| Motion planning | MoveIt2 (data collection only) |
| VLA Model | Pi0 (PaliGemma 2B + Action Expert 300M) |
| Training | LeRobot |

## Directory Layout

```
~/ur3_ft300_ws/
├── src/ur_simulation_gz/    # Simulation package + all scripts
├── ai-models/
│   ├── lerobot/pi0/          # Base Pi0 pre-trained model (~40 GB)
│   ├── paligemma_tokenizer/  # PaliGemma tokenizer
│   ├── ur3_pick_place_raw/   # Recorded trajectory data (.npz)
│   └── ur3_pick_place_lerobot/ # Converted LeRobot dataset
├── outputs/train/            # Fine-tuned checkpoints
└── README.md
```

---

## 1. One-Time Setup

### 1.1 System Dependencies

```bash
sudo apt install ros-humble-ros-gz ros-humble-moveit ros-humble-ros2-control \
                 ros-humble-cv-bridge ros-humble-rqt-image-view
pip install pymoveit2
```

### 1.2 Clone Repos

```bash
git clone https://github.com/cjx-cell/ur3_ft300_ws.git ~/ur3_ft300_ws
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```

### 1.3 Pi0 Environment

```bash
conda create -n pi0-env python=3.12
conda activate pi0-env
cd ~/lerobot && pip install -e .
pip install safetensors torch torchvision transformers accelerate
```

### 1.4 Build Workspace

```bash
cd ~/ur3_ft300_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 1.5 Model Weights

Copy base Pi0 weights and tokenizer into `~/ur3_ft300_ws/ai-models/`:

```
ai-models/lerobot/pi0/*.safetensors   # ~40 GB
ai-models/paligemma_tokenizer/*        # tokenizer.json, tokenizer.model
```

---

## 2. Launch Simulation

**Terminal 1 — Gazebo:**

```bash
cd ~/ur3_ft300_ws
source install/setup.bash
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py

# Headless (no GUI):
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py gazebo_gui:=false
```

Wait ~30 seconds for controllers to start (joint_state_broadcaster, joint_trajectory_controller).

---

## 3. Data Collection

### 3.1 Start MoveIt (planning only)

**Terminal 2:**

```bash
cd ~/ur3_ft300_ws
source install/setup.bash
ros2 launch ur3_ft300_moveit_config move_group.launch.py
```

Wait for `You can start planning now!`.

### 3.2 Record Episodes

**Terminal 2 (same terminal, after MoveIt is ready):**

```bash
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_record_pick_place.py \
    --episodes 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 1 | Number of episodes to record |
| `--output` | `~/ur3_ft300_ws/ai-models/ur3_pick_place_raw` | Output directory |

Each episode saves as `episode_XXXX/data.npz`:
- `state`: (N, 7) actual joint positions from `/joint_states`
- `action`: (N, 7) target waypoint being moved toward
- `camera0`: (N, 224, 224, 3) wrist camera
- `camera1`: (N, 224, 224, 3) global camera
- `task`: language instruction

### 3.3 Visualize Recordings

```bash
# Generate videos (all episodes, camera + joint curves):
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/make_video.py

# Single episode:
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/make_video.py --episode 0

# Static per-joint analysis charts:
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/visualize_dataset.py

# Watch a video:
mpv ~/ur3_ft300_ws/ai-models/ur3_pick_place_raw/trajectory_viz/episode_0000.mp4
```

### 3.4 Convert to LeRobot Format

```bash
conda activate pi0-env

python src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_convert_to_lerobot.py \
    --input ~/ur3_ft300_ws/ai-models/ur3_pick_place_raw

# Push to HuggingFace Hub:
python src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_convert_to_lerobot.py \
    --input ~/ur3_ft300_ws/ai-models/ur3_pick_place_raw \
    --push_to_hub
```

Output: `~/ur3_ft300_ws/ai-models/ur3_pick_place_lerobot/` (LeRobot v3.0 format, AV1 videos).

---

## 4. Training

```bash
conda activate pi0-env

# Action-Expert only (~15 GB VRAM, single GPU)
python -m lerobot.scripts.lerobot_train \
    --policy.path=./ai-models/lerobot/pi0 \
    --dataset.repo_id=cjx-cell/ur3_pick_place \
    --policy.train_expert_only=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.gradient_checkpointing=true \
    --batch_size=2 \
    --steps=5000 \
    --output_dir=./outputs/train/pi0_ur3_expert

# Full VLM fine-tuning (2× A100 80GB recommended)
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
    --output_dir=./outputs/train/pi0_ur3_full
```

**Key training flags:**

| Flag | Effect |
|------|--------|
| `--policy.train_expert_only=true` | Freeze VLM, only train action head (~15 GB VRAM) |
| `--policy.train_expert_only=false` | Full VLM fine-tuning (70+ GB VRAM) |
| `--policy.dtype=bfloat16` | Halve VRAM usage |
| `--policy.gradient_checkpointing=true` | Trade 30% speed for ~40% less VRAM |

---

## 5. Inference

**Data flow (no MoveIt needed — Pi0 outputs go directly to the controller):**

```
Gazebo → /joint_states → ros_side.py → /tmp/joint_state.txt → pi0_inference.py
Gazebo → /camera/image_raw → ros_side.py → /tmp/camera{0,1}.npy → pi0_inference.py
Pi0 action → /tmp/ur3_action.txt → ros_side.py → FollowJointTrajectory → UR3
```

### Terminal 1: Gazebo

```bash
cd ~/ur3_ft300_ws && source install/setup.bash
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py
```

### Terminal 2: ROS Bridge

```bash
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_ros_side.py
```

### Terminal 3: Pi0 Inference

```bash
conda activate pi0-env

# GPU (recommended, ~7.5 GB VRAM):
python src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_inference.py --mode bf16 --hz 5

# CPU fallback:
python src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_inference.py --mode cpu --hz 2
```

---

## 6. Scripts

| Script | Python | Purpose |
|--------|--------|---------|
| `ur3_record_pick_place.py` | 3.10 (system) | Record pick-and-place trajectories |
| `ur3_convert_to_lerobot.py` | pi0-env | Convert .npz → LeRobot dataset |
| `make_video.py` | 3.10 (system) | Generate video from recorded data |
| `visualize_dataset.py` | 3.10 (system) | Static joint trajectory charts |
| `compute_ik.py` | 3.10 (system) | Compute IK for new waypoints |
| `fix_checkpoint_keys.py` | pi0-env | Fix safetensors key mismatches |
| `ur3_pi0_inference.py` | pi0-env | Pi0 inference loop |
| `ur3_pi0_ros_side.py` | 3.10 (system) | ROS ↔ /tmp/ bridge for inference |
| `ur3_pi0_test.py` | pi0-env | Single-shot inference test |

---

## 7. Troubleshooting

- **Controller not responding**: Wait 30s after Gazebo launch for controller spawn delay.
- **`STATUS_ABORTED` during recording**: Normal for gripper — occurs when it's already at the target position (open at 0.0, close at 0.75). Harmless.
- **MoveIt service not found**: Make sure `move_group.launch.py` is running *after* Gazebo is fully up.
- **Tokenizer not found**: Ensure `ai-models/paligemma_tokenizer/` contains `tokenizer.json` and `tokenizer.model`.
- **OOM during training**: Reduce `--batch_size=1`, enable `--policy.gradient_checkpointing=true`, use `--policy.dtype=bfloat16`.
- **Gripper doesn't fully close**: With block in hand, gripper physically stops at ~0.45. The inference state machine uses time-based freeze to handle this.

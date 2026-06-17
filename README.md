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
├── src/ur_simulation_gz/       # Simulation package + all scripts
├── ai-models/
│   ├── pi0_libero_base/         # Pi0 fine-tuned on LIBERO (starting point)
│   ├── paligemma_tokenizer/     # PaliGemma tokenizer
│   ├── ur3_pick_place_raw/      # Recorded trajectory data (.npz)
│   └── ur3_pick_place_lerobot/  # Converted LeRobot dataset
├── outputs/train/               # Fine-tuned checkpoints
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
pip install safetensors torch torchvision transformers accelerate peft
```

### 1.4 Build Workspace

```bash
cd ~/ur3_ft300_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 1.5 Model Weights

Download the **Pi0 Libero base model** (VLM already fine-tuned on 130+ robot manipulation tasks):

```bash
# From HuggingFace (with mirror for China):
export HF_ENDPOINT=https://hf-mirror.com
conda activate pi0-env
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/pi0_libero_base',
                   local_dir='./ai-models/pi0_libero_base')
"
```

Also copy the PaliGemma tokenizer:

```bash
# Tokenizer (from original Pi0 or download from HuggingFace):
# Place tokenizer.json and tokenizer.model in:
#   ai-models/paligemma_tokenizer/
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

Wait ~30 seconds for controllers to start.

---

## 3. Data Collection

### 3.1 Start MoveIt

**Terminal 2:**

```bash
cd ~/ur3_ft300_ws
source install/setup.bash
ros2 launch ur3_ft300_moveit_config move_group.launch.py
```

### 3.2 Record Episodes

```bash
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_record_pick_place.py \
    --episodes 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 1 | Number of episodes to record |
| `--output` | `ai-models/ur3_pick_place_raw` | Output directory |

Each episode saves as `<task>_episode_XXXX/<status>/data.npz`:
- `state`: (N, 7) joint positions from `/joint_states`
- `action`: (N, 7) next state = absolute joint positions
- `camera0`: (N, 224, 224, 3) wrist camera
- `camera1`: (N, 224, 224, 3) global camera
- `task`: language instruction

### 3.3 Visualize Recordings

```bash
# Generate videos (all episodes):
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/make_video.py

# Single episode:
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/make_video.py --episode 0

# Watch:
mpv ai-models/ur3_pick_place_raw/trajectory_viz/episode_0000.mp4
```

### 3.4 Convert to LeRobot Format

```bash
conda activate pi0-env

python src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_convert_to_lerobot.py \
    --input ai-models/ur3_pick_place_raw

# Push to HuggingFace Hub:
python src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_convert_to_lerobot.py \
    --input ai-models/ur3_pick_place_raw \
    --push_to_hub
```

---

## 4. Training

### Strategy Comparison

| Strategy | GPU | VRAM | Trainable Params | Best For |
|----------|-----|------|-----------------|----------|
| LoRA rank=16 | RTX 5080 16GB | ~9 GB | ~48M | Quick experiments |
| LoRA rank=16 (VLM+Expert) | RTX 5080 16GB | ~10 GB | ~48M | Better visual adaptation |
| Full Expert Only | RTX 5080 16GB | ~12 GB | ~300M | Max capacity, frozen VLM |
| **Full Fine-tune** | **A100 40/80GB** | **~35 GB** | **~2.3B** | **Best quality** |

### 4.1 LoRA Fine-tuning (RTX 5080, Local)

For quick experiments on a 5080 GPU. LoRA adapts Q/V attention in both VLM and action expert.

```bash
conda activate pi0-env

/home/ubuntu/miniconda3/envs/pi0-env/bin/python -m lerobot.scripts.lerobot_train \
  --policy.path=./ai-models/pi0_libero_base \
  --dataset.repo_id=cjx-cell/ur3_pick_place \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --batch_size=1 \
  --steps=40000 \
  --peft.r=16 \
  --tolerance_s=0.001 \
  --output_dir=./outputs/train/ur3_pi0_lora \
  --save_freq=20000 \
  --log_freq=100
```

### 4.2 Full Fine-tune (A100, Remote Server)

Full model training on A100. All 2.3B parameters updated — VLM learns Gazebo visual features from scratch.

```bash
conda activate pi0-env

python -m lerobot.scripts.lerobot_train \
  --policy.path=./ai-models/pi0_libero_base \
  --dataset.repo_id=cjx-cell/ur3_pick_place \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.gradient_checkpointing=false \
  --policy.optimizer_lr=1e-5 \
  --batch_size=2 \
  --steps=60000 \
  --tolerance_s=0.001 \
  --output_dir=./outputs/train/ur3_pi0_full \
  --save_freq=10000 \
  --log_freq=50
```

| Flag | Effect |
|------|--------|
| `train_expert_only=false` | Train both VLM + action expert (not just expert) |
| `freeze_vision_encoder=false` | Unfreeze SigLIP vision encoder |
| `gradient_checkpointing=false` | A100 has enough VRAM, no need to trade speed |
| `optimizer_lr=1e-5` | Lower LR for full model (vs 2.5e-5 for LoRA) |
| `batch_size=2` | A100 80GB can handle 2-4 |
| `steps=60000` | ~35% of dataset, sufficient for full fine-tune |

### 4.3 Monitor Training

```bash
# Watch loss in real-time (training prints every --log_freq steps)
# Check checkpoint quality:
python src/ur_simulation_gz/ur_simulation_gz/scripts/eval_lora_model.py \
    --model ./outputs/train/ur3_pi0_full/checkpoints/20000/pretrained_model \
    --ckpt ./outputs/train/ur3_pi0_full/checkpoints/20000/pretrained_model
```

> Note: for full fine-tune (not LoRA), skip the `merge_lora.py` step. Load checkpoint directly.

---

## 5. Evaluation

### 5.1 Dataset Accuracy (MAE)

```bash
conda activate pi0-env

# For LoRA models (need to merge first):
python src/ur_simulation_gz/ur_simulation_gz/scripts/merge_lora.py \
    --base ./ai-models/pi0_libero_base \
    --lora ./outputs/train/ur3_pi0_lora/checkpoints/040000/pretrained_model \
    --output ./ai-models/ur3_pi0_lora_merged

python src/ur_simulation_gz/ur_simulation_gz/scripts/eval_lora_model.py \
    --model ./ai-models/ur3_pi0_lora_merged \
    --ckpt ./outputs/train/ur3_pi0_lora/checkpoints/040000/pretrained_model

# For full fine-tune models (use checkpoint directly):
python src/ur_simulation_gz/ur_simulation_gz/scripts/eval_lora_model.py \
    --model ./outputs/train/ur3_pi0_full/checkpoints/40000/pretrained_model \
    --ckpt ./outputs/train/ur3_pi0_full/checkpoints/40000/pretrained_model
```

MAE interpretation:
- **< 0.10 rad**: Excellent, ready for Gazebo
- **0.10–0.20 rad**: Good, may work in Gazebo
- **> 0.30 rad**: Needs more training

### 5.2 Gazebo Test

**Terminal 1 — Gazebo:**
```bash
cd ~/ur3_ft300_ws && source install/setup.bash
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py
```

**Terminal 2 — ROS Bridge (with spawn):**
```bash
/usr/bin/python3.10 src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_ros_side.py --spawn
```

**Terminal 3 — Pi0 Inference:**
```bash
conda activate pi0-env

# LoRA merged model:
python src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_inference.py \
    --model ./ai-models/ur3_pi0_lora_merged --mode bf16 --hz 10

# Full fine-tune model:
python src/ur_simulation_gz/ur_simulation_gz/scripts/ur3_pi0_inference.py \
    --model ./outputs/train/ur3_pi0_full/checkpoints/40000/pretrained_model --mode bf16 --hz 10
```

### 5.3 Data Flow

```
Gazebo → /joint_states → ros_side.py → /tmp/ur3_joint_state.txt → pi0_inference.py
Gazebo → /camera/image_raw → ros_side.py → /tmp/ur3_camera{0,1}.npy → pi0_inference.py
Pi0 action → /tmp/ur3_action.txt → ros_side.py → FollowJointTrajectory → UR3
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
| `ur3_pi0_inference.py` | pi0-env | Pi0 inference loop (w/ state machine, EMA, atomic write) |
| `ur3_pi0_ros_side.py` | 3.10 (system) | ROS ↔ /tmp/ bridge, block/bowl spawn |
| `merge_lora.py` | pi0-env | Offline merge LoRA adapter → single safetensors |
| `eval_lora_model.py` | pi0-env | Dataset prediction MAE evaluation |
| `fix_checkpoint_keys.py` | pi0-env | Fix safetensors key mismatches |

---

## 7. Troubleshooting

- **Controller not responding**: Wait 30s after Gazebo launch for controller spawn.
- **OOM during training (5080)**: Reduce `--batch_size=1`, enable `--policy.gradient_checkpointing=true`, use `--policy.dtype=bfloat16`.
- **OOM during model loading**: Script uses `init_empty_weights()` + `to_empty()`. If it still OOMs, reduce system memory usage.
- **Model predicts wrong actions**: Check that `empty_cameras=0` in inference (matches training). Verify normalizer stats are from the correct checkpoint.
- **Gripper state machine cycles**: Normal — model needs to see the block in camera to guide approach. If MAE > 0.3, model isn't accurate enough yet.
- **Tokenizer not found**: Ensure `ai-models/paligemma_tokenizer/` contains tokenizer files.
- **Network unreachable for HF**: Use `export HF_ENDPOINT=https://hf-mirror.com` for China mirror, or download models via browser and scp to server.

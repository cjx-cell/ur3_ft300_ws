# UR3 + FT300 + Robotiq 2F85 — Pi0 / SA-MOE

Gazebo Fortress simulation of a UR3 collaborative robot with Robotiq FT300 F/T sensor and
2F-85 adaptive gripper. Two task pipelines:
- **Pi0**: "pick up the red cube and place it into the bowl"
- **SA-MOE**: "pick up the peg and insert it into the hole" (force-aware, stage-aware)

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
| VLA Model (Pick&Place) | Pi0 (PaliGemma 2B + Action Expert 300M) |
| VLA Model (Peg-in-Hole) | SA-MOE (Stage-Aware Mixture of Experts on Pi0) |
| Training | LeRobot |

## Directory Layout

```
~/ur3_ft300_ws/
├── ai-models/
│   ├── pi0/                          # Pi0 model weights
│   │   ├── pi0_libero_base/          # Pretrained base (6.6GB)
│   │   ├── pi0_full_v1/              # V1 full fine-tune (84K steps)
│   │   ├── pi0_full_v2/              # V2 full fine-tune (84K, vision fix)
│   │   └── pi0_full_v3/              # V3 full fine-tune (135K, 10fps)
│   ├── samoe/                        # SA-MOE model weights
│   │   ├── v1/ v2/                   # Early experiments
│   │   ├── delta_full/               # Delta action 100K
│   │   └── v7/                       # ForceVLA-style (30K)
│   ├── datasets/                     # All datasets
│   │   ├── ur3_pick_place_raw/       # Raw pick-and-place (49 eps, 50fps)
│   │   ├── ur3_pick_place_lerobot/   # LeRobot format (50fps)
│   │   ├── ur3_pick_place_10hz_lerobot/  # LeRobot format (10fps, V3)
│   │   ├── ur3_peg_in_hole_raw/      # Raw peg-in-hole (force + stage labels)
│   │   └── ur3_peg_in_hole_lerobot/  # LeRobot format
│   ├── paligemma_tokenizer/          # PaliGemma tokenizer
│   └── run_train_sa_moe.sh           # SA-MOE training launcher
├── src/ur_simulation_gz/
│   └── ur_simulation_gz/
│       ├── launch/ur3_ft300_robotiq.launch.py  # Gazebo launch
│       ├── src/
│       │   ├── pick_and_place.cpp    # C++ pick-and-place controller
│       │   └── peg_in_hole.cpp       # C++ peg-in-hole controller (force + stage)
│       └── scripts/
│           ├── pick_and_place/       # Pi0 pick-and-place scripts
│           │   ├── ur3_pi0_pick_place_record.py              # Data recording
│           │   ├── ur3_pi0_pick_place_make_video.py          # Visualization
│           │   ├── ur3_pi0_pick_place_convert_to_lerobot.py  # npz → LeRobot
│           │   ├── ur3_pi0_pick_place_ros_side.py            # ROS bridge
│           │   ├── ur3_pi0_pick_place_inference.py           # Pi0 inference
│           │   ├── ur3_pi0_pick_place_eval_offline.py        # Offline eval
│           │   ├── ur3_pi0_pick_place_eval_lora.py           # LoRA eval
│           │   ├── ur3_pi0_pick_place_compute_ik.py          # IK solver
│           │   ├── ur3_pi0_pick_place_merge_lora.py          # LoRA merge
│           │   └── ur3_pi0_pick_place_fix_checkpoint_keys.py # Key fixer
│           └── peg_in_hole/          # SA-MOE peg-in-hole scripts
│               ├── ur3_samoe_peg_in_hole_record.py           # Data recording
│               ├── ur3_samoe_peg_in_hole_make_video.py       # Visualization
│               ├── ur3_samoe_peg_in_hole_convert_to_lerobot.py # npz → LeRobot
│               ├── ur3_samoe_peg_in_hole_inference.py        # SA-MOE inference
│               └── ur3_samoe_peg_in_hole_ros_side.py         # ROS bridge
├── docs/                           # Training guides & design docs
│   ├── A100_RETRAIN.md
│   ├── A100_TRAINING_V2.md
│   ├── A100_V3_TRAINING_GUIDE.md
│   └── SA_MOE_DESIGN.md
├── PROJECT_REFERENCE.md            # Complete project reference
├── PI0_TRAINING_ANALYSIS.md        # Pi0 V1/V2 analysis
├── SA_MOE_CHANGELOG.md             # SA-MOE version history
└── README.md
```

---

## 1. One-Time Setup

### 1.1 System Dependencies

```bash
sudo apt install ros-humble-ros-gz ros-humble-moveit ros-humble-ros2-control \
                 ros-humble-cv-bridge ros-humble-rqt-image-view
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

```bash
# Pi0 base model (HuggingFace)
export HF_ENDPOINT=https://hf-mirror.com
conda activate pi0-env
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/pi0_libero_base',
                   local_dir='./ai-models/pi0/pi0_libero_base')
"

# PaliGemma tokenizer — copy tokenizer files to:
#   ai-models/paligemma_tokenizer/
```

---

## 2. Launch Simulation

```bash
cd ~/ur3_ft300_ws
source install/setup.bash
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py

# Headless:
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py gazebo_gui:=false
```

---

## 3. Data Collection

### 3.1 Pick-and-Place (Pi0)

```bash
# Gazebo must be running first
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_record.py \
    --episodes 50
```

The script automatically: spawns block + bowl → runs C++ `pick_and_place` → records at 50Hz → saves `.npz`.

Output: `ai-models/datasets/ur3_pick_place_raw/`

### 3.2 Peg-in-Hole (SA-MOE)

```bash
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_record.py \
    --episodes 50
```

Records at 10Hz with force/torque data + stage labels (0=approach, 1=align, 2=grasp, 3=insert, 4=confirm).

Output: `ai-models/datasets/ur3_peg_in_hole_raw/`

### 3.3 Visualize Recordings

```bash
conda activate pi0-env

# Pick-and-place
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_make_video.py \
    --input ai-models/datasets/ur3_pick_place_raw --episode 0

# Peg-in-hole
python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_make_video.py \
    --input ai-models/datasets/ur3_peg_in_hole_raw --episode 0
```

### 3.4 Convert to LeRobot Format

```bash
conda activate pi0-env

# Pick-and-place (10fps)
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_pick_place_raw \
    --repo_id local/ur3_pick_place_10hz \
    --fps 10 --source_fps 50

# Peg-in-hole
python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_peg_in_hole_raw \
    --repo_id local/ur3_peg_in_hole
```

---

## 4. Training

### 4.1 Pi0 Full Fine-tune (A100)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env

python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/pi0 \
    --policy.pretrained_path=ai-models/pi0/pi0_libero_base \
    --policy.num_inference_steps=50 \
    --policy.dtype=bfloat16 --policy.device=cuda \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --dataset.repo_id=local/ur3_pick_place_10hz \
    --dataset.root=ai-models/datasets/ur3_pick_place_10hz_lerobot \
    --batch_size=2 --steps=135000 \
    --output_dir=outputs/train/ur3_pi0_v4
```

> See [docs/A100_V3_TRAINING_GUIDE.md](docs/A100_V3_TRAINING_GUIDE.md) for full A100 training guide.

### 4.2 SA-MOE Training

```bash
conda activate pi0-env
# See ai-models/run_train_sa_moe.sh for config
```

See [SA_MOE_CHANGELOG.md](SA_MOE_CHANGELOG.md) for version history.

---

## 5. Gazebo Test

### 5.1 Pi0 Pick-and-Place

**Terminal 1 — Gazebo:**
```bash
cd ~/ur3_ft300_ws && source install/setup.bash
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py
```

**Terminal 2 — ROS Bridge:**
```bash
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_ros_side.py --spawn
```

**Terminal 3 — Inference:**
```bash
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_inference.py \
    --mode bf16 --hz 10 \
    --model ai-models/pi0/pi0_full_v3/checkpoints/135000/pretrained_model
```

### 5.2 SA-MOE Peg-in-Hole

**Terminal 2 — ROS Bridge:**
```bash
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_ros_side.py
```

**Terminal 3 — Inference:**
```bash
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_inference.py \
    --checkpoint ai-models/samoe/v7/checkpoints/030451/pretrained_model
```

### 5.3 Data Flow

```
Gazebo → /joint_states        → ros_side.py → /tmp/ur3_joint_state.txt → inference.py
Gazebo → /camera/image_raw     → ros_side.py → /tmp/ur3_camera{0,1}.npy → inference.py
Gazebo → /force_torque/wrench  → ros_side.py → /tmp/ur3_force.npy       → samoe_inference.py
Inference → /tmp/ur3_action.txt → ros_side.py → FollowJointTrajectory → UR3
```

---

## 6. Known Issues

- **Pi0 Identity Shortcut**: Absolute actions at high fps cause `action ≈ state`. Model learns identity mapping. Fix: use delta actions (`action = next_state - state`).
- **SA-MOE V8**: Training stopped at 10K/30K. `stage_acc=0%`, `alpha=0.05` (ModalGate collapse).
- **wrist_2 joint**: Near-zero variance (std=0.003 rad) due to UR3 mechanical coupling.

See [PROJECT_REFERENCE.md](PROJECT_REFERENCE.md) for complete details.

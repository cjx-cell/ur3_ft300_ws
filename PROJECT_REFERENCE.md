# UR3 FT300 Pi0 / SA-MOE 项目完整参考

> 最后更新: 2026-06-23

---

## 目录结构

```
~/ur3_ft300_ws/
├── ai-models/                          # 所有模型权重 + 数据集
│   ├── pi0/                            # 【Pi0 模型权重】
│   │   ├── pi0_libero_base/            # 预训练基座模型 (6.6GB)
│   │   ├── pi0_full_v1/                # V1 全量微调 (84K steps)
│   │   ├── pi0_full_v2/                # V2 全量微调 (84K steps, vision fix)
│   │   └── pi0_full_v3/                # V3 全量微调 (135K steps, 10fps)
│   ├── samoe/                          # 【SA-MOE 模型权重】
│   │   ├── v1_z1006/                   # V1 绝对动作 100K
│   │   ├── v1_z0044/                   # V1 checkpoint 0044
│   │   ├── v1/                         # V1 初始
│   │   ├── v2/                         # V2 delta 5K test
│   │   ├── delta_full/                 # Delta 100K full
│   │   ├── delta_balanced/             # Class weights (未完成)
│   │   └── v7/                         # V7 ForceVLA-style (30K, 完成)
│   ├── datasets/                       # 【数据集】
│   │   ├── ur3_pick_place_raw/         # 抓取放置原始数据 (49 episodes, 50fps)
│   │   ├── ur3_pick_place_lerobot/     # V1 LeRobot 格式 (50fps)
│   │   ├── ur3_pick_place_10hz_lerobot/# V3 训练用 (10fps 降采样)
│   │   ├── ur3_peg_in_hole_raw/        # 孔轴装配原始数据 (带力+stage标签)
│   │   └── ur3_peg_in_hole_lerobot/    # 孔轴装配 LeRobot 格式
│   ├── paligemma_tokenizer/            # PaliGemma tokenizer
│   ├── run_train_sa_moe.sh              # SA-MOE 训练启动脚本
│   └── run_train_sa_moe_v8.sh           # SA-MOE V8 训练启动脚本
├── src/ur_simulation_gz/               # 【Gazebo 仿真 + 脚本】
│   └── ur_simulation_gz/
│       ├── launch/                     # Gazebo 启动文件
│       │   └── ur3_ft300_robotiq.launch.py
│       ├── worlds/                     # 世界文件
│       │   └── simulation_world.sdf
│       ├── urdf/                       # 机器人描述
│       │   └── ur3_ft300_robotiq_2f85.urdf.xacro
│       ├── src/                        # C++ 控制插件
│       │   ├── pick_and_place.cpp      # 抓取放置控制器
│       │   └── peg_in_hole.cpp         # 孔轴装配控制器 (带力+stage)
│       └── scripts/                    # Python 脚本
│           ├── pick_and_place/         # 【Pi0 抓取放置脚本】
│           │   ├── ur3_pi0_pick_place_record.py              # 数据录制
│           │   ├── ur3_pi0_pick_place_make_video.py          # 数据可视化
│           │   ├── ur3_pi0_pick_place_convert_to_lerobot.py  # npz→LeRobot
│           │   ├── ur3_pi0_pick_place_ros_side.py            # ROS 通信端
│           │   ├── ur3_pi0_pick_place_inference.py           # Pi0 推理端
│           │   ├── ur3_pi0_pick_place_eval_offline.py        # 离线评估
│           │   ├── ur3_pi0_pick_place_eval_lora.py           # LoRA 评估
│           │   ├── ur3_pi0_pick_place_compute_ik.py          # IK 计算
│           │   ├── ur3_pi0_pick_place_merge_lora.py          # LoRA 合并
│           │   └── ur3_pi0_pick_place_fix_checkpoint_keys.py # key 修复
│           └── peg_in_hole/            # 【孔轴装配脚本】
│               ├── ur3_samoe_peg_in_hole_record.py           # 数据录制
│               ├── ur3_samoe_peg_in_hole_make_video.py       # 数据可视化
│               ├── ur3_samoe_peg_in_hole_convert_to_lerobot.py  # npz→LeRobot
│               ├── ur3_samoe_peg_in_hole_inference.py        # SA-MOE 推理
│               └── ur3_samoe_peg_in_hole_ros_side.py         # SA-MOE ROS 端
├── docs/                               # 文档
│   ├── A100_RETRAIN.md                 # Pi0 V1→V2 vision fix
│   ├── A100_TRAINING_V2.md             # Pi0 V2 训练指南
│   ├── A100_V3_TRAINING_GUIDE.md       # Pi0 V3 训练指南 (10fps)
│   └── SA_MOE_DESIGN.md               # SA-MOE V6 设计
├── PROJECT_REFERENCE.md                # 【本文档】
├── PI0_TRAINING_ANALYSIS.md            # Pi0 V1/V2 分析报告
├── SA_MOE_CHANGELOG.md                 # SA-MOE 版本改动记录
└── ForceVLA/                           # ForceVLA 参考实现
```

---

# Part 1 — Pi0 微调

## 1.1 背景

在 UR3 + FT300 + Robotiq 2F85 + Gazebo Fortress 环境中，微调 Pi0 (PaliGemma 2B + Action Expert 300M) 完成 "抓取红色方块放入蓝色碗中" 任务。

## 1.2 数据集

### 原始采集数据
```
ai-models/datasets/ur3_pick_place_raw/
```
- **来源**: `pick_and_place.cpp` (C++ 控制器) + `ur3_record_pick_place.py` (Python 录制)
- **格式**: 每 episode 一个目录，内含 `data.npz`
- **内容**: state(7), action(7), camera0(224x224x3), camera1(224x224x3)
- **规模**: 49 episodes, ~168K 帧, 50fps
- **动作格式**: 绝对关节位置 (action ≈ next_state)

### LeRobot 格式数据集 (50fps)
```
ai-models/datasets/ur3_pick_place_lerobot/
```
- **生成命令**:
  ```bash
  conda activate pi0-env
  python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_pick_place_raw \
    --repo_id local/ur3_pick_place \
    --fps 50
  ```

### LeRobot 格式数据集 (10fps, V3使用)
```
ai-models/datasets/ur3_pick_place_10hz_lerobot/
```
- **生成命令**:
  ```bash
  conda activate pi0-env
  python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_pick_place_raw \
    --repo_id local/ur3_pick_place_10hz \
    --fps 10 --source_fps 50
  ```

## 1.3 预训练模型

```
ai-models/pi0/pi0_libero_base/
├── model.safetensors    # 6.6GB
└── config.json
```
- 来源: lerobot/pi0_libero_base (HuggingFace)
- 架构: PaliGemma 2B + Action Expert 300M (4B 总参数)
- SigLIP 400M 视觉编码器 (LIBERO 任务预训练)

## 1.4 训练版本

### V1 — 首次尝试 (失败)

| 参数 | 值 |
|------|-----|
| 数据集 | ur3_pick_place_lerobot (50fps) |
| LR | 1e-5 → 2.5e-6 (cosine, 30K decay) |
| Steps | 84,000 |
| Batch | 2 |
| 最终 Loss | 0.012 |
| 推理 MAE | 0.37 rad |

**Bug**: `modeling_pi0.py` 的 `_fix_pytorch_state_dict_keys` 未处理 `.vision_model.` 路径差异 → **437 个 vision keys 静默跳过** → 视觉编码器随机初始化训练。

### V2 — Vision Fix (失败)

| 参数 | 值 |
|------|-----|
| 数据集 | ur3_pick_place_lerobot (50fps) |
| LR | 2e-5 → 2.5e-6 (cosine, 80K decay) |
| Steps | 84,000 |
| Batch | 2 |
| 最终 Loss | 0.006 |
| 推理 MAE | 0.34 rad |

**修复**: 在 `modeling_pi0.py:~1096` 添加:
```python
new_key = new_key.replace(".vision_model.", ".")
if not new_key.startswith("model.") and not new_key.startswith("normalize_"):
    new_key = "model." + new_key
```

**仍然失败的原因**: 50fps 下 action ≈ state（帧间差 ≈ 0.0004 rad），模型学到恒等映射 (identity shortcut)。Loss 极低但 MAE 极高。

### V3 — 10fps 降采样 (当前版本，效果不佳)

| 参数 | 值 |
|------|-----|
| 数据集 | ur3_pick_place_10hz_lerobot (10fps) |
| LR | 2e-5 → 2.5e-6 (cosine, 130K decay) |
| Steps | 135,000 |
| Batch | 2 |
| num_inference_steps | 50 |
| 最终 Loss | 0.006 |
| 推理 MAE | **0.35 rad (1.1×std)** |

**关键发现**: 即使降到 10fps，action ≈ state 的问题仍然存在（10Hz 差异 ~0.001 rad）。模型仍然学 identity shortcut。

**训练命令**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env

python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/pi0 \
    --policy.pretrained_path=/home/a/ur3_ft300_ws/ai-models/pi0/pi0_libero_base \
    --policy.num_inference_steps=50 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --policy.n_obs_steps=1 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.use_relative_actions=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --dataset.repo_id=local/ur3_pick_place_10hz \
    --dataset.root=/home/a/ur3_ft300_ws/ai-models/datasets/ur3_pick_place_10hz_lerobot \
    --dataset.streaming=false \
    --dataset.use_imagenet_stats=true \
    --dataset.image_transforms.enable=true \
    --batch_size=2 \
    --steps=135000 \
    --num_workers=4 \
    --optimizer.type=adamw \
    --optimizer.lr=2e-5 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip_norm=1.0 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=130000 \
    --scheduler.peak_lr=2e-5 \
    --scheduler.decay_lr=2.5e-6 \
    --seed=1000 \
    --output_dir=outputs/train/ur3_pi0_full_v3 \
    --save_freq=10000 \
    --log_freq=50
```

### 根本问题 (V1-V3 共通)

**Identity Shortcut**: 绝对动作格式下 `action ≈ state`（即使在 10fps），模型不需要看图像就能通过恒等映射获得极低 loss。

**解决方案**: 训练时使用 `use_relative_actions=true` — 模型内部自动处理 `action_rel = action - state`。推理时模型输出 delta，脚本自动加回 state 得到绝对关节角。这是 Pi0 官方支持的相对动作机制，比手动在数据中存 delta 更标准。

## 1.5 Pi0 脚本说明

### 数据采集

```bash
# 1. 启动 Gazebo
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py

# 2. 运行录制 (Python 自动 spawn 方块+碗 → 启动 C++ pick_and_place → 录制 50Hz)
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_record.py --episodes 50
```

### 数据转换
```bash
# npz → LeRobot 格式
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_pick_place_raw \
    --repo_id local/ur3_pick_place_10hz \
    --fps 10 --source_fps 50
```

### 数据可视化
```bash
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_make_video.py \
    --input ai-models/datasets/ur3_pick_place_raw \
    --episode 0
```

### Gazebo 实测 (Pi0 V3)
```bash
# 终端 1: ROS 通信端 (系统 Python 3.10)
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_ros_side.py --spawn

# 终端 2: 推理端 (conda pi0-env)
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_inference.py \
    --mode bf16 --hz 10 \
    --model ai-models/pi0/pi0_full_v3/checkpoints/135000/pretrained_model
```

### 离线评估
```bash
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_eval_offline.py \
    --model ai-models/pi0/pi0_full_v3/checkpoints/135000/pretrained_model \
    --dataset ai-models/datasets/ur3_pick_place_10hz_lerobot \
    --episodes 0 5 10  # 前5个episode
```

---

# Part 2 — SA-MOE (孔轴装配)

## 2.1 背景

在 UR3 + FT300 + Gazebo 中完成 Peg-in-Hole 精密装配任务。SA-MOE (Stage-Aware Mixture of Experts) 利用力传感数据+阶段标签实现分阶段专家路由。

## 2.2 数据集

### 原始采集数据
```
ai-models/datasets/ur3_peg_in_hole_raw/
```
- **来源**: `peg_in_hole.cpp` (C++ 控制器 + 力数据 + stage 标签) + `ur3_record_peg_in_hole.py` (Python 录制)
- **格式**: data.npz (state, action, force, stage, images)
- **特点**: 含 6 维力数据 + C++ 真实 stage 标签 (0-4)
- **Stage 定义**: 0=approach, 1=contact, 2=insert, 3=confirm, 4=retract

### LeRobot 格式数据集
```
ai-models/datasets/ur3_peg_in_hole_lerobot/
```
- **生成命令**:
  ```bash
  conda activate pi0-env
  python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_peg_in_hole_raw
  ```

## 2.3 SA-MOE 架构 (V8 最终版)

```
Input: [PaliGemma E_VL | force_tokens]
         ↓
    StageGate (Attention Pool) → 选择 expert (per-sample routing)
         ↓
    TransformerExpert (2-layer self-attn) → 阶段特化处理
         ↓
    ModalGate → α 调制
         ↓
    Residual: α × expert_out[:, -50:] + suffix_out[:, -50:]
         ↓
    Action decoder → 7-dim 关节动作
```

- **172M 可训练参数**
- **数据流**: residual addition (不是 condition prepend)
- **三个 Active 组件**: StageGate, TransformerExpert, ModalGate

## 2.4 SA-MOE 训练版本

| 版本 | 日期 | Steps | Loss | 说明 |
|------|------|-------|------|------|
| V1 | 0620 | 100K | 2.2→1.4 | 绝对动作，伪标签 |
| V2 | 0620 | 5K | 1.99→1.62 | Delta 动作测试 |
| V3 | 0620 | 100K | flat 1.60 | Delta + 伪标签，未收敛 |
| V4 | 0622 | 16K | ? | class_weights，未完成 |
| V7 | 0622 | 30K | 1.15→1.04 | ForceVLA-style，真实标签 |
| V8 | 0622 | **10K/30K** | 1.87→**1.08** | Residual addition (未完成) |

### V8 当前状态

- **日志**: `/tmp/sa_moe_v8.log`
- **进度**: 10K/30K steps (33%), 训练已停止
- **Loss**: 1.87 (step 200) → 1.08 (step 10K)
- **问题**:
  - stage_acc = 0.0% — stage 分类器未学到任何东西
  - alpha 锁定在 0.05 — ModalGate 几乎关闭 SA-MOE 的 residual 贡献
  - expert 4 占 50% — routing 崩溃

### V8 训练命令
```bash
# 见 ai-models/run_train_sa_moe.sh (v1)
# 和 /tmp/sa_moe_v8_config (v8 配置)
```

## 2.5 SA-MOE 脚本说明

### 数据采集

```bash
# 1. 启动 Gazebo
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py

# 2. 运行录制 (Python 自动 spawn peg+hole → 启动 C++ peg_in_hole → 录制 10Hz + 力+stage)
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_record.py --episodes 50
```

### 数据可视化
```bash
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_make_video.py \
    --input ai-models/datasets/ur3_peg_in_hole_raw \
    --episode 0
```

### 数据转换
```bash
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_peg_in_hole_raw
```

### Gazebo 实测 (SA-MOE)
```bash
# 终端 1: ROS 端 (系统 Python)
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_ros_side.py

# 终端 2: 推理端 (conda pi0-env)
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_inference.py \
    --mode bf16 --hz 10 \
    --model ai-models/samoe/v7/checkpoints/030451/pretrained_model
```

---

# Part 3 — 文件速查表

## 启动/仿真

| 文件 | 用途 | 运行方式 |
|------|------|---------|
| `launch/ur3_ft300_robotiq.launch.py` | 启动 Gazebo 仿真 | `ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py` |

## 抓取放置 (Pi0)

| 文件 | 用途 | 环境 |
|------|------|------|
| `src/pick_and_place.cpp` | C++ 控制节点 (由 record.py 通过 ros2 run 启动) | Gazebo 编译时加载 |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_record.py` | 数据录制 (spawn→启C++→录50Hz→存npz) | 系统 Python 3.10 |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_make_video.py` | 将采集数据可视化为视频 | pi0-env |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_convert_to_lerobot.py` | npz → LeRobot 格式转换 | pi0-env |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_ros_side.py` | ROS 通信端 (订阅相机+关节，发送动作) | 系统 Python 3.10 |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_inference.py` | Pi0 推理端 (加载模型，输出动作) | pi0-env |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_eval_offline.py` | 离线评估 Pi0 模型 | pi0-env |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_eval_lora.py` | LoRA 模型评估 | pi0-env |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_compute_ik.py` | IK 计算 | 系统 Python |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_merge_lora.py` | LoRA 权重合并 | pi0-env |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_fix_checkpoint_keys.py` | 修复 checkpoint key 名称 | pi0-env |

## 孔轴装配 (SA-MOE)

| 文件 | 用途 | 环境 |
|------|------|------|
| `src/peg_in_hole.cpp` | C++ 控制节点 (由 record.py 通过 ros2 run 启动, 含力+stage) | Gazebo 编译时加载 |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_record.py` | 数据录制 (spawn→启C++→录10Hz+力+stage→存npz) | 系统 Python 3.10 |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_make_video.py` | 数据可视化 | pi0-env |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_convert_to_lerobot.py` | npz → LeRobot 转换 (含力+stage) | pi0-env |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_ros_side.py` | ROS 通信端 (含力传感器订阅) | 系统 Python 3.10 |
| `src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_inference.py` | SA-MOE 推理端 | pi0-env |

## 文档

| 文件 | 内容 |
|------|------|
| `PROJECT_REFERENCE.md` | 【本文档】完整项目参考 |
| `PI0_TRAINING_ANALYSIS.md` | Pi0 V1/V2 失败分析 |
| `docs/A100_RETRAIN.md` | Pi0 V1→V2 vision fix 详细 |
| `docs/A100_TRAINING_V2.md` | Pi0 V2 训练步骤 |
| `docs/A100_V3_TRAINING_GUIDE.md` | Pi0 V3 训练指南 (10fps) |
| `docs/SA_MOE_DESIGN.md` | SA-MOE V6 架构设计 |
| `SA_MOE_CHANGELOG.md` | SA-MOE 各版本改动记录 |

## 工具脚本

| 文件 | 用途 |
|------|------|
| `scripts/fix_checkpoint_keys.py` | 修复 checkpoint key 名称不匹配 |

---

# Part 4 — 常用命令速查

## 采集数据

```bash
# 先启动 Gazebo
ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py

# 抓取放置 (50Hz)
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_record.py --episodes 50

# 孔轴装配 (10Hz, 含力+stage标签)
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_record.py --episodes 50
```

## 训练 Pi0

```bash
conda activate pi0-env

# V3 训练 (10fps, 135K steps)
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/pi0 \
    --policy.pretrained_path=ai-models/pi0/pi0_libero_base \
    --policy.num_inference_steps=50 \
    --policy.dtype=bfloat16 --policy.device=cuda \
    --dataset.repo_id=local/ur3_pick_place_10hz \
    --dataset.root=ai-models/datasets/ur3_pick_place_10hz_lerobot \
    --batch_size=2 --steps=135000 \
    --output_dir=outputs/train/ur3_pi0_v4
```

## 测试 Pi0

```bash
# 终端 1:
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_ros_side.py --spawn

# 终端 2:
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_inference.py \
    --mode bf16 --hz 10 \
    --model ai-models/pi0/pi0_full_v3/checkpoints/135000/pretrained_model
```

## 评估 Pi0

```bash
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_eval_offline.py \
    --model ai-models/pi0/pi0_full_v3/checkpoints/135000/pretrained_model \
    --dataset ai-models/datasets/ur3_pick_place_10hz_lerobot
```

## 数据转换

```bash
# 抓取放置: npz → LeRobot
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_pick_place_raw \
    --repo_id local/ur3_pick_place \
    --fps 10 --source_fps 50

# 孔轴装配: npz → LeRobot
python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_samoe_peg_in_hole_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_peg_in_hole_raw
```

---

# Part 5 — 已知问题 & 下一步

## Pi0
- [x] **Identity Shortcut**: 已通过 `use_relative_actions=true` 修复 ✅
- [x] V3 推理 `num_inference_steps` 已从 10 修复为 50 ✅
- [x] **图像质量**: 相机分辨率 224→640 + 补光灯 ✅
- [x] **摄像头频率**: parameter_bridge → ros_gz_image/image_bridge (无损二进制), update_rate 30→50 ✅
- [x] **图像缩放**: cv2.resize INTER_LINEAR → INTER_AREA (区域平均，全像素参与) ✅ (2026-06-23)
- [x] **直接10Hz录制**: 新增 --hz 参数, 默认10Hz, 帧间delta有意义 ✅ (2026-06-23)
- [x] **失败数据过滤**: convert 脚本自动跳过 _failed 目录 ✅ (2026-06-23)
- [ ] **夹爪重力下垂**: Humble gz_ros2_control `position` 接口不做力矩级 PID
- [ ] wrist_2 关节方差 ~0 (机械耦合)，可能影响学习

## SA-MOE
- [x] **数据管道同步**: peg-in-hole 脚本同步 INTER_AREA + --hz + failed 过滤 ✅ (2026-06-23)
- [ ] V8 训练中断于 10K/30K，需恢复或重启
- [ ] **stage_acc 持续为 0%** — 诊断: V8 (1024-dim) 加载 delta_full (2048-dim) 维度不匹配，SA-MOE 头随机初始化
- [ ] **alpha 锁定 0.05** — 诊断: stage_acc=0% → 错误 stage 预测 → stage_priors 压垮 alpha logit → clamp
- [ ] V8_balanced 因 episode index 越界崩溃

---

# Part 6 — 2026-06-23 改动记录

## 6.1 摄像头频率修复

**根因**: Gazebo 原生 camera ~51Hz, 但 `ros_gz_bridge/parameter_bridge` 对 900KB 图像做文本序列化, ROS 2 只收到 10-18Hz。

**修复**: `parameter_bridge` → `ros_gz_image/image_bridge` (专用二进制图像传输, 无损)

**文件**: `launch/ur3_ft300_robotiq.launch.py`
- 两个 `parameter_bridge` 节点 → 两个独立 `image_bridge` 节点
- wrist_camera: 15→22-34Hz, global_camera: 28→18-23Hz (总吞吐持平, 手腕更优)

**URDF**: `update_rate` 30→50 (匹配录制频率)

## 6.2 图像缩放优化

**文件**:
- `scripts/pick_and_place/ur3_pi0_pick_place_record.py`
- `scripts/peg_in_hole/ur3_samoe_peg_in_hole_record.py`
- `scripts/peg_in_hole/ur3_samoe_peg_in_hole_ros_side.py`

**改动**: `cv2.resize()` 默认 INTER_LINEAR (2×2邻域) → `interpolation=cv2.INTER_AREA` (区域平均, 所有源像素参与)
- 640×480→224×224 下采样, INTER_AREA 是 OpenCV 推荐的下采样插值法
- 本质: 640 渲染提供超采样抗锯齿 (SSAA), INTER_AREA 在 CPU 端做高质量平均

## 6.3 直接 10Hz 录制

**根因**: 50Hz 录制 → 降采样 10Hz 浪费磁盘 + action delta ~0.0004 rad 导致 Identity Shortcut

**修复**: 录制脚本新增 `--hz` 参数 (默认 10), 直接 10Hz 录制
- 10Hz 帧间 100ms, action delta ~0.01-0.05 rad (有意义)
- 摄像头 22+Hz > 10Hz → 每帧图像唯一
- 转换脚本 `--source_fps` 匹配录制 Hz, 无需降采样

**文件**:
- `scripts/pick_and_place/ur3_pi0_pick_place_record.py` — 新增 `--hz` 参数
- `scripts/peg_in_hole/ur3_samoe_peg_in_hole_record.py` — 新增 `--hz` 参数
- `scripts/pick_and_place/ur3_pi0_pick_place_convert_to_lerobot.py` — find_episodes() 跳过 `_failed`
- peg-in-hole convert 已有 failed 过滤, 无需改

## 6.4 失败数据过滤

**文件**: `scripts/pick_and_place/ur3_pi0_pick_place_convert_to_lerobot.py`

**改动**: `find_episodes()` 检测目录名含 `_failed` → 打印 `⏭ 跳过` → 不进入 LeRobot

## 6.5 SA-MOE V8 诊断

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| stage_acc=0% | V8 config `sa_moe_feature_dim=1024` 但 pretrained delta_full 是 2048-dim → 维度不匹配 → SA-MOE 头随机初始化 | 设置 `sa_moe_feature_dim=2048` 以匹配 pretrained checkpoint |
| alpha=0.05 | stage_acc=0% → argmax(stage_probs) 随机 → stage_priors[-3.0] 加到 alpha logit → softmax 后接近 0 → clamp 到 0.05 | 修复 stage_acc 后 alpha 应自然恢复 |
| batch_size=1 | 稀有 stage (insert/tighten) 每 15-30 步才出现一次 | 尝试 batch_size=2 或重采样 |

### 维度不匹配详情

```
V8 config:     sa_moe_feature_dim = 1024
delta_full:    transformer_encoder dim = 2048

加载时 transformer_encoder 所有参数 size mismatch:
  copying param from [6144, 2048] → model expects [3072, 1024]
  结果: 跳过加载, 使用随机初始化
```

### 修复步骤 (待实施)

1. 修改 V8 config: `sa_moe_feature_dim: 1024 → 2048`
2. 确认 `fusion_dim` 也匹配 (delta_full 是 2048)
3. batch_size 1→2 (如果显存够)
4. 可选: 降低 `stage_loss_weight` 0.2→0.05 (stage 分类弱, 降低权重让主 loss 主导)
5. 可选: 对齐 `stage_priors` 到实际数据分布 (当前 prior 偏向 stage 0, 但数据以 stage 0/4 为主)

## 6.6 当前操作顺序

1. ✅ 修改所有脚本 (pick-and-place + peg-in-hole)
2. ⏳ 采集 pick-and-place 数据 (50ep, 10Hz)
3. ⏳ 转换 → LoRA 训练 → 评估
4. ⏳ 采集 peg-in-hole 数据
5. ⏳ 修复 SA-MOE V8 维度问题 → 训练 → 评估

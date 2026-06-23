# Pi0 V5 全量微调指南 (A100 80GB)

> 基于 2026-06-23 数据管道优化后的重新训练
> 从本地推送到 GitHub 后在 A100 上拉取执行

---

## 前置条件

A100 上需要：
- git clone 本仓库
- conda 环境 `pi0-env`（与本地一致）
- 数据集上传到 `/home/a/ur3_ft300_ws/ai-models/datasets/ur3_pick_place_10hz_v4_lerobot/`
- 预训练模型: `/home/a/ur3_ft300_ws/ai-models/pi0/pi0_libero_base/`

---

## 数据管道（本地已完成）

数据已在本地采集并转换：

```bash
# 本地采集（A100 无需执行）
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_record.py \
    --episodes 50 --hz 10

# 本地转换（A100 无需执行）
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_convert_to_lerobot.py \
    --input ai-models/datasets/ur3_pick_place_raw \
    --repo_id local/ur3_pick_place_10hz_v4 \
    --fps 10 --source_fps 10 \
    --output_dir ai-models/datasets/ur3_pick_place_10hz_v4_lerobot
```

**A100 上需要做的事**: 把 `ur3_pick_place_10hz_v4_lerobot/` 整个目录 scp 到 A100 对应路径。

---

## Step 1: 上传数据集到 A100

```bash
# 本地执行
scp -r ai-models/datasets/ur3_pick_place_10hz_v4_lerobot \
    a100:~/ur3_ft300_ws/ai-models/datasets/
```

## Step 2: A100 上训练

```bash
# SSH 到 A100
ssh a100
cd ~/ur3_ft300_ws

# 启动训练
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
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --dataset.repo_id=local/ur3_pick_place_10hz_v4 \
    --dataset.root=/home/a/ur3_ft300_ws/ai-models/datasets/ur3_pick_place_10hz_v4_lerobot \
    --dataset.streaming=false \
    --dataset.use_imagenet_stats=true \
    --dataset.image_transforms.enable=true \
    --batch_size=4 \
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
    --output_dir=outputs/train/ur3_pi0_v5_full \
    --save_freq=10000 \
    --log_freq=50
```

### 关键参数 vs V3/V4

| 参数 | V3 (失败) | V4 Expert | **V5 Full** |
|------|----------|-----------|-------------|
| `use_relative_actions` | **false** ❌ | true | **true** |
| `train_expert_only` | false | **true** ❌ | **false** |
| `batch_size` | 2 | 2 | **4** (A100) |
| `steps` | 135K | 50K | **135K** |
| 数据集 fps | 10 (降采样) | **10 (直接录制)** | 10 (直接录制) |
| 图像 resize | INTER_LINEAR | **INTER_AREA** | INTER_AREA |
| 预计时间 | ~14h | ~4.5h | **~8-10h** |

### 训练监控

```
正常现象:
  - 初始 loss: ~40,000 (flow matching noise prediction loss)
  - 最终 loss: < 0.5 为佳
  - batch_size=4, ~3.5-4 steps/sec on A100

异常标志:
  - loss 不下降超过 5000 steps → 检查数据/配置
  - loss 降到 ~0.001 但 MAE 高 → Identity Shortcut 复发
```

## Step 3: 离线评估

```bash
conda activate pi0-env
python -m lerobot.scripts.lerobot_eval \
    --policy.path=outputs/train/ur3_pi0_v5_full/checkpoints/last/pretrained_model \
    --dataset.repo_id=local/ur3_pick_place_10hz_v4 \
    --dataset.root=/home/a/ur3_ft300_ws/ai-models/datasets/ur3_pick_place_10hz_v4_lerobot
```

**成功标准**: MAE < 0.15 rad (V3=0.35 rad, 零增量基线≈0.0004 rad)

## Step 4: 复制结果回本地

```bash
# 本地执行
scp -r a100:~/outputs/train/ur3_pi0_v5_full ~/ur3_ft300_ws/ai-models/pi0/
```

## Step 5: Gazebo 实测

```bash
# 终端 1: 先复位机器人到 READY
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
    "{joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', \
    'wrist_2_joint', 'wrist_3_joint', 'robotiq_85_left_knuckle_joint'], \
    points: [{positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.1], \
    time_from_start: {sec: 3, nanosec: 0}}]}" --once

# 终端 2: ROS 通信端
/usr/bin/python3 src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_ros_side.py --spawn

# 终端 3: 推理端
conda activate pi0-env
python src/ur_simulation_gz/ur_simulation_gz/scripts/pick_and_place/ur3_pi0_pick_place_inference.py \
    --mode bf16 --hz 10 \
    --model ai-models/pi0/pi0_v5_full/checkpoints/135000/pretrained_model
```

## 故障排除

| 问题 | 原因 | 解决 |
|------|------|------|
| loss 不下降 | lr 太高/太低 | 尝试 lr=1e-5 或 5e-5 |
| MAE ≈ state_std | Identity Shortcut | 确认 `use_relative_actions=true` |
| OOM | batch_size 太大 | 降到 2 或 1 |
| `HF_HUB_OFFLINE` 错误 | 无网络 | `export HF_HUB_OFFLINE=1` |
| `policy.repo_id` 缺失 | config 验证 | 加 `--policy.repo_id=local/xxx --policy.push_to_hub=false` |

---

> 本地已完成的代码修改已推送到 GitHub。A100 上 `git pull` 即可获取数据管道优化。

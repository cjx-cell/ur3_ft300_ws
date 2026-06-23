# Pi0 UR3 Pick-and-Place V3 训练指南

> 目标：在 A100 服务器上从头训练 Pi0 (4B params)，使用降采样 10fps 数据集修复 V2 的高 MAE 问题。

## 1. V2 为什么失败

| | V2 (失败) | 根因 |
|---|---|---|
| 训练 Loss | 0.006 | Diffusion noise prediction loss，不代表动作准确度 |
| 推理 MAE | **0.37 rad** (21°/关节) | 比预测均值 (0.28 rad) 还差 33% |
| 比"不动" | 差 **92,000%** | 零增量基线 MAE = 0.0004 rad |
| Gazebo 表现 | 随机游走 | 模型输出 ≈ 训练集均值 + 有害噪声 |

**三个根因叠加**：
1. **50fps 帧间变化太小** — `action[t+1] - action[t] ≈ 0.0004 rad`，连续帧几乎一样，模型学不到有意义的映射
2. **1 epoch 不够** — 84K steps / (167,962 frames / batch_size=2) = 1.0 epoch，4B 参数模型严重欠拟合
3. **num_inference_steps=10 不充分** — Euler 积分步数太少，去噪过程累积误差

## 2. V3 的修改

| 参数 | V2 | V3 | 原因 |
|------|----|----|------|
| 数据集 fps | 50 | **10** | 帧间动作差 5x，模型能学到有意义的轨迹变化 |
| 数据集 frames | 167,962 | **33,610** | 降采样 5:1，同样物理时间 |
| num_inference_steps | 10 | **50** | Euler 积分更充分，减少去噪累积误差 |
| steps | 84,000 | **135,000** | ~8 epochs over 33,610 frames |
| scheduler_decay_steps | 80,000 | **130,000** | 覆盖大部分训练步数 |
| use_relative_actions | false | **true** | ✓ 修复 Identity Shortcut: 模型内部 action_rel = action - state |
| 其他参数 | — | **不变** | LR=2e-5, batch_size=2, 预训练权重从头加载 |

## 3. 前置准备

### 3.1 复制数据集到 A100

```bash
# 在本地机器上
scp -r ~/ur3_ft300_ws/ai-models/datasets/ur3_pick_place_10hz_lerobot a100:~/ur3_ft300_ws/ai-models/
```

### 3.2 确认代码中已有 vision key fix

检查 A100 上的 `/home/a/lerobot/src/lerobot/policies/pi0/modeling_pi0.py` 第 ~1096-1104 行是否包含：

```python
# Fix 1: Remove ".vision_model." from vision tower path
new_key = new_key.replace(".vision_model.", ".")

# Fix 2: Add "model." prefix to keys that lack it
if not new_key.startswith("model.") and not new_key.startswith("normalize_"):
    new_key = "model." + new_key
```

如果 A100 上的 lerobot 是旧版本（无此修复），先同步：

```bash
# 方案 A: 从本地 rsync 整个 lerobot 目录
rsync -avz ~/lerobot/ a100:~/lerobot/

# 方案 B: 只复制修复后的文件
scp ~/lerobot/src/lerobot/policies/pi0/modeling_pi0.py a100:~/lerobot/src/lerobot/policies/pi0/
```

### 3.3 确认预训练模型

```bash
ls ~/ur3_ft300_ws/ai-models/pi0/pi0_libero_base/model.safetensors
# 应该存在，~6.6GB
```

## 4. 训练命令

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
    --policy.use_relative_actions=true \
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
    --prefetch_factor=4 \
    --persistent_workers=true \
    --optimizer.type=adamw \
    --optimizer.lr=2e-5 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip_norm=1.0 \
    --optimizer.betas='[0.9,0.95]' \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=130000 \
    --scheduler.peak_lr=2e-5 \
    --scheduler.decay_lr=2.5e-6 \
    --seed=1000 \
    --output_dir=outputs/train/ur3_pi0_full_v3 \
    --job_name=pi0_v3 \
    --save_freq=10000 \
    --eval_freq=20000 \
    --log_freq=50 \
    --save_checkpoint=true \
    --resume=false
```

### 关键参数说明

```
batch_size=2          A100 80GB 可以尝试 batch_size=4（如 OOM 则回退到 2）
steps=135000          135K / (33610/2) ≈ 8 epochs
num_decay_steps=130000 LR 从 2e-5 cosine decay 到 2.5e-6，覆盖绝大部分训练
num_inference_steps=50 推理时 Euler 积分步数（训练时不使用，仅影响 eval）
```

## 5. 训练时间预估

- V2 基准: 84K steps / batch_size=2 ≈ 8.5h (A100 80GB)
- V3: 135K steps / batch_size=2 ≈ **13-14h**
- 如果 batch_size=4 可行: **7-8h**

## 6. 训练后验证

### 6.1 检查 Loss 曲线

Loss 降到 0.005 以下是正常的，但**不能只看 loss**。

### 6.2 评估 MAE

```bash
python -m lerobot.scripts.lerobot_eval \
    --policy.path=outputs/train/ur3_pi0_full_v3/checkpoints/last/pretrained_model \
    --dataset.repo_id=local/ur3_pick_place_10hz \
    --dataset.root=/home/a/ur3_ft300_ws/ai-models/datasets/ur3_pick_place_10hz_lerobot
```

**V3 成功标准**：
- 总体 MAE < **0.15 rad**（V2 是 0.37，零增量基线是 0.0004）
- 比"预测均值"基线 (0.28 rad) 要好 50%+
- 如果 MAE 仍然 > 0.25 rad → 训练失败，需要进一步诊断

### 6.3 Gazebo 实测

模型复制回本地后，在 Gazebo 中跑 rollout：
```bash
# 本地机器
python3 ur3_pi0_inference.py --model_path <v3_checkpoint_path>
```

## 7. 复制结果回本地

```bash
# 在本地机器上
scp -r a100:~/outputs/train/ur3_pi0_full_v3 ~/ur3_ft300_ws/ai-models/
```

## 8. 如果 V3 仍然失败

如果 MAE 仍然 > 0.25 rad，可能的原因和解决方案：

1. **Action 表示问题** — 已通过 `use_relative_actions=true` 修复。如果仍失败，检查 `relative_exclude_joints` 是否排除了正确的关节。gripper 关节通常应排除（非连续运动）。

2. **chunk_size 太大** — 50 tokens × 5s horizon 跨度太大。尝试 `chunk_size=20` (2s horizon at 10fps)

3. **仍需更多数据** — 49 episodes 可能仍不够。扩展数据集到 100+ episodes

4. **LoRA 替代全量微调** — 只微调 ~2% 参数可能更稳定 (`use_peft=true`)

## 9. 文件清单

| 路径 | 说明 |
|------|------|
| `~/ur3_ft300_ws/ai-models/datasets/ur3_pick_place_10hz_lerobot/` | 10fps 数据集 |
| `~/ur3_ft300_ws/ai-models/pi0/pi0_libero_base/` | Pi0 预训练权重 (6.6GB) |
| `~/lerobot/src/lerobot/policies/pi0/modeling_pi0.py` | 含 vision key fix (line 1096-1104) |
| `~/ur3_ft300_ws/docs/A100_V3_TRAINING_GUIDE.md` | 本文档 |

# A100 全量微调 v2 —— 训练指南

> **写给另一台服务器上的 Claude Code**
> 本文档包含完整的上下文、诊断和操作指令。直接按步骤执行即可。

---

## 1. 背景

### 项目目标
在 Gazebo Fortress 仿真 UR3 机器人上微调 Pi0 VLA 模型，执行 "pick up the red cube and place it into the bowl" 任务。

### 已完成的工作
- 50 条 UR3 pick-place 轨迹已录制并转换为 LeRobot 格式
- 数据集已上传 HuggingFace: `cjx-cell/ur3_pick_place`
- Pi0 基础模型: `lerobot/pi0_libero_base`（基于 LIBERO 130+ 任务预训练）

### v1 训练（失败）
- 配置: 全量微调, LR=1e-5, batch_size=2, steps=84000
- Scheduler: cosine decay, warmup=1000, decay_steps=30000, decay_lr=2.5e-6
- 数据集: 168K 帧, 49 episodes
- **结果: 训练 loss = 0.012, 但推理 MAE = 0.379 rad（极差）**

### 根因诊断
```
Step 0-1K:    warmup
Step 1K-30K:  LR 从 1e-5 → 2.5e-6   ← 唯一有效学习期 (~29K 步)
Step 30K-84K: LR = 2.5e-6 (恒定)     ← 54K 步浪费，学习率太低
```

**scheduler_decay_steps=30000 但总 steps=84000 → 64% 的训练时间在最低 LR 白跑。**
全量微调 2.3B 参数需要更强的优化信号。

证据——模型连训练数据都拟合不了：
```
Episode 0 (训练数据) 单集评估:
  Mean MAE:   0.373 rad
  shldr_pan:  1.074 rad  ← 完全没学到
  wrist_3:    0.954 rad  ← 完全没学到
```

### Gazebo 推理表现
- 模型加载正常（2.3s, 7.5GB GPU, 25ms/步）
- 机械臂在动但**不朝目标移动**——没有建立起视觉→动作映射
- 夹爪在 FREE↔CLOSING↔OPENING 之间无限循环——没学会"抓住后保持"

---

## 2. 环境准备

### 2.1 克隆仓库
```bash
git clone https://github.com/cjx-cell/ur3_ft300_ws.git ~/ur3_ft300_ws
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```

### 2.2 Conda 环境
```bash
conda create -n pi0-env python=3.12 -y
conda activate pi0-env
cd ~/lerobot
pip install -e .
pip install safetensors torch torchvision transformers accelerate peft
```

### 2.3 下载基础模型
```bash
conda activate pi0-env
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/pi0_libero_base',
                   local_dir='./ai-models/lerobot/pi0_libero_base')
"
```

### 2.4 验证基础模型 config
确认 `~/ur3_ft300_ws/ai-models/lerobot/pi0_libero_base/config.json` 内容正确：
- `input_features`: `observation.images.camera0` (3,480,640), `observation.images.camera1` (3,480,640), `observation.state` (7,)
- `output_features`: `action` (7,)
- `empty_cameras`: 0
- `dtype`: "bfloat16"
- `n_action_steps`: 50

（config.json 已在主仓库中修改好，直接使用即可）

### 2.5 下载 Tokenizer
需要 PaliGemma tokenizer 文件放在 `~/ur3_ft300_ws/ai-models/paligemma_tokenizer/`:
- `tokenizer.json`
- `tokenizer.model`
- `tokenizer_config.json`

如果 A100 服务器能访问 HuggingFace:
```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')
tok.save_pretrained('./ai-models/paligemma_tokenizer')
"
```

如果不能，从主工作站 scp:
```bash
# 在主工作站上:
scp -r ~/ur3_ft300_ws/ai-models/paligemma_tokenizer user@a100-server:~/ur3_ft300_ws/ai-models/
```

---

## 3. 训练命令

### 3.1 v2 训练（核心修复）

```bash
cd ~/ur3_ft300_ws
conda activate pi0-env

python -m lerobot.scripts.lerobot_train \
  --policy.path=./ai-models/lerobot/pi0_libero_base \
  --dataset.repo_id=cjx-cell/ur3_pick_place \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.n_action_steps=50 \
  --policy.optimizer_lr=3e-5 \
  --policy.scheduler_warmup_steps=2000 \
  --policy.scheduler_decay_steps=150000 \
  --policy.scheduler_decay_lr=1e-6 \
  --batch_size=2 \
  --steps=150000 \
  --tolerance_s=0.001 \
  --output_dir=./outputs/train/ur3_pi0_full_v2 \
  --save_freq=25000 \
  --log_freq=50
```

### 3.2 关键参数说明

| 参数 | v1 (失败) | v2 (本次) | 原因 |
|------|-----------|-----------|------|
| `optimizer_lr` | 1e-5 | **3e-5** | 全量微调 2.3B 需要更大步长 |
| `scheduler_warmup_steps` | 1000 | **2000** | 更平滑的 warmup |
| `scheduler_decay_steps` | 30000 | **150000** | 覆盖全训练周期 |
| `scheduler_decay_lr` | 2.5e-6 | **1e-6** | 更低终值，给后期留学习空间 |
| `steps` | 84000 | **150000** | ~2 epoch（168K 帧 / batch_size 2 = 84K/epoch） |
| `save_freq` | 10000 | **25000** | 更合理的 checkpoint 间隔 |

### 3.3 训练监控

训练过程中关注 `log_freq=50` 输出的 loss：

```
期望 loss 曲线:
  Step 0-2K:     loss ~0.5 → 0.05    (warmup + 快速下降)
  Step 2K-50K:   loss ~0.05 → 0.005  (持续下降)
  Step 50K-150K: loss ~0.005 → 0.001  (精调)
```

**如果 loss 在 50K 步后仍 > 0.01，说明 LR 太低或模型容量不足。**

---

## 4. 评估

### 4.1 训练中快速检查

每 25K 步保存 checkpoint，可以在训练进行中评估最新 checkpoint：

```bash
conda activate pi0-env
cd ~/ur3_ft300_ws

python src/ur_simulation_gz/ur_simulation_gz/scripts/eval_lora_model.py \
  --model ./outputs/train/ur3_pi0_full_v2/checkpoints/050000/pretrained_model \
  --ckpt ./outputs/train/ur3_pi0_full_v2/checkpoints/050000/pretrained_model
```

**合格标准**:
| MAE | 判断 |
|-----|------|
| < 0.10 rad | 优秀，可以部署 Gazebo 推理 |
| 0.10-0.20 rad | 良好，Gazebo 中可能工作 |
| 0.20-0.30 rad | 勉强，需要更多训练 |
| > 0.30 rad | 不合格，还是 v1 水平 |

### 4.2 最终评估

训练完成后评估最终 checkpoint：

```bash
python src/ur_simulation_gz/ur_simulation_gz/scripts/eval_lora_model.py \
  --model ./outputs/train/ur3_pi0_full_v2/checkpoints/150000/pretrained_model \
  --ckpt ./outputs/train/ur3_pi0_full_v2/checkpoints/150000/pretrained_model
```

**特别关注 shldr_pan 和 wrist_3 的 MAE**——这两个关节的 std 最大（0.53 rad），在 v1 中是完全没学到的（MAE > 0.9 rad）。

---

## 5. 文件传输

训练完成后，将 checkpoint 传回主工作站：

```bash
# 在 A100 服务器上:
scp -r ./outputs/train/ur3_pi0_full_v2/checkpoints/150000 \
    user@main-workstation:~/ur3_ft300_ws/ai-models/ur3_pi0_full_v2/checkpoints/
```

---

## 6. 常见问题

### Q: OOM (Out of Memory)
A100 80GB 应该足够。如果 OOM:
- 减小 `--batch_size=1`
- 但不建议减小——batch_size=1 会降低训练稳定性

### Q: 网络不可达
如果 HuggingFace 无法访问:
- `export HF_ENDPOINT=https://hf-mirror.com`
- 或者从主工作站 SCP 整个 `ai-models/` 目录

### Q: 训练中断
从 checkpoint 恢复:
```bash
# 在 train 命令中加:
#   --resume=true
#   --checkpoint_path=./outputs/train/ur3_pi0_full_v2/checkpoints/050000/training_state
```

### Q: loss 不下降
可能原因:
1. 数据集加载失败——检查 `cjx-cell/ur3_pick_place` 是否可访问
2. 网络问题导致数据流中断——检查日志
3. LR 仍需调整——如果 loss 平坦 > 5K 步，尝试 `--policy.optimizer_lr=5e-5`

---

## 7. 附录：v1 完整诊断

### v1 训练配置
- LR: 1e-5, scheduler_decay_steps: 30000, scheduler_warmup_steps: 1000
- decay_lr: 2.5e-6, steps: 84000, batch_size: 2
- freeze_vision_encoder: false, train_expert_only: false

### v1 评估结果
```
整体 MAE: 0.379 rad  (需要 < 0.10)
shldr_pan: 0.704 rad  (需要 < 0.10)
wrist_3:   0.750 rad  (需要 < 0.10)
wrist_2:   0.005 rad  (几乎不动，完美)

Episode 0 单集过拟合测试:
  Mean MAE: 0.373 rad  ← 训练数据，应该接近 0
  shldr_pan: 1.074 rad
  wrist_3: 0.954 rad
```

### v1 Gazebo 行为
- 机械臂在动但不朝目标移动
- 夹爪振荡 FREE↔CLOSING↔OPENING，从不保持
- 模型没有建立起视觉→动作的映射

### v1 根因
```
有效学习时间: 仅 1K-30K 步 (29K 步)
浪费学习时间: 30K-84K 步 (54K 步, LR=2.5e-6 几乎无参数更新)
1 epoch 不够 + 64% 时间在无效 LR = 模型没收敛
```

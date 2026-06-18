# UR3 Pi0 全量微调 v2

## 当前状态

v1 全量微调跑完了但模型没学到东西：

| 指标 | v1 结果 | 目标 |
|------|---------|------|
| 训练 loss | 0.012 | < 0.001 |
| 推理 MAE | 0.379 rad | < 0.10 rad |
| shldr_pan MAE | 1.074 rad | < 0.10 rad |
| wrist_3 MAE | 0.954 rad | < 0.10 rad |

Gazebo 表现：机械臂在动但不朝目标去，夹爪无限振荡，根本没建立视觉→动作映射。

**模型连单条训练轨迹都拟合不了（episode 0 mean MAE=0.373），不是架构问题，是优化没收敛。**

## 根因

```
scheduler_decay_steps = 30000   ← LR 在 30K 步就衰减到底
总 steps = 84000                ← 剩下 54K 步 LR=2.5e-6，几乎不更新参数
batch_size = 2, LR = 1e-5       ← 对 2.3B 全量微调太保守
```

## v2 训练命令

```bash
cd ~/ur3_ft300_ws && conda activate pi0-env

python -m lerobot.scripts.lerobot_train \
  --policy.path=./ai-models/lerobot/pi0_libero_base \
  --dataset.repo_id=cjx-cell/ur3_pick_place \
  --policy.dtype=bfloat16 --policy.device=cuda \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.n_action_steps=50 \
  --policy.optimizer_lr=3e-5 \
  --policy.scheduler_warmup_steps=2000 \
  --policy.scheduler_decay_steps=150000 \
  --policy.scheduler_decay_lr=1e-6 \
  --batch_size=2 --steps=150000 \
  --tolerance_s=0.001 \
  --output_dir=./outputs/train/ur3_pi0_full_v2 \
  --save_freq=25000 --log_freq=50
```

## v1→v2 改动

| 参数 | v1 | v2 | 原因 |
|------|----|----|------|
| `lr` | 1e-5 | **3e-5** | 步长太小，2.3B 全量微调推不动 |
| `decay_steps` | 30000 | **150000** | v1 在 30K 后 LR 见底，54K 步白跑 |
| `decay_lr` | 2.5e-6 | **1e-6** | 更低的终值但配更长衰减 |
| `steps` | 84000 | **150000** | ~2 epoch（168K帧 / batch 2 = 84K/epoch） |
| `warmup` | 1000 | **2000** | 稍长 warmup |

## 每 25K 步评估

```bash
python src/ur_simulation_gz/ur_simulation_gz/scripts/eval_lora_model.py \
  --model ./outputs/train/ur3_pi0_full_v2/checkpoints/050000/pretrained_model \
  --ckpt ./outputs/train/ur3_pi0_full_v2/checkpoints/050000/pretrained_model
```

合格线：**整体 MAE < 0.10 rad，shldr_pan 和 wrist_3 < 0.15 rad**。

## 训练完成后

把最终 checkpoint 传回这台机器：
```bash
scp -r ./outputs/train/ur3_pi0_full_v2/checkpoints/150000 \
    cjx-cell@<this-machine>:~/ur3_ft300_ws/ai-models/ur3_pi0_full_v2/checkpoints/
```

## 模型和数据位置

- Base model: `./ai-models/lerobot/pi0_libero_base/`（含 model.safetensors + config.json）
- 数据集: HuggingFace `cjx-cell/ur3_pick_place`（49 episodes, 168K frames）
- Tokenizer: `./ai-models/paligemma_tokenizer/`
- Config 已验证：camera0/camera1, state[7], action[7], empty_cameras=0, n_action_steps=50

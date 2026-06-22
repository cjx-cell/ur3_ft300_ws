# UR3 Pi0 全量微调 v2

## v1 根因

**视觉编码器权重从未加载——437/783 个 safetensor keys 被静默丢弃。**

`pi0_libero_base/model.safetensors` 的 vision key 名与当前 `lerobot` 代码创建的模型不匹配：

| 差异 | Safetensors（预训练） | 模型代码 |
|------|----------------------|---------|
| 前缀 | 部分 key 有 `model.`，部分没有 | 全部需要 `model.` |
| Vision 路径 | `...vision_tower.vision_model.encoder...` | `...vision_tower.encoder...` |

`load_state_dict(strict=False)` 静默跳过了全部 437 个 vision keys，SigLIP 400M 以随机权重训练。

**已修复**：`lerobot/policies/pi0/modeling_pi0.py` 的 `_fix_pytorch_state_dict_keys()` 添加了：
```python
new_key = new_key.replace(".vision_model.", ".")
if not new_key.startswith("model.") and not new_key.startswith("normalize_"):
    new_key = "model." + new_key
```

**验证结果**：437/437 vision keys 正确加载，0 缺失。

## 重新训练

v1 的 scheduler/LR/epoch 配置没有问题（根因是 vision weights 没加载，不是优化）。直接用原始参数重跑即可：

```bash
cd ~/ur3_ft300_ws && conda activate pi0-env

# 先 pull 最新代码（含 vision key fix）
git pull origin main

# 注意：必须用本地的 lerobot（含 fix），不能用 pip 安装的旧版
pip install -e ~/lerobot

python -m lerobot.scripts.lerobot_train \
  --policy.path=./ai-models/pi0/pi0_libero_base \
  --dataset.repo_id=local/ur3_pick_place \
  --dataset.root=./ai-models/datasets/ur3_pick_place_lerobot \
  --policy.dtype=bfloat16 --policy.device=cuda \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.n_action_steps=50 \
  --policy.optimizer_lr=1e-5 \
  --batch_size=2 --steps=84000 \
  --tolerance_s=0.001 \
  --output_dir=./outputs/train/ur3_pi0_full_v2 \
  --save_freq=10000 --log_freq=50
```

## 训练后评估

```bash
python src/ur_simulation_gz/ur_simulation_gz/scripts/eval_lora_model.py \
  --model ./outputs/train/ur3_pi0_full_v2/checkpoints/084000/pretrained_model \
  --ckpt ./outputs/train/ur3_pi0_full_v2/checkpoints/084000/pretrained_model
```

合格线：**整体 MAE < 0.10 rad**。

## 数据位置

- Base model: `./ai-models/pi0/pi0_libero_base/`（含 model.safetensors + config.json）
- 数据集: HuggingFace `cjx-cell/ur3_pick_place`（完整 49 episodes）
- Tokenizer: `./ai-models/paligemma_tokenizer/`
- Config 已验证：camera0/camera1, state[7], action[7], empty_cameras=0, n_action_steps=50

## 注意事项

1. **必须用最新的 lerobot 代码**（含 vision key fix），`pip install -e ~/lerobot` 不要用旧版本
2. **```dataset.repo_id```**如果使用`cjx-cell/ur3_pick_place` HF 有缓存的旧数据集，检查一下episode是否够49个或删除缓存重新下载
3. eval 脚本的数据集路径可能需要改为本地路径如果 HF 上的数据不完整

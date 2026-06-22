# UR3 Pi0 重新训练指南

## 背景

v1 全量微调 84K 步后模型完全没学到任务——Gazebo 中机械臂随机游走，夹爪振荡。

**根因：`pi0_libero_base` 的视觉编码器权重从未被加载。**

`pi0_libero_base/model.safetensors` 的 key 名与当前 lerobot 代码创建的模型不匹配：

| Safetensors（预训练权重） | 模型代码 |
|---|---|
| `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder...` | `model.paligemma_with_expert.paligemma.model.vision_tower.encoder...` |
| `action_in_proj.bias`（无 `model.` 前缀） | `model.action_in_proj.bias`（需要 `model.` 前缀） |

`load_state_dict(strict=False)` 静默跳过了全部 437 个 vision keys，SigLIP 以随机权重训练——模型"看不见"。

## 必须做的修复

编辑 `~/lerobot/src/lerobot/policies/pi0/modeling_pi0.py`，找到 `_fix_pytorch_state_dict_keys` 方法（约第 1090 行）。

**在 `for key, value in state_dict.items():` 下一行**，把：

```python
            new_key = key
```

替换为：

```python
            new_key = key

            # Fix 1: remove ".vision_model." from vision tower path
            new_key = new_key.replace(".vision_model.", ".")

            # Fix 2: add "model." prefix to keys that lack it
            if not new_key.startswith("model.") and not new_key.startswith("normalize_"):
                new_key = "model." + new_key
```

## 验证修复

```bash
conda activate pi0-env
python -c "
from safetensors.torch import load_file
sd = load_file('./ai-models/pi0/pi0_libero_base/model.safetensors', device='cpu')
from accelerate import init_empty_weights
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
cfg = PI0Config(device='cpu', dtype='bfloat16')
cfg.device = 'meta'
with init_empty_weights():
    policy = PI0Policy(cfg)
fixed = policy._fix_pytorch_state_dict_keys(sd, cfg)
miss, _ = policy.load_state_dict(fixed, strict=False, assign=True)
vision_miss = [k for k in miss if 'vision' in k.lower()]
print(f'Vision keys missing: {len(vision_miss)} / 437')
assert len(vision_miss) == 0, 'Fix NOT applied!'
print('OK')
"
```

预期输出：`Vision keys missing: 0 / 437`

## 训练

```bash
cd ~/ur3_ft300_ws && conda activate pi0-env

# 确保用的 lerobot 包含上述 fix
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

**注意**：数据集用**本地路径** `local/ur3_pick_place` + `--dataset.root`，不要用 `cjx-cell/ur3_pick_place`（HF 上那个只有 3 episodes）。

## 评估

```bash
python src/ur_simulation_gz/ur_simulation_gz/scripts/eval_lora_model.py \
  --model ./outputs/train/ur3_pi0_full_v2/checkpoints/084000/pretrained_model \
  --ckpt ./outputs/train/ur3_pi0_full_v2/checkpoints/084000/pretrained_model
```

目标：整体 MAE < 0.10 rad。

## 训练完成后

传回 checkpoint：
```bash
scp -r ./outputs/train/ur3_pi0_full_v2/checkpoints/084000 \
    <user>@<main-machine>:~/ur3_ft300_ws/ai-models/ur3_pi0_full_v2/checkpoints/
```

## 文件位置

| 文件 | 路径 |
|------|------|
| Base model | `./ai-models/pi0/pi0_libero_base/` |
| 本地数据集 | `./ai-models/datasets/ur3_pick_place_lerobot/` |
| Tokenizer | `./ai-models/paligemma_tokenizer/` |
| 待修复的文件 | `~/lerobot/src/lerobot/policies/pi0/modeling_pi0.py` |

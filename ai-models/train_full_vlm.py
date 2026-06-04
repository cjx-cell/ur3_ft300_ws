#!/usr/bin/env python3
"""
Pi0 全模型微调 — 用于 2× A100 (80GB) 服务器

用法:
  source /path/to/conda/etc/profile.d/conda.sh && conda activate pi0-env
  export WANDB_MODE=disabled
  python3 train_full_vlm.py
"""

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature

config = TrainPipelineConfig()

# ── 数据集 ──
config.dataset.repo_id = "cjx-cell/ur3_pick_place"
config.dataset.image_transforms.enable = True
config.dataset.use_imagenet_stats = True

# ── 模型（全微调） ──
config.policy.type = "pi0"
config.policy.pretrained_path = "./ai-models/lerobot/pi0"  # 预训练权重
config.policy.paligemma_variant = "gemma_2b"
config.policy.action_expert_variant = "gemma_300m"
config.policy.dtype = "bfloat16"
config.policy.device = "cuda"

# ★ 关键：全模型微调 ★
config.policy.freeze_vision_encoder = False  # 视觉编码器也训练
config.policy.train_expert_only = False       # PaliGemma + Action Expert 一起训练

config.policy.n_obs_steps = 1
config.policy.chunk_size = 50
config.policy.n_action_steps = 50
config.policy.num_inference_steps = 10
config.policy.use_relative_actions = False
config.policy.normalization_mapping = {
    "VISUAL": "IDENTITY",
    "STATE": "MEAN_STD",
    "ACTION": "MEAN_STD",
}
config.policy.input_features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    "observation.images.camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
}
config.policy.output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
}

# ── 训练参数（适配 A100 × 2） ──
config.policy.gradient_checkpointing = True   # 节省显存
config.policy.compile_model = False           # torch.compile 对 Pi0 支持有限
config.policy.optimizer_lr = 2e-5             # 全模型微调用较低学习率
config.policy.optimizer_weight_decay = 0.01
config.policy.optimizer_grad_clip_norm = 1.0

config.batch_size = 1                          # A100 80GB: bs=1 per GPU
config.steps = 3000                            # 全微调步数可少一些
config.num_workers = 4
config.prefetch_factor = 4
config.save_freq = 500
config.log_freq = 50

# Scheduler
config.policy.scheduler_warmup_steps = 200
config.policy.scheduler_decay_steps = 3000
config.policy.scheduler_decay_lr = 2e-6

# ── 输出目录 ──
config.output_dir = "./outputs/train/pi0_ur3_full_vlm"
config.job_name = "pi0_full_vlm"

print("=" * 60)
print("Full VLM fine-tuning config")
print("=" * 60)
print(f"  train_expert_only: {config.policy.train_expert_only}")
print(f"  freeze_vision_encoder: {config.policy.freeze_vision_encoder}")
print(f"  batch_size: {config.batch_size}")
print(f"  steps: {config.steps}")
print(f"  lr: {config.policy.optimizer_lr}")
print(f"  output: {config.output_dir}")

# Run training
from lerobot.scripts.lerobot_train import train
train(config)

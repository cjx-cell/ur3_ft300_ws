#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Stage 1 Baseline Overfitting Training Launcher
# Dataset: 5 Episodes (0301-0305) LeRobot format
# Policy:  PI0.5 OpenPI 3.2B Pretrained Weights + Gemma-300m Action Expert
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# 1. Setup config directory and pretrained weight links
echo "[1/3] Preparing model configuration and pretrained weights..."
mkdir -p /tmp/pap_moe_config
cp /home/ubuntu/ur3_ft300_ws/ai-models/sa_moe_config/config_pap_moe.json /tmp/pap_moe_config/config.json
ln -sf /home/ubuntu/ur3_ft300_ws/ai-models/pi05/pi05_libero_base/model.safetensors /tmp/pap_moe_config/model.safetensors
ln -sf /home/ubuntu/ur3_ft300_ws/ai-models/pi05/pi05_libero_base/policy_preprocessor.json /tmp/pap_moe_config/policy_preprocessor.json
ln -sf /home/ubuntu/ur3_ft300_ws/ai-models/pi05/pi05_libero_base/policy_postprocessor.json /tmp/pap_moe_config/policy_postprocessor.json

# 2. Setup environment variables
export PYTHONPATH=/home/ubuntu/lerobot/src
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTDIR="/home/ubuntu/ur3_ft300_ws/outputs/train/overfit_baseline_stage1_5ep"

echo "================================================================="
echo "  PAP-MoE Stage 1 Baseline Overfitting Training Launcher"
echo "  Dataset:  pap_moe/ur3_peg_in_hole_overfit_5ep (5 Episodes)"
echo "  Pretrain: /home/ubuntu/ur3_ft300_ws/ai-models/pi05/pi05_libero_base"
echo "  Workers:  num_workers=0 (OOM Safe)"
echo "  Steps:    1000"
echo "  Output:   $OUTDIR"
echo "================================================================="
echo ""

# 3. Launch training with real-time output
cd /home/ubuntu/ur3_ft300_ws

exec /home/ubuntu/miniconda3/envs/pi0-env/bin/python -m lerobot.scripts.lerobot_train \
    --policy.path=/tmp/pap_moe_config \
    --dataset.repo_id=pap_moe/ur3_peg_in_hole_overfit_5ep \
    --dataset.root=/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_overfit_5ep \
    --batch_size=2 \
    --steps=1000 \
    --num_workers=0 \
    --save_freq=500 \
    --log_freq=20 \
    --output_dir="$OUTDIR" \
    --policy.push_to_hub=false

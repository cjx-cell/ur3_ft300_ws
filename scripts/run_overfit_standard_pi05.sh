#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Baseline A: Standard Pi0.5 (Pure Vision/State Baseline, bfloat16 RAM Safe)
# Dataset: 5 Episodes (0301-0305) LeRobot format
# Policy:  Official PI05Policy with 3.2B Pretrained Weights in bfloat16
# Usage:   ./scripts/run_overfit_standard_pi05.sh [--prepare-only]
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

PREPARE_ONLY="${1:-}"

RELATIVE_VIEW="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_overfit_5ep_relative_h50"
GLOBAL_TASK_VIEW="${RELATIVE_VIEW}_global_task"

echo "[1/4] Preparing 50-step relative-action dataset statistics..."
/home/ubuntu/miniconda3/envs/pi0-env/bin/python \
  /home/ubuntu/ur3_ft300_ws/scripts/compute_relative_action_chunk_stats.py \
  --dataset-root /home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_overfit_5ep \
  --chunk-size 50 \
  --view-root "$RELATIVE_VIEW"

echo "[2/4] Creating global-language baseline dataset view..."
/home/ubuntu/miniconda3/envs/pi0-env/bin/python \
  /home/ubuntu/ur3_ft300_ws/scripts/create_global_task_dataset_view.py \
  --source-root "$RELATIVE_VIEW" \
  --view-root "$GLOBAL_TASK_VIEW"

echo "[3/4] Preparing bfloat16 memory-safe Pi0.5 configuration..."
mkdir -p /tmp/standard_pi05_config
cat << 'EOF' > /tmp/standard_pi05_config/config.json
{
  "n_obs_steps": 1,
  "input_features": {},
  "output_features": {},
  "device": "cuda",
  "use_amp": false,
  "use_peft": false,
  "push_to_hub": false,
  "pretrained_path": "/home/ubuntu/ur3_ft300_ws/ai-models/pi05/pi05_libero_base",
  "paligemma_variant": "gemma_2b",
  "action_expert_variant": "gemma_300m",
  "dtype": "bfloat16",
  "chunk_size": 50,
  "n_action_steps": 50,
  "max_state_dim": 32,
  "max_action_dim": 32,
  "num_inference_steps": 10,
  "time_sampling_beta_alpha": 1.5,
  "time_sampling_beta_beta": 1.0,
  "time_sampling_scale": 0.999,
  "time_sampling_offset": 0.001,
  "min_period": 0.004,
  "max_period": 4.0,
  "use_relative_actions": true,
  "relative_exclude_joints": [
    "gripper"
  ],
  "image_resolution": [
    224,
    224
  ],
  "empty_cameras": 0,
  "normalization_mapping": {
    "VISUAL": "IDENTITY",
    "STATE": "QUANTILES",
    "ACTION": "QUANTILES"
  },
  "gradient_checkpointing": true,
  "freeze_vision_encoder": true,
  "train_expert_only": true,
  "optimizer_lr": 2.5e-05,
  "optimizer_betas": [0.9, 0.95],
  "optimizer_eps": 1e-08,
  "optimizer_weight_decay": 0.01,
  "optimizer_grad_clip_norm": 1.0,
  "scheduler_warmup_steps": 1000,
  "scheduler_decay_steps": 30000,
  "scheduler_decay_lr": 2.5e-06,
  "tokenizer_max_length": 48,
  "tokenizer_name": "/home/ubuntu/ur3_ft300_ws/ai-models/paligemma_tokenizer",
  "type": "pi05"
}
EOF

ln -sf /home/ubuntu/ur3_ft300_ws/ai-models/pi05/pi05_libero_base/model.safetensors /tmp/standard_pi05_config/model.safetensors
ln -sf /home/ubuntu/ur3_ft300_ws/ai-models/pi05/pi05_libero_base/policy_preprocessor.json /tmp/standard_pi05_config/policy_preprocessor.json
ln -sf /home/ubuntu/ur3_ft300_ws/ai-models/pi05/pi05_libero_base/policy_postprocessor.json /tmp/standard_pi05_config/policy_postprocessor.json

export PYTHONPATH=/home/ubuntu/lerobot/src
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTDIR="/home/ubuntu/ur3_ft300_ws/outputs/train/pi05_relative_h50_global_task_overfit_5ep"

echo "================================================================="
echo "  Baseline A: Standard Pi0.5 Overfitting Training Launcher"
echo "  Dataset:  pap_moe/ur3_peg_in_hole_overfit_5ep_relative_h50_global_task"
echo "  Language: one global instruction for every frame"
echo "  Policy:   Standard PI05Policy (bfloat16 RAM Safe)"
echo "  Modality: Standard Vision + Joint State (No 6D Force, No MoE)"
echo "  Workers:  num_workers=0, batch_size=1 (16GB GPU safe)"
echo "  Steps:    1000"
echo "  Output:   $OUTDIR"
echo "================================================================="
echo ""

cd /home/ubuntu/ur3_ft300_ws

if [ "$PREPARE_ONLY" = "--prepare-only" ]; then
  echo "Preparation complete; training was not started."
  exit 0
fi

echo "[4/4] Starting Pi0.5 relative-action overfit training..."
exec /home/ubuntu/miniconda3/envs/pi0-env/bin/python -m lerobot.scripts.lerobot_train \
    --policy.path=/tmp/standard_pi05_config \
    --dataset.repo_id=pap_moe/ur3_peg_in_hole_overfit_5ep_relative_h50_global_task \
    --dataset.root="$GLOBAL_TASK_VIEW" \
    --batch_size=1 \
    --steps=1000 \
    --num_workers=0 \
    --save_freq=500 \
    --log_freq=20 \
    --output_dir="$OUTDIR" \
    --policy.push_to_hub=false

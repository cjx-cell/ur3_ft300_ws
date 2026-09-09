#!/usr/bin/env bash
set -euo pipefail

WS_DIR=/home/ubuntu/ur3_ft300_ws
LEROBOT_DIR=/home/ubuntu/lerobot
PYTHON_BIN=/home/ubuntu/miniconda3/envs/pi0-env/bin/python
DATASET_ROOT=${WORKSPACE50_DATASET_ROOT:-$WS_DIR/pap_moe_framework/datasets/lerobot_v3_workspace50_v10_baseline_global_stats_v1}
SOURCE_CONFIG=$WS_DIR/outputs/train/diffusion_workspace50_v10_official_20k/checkpoints/020000/pretrained_model/train_config.json
RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${DIFFUSION_OUTPUT_DIR:-$WS_DIR/outputs/train/diffusion_workspace50_global_stats_20k_$RUN_TAG}
LOG_FILE=${DIFFUSION_LOG_FILE:-$WS_DIR/artifacts/diffusion_workspace50_global_stats_20k_$RUN_TAG.log}

test -f "$DATASET_ROOT/meta/exact_global_stats_receipt.json"
test -f "$SOURCE_CONFIG"
export PYTHONPATH=$LEROBOT_DIR/src
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_REBUILD_PROCESSORS=1

cd "$LEROBOT_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --config_path="$SOURCE_CONFIG" --resume=false \
  --dataset.root="$DATASET_ROOT" \
  --dataset.repo_id=local/pap_moe_workspace50_global_stats_v1 \
  --policy.pretrained_path=null \
  --output_dir="$OUTPUT_DIR" \
  --steps=20000 --save_freq=5000 --log_freq=100 \
  2>&1 | tee "$LOG_FILE"

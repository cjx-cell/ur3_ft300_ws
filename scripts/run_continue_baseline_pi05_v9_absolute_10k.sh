#!/usr/bin/env bash
# Resume the 30-episode exact-fit absolute-action baseline from step 1k to 10k.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
RUN_DIR="${1:-$WS_DIR/outputs/train/pi05_v9_absolute_1000step_20260818_233017}"
CHECKPOINT_DIR="$RUN_DIR/checkpoints/001000"
CONFIG_PATH="$CHECKPOINT_DIR/pretrained_model/train_config.json"
LOG_FILE="$WS_DIR/artifacts/pi05_v9_absolute_resume_1k_to_10k_$(date +%Y%m%d_%H%M%S).log"

for required in \
  "$CONFIG_PATH" \
  "$CHECKPOINT_DIR/pretrained_model/model.safetensors" \
  "$CHECKPOINT_DIR/training_state/optimizer_state.safetensors" \
  "$CHECKPOINT_DIR/training_state/scheduler_state.json"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: resume input is missing: $required" >&2
    exit 2
  fi
done
if [[ -d "$RUN_DIR/checkpoints/010000" ]]; then
  echo "ERROR: step-10000 checkpoint already exists: $RUN_DIR/checkpoints/010000" >&2
  exit 2
fi

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Resume uses the exact same dataset and must keep the two independently saved
# state/action normalizers.  Reapplying one name-based override to both
# `normalizer_processor` steps is ambiguous and can silently double-normalize.
unset LEROBOT_REBUILD_PROCESSORS || true
export LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS=1

echo "Resume optimized-data Pi0.5 baseline"
echo "  source step: 1000"
echo "  target step: 10000"
  echo "  dataset:     30-episode exact-fit LeRobot v3 baseline view"
echo "  output:      $RUN_DIR"
echo "  log:         $LOG_FILE"

cd "$WS_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --config_path="$CONFIG_PATH" \
  --resume=true \
  --steps=10000 \
  --save_freq=1000 \
  --log_freq=50 \
  2>&1 | tee "$LOG_FILE"

#!/usr/bin/env bash
# PAP-MoE v9 PhysicsGate smoke training on the frozen v7/v9/v5 data contract.
# This verifies routing learnability without writing an unnecessary 8 GB checkpoint.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_ENV="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
PAP_CONFIG="$WS_DIR/ai-models/pap_moe_v6/config.json"
PI05_BASE="$WS_DIR/ai-models/pi05/pi05_libero_base"
DATASET_ROOT="${1:-$WS_DIR/pap_moe_framework/datasets/lerobot_v6_v9_smoke_2ep}"
STEPS="${2:-1000}"
BATCH_SIZE="${3:-2}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$WS_DIR/outputs/train/pap_moe_v9_physicsgate_smoke_$RUN_TAG"
LOG_FILE="$WS_DIR/artifacts/pap_moe_v9_physicsgate_smoke_$RUN_TAG.log"
MODEL_VIEW="$(mktemp -d /tmp/pap_moe_v9_physicsgate_smoke.XXXXXX)"

cleanup() {
  rm -f \
    "$MODEL_VIEW/config.json" \
    "$MODEL_VIEW/model.safetensors" \
    "$MODEL_VIEW/policy_preprocessor.json" \
    "$MODEL_VIEW/policy_postprocessor.json"
  rmdir "$MODEL_VIEW" 2>/dev/null || true
}
trap cleanup EXIT

if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: steps and batch size must be positive integers" >&2
  exit 2
fi
for required in "$PAP_CONFIG" "$PI05_BASE/model.safetensors" "$DATASET_ROOT/meta/info.json"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 2
  fi
done

jq \
  '.train_expert_only = false
   | del(.train_stagegate_only)
   | .train_physicsgate_only = true
   | .train_gate_calibration_only = false
   | .train_conditioner_only = false
   | .train_pap_moe_joint = false' \
  "$PAP_CONFIG" > "$MODEL_VIEW/config.json"
ln -s "$PI05_BASE/model.safetensors" "$MODEL_VIEW/model.safetensors"
for processor in policy_preprocessor.json policy_postprocessor.json; do
  if [[ -e "$PI05_BASE/$processor" ]]; then
    ln -s "$PI05_BASE/$processor" "$MODEL_VIEW/$processor"
  fi
done

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$WS_DIR/artifacts"
echo "PAP-MoE v9 PhysicsGate smoke training"
echo "  dataset:   $DATASET_ROOT"
echo "  steps:     $STEPS"
echo "  batch:     $BATCH_SIZE"
echo "  checkpoint:false"
echo "  log:       $LOG_FILE"

cd "$WS_DIR"
"$PYTHON_ENV" -m lerobot.scripts.lerobot_train \
  --policy.path="$MODEL_VIEW" \
  --dataset.repo_id=pap_moe/pap_moe_v6_v9_smoke_2ep \
  --dataset.root="$DATASET_ROOT" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers=0 \
  --save_checkpoint=false \
  --log_freq=25 \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=false \
  2>&1 | tee "$LOG_FILE"

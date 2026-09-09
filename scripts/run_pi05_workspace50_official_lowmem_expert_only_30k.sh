#!/usr/bin/env bash
# Workspace50 Pi0.5 baseline using LeRobot's documented low-memory option:
# frozen VLM, full action expert/projections, and the pretrained three-view
# layout (two real cameras plus one empty camera).
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
BASE_MODEL="${PI05_LEGACY_BASE_MODEL:-$WS_DIR/ai-models/pi05/pi05_libero_base}"
DATASET_ROOT="${PI05_LEGACY_DATASET_ROOT:-$WS_DIR/pap_moe_framework/datasets/lerobot_v3_workspace50_v10_baseline_global_stats_v1}"
STEPS="${1:-30000}"
SAVE_FREQ="${2:-3000}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PI05_LEGACY_OUTPUT_DIR:-$WS_DIR/outputs/train/pi05_workspace50_official_lowmem_expert_only_30k_$RUN_TAG}"
LOG_FILE="${PI05_LEGACY_LOG_FILE:-$WS_DIR/artifacts/pi05_workspace50_official_lowmem_expert_only_30k_$RUN_TAG.log}"

for value in "$STEPS" "$SAVE_FREQ"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: steps/save_freq must be positive integers" >&2; exit 2; }
done
for required in "$BASE_MODEL/config.json" "$BASE_MODEL/model.safetensors" \
  "$DATASET_ROOT/meta/info.json" "$DATASET_ROOT/meta/stats.json" \
  "$DATASET_ROOT/meta/exact_global_stats_receipt.json"; do
  [[ -f "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "ERROR: output already exists: $OUTPUT_DIR" >&2; exit 2; }

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_REBUILD_PROCESSORS=1
unset LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS || true

mkdir -p "$WS_DIR/artifacts" "$WS_DIR/outputs/train"
echo "Pi0.5 Workspace50 LeRobot official low-memory expert-only training"
echo "  base:              $BASE_MODEL"
echo "  dataset:           $DATASET_ROOT (50 episodes only)"
echo "  normalization:     exact dataset-wide QUANTILES q01/q99"
echo "  cameras:           2 real + 1 empty (pretrained layout)"
echo "  frozen:            VLM and vision encoder"
echo "  trainable:         full action expert"
echo "  chunk/action steps:50 / 10"
echo "  tokenizer length:  200 (pretrained/LeRobot default)"
echo "  initial sampling:  first 20 frames x5"
echo "  lr:                2.5e-5 -> 2.5e-6"
echo "  steps/batch:       $STEPS / 1"
echo "  output:            $OUTPUT_DIR"
echo "  log:               $LOG_FILE"

cd "$LEROBOT_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=pap_moe/workspace50_v10_pi05_legacy_contract \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path="$BASE_MODEL" \
  --policy.repo_id=pap_moe/pi05_workspace50_legacy_expert_only \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.empty_cameras=1 \
  --policy.tokenizer_name="$WS_DIR/ai-models/paligemma_tokenizer" \
  --policy.tokenizer_max_length=200 \
  --policy.use_relative_actions=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --batch_size=1 \
  --steps="$STEPS" \
  --num_workers=0 \
  --initial_frame_sampling_count=20 \
  --initial_frame_sampling_weight=5 \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=10 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi05_workspace50_legacy_expert_only \
  --policy.push_to_hub=false \
  2>&1 | tee "$LOG_FILE"

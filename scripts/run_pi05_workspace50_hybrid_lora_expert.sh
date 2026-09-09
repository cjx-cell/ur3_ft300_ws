#!/usr/bin/env bash
# Workspace50 Pi0.5 baseline: low-rank Gazebo visual adaptation plus a fully
# trainable action expert. The original PaliGemma/SigLIP weights stay frozen.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
BASE_MODEL="${PI05_HYBRID_BASE_MODEL:-$WS_DIR/ai-models/pi05/pi05_libero_base}"
DATASET_ROOT="${PI05_HYBRID_DATASET_ROOT:-$WS_DIR/pap_moe_framework/datasets/lerobot_v3_workspace50_v10_baseline}"
STEPS="${1:-${PI05_HYBRID_STEPS:-14418}}"
BATCH_SIZE="${2:-${PI05_HYBRID_BATCH_SIZE:-1}}"
SAVE_FREQ="${3:-${PI05_HYBRID_SAVE_FREQ:-2000}}"
LORA_R="${PI05_HYBRID_LORA_R:-8}"
PEAK_LR="${PI05_HYBRID_PEAK_LR:-2.5e-5}"
DECAY_LR="${PI05_HYBRID_DECAY_LR:-2.5e-6}"
WARMUP_STEPS="${PI05_HYBRID_WARMUP_STEPS:-500}"
COMPILE_MODEL="${PI05_HYBRID_COMPILE_MODEL:-false}"
INITIAL_FRAME_COUNT="${PI05_HYBRID_INITIAL_FRAME_COUNT:-0}"
INITIAL_FRAME_WEIGHT="${PI05_HYBRID_INITIAL_FRAME_WEIGHT:-1.0}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PI05_HYBRID_OUTPUT_DIR:-$WS_DIR/outputs/train/pi05_workspace50_visual_lora_r${LORA_R}_expert_full_${RUN_TAG}}"
LOG_FILE="${PI05_HYBRID_LOG_FILE:-$WS_DIR/artifacts/pi05_workspace50_visual_lora_r${LORA_R}_expert_full_${RUN_TAG}.log}"

# LoRA is deliberately restricted to SigLIP q/v and the multimodal projector.
# It never targets gemma_expert, so the expert is represented exactly once as
# a fully trained modules_to_save subtree rather than as an adapter.
# Keep this deliberately tolerant of Transformers' internal SigLIP container
# names. A more specific encoder path silently matched only the projector in
# PEFT 0.17, leaving the intended visual q/v adapters absent.
VISUAL_TARGETS='(.*vision_tower.*self_attn\.(q_proj|v_proj)|.*multi_modal_projector\.linear)'
FULL_EXPERT_MODULES='["paligemma_with_expert.gemma_expert.model","action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]'

for value in "$STEPS" "$BATCH_SIZE" "$SAVE_FREQ" "$LORA_R"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: steps/batch/save/rank must be positive integers" >&2; exit 2; }
done
[[ "$COMPILE_MODEL" == "true" || "$COMPILE_MODEL" == "false" ]] || {
  echo "ERROR: PI05_HYBRID_COMPILE_MODEL must be true or false" >&2
  exit 2
}
for required in "$BASE_MODEL/config.json" "$BASE_MODEL/model.safetensors" \
  "$DATASET_ROOT/meta/info.json" "$DATASET_ROOT/meta/stats.json"; do
  [[ -f "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "ERROR: output already exists: $OUTPUT_DIR" >&2; exit 2; }

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_REBUILD_PROCESSORS=1

mkdir -p "$WS_DIR/artifacts" "$WS_DIR/outputs/train"
echo "Pi0.5 Workspace50 hybrid adaptation"
echo "  base:             $BASE_MODEL"
echo "  dataset:          $DATASET_ROOT"
echo "  visual LoRA:      SigLIP q/v + multimodal projector, r=$LORA_R"
echo "  full train:       action expert + action/time projections"
echo "  steps/batch:      $STEPS / $BATCH_SIZE"
echo "  sample exposure:  $((STEPS * BATCH_SIZE))"
echo "  lr:               $PEAK_LR -> $DECAY_LR; warmup=$WARMUP_STEPS"
echo "  initial sampling: first $INITIAL_FRAME_COUNT frames x$INITIAL_FRAME_WEIGHT"
echo "  output:           $OUTPUT_DIR"
echo "  log:              $LOG_FILE"

cd "$LEROBOT_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=pap_moe/workspace50_v10_pi05_baseline \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path="$BASE_MODEL" \
  --policy.repo_id=pap_moe/pi05_workspace50_visual_lora_expert_full \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.empty_cameras=0 \
  --policy.tokenizer_name="$WS_DIR/ai-models/paligemma_tokenizer" \
  --policy.tokenizer_max_length=200 \
  --policy.use_relative_actions=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model="$COMPILE_MODEL" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --peft.method_type=LORA \
  --peft.r="$LORA_R" \
  --peft.target_modules="$VISUAL_TARGETS" \
  --peft.full_training_modules="$FULL_EXPERT_MODULES" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers=0 \
  --initial_frame_sampling_count="$INITIAL_FRAME_COUNT" \
  --initial_frame_sampling_weight="$INITIAL_FRAME_WEIGHT" \
  --policy.optimizer_lr="$PEAK_LR" \
  --policy.scheduler_decay_lr="$DECAY_LR" \
  --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$STEPS" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=10 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi05_workspace50_visual_lora_expert_full \
  --policy.push_to_hub=false \
  2>&1 | tee "$LOG_FILE"

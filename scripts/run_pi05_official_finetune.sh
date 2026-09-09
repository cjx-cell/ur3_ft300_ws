#!/usr/bin/env bash
# Clean LeRobot PI0.5 fine-tuning entry point for the UR3 baseline.
# It follows docs/source/pi05.mdx and deliberately supplies no project-specific
# sampling, auxiliary-head, force, stage, progress, RTC, or action-repair flags.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
BASE_MODEL="${PI05_OFFICIAL_BASE_MODEL:-$WS_DIR/ai-models/pi05/pi05_libero_base}"
DATASET_ROOT="${PI05_OFFICIAL_DATASET_ROOT:-$WS_DIR/pap_moe_framework/datasets/lerobot_v3_exactfit_pilot30_baseline}"
STEPS="${1:-3000}"
BATCH_SIZE="${2:-1}"
SAVE_FREQ="${3:-500}"
COMPILE_MODEL="${PI05_OFFICIAL_COMPILE_MODEL:-true}"
PEFT_R="${PI05_OFFICIAL_PEFT_R:-0}"
PEFT_TARGET_MODULES="${PI05_OFFICIAL_PEFT_TARGET_MODULES:-}"
TARGET_SAMPLES="${PI05_OFFICIAL_TARGET_SAMPLES:-}"

for value in "$STEPS" "$BATCH_SIZE" "$SAVE_FREQ"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: numeric arguments must be positive integers" >&2; exit 2; }
done
[[ "$COMPILE_MODEL" == "true" || "$COMPILE_MODEL" == "false" ]] || {
  echo "ERROR: PI05_OFFICIAL_COMPILE_MODEL must be true or false" >&2
  exit 2
}
[[ "$PEFT_R" =~ ^[0-9]+$ ]] || { echo "ERROR: PI05_OFFICIAL_PEFT_R must be a non-negative integer" >&2; exit 2; }
if [[ -n "$TARGET_SAMPLES" ]]; then
  [[ "$TARGET_SAMPLES" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: PI05_OFFICIAL_TARGET_SAMPLES must be a positive integer" >&2
    exit 2
  }
  # Keep training exposure comparable when a smaller local batch replaces the
  # batch=32 reference recipe.
  STEPS=$(( (TARGET_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))
fi
for required in "$BASE_MODEL/config.json" "$BASE_MODEL/model.safetensors" "$DATASET_ROOT/meta/info.json"; do
  [[ -e "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 2; }
done

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
ADAPTATION_TAG="full"
if (( PEFT_R > 0 )); then
  ADAPTATION_TAG="peft_r${PEFT_R}"
fi
OUTPUT_DIR="$WS_DIR/outputs/train/pi05_official_${ADAPTATION_TAG}_${STEPS}step_$RUN_TAG"
LOG_FILE="$WS_DIR/artifacts/pi05_official_${ADAPTATION_TAG}_${STEPS}step_$RUN_TAG.log"

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_REBUILD_PROCESSORS=1

mkdir -p "$WS_DIR/artifacts" "$WS_DIR/outputs/train"
echo "Official LeRobot Pi0.5 fine-tuning"
echo "  pretrained_path: $BASE_MODEL"
echo "  dataset:         $DATASET_ROOT"
echo "  training chunk:  50 (deployment execution prefix is evaluated separately)"
echo "  vision frozen:   false"
echo "  expert only:     false"
echo "  sampling:        uniform"
echo "  batch:           $BATCH_SIZE (adapted to local GPU memory)"
echo "  steps:           $STEPS"
echo "  seen samples:    $((STEPS * BATCH_SIZE))"
if (( PEFT_R > 0 )); then
  if [[ -n "$PEFT_TARGET_MODULES" ]]; then
    echo "  adaptation:      PEFT LoRA r=$PEFT_R, targets=$PEFT_TARGET_MODULES"
  else
    echo "  adaptation:      LeRobot Pi0.5 default PEFT LoRA r=$PEFT_R (action expert q/v + task projections)"
  fi
else
  echo "  adaptation:      full fine-tuning"
fi
echo "  output:          $OUTPUT_DIR"
echo "  log:             $LOG_FILE"

PEFT_ARGS=()
if (( PEFT_R > 0 )); then
  # Use PI05Policy._get_default_peft_targets(). LeRobot's Pi0.5 default adapts
  # action-expert q/v plus task-dependent state/action projections.
  PEFT_ARGS+=(
    --peft.method_type=LORA
    --peft.r="$PEFT_R"
  )
  if [[ -n "$PEFT_TARGET_MODULES" ]]; then
    PEFT_ARGS+=(--peft.target_modules="$PEFT_TARGET_MODULES")
  fi
fi

cd "$LEROBOT_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=pap_moe/pi05_official_pilot30_baseline \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path="$BASE_MODEL" \
  --policy.repo_id=pap_moe/pi05_official_ur3 \
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
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers=0 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=10 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi05_official_ur3 \
  --policy.push_to_hub=false \
  "${PEFT_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

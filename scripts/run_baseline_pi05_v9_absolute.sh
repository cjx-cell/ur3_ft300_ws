#!/usr/bin/env bash
# Standard Pi0.5 baseline on the optimized v9 absolute-action data contract.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
BASE_MODEL="${5:-$WS_DIR/ai-models/pi05/pi05_libero_base}"
DATASET_ROOT="${1:-$WS_DIR/pap_moe_framework/datasets/lerobot_v8_narrow_success_8ep_baseline_absolute}"
STEPS="${2:-1000}"
BATCH_SIZE="${3:-1}"
SAVE_FREQ="${4:-$STEPS}"
INITIAL_FRAME_COUNT="${PI05_INITIAL_FRAME_SAMPLING_COUNT:-20}"
INITIAL_FRAME_WEIGHT="${PI05_INITIAL_FRAME_SAMPLING_WEIGHT:-5.0}"
GRIPPER_WINDOW="${PI05_GRIPPER_TRANSITION_SAMPLING_WINDOW:-10}"
GRIPPER_WEIGHT="${PI05_GRIPPER_TRANSITION_SAMPLING_WEIGHT:-3.0}"

if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$SAVE_FREQ" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: steps, batch size, and save frequency must be positive integers" >&2
  exit 2
fi
for required in \
  "$BASE_MODEL/config.json" \
  "$BASE_MODEL/model.safetensors" \
  "$DATASET_ROOT/meta/info.json" \
  "$DATASET_ROOT/MODEL_VIEWS.json" \
  "$DATASET_ROOT/GLOBAL_TASK_VIEW.json"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 2
  fi
done
if [[ "$(jq -r '.materialized_view // "legacy"' "$DATASET_ROOT/MODEL_VIEWS.json")" != "baseline" ]]; then
  echo "ERROR: pure baseline training requires a materialized_view=baseline dataset" >&2
  echo "       full PAP-MoE metadata can retain unused force statistics" >&2
  exit 2
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$WS_DIR/outputs/train/pi05_v9_absolute_${STEPS}step_$RUN_TAG"
LOG_FILE="$WS_DIR/artifacts/pi05_v9_absolute_${STEPS}step_$RUN_TAG.log"
MODEL_VIEW="$(mktemp -d /tmp/pi05_v9_absolute.XXXXXX)"

cleanup() {
  rm -f \
    "$MODEL_VIEW/config.json" \
    "$MODEL_VIEW/model.safetensors" \
    "$MODEL_VIEW/policy_preprocessor.json" \
    "$MODEL_VIEW/policy_postprocessor.json"
  rmdir "$MODEL_VIEW" 2>/dev/null || true
}
trap cleanup EXIT

# Explicit features prevent force, soft routing labels, and subtask metadata
# from entering the standard baseline preprocessing contract.
jq '
  .device = "cuda"
  | .dtype = "bfloat16"
  | .use_amp = false
  | .gradient_checkpointing = true
  | .tokenizer_name = "/home/ubuntu/ur3_ft300_ws/ai-models/paligemma_tokenizer"
  | .tokenizer_max_length = 48
  | .use_relative_actions = false
  | .relative_exclude_joints = []
  | .gripper_action_index = 6
  | .chunk_size = 50
  | .n_action_steps = 10
  | .image_resolution = [224, 224]
  | .freeze_vision_encoder = true
  | .train_expert_only = true
  | .input_features = {
      "observation.images.camera0": {"type": "VISUAL", "shape": [3,224,224]},
      "observation.images.camera1": {"type": "VISUAL", "shape": [3,224,224]},
      "observation.state": {"type": "STATE", "shape": [7]}
    }
  | .output_features = {"action": {"type": "ACTION", "shape": [7]}}
' "$BASE_MODEL/config.json" > "$MODEL_VIEW/config.json"
ln -s "$BASE_MODEL/model.safetensors" "$MODEL_VIEW/model.safetensors"

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_REBUILD_PROCESSORS=1

mkdir -p "$WS_DIR/artifacts" "$WS_DIR/outputs/train"
echo "Optimized-data standard Pi0.5 absolute-action baseline"
echo "  initialization: $BASE_MODEL"
echo "  dataset:        $DATASET_ROOT"
echo "  modalities:     two cameras + 7D joint state"
echo "  action:         absolute, chunk=50, execute=10"
echo "  steps:          $STEPS"
echo "  batch:          $BATCH_SIZE"
echo "  save frequency: $SAVE_FREQ"
echo "  initial sampling: first $INITIAL_FRAME_COUNT frames x$INITIAL_FRAME_WEIGHT"
echo "  gripper sampling: +/-$GRIPPER_WINDOW frames x$GRIPPER_WEIGHT"
echo "  output:         $OUTPUT_DIR"
echo "  log:            $LOG_FILE"

cd "$WS_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path="$MODEL_VIEW" \
  --dataset.repo_id=pap_moe/pi05_v9_absolute_global_task \
  --dataset.root="$DATASET_ROOT" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers=0 \
  --initial_frame_sampling_count="$INITIAL_FRAME_COUNT" \
  --initial_frame_sampling_weight="$INITIAL_FRAME_WEIGHT" \
  --gripper_transition_sampling_window="$GRIPPER_WINDOW" \
  --gripper_transition_sampling_weight="$GRIPPER_WEIGHT" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=50 \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=false \
  2>&1 | tee "$LOG_FILE"

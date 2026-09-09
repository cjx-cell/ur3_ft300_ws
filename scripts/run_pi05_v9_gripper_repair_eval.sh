#!/usr/bin/env bash
# Evaluate a short gripper repair against both its 20k parent and original source.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
RUN_DIR="${1:?usage: run_pi05_v9_gripper_repair_eval.sh RUN_DIR}"
ORIGINAL_SOURCE="$WS_DIR/outputs/train/pi05_v9_absolute_30000step_20260804_231244/checkpoints/030000/pretrained_model"
REPAIR_PARENT="$WS_DIR/outputs/train/pi05_v9_semantic_rebalanced_action_20000step_20260808_103251/checkpoints/020000/pretrained_model"
CANDIDATE="$RUN_DIR/checkpoints/002500/pretrained_model"
EVALUATOR="$WS_DIR/scripts/eval_pi05_checkpoint_stages.py"
PROCESSOR_REPAIR="$WS_DIR/scripts/repair_pi05_global_task_processor.py"
CHECKPOINT_AUDITOR="$WS_DIR/scripts/audit_pi05_action_repair_checkpoint.py"
SELECTOR="$WS_DIR/scripts/select_pi05_gripper_repair_candidate.py"
TRAIN_EPISODE="$WS_DIR/pap_moe_framework/datasets/raw_v6_admittance/pick_up_the_peg_and_insert_it_into_the_hole_episode_13003_success/data.npz"
HOLDOUT_EPISODE="$WS_DIR/pap_moe_framework/datasets/raw_v6_admittance/pick_up_the_peg_and_insert_it_into_the_hole_episode_14003_success/data.npz"
OUTPUT_DIR="$WS_DIR/artifacts/$(basename "$RUN_DIR")_deployment_eval"

for required in \
  "$ORIGINAL_SOURCE/model.safetensors" \
  "$REPAIR_PARENT/model.safetensors" \
  "$CANDIDATE/model.safetensors" \
  "$EVALUATOR" \
  "$PROCESSOR_REPAIR" \
  "$CHECKPOINT_AUDITOR" \
  "$SELECTOR" \
  "$TRAIN_EPISODE" \
  "$HOLDOUT_EPISODE"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 2
  fi
done

if pgrep -af 'lerobot.scripts.lerobot_train' \
  | grep -F -- "--output_dir=$RUN_DIR" >/dev/null; then
  echo "ERROR: training is still writing $RUN_DIR" >&2
  exit 3
fi

export PYTHONPATH="/home/ubuntu/lerobot/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON_BIN" "$PROCESSOR_REPAIR" "$RUN_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR"

AUDIT="$OUTPUT_DIR/step_002500_parameter_audit.json"
"$PYTHON_BIN" "$CHECKPOINT_AUDITOR" \
  --source-checkpoint "$REPAIR_PARENT" \
  --checkpoint "$CANDIDATE" \
  --output "$AUDIT"

tags=(original_source_030000 repair_parent_020000 candidate_002500)
checkpoints=("$ORIGINAL_SOURCE" "$REPAIR_PARENT" "$CANDIDATE")
for index in "${!tags[@]}"; do
  tag="${tags[$index]}"
  checkpoint="${checkpoints[$index]}"
  echo "Evaluating $tag: $checkpoint"
  "$PYTHON_BIN" "$EVALUATOR" \
    --checkpoint "$checkpoint" \
    --episode-npz "$TRAIN_EPISODE" \
    --episode-npz "$HOLDOUT_EPISODE" \
    --seed 20260812 \
    --num-seeds 5 \
    --output "$OUTPUT_DIR/$tag.json" \
    >"$OUTPUT_DIR/$tag.log" 2>&1
done

"$PYTHON_BIN" "$SELECTOR" \
  --original-source "$OUTPUT_DIR/original_source_030000.json" \
  --repair-parent "$OUTPUT_DIR/repair_parent_020000.json" \
  --candidate "$OUTPUT_DIR/candidate_002500.json" \
  --parameter-audit "$AUDIT" \
  --output "$OUTPUT_DIR/closed_loop_candidate.json" \
  >"$OUTPUT_DIR/selector.log" 2>&1

cat "$OUTPUT_DIR/selector.log"
echo "Evaluation complete: $OUTPUT_DIR/closed_loop_candidate.json"

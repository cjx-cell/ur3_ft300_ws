#!/usr/bin/env bash
# Migrate a successful repaired baseline into PAP-MoE, audit it, then run train-position Gazebo.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
CANDIDATE_JSON="${1:-$WS_DIR/artifacts/pi05_v9_semantic_rebalanced_action_20000step_20260808_103251_decode_sweep/closed_loop_candidate.json}"
PAP_SOURCE="$WS_DIR/outputs/train/pap_moe_v9_conditioner_20260808_021028/checkpoints/001000/pretrained_model"
TRANSPLANT="$WS_DIR/scripts/transplant_pi05_action_backbone_to_pap.py"
PAP_EVALUATOR="$WS_DIR/scripts/eval_pap_moe_offline_action_chunk.py"
PAP_AUDITOR="$WS_DIR/scripts/audit_transplanted_pap_moe.py"
PAP_ABLATION="$WS_DIR/scripts/eval_pap_moe_expert_ablation.py"
TRAIN_EPISODE="$WS_DIR/pap_moe_framework/datasets/raw_v6_admittance/pick_up_the_peg_and_insert_it_into_the_hole_episode_13003_success/data.npz"
HOLDOUT_EPISODE="$WS_DIR/pap_moe_framework/datasets/raw_v6_admittance/pick_up_the_peg_and_insert_it_into_the_hole_episode_14003_success/data.npz"

for required in \
  "$CANDIDATE_JSON" \
  "$PAP_SOURCE/model.safetensors" \
  "$TRANSPLANT" \
  "$PAP_EVALUATOR" \
  "$PAP_AUDITOR" \
  "$PAP_ABLATION" \
  "$TRAIN_EPISODE" \
  "$HOLDOUT_EPISODE"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 2
  fi
done

candidate="$(jq -r '.recommended_checkpoint // empty' "$CANDIDATE_JSON")"
if [[ -z "$candidate" ]]; then
  echo "STOP: no repaired baseline passed the offline gates." >&2
  exit 5
fi

baseline_result="$($PYTHON_BIN - "$candidate" <<'PY'
import json
import sys
from pathlib import Path

candidate = str(Path(sys.argv[1]).resolve())
matches = []
for path in Path("/home/ubuntu/ur3_ft300_ws/artifacts").glob(
    "gazebo_pi05_v9_absolute_*_ep13003/result.json"
):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("checkpoint") == candidate and data.get("success") is True:
        matches.append(path)
if matches:
    print(max(matches, key=lambda path: path.stat().st_mtime))
PY
)"
if [[ -z "$baseline_result" ]]; then
  echo "STOP: no successful episode-13003 Gazebo result matches $candidate" >&2
  exit 6
fi

step_tag="$(basename "$(dirname "$candidate")")"
OUTPUT_ROOT="$WS_DIR/outputs/deploy/pap_moe_v9_action_repaired_$step_tag"
PAP_CHECKPOINT="$OUTPUT_ROOT/pretrained_model"
ARTIFACT_ROOT="$WS_DIR/artifacts/pap_moe_v9_action_repaired_$step_tag"
mkdir -p "$OUTPUT_ROOT" "$ARTIFACT_ROOT"

if [[ ! -f "$PAP_CHECKPOINT/action_backbone_transplant.json" ]]; then
  if [[ -e "$PAP_CHECKPOINT" ]]; then
    echo "ERROR: incomplete transplant output already exists: $PAP_CHECKPOINT" >&2
    exit 2
  fi
  "$PYTHON_BIN" "$TRANSPLANT" \
    --pi05-checkpoint "$candidate" \
    --pap-checkpoint "$PAP_SOURCE" \
    --output "$PAP_CHECKPOINT" \
    >"$ARTIFACT_ROOT/transplant.log" 2>&1
fi

export PYTHONPATH="/home/ubuntu/lerobot/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON_BIN" \
  "$WS_DIR/src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_pap_moe_peg_in_hole_inference.py" \
  --checkpoint "$PAP_CHECKPOINT" --validate-only \
  >"$ARTIFACT_ROOT/validate.log" 2>&1

"$PYTHON_BIN" "$PAP_EVALUATOR" \
  --checkpoint "$PAP_CHECKPOINT" \
  --episode-npz "$TRAIN_EPISODE" \
  --frame 0 --frame 100 --frame 163 --frame 193 --frame 219 \
  --frame 401 --frame 431 --frame 490 \
  --seed 20260808 --num-seeds 5 \
  --output "$ARTIFACT_ROOT/ep13003_decoded.json" \
  >"$ARTIFACT_ROOT/ep13003_decoded.log" 2>&1

"$PYTHON_BIN" "$PAP_EVALUATOR" \
  --checkpoint "$PAP_CHECKPOINT" \
  --episode-npz "$HOLDOUT_EPISODE" \
  --frame 0 --frame 100 --frame 172 --frame 211 --frame 514 \
  --frame 527 --frame 707 --frame 734 --frame 798 \
  --seed 20260808 --num-seeds 5 \
  --output "$ARTIFACT_ROOT/ep14003_decoded.json" \
  >"$ARTIFACT_ROOT/ep14003_decoded.log" 2>&1

"$PYTHON_BIN" "$PAP_ABLATION" \
  --checkpoint "$PAP_CHECKPOINT" \
  --episode-npz "$TRAIN_EPISODE" \
  --episode-npz "$HOLDOUT_EPISODE" \
  --max-frames-per-expert 25 \
  --batch-size 2 \
  --seed 20260808 --num-seeds 3 \
  --routing-source predicted \
  --output "$ARTIFACT_ROOT/expert_ablation_real.json" \
  >"$ARTIFACT_ROOT/expert_ablation_real.log" 2>&1

"$PYTHON_BIN" "$PAP_ABLATION" \
  --checkpoint "$PAP_CHECKPOINT" \
  --episode-npz "$TRAIN_EPISODE" \
  --episode-npz "$HOLDOUT_EPISODE" \
  --max-frames-per-expert 10 \
  --batch-size 2 \
  --seed 20260808 --num-seeds 3 \
  --routing-source predicted --wrist-dropout \
  --mask full --mask all_zero --mask drop_E2 \
  --output "$ARTIFACT_ROOT/expert_ablation_e2_wrist_dropout.json" \
  >"$ARTIFACT_ROOT/expert_ablation_e2_wrist_dropout.log" 2>&1

"$PYTHON_BIN" "$PAP_AUDITOR" \
  --candidate-audit "$CANDIDATE_JSON" \
  --train-eval "$ARTIFACT_ROOT/ep13003_decoded.json" \
  --holdout-eval "$ARTIFACT_ROOT/ep14003_decoded.json" \
  --expert-ablation "$ARTIFACT_ROOT/expert_ablation_real.json" \
  --e2-ablation "$ARTIFACT_ROOT/expert_ablation_e2_wrist_dropout.json" \
  --output "$ARTIFACT_ROOT/offline_gate.json" \
  >"$ARTIFACT_ROOT/offline_gate.log" 2>&1

export PAP_MOE_CHECKPOINT="$PAP_CHECKPOINT"
export PAP_MOE_SEED=20260812
export PAP_MOE_EXECUTE_STEPS=10
"$WS_DIR/scripts/run_pap_moe_v9_gazebo_eval.sh" false 13003

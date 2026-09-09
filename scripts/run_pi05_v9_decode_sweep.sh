#!/usr/bin/env bash
# Decode the source and every semantic-rebalanced Pi0.5 checkpoint on train/holdout phases.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
RUN_DIR="${1:-$WS_DIR/outputs/train/pi05_v9_semantic_rebalanced_action_20000step_20260808_103251}"
SOURCE_CHECKPOINT="$WS_DIR/outputs/train/pi05_v9_absolute_30000step_20260804_231244/checkpoints/030000/pretrained_model"
EVALUATOR="$WS_DIR/scripts/eval_pi05_checkpoint_stages.py"
PROCESSOR_REPAIR="$WS_DIR/scripts/repair_pi05_global_task_processor.py"
CHECKPOINT_AUDITOR="$WS_DIR/scripts/audit_pi05_action_repair_checkpoint.py"
TRAIN_EPISODE="$WS_DIR/pap_moe_framework/datasets/raw_v6_admittance/pick_up_the_peg_and_insert_it_into_the_hole_episode_13003_success/data.npz"
HOLDOUT_EPISODE="$WS_DIR/pap_moe_framework/datasets/raw_v6_admittance/pick_up_the_peg_and_insert_it_into_the_hole_episode_14003_success/data.npz"
OUTPUT_DIR="$WS_DIR/artifacts/$(basename "$RUN_DIR")_decode_sweep"

for required in \
  "$SOURCE_CHECKPOINT/model.safetensors" \
  "$RUN_DIR/checkpoints" \
  "$EVALUATOR" \
  "$PROCESSOR_REPAIR" \
  "$CHECKPOINT_AUDITOR" \
  "$TRAIN_EPISODE" \
  "$HOLDOUT_EPISODE"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 2
  fi
done

if pgrep -af 'lerobot.scripts.lerobot_train' \
  | grep -F -- "--output_dir=$RUN_DIR" >/dev/null; then
  echo "ERROR: training is still writing $RUN_DIR; wait for completion." >&2
  exit 3
fi

export PYTHONPATH="/home/ubuntu/lerobot/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON_BIN" "$PROCESSOR_REPAIR" "$RUN_DIR/checkpoints"

mkdir -p "$OUTPUT_DIR"
checkpoints=("$SOURCE_CHECKPOINT")
expected_steps=(002500 005000 007500 010000 012500 015000 017500 020000)
for step in "${expected_steps[@]}"; do
  checkpoint="$RUN_DIR/checkpoints/$step/pretrained_model"
  if [[ ! -f "$checkpoint/model.safetensors" ]]; then
    echo "ERROR: required 2.5k checkpoint is incomplete: $checkpoint" >&2
    exit 4
  fi
  checkpoints+=("$checkpoint")
done

for checkpoint in "${checkpoints[@]}"; do
  if [[ "$checkpoint" == "$SOURCE_CHECKPOINT" ]]; then
    tag="source_030000"
  else
    tag="step_$(basename "$(dirname "$checkpoint")")"
    audit_output="$OUTPUT_DIR/${tag}_parameter_audit.json"
    if [[ -s "$audit_output" ]] && "$PYTHON_BIN" - \
      "$audit_output" "$SOURCE_CHECKPOINT" "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert data["source_checkpoint"] == str(Path(sys.argv[2]).resolve())
assert data["checkpoint"] == str(Path(sys.argv[3]).resolve())
assert data["passed"] is True
PY
    then
      echo "SKIP: $tag parameter integrity already audited: $audit_output"
    else
      echo "Auditing $tag parameter integrity"
      "$PYTHON_BIN" "$CHECKPOINT_AUDITOR" \
        --source-checkpoint "$SOURCE_CHECKPOINT" \
        --checkpoint "$checkpoint" \
        --output "$audit_output"
    fi
  fi
  output="$OUTPUT_DIR/$tag.json"
  log="$OUTPUT_DIR/$tag.log"
  if [[ -s "$output" ]] && "$PYTHON_BIN" - "$output" "$checkpoint" <<'PY'
import json
import os
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert data["checkpoint"] == str(Path(sys.argv[2]).resolve())
assert len(data["episodes"]) == 2
assert len(data["observed_phases"]) == 8
PY
  then
    echo "SKIP: $tag already evaluated: $output"
    continue
  fi
  echo "Evaluating $tag: $checkpoint"
  "$PYTHON_BIN" "$EVALUATOR" \
    --checkpoint "$checkpoint" \
    --episode-npz "$TRAIN_EPISODE" \
    --episode-npz "$HOLDOUT_EPISODE" \
    --seed 20260808 \
    --num-seeds 5 \
    --output "$output" \
    >"$log" 2>&1
  echo "DONE: $output"
done

"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []


def phase_means(episode):
    return {
        phase: metrics["mean_executed_arm_mae_rad"]
        for phase, metrics in episode["phase_metrics"].items()
    }


expected_tags = ["source_030000"] + [
    f"step_{step}"
    for step in (
        "002500",
        "005000",
        "007500",
        "010000",
        "012500",
        "015000",
        "017500",
        "020000",
    )
]
evaluation_paths = [root / f"{tag}.json" for tag in expected_tags]
missing = [str(path) for path in evaluation_paths if not path.is_file()]
if missing:
    raise ValueError(f"Missing required evaluation outputs: {missing}")
for path in evaluation_paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "tag": path.stem,
            "checkpoint": data["checkpoint"],
            "mean_executed_arm_mae_rad": data[
                "aggregate_mean_executed_arm_mae_rad"
            ],
            "worst_executed_arm_mae_rad": data[
                "aggregate_worst_executed_arm_mae_rad"
            ],
            "mean_prediction_std": data["aggregate_mean_prediction_std"],
            "max_executed_predicted_step_rad": data[
                "aggregate_max_executed_predicted_step_rad"
            ],
            "mean_gripper_accuracy": data[
                "aggregate_mean_gripper_accuracy"
            ],
            "mean_full_chunk_gripper_accuracy": data[
                "aggregate_mean_full_chunk_gripper_accuracy"
            ],
            "train_mean_executed_arm_mae_rad": data["episodes"][0][
                "mean_executed_arm_mae_rad"
            ],
            "holdout_mean_executed_arm_mae_rad": data["episodes"][1][
                "mean_executed_arm_mae_rad"
            ],
            "train_phase_mean_executed_arm_mae_rad": phase_means(
                data["episodes"][0]
            ),
            "holdout_phase_mean_executed_arm_mae_rad": phase_means(
                data["episodes"][1]
            ),
            "train_startup_mean_executed_arm_mae_rad": sum(
                frame["mean_executed_arm_mae_rad"]
                for frame in data["episodes"][0]["frames"]
                if frame["phase"] == "grasp the peg"
            )
            / sum(
                1
                for frame in data["episodes"][0]["frames"]
                if frame["phase"] == "grasp the peg"
            ),
            "holdout_startup_mean_executed_arm_mae_rad": sum(
                frame["mean_executed_arm_mae_rad"]
                for frame in data["episodes"][1]["frames"]
                if frame["phase"] == "grasp the peg"
            )
            / sum(
                1
                for frame in data["episodes"][1]["frames"]
                if frame["phase"] == "grasp the peg"
            ),
        }
    )
rows.sort(
    key=lambda item: (
        item["mean_executed_arm_mae_rad"],
        item["worst_executed_arm_mae_rad"],
    )
)
summary = {"ranking": rows, "best_decoded_candidate": rows[0] if rows else None}
summary_path = root / "summary.json"
temporary_summary = summary_path.with_suffix(".json.tmp")
temporary_summary.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary_summary, summary_path)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "Sweep complete: $OUTPUT_DIR/summary.json"
"$PYTHON_BIN" "$WS_DIR/scripts/select_pi05_closed_loop_candidate.py" \
  --summary "$OUTPUT_DIR/summary.json" \
  --output "$OUTPUT_DIR/closed_loop_candidate.json"
echo "Candidate gate complete: $OUTPUT_DIR/closed_loop_candidate.json"

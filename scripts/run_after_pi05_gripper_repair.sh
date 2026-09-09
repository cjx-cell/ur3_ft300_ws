#!/usr/bin/env bash
# Wait for a named repair service, evaluate it, and launch Gazebo only if qualified.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
TRAIN_UNIT="${1:?usage: run_after_pi05_gripper_repair.sh TRAIN_UNIT RUN_DIR}"
RUN_DIR="${2:?usage: run_after_pi05_gripper_repair.sh TRAIN_UNIT RUN_DIR}"
TRAIN_LOG="$WS_DIR/artifacts/$(basename "$RUN_DIR" | sed 's/^pi05_/pi05_/').log"

while systemctl --user is-active --quiet "$TRAIN_UNIT"; do
  sleep 20
done

CHECKPOINT="$RUN_DIR/checkpoints/002500/pretrained_model/model.safetensors"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: training ended without a complete 2.5k checkpoint: $CHECKPOINT" >&2
  [[ -f "$TRAIN_LOG" ]] && tail -n 80 "$TRAIN_LOG" >&2
  exit 4
fi

"$WS_DIR/scripts/run_pi05_v9_gripper_repair_eval.sh" "$RUN_DIR"
CANDIDATE_JSON="$WS_DIR/artifacts/$(basename "$RUN_DIR")_deployment_eval/closed_loop_candidate.json"
if [[ "$(jq -r '.recommended_checkpoint // empty' "$CANDIDATE_JSON")" == "" ]]; then
  echo "STOP: repaired checkpoint did not pass every deployment-consistent gate."
  echo "Audit: $CANDIDATE_JSON"
  exit 0
fi

exec "$WS_DIR/scripts/run_qualified_pi05_v9_closed_loop.sh" "$CANDIDATE_JSON"

#!/usr/bin/env bash
# Run deterministic train-position Gazebo only for a checkpoint that passed every offline gate.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
CANDIDATE_JSON="${1:-$WS_DIR/artifacts/pi05_v9_semantic_rebalanced_action_20000step_20260808_103251_decode_sweep/closed_loop_candidate.json}"

if [[ ! -s "$CANDIDATE_JSON" ]]; then
  echo "ERROR: candidate audit is missing: $CANDIDATE_JSON" >&2
  exit 2
fi

checkpoint="$(jq -r '.recommended_checkpoint // empty' "$CANDIDATE_JSON")"
if [[ -z "$checkpoint" ]]; then
  echo "STOP: no checkpoint passed every offline closed-loop gate."
  echo "Audit: $CANDIDATE_JSON"
  exit 5
fi
if [[ ! -f "$checkpoint/model.safetensors" ]]; then
  echo "ERROR: qualified checkpoint is incomplete: $checkpoint" >&2
  exit 2
fi

echo "Qualified Pi0.5 deterministic closed-loop evaluation"
echo "  checkpoint: $checkpoint"
echo "  audit:      $CANDIDATE_JSON"
echo "  episode:    13003 (training position)"
echo "  inference:  seeds 20260812..20260816, fixed at every replan"
echo "  safety:     physical ensemble mean + causal 0.12 rad arm limit"

export PI05_CHECKPOINT="$checkpoint"
export PI05_SEED=20260812
export PI05_FIXED_NOISE_PER_REPLAN=true
export PI05_ENSEMBLE_SIZE=5
export PI05_MAX_ARM_STEP_RAD=0.12
exec "$WS_DIR/scripts/run_pi05_v9_absolute_gazebo_eval.sh" false 13003

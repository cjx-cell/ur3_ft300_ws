#!/usr/bin/env bash
# Wait for the unified PAP-MoE Stage-1 run, then perform matched oracle-route
# action evaluation on representative checkpoints. This stage has no trained
# 50-step PhysicsGate yet, so predicted-route closed-loop evaluation belongs
# to the next stage rather than this watcher.
set -euo pipefail

TRAIN_PID="${1:?usage: $0 TRAIN_PID}"
ROOT=/home/ubuntu/ur3_ft300_ws
RUN=$ROOT/outputs/train/pap_moe_unified_v2_expert_action_joint_20260904_235835
TRAIN_LOG=$ROOT/artifacts/pap_moe_unified_v2_expert_action_joint_20260904_235835.log
RESUME_LOG=$ROOT/artifacts/pap_moe_unified_v2_expert_action_joint_20260904_235835_resume.log
OUT=$ROOT/artifacts/pap_moe_unified_v2_stage1_auto_eval_20260905
PY=/home/ubuntu/miniconda3/envs/pi0-env/bin/python
EVAL=$ROOT/scripts/eval_pap_moe_expert_ablation.py
DATA=$ROOT/pap_moe_framework/datasets/workspace_50_v10_canonical

mkdir -p "$OUT"
echo "waiting for training pid=$TRAIN_PID" 
while [[ "$TRAIN_PID" != completed ]] && kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
done

# User services do not inherit the IDE extension's ripgrep executable.
if ! /usr/bin/grep -qF "End of training" "$TRAIN_LOG" "$RESUME_LOG"; then
  echo "ERROR: training process ended without a clean completion marker" >&2
  tail -n 100 "$RESUME_LOG" >&2
  exit 1
fi

EPISODE_ARGS=()
for episode in 1 11 21 31 41; do
  EPISODE_ARGS+=(
    --episode-npz
    "$DATA/pick_up_the_peg_and_insert_it_into_the_hole_episode_$(printf '%04d' "$episode")_success/data.npz"
  )
done

export PYTHONPATH=/home/ubuntu/lerobot/src
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for step in 012000 021000 030000; do
  checkpoint=$RUN/checkpoints/$step/pretrained_model
  if [[ ! -s "$checkpoint/model.safetensors" ]]; then
    echo "ERROR: incomplete checkpoint: $checkpoint" >&2
    exit 1
  fi
  echo "START action evaluation $step $(date '+%F %T')"
  "$PY" "$EVAL" \
    --checkpoint "$checkpoint" \
    "${EPISODE_ARGS[@]}" \
    --stride 20 \
    --batch-size 2 \
    --seed 1000 \
    --num-seeds 3 \
    --routing-source dataset \
    --mask full \
    --mask all_zero \
    --continuous-gripper \
    --output "$OUT/action_$step.json" \
    > "$OUT/action_$step.log" 2>&1
  echo "DONE action evaluation $step $(date '+%F %T')"
done

{
  printf 'checkpoint\tfull_mse\tall_zero_mse\texpert_delta\tci95_low\tci95_high\texpert_better_fraction\n'
  for step in 012000 021000 030000; do
    jq -r --arg step "$step" '[
      $step,
      .masks.full.mse,
      .masks.all_zero.mse,
      .masks.all_zero.delta_vs_full,
      .masks.all_zero.delta_ci95[0],
      .masks.all_zero.delta_ci95[1],
      .masks.all_zero.worse_than_full_fraction
    ] | @tsv' "$OUT/action_$step.json"
  done
} > "$OUT/summary.tsv"

{
  read -r _
  sort -t $'\t' -k2,2g | head -n 1
} < "$OUT/summary.tsv" > "$OUT/best_by_full_mse.tsv"

echo "Automatic Stage-1 evaluation complete"
cat "$OUT/summary.tsv"
echo "best: $(cat "$OUT/best_by_full_mse.tsv")"

#!/usr/bin/env bash
# Matched five-position routing and action-flow audit for one stepwise PAP-MoE checkpoint.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
CHECKPOINT="${1:?usage: $0 CHECKPOINT OUTPUT_TAG}"
OUTPUT_TAG="${2:?usage: $0 CHECKPOINT OUTPUT_TAG}"
DATA_ROOT="$WS_DIR/pap_moe_framework/datasets/workspace_50_v10_canonical"
EPISODES=(1 11 21 31 41)
EPISODE_ARGS=()
for episode in "${EPISODES[@]}"; do
  EPISODE_ARGS+=(
    --episode-npz
    "$DATA_ROOT/pick_up_the_peg_and_insert_it_into_the_hole_episode_$(printf '%04d' "$episode")_success/data.npz"
  )
done

export PYTHONPATH="/home/ubuntu/lerobot/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROUTING_OUTPUT="$WS_DIR/artifacts/${OUTPUT_TAG}_routing.json"
ACTION_OUTPUT="$WS_DIR/artifacts/${OUTPUT_TAG}_predicted_action_3seeds.json"

/home/ubuntu/miniconda3/envs/pi0-env/bin/python \
  "$WS_DIR/scripts/eval_pap_moe_routing.py" \
  --checkpoint "$CHECKPOINT" \
  "${EPISODE_ARGS[@]}" \
  --stride 10 \
  --batch-size 2 \
  --wrist-dropout \
  --all-camera-dropout \
  --output "$ROUTING_OUTPUT"

/home/ubuntu/miniconda3/envs/pi0-env/bin/python \
  "$WS_DIR/scripts/eval_pap_moe_expert_ablation.py" \
  --checkpoint "$CHECKPOINT" \
  "${EPISODE_ARGS[@]}" \
  --stride 20 \
  --batch-size 2 \
  --seed 1000 \
  --num-seeds 3 \
  --routing-source predicted \
  --mask full \
  --mask all_zero \
  --continuous-gripper \
  --output "$ACTION_OUTPUT"

echo "routing: $ROUTING_OUTPUT"
echo "action:  $ACTION_OUTPUT"

#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ubuntu/ur3_ft300_ws
RUN=$ROOT/outputs/train/pap_moe_unified_v2_expert_action_joint_20260904_235835
export PYTHONPATH=/home/ubuntu/lerobot/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LEROBOT_REBUILD_PROCESSORS=1
cd /home/ubuntu/lerobot
/home/ubuntu/miniconda3/envs/pi0-env/bin/python -m lerobot.scripts.lerobot_train \
  --config_path="$RUN/checkpoints/024000/pretrained_model/train_config.json" \
  --resume=true 2>&1 | tee -a "$ROOT/artifacts/pap_moe_unified_v2_expert_action_joint_20260904_235835_resume.log"
exec bash "$ROOT/scripts/auto_eval_pap_moe_unified_v2_stage1.sh" completed

#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ubuntu/ur3_ft300_ws
EXP="$ROOT/artifacts/pap_lr_ab_10k_20260905"
PY=/home/ubuntu/miniconda3/envs/pi0-env/bin/python
export PYTHONPATH=/home/ubuntu/lerobot/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LEROBOT_REBUILD_PROCESSORS=0
export LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS=1
cd /home/ubuntu/lerobot
for label in A B; do
  "$PY" - "$EXP/manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
for path,digest in m['hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, f'Source/config changed: {path}'
print('Source/config/statistics fingerprint check passed',flush=True)
PY
  echo "START training $label $(date -Is)"
  "$PY" -u -m lerobot.scripts.lerobot_train --config_path="$EXP/$label.json" 2>&1 | tee "$EXP/train_$label.log"
  RUN="$ROOT/outputs/train/pap_lr_ab_${label}_10k_20260905"
  test "$(jq -r '.step' "$RUN/checkpoints/010000/training_state/training_step.json")" = 10000
  echo "START final action evaluation $label $(date -Is)"
  episodes=()
  for index in 0001 0011 0021 0031 0041; do
    episodes+=(--episode-npz "$ROOT/pap_moe_framework/datasets/workspace_50_v10_canonical/pick_up_the_peg_and_insert_it_into_the_hole_episode_${index}_success/data.npz")
  done
  "$PY" -u "$ROOT/scripts/audit_pap_generated_conditions.py" \
    --checkpoint "$RUN/checkpoints/010000/pretrained_model" \
    --output "$EXP/eval_$label" "${episodes[@]}" 2>&1 | tee "$EXP/eval_$label.log"
  echo "DONE $label $(date -Is)"
done
echo "A/B training and final offline evaluations complete $(date -Is)"

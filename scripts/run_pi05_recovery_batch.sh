#!/usr/bin/env bash
# Collect a DAgger batch before rebuilding the dataset or retraining Pi0.5.
set -uo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
COLLECTOR="$WS_DIR/scripts/run_pi05_ep13001_grasp_recovery_collection.sh"
CHECKPOINT="${PI05_CHECKPOINT:-$WS_DIR/outputs/train/pi05_grasp_bucket_v3_preserve_v2_stats_20260819_2055/checkpoints/003000/pretrained_model}"
GUI="${1:-false}"
INTER_CASE_DELAY_S="${PI05_RECOVERY_INTER_CASE_DELAY_S:-15}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$WS_DIR/artifacts/pi05_recovery_batch_$RUN_TAG"
mkdir -p "$LOG_DIR"
trap 'echo "Batch interrupted; no further cases will be started." >&2; exit 130' INT TERM

# episode:seed:fixed_noise.  Recovery references must match the exact fixture
# scene, so the default batch varies policy noise only within episode 15001.
# Other scenes require their own exact-scene successful reference episodes.
read -r -a SPECS <<<"${PI05_RECOVERY_BATCH_SPECS:-15001:0:false 15001:1:false 15001:2:false 15001:3:false 15001:4:false 15001:5:false 15001:6:false 15001:7:false}"

if [[ "$GUI" != "true" && "$GUI" != "false" ]]; then
  echo "ERROR: GUI must be true or false" >&2
  exit 2
fi
if ! [[ "$INTER_CASE_DELAY_S" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: PI05_RECOVERY_INTER_CASE_DELAY_S must be non-negative" >&2
  exit 2
fi
[[ -x "$COLLECTOR" ]] || { echo "ERROR: missing collector: $COLLECTOR" >&2; exit 2; }
if [[ ! -f "$CHECKPOINT/model.safetensors" && ! -f "$CHECKPOINT/adapter_model.safetensors" ]]; then
  echo "ERROR: missing full-policy or LoRA checkpoint weights: $CHECKPOINT" >&2
  exit 2
fi

echo "Pi0.5 rollout-recovery DAgger batch"
echo "  checkpoint: $CHECKPOINT"
echo "  cases:      ${#SPECS[@]}"
echo "  log dir:    $LOG_DIR"
echo "  restart cooldown: ${INTER_CASE_DELAY_S}s"

success=0
failed=0
for index in "${!SPECS[@]}"; do
  spec="${SPECS[$index]}"
  IFS=: read -r episode seed fixed_noise <<<"$spec"
  if [[ -z "$episode" || -z "$seed" || ( "$fixed_noise" != "true" && "$fixed_noise" != "false" ) ]]; then
    echo "ERROR: invalid batch spec: $spec" >&2
    exit 2
  fi
  case_no=$((index + 1))
  case_log="$LOG_DIR/case_${case_no}_ep${episode}_seed${seed}_${fixed_noise}.log"
  echo "[$case_no/${#SPECS[@]}] episode=$episode seed=$seed fixed_noise=$fixed_noise"
  env \
    PI05_CHECKPOINT="$CHECKPOINT" \
    PI05_SEED="$seed" \
    PI05_FIXED_NOISE_PER_REPLAN="$fixed_noise" \
    PI05_RECORD_VIDEO=true \
    PI05_RECOVERY_PHASE="${PI05_RECOVERY_PHASE:-full_task}" \
    PI05_RECOVERY_ROLLOUT_ACTION_CHUNK_SIZE="${PI05_RECOVERY_ROLLOUT_ACTION_CHUNK_SIZE:-10}" \
    PI05_RECOVERY_RTC_ENABLED="${PI05_RECOVERY_RTC_ENABLED:-true}" \
    PI05_RECOVERY_RTC_HORIZON="${PI05_RECOVERY_RTC_HORIZON:-10}" \
    PI05_RECOVERY_ROLLOUT_MAX_ARM_STEP_RAD="${PI05_RECOVERY_ROLLOUT_MAX_ARM_STEP_RAD:-0}" \
    PI05_RECOVERY_POLICY_ALIGNMENT_ENTRY_M="${PI05_RECOVERY_POLICY_ALIGNMENT_ENTRY_M:-0.012}" \
    PI05_RECOVERY_POLICY_CONTACT_FORCE_N="${PI05_RECOVERY_POLICY_CONTACT_FORCE_N:-20}" \
    PI05_RECOVERY_POLICY_ALIGNMENT_STALL_S="${PI05_RECOVERY_POLICY_ALIGNMENT_STALL_S:-15.0}" \
    PI05_RECOVERY_TRANSPORT_STALL_S="${PI05_RECOVERY_TRANSPORT_STALL_S:-30.0}" \
    PI05_RECOVERY_POLICY_ALIGNMENT_MIN_DESCENT_M="${PI05_RECOVERY_POLICY_ALIGNMENT_MIN_DESCENT_M:-0.003}" \
    PI05_RECOVERY_REJOIN_RELEASE_PHYSICAL_M="${PI05_RECOVERY_REJOIN_RELEASE_PHYSICAL_M:-0.004}" \
    PI05_RECOVERY_REJOIN_TRIGGER_L2_RAD="${PI05_RECOVERY_REJOIN_TRIGGER_L2_RAD:-1.0}" \
    PI05_RECOVERY_GRASP_TRIGGER_MODE="${PI05_RECOVERY_GRASP_TRIGGER_MODE:-gripper_misaligned}" \
    PI05_RECOVERY_TRIGGER_CLOSED_RAD="${PI05_RECOVERY_TRIGGER_CLOSED_RAD:-0.300}" \
    PI05_RECOVERY_TRIGGER_AFTER_S="${PI05_RECOVERY_TRIGGER_AFTER_S:-25.0}" \
    PI05_RECOVERY_TRIGGER_XY_M="${PI05_RECOVERY_TRIGGER_XY_M:-0.020}" \
    PI05_RECOVERY_TRIGGER_MIN_HEIGHT_M="${PI05_RECOVERY_TRIGGER_MIN_HEIGHT_M:-0.160}" \
    PI05_RECOVERY_DESCENT_XY_GATE_M="${PI05_RECOVERY_DESCENT_XY_GATE_M:-0.0005}" \
    PI05_RECOVERY_INSERTION_CHUNK_SIZE="${PI05_RECOVERY_INSERTION_CHUNK_SIZE:-20}" \
    PI05_RECOVERY_EXPERT_CHUNK_SIZE="${PI05_RECOVERY_EXPERT_CHUNK_SIZE:-30}" \
    PI05_RECOVERY_ALIGNMENT_CHUNK_SIZE="${PI05_RECOVERY_ALIGNMENT_CHUNK_SIZE:-30}" \
    bash "$COLLECTOR" "$GUI" "$episode" 2>&1 | tee "$case_log"
  status=${PIPESTATUS[0]}
  if (( status == 0 )); then
    success=$((success + 1))
  else
    failed=$((failed + 1))
    echo "WARN: case $spec failed with status $status; continuing batch" >&2
  fi
  if (( case_no < ${#SPECS[@]} )); then
    echo "Cooling down Gazebo/ROS resources for ${INTER_CASE_DELAY_S}s..."
    sleep "$INTER_CASE_DELAY_S"
  fi
done

printf 'success=%d\nfailed=%d\ntotal=%d\ncheckpoint=%s\n' \
  "$success" "$failed" "${#SPECS[@]}" "$CHECKPOINT" | tee "$LOG_DIR/summary.txt"

# A partial batch is still useful, but fail the wrapper if fewer than half of
# the requested recoveries passed strict validation.
(( success * 2 >= ${#SPECS[@]} ))

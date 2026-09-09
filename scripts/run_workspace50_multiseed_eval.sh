#!/usr/bin/env bash
# Sequential, contract-matched Workspace50 closed-loop evaluation.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
RUNNER="${WORKSPACE50_EVAL_RUNNER:-$WS_DIR/scripts/run_frozen_workspace50_eval.py}"
OUTPUT_ROOT="${1:?usage: $0 OUTPUT_ROOT POLICY LABEL CHECKPOINT [POLICY LABEL CHECKPOINT ...]}"
shift
if (( $# == 0 || $# % 3 != 0 )); then
  echo "ERROR: model arguments must be POLICY LABEL CHECKPOINT triples" >&2
  exit 2
fi

read -r -a EPISODES <<<"${WORKSPACE50_EVAL_EPISODES:-1 11 21 31 41}"
read -r -a SEEDS <<<"${WORKSPACE50_EVAL_SEEDS:-0 1 2}"
RECORD_SEED="${WORKSPACE50_EVAL_RECORD_SEED:-0}"
PAP_ROUTING_SOURCE="${PAP_MOE_ROUTING_SOURCE:-physicsgate}"
if [[ "${POLICY_ACTION_CHUNK_MAX_STEP_RAD:-0.13}" != "0.13" ]]; then
  echo "ERROR: experimental controller limit must be validated before formal batching" >&2
  exit 2
fi
if [[ "${POLICY_GOAL_TIME_TOLERANCE_S:-0}" != "0" ]]; then
  echo "ERROR: experimental goal grace must be validated before formal batching" >&2
  exit 2
fi
if [[ "${WORKSPACE50_MAX_EPISODE_DURATION_S:-120}" != "120" && "${WORKSPACE50_MAX_EPISODE_DURATION_S:-120}" != "120.0" ]]; then
  echo "ERROR: short engineering smoke tests cannot enter the formal multiseed batch" >&2
  exit 2
fi
export WORKSPACE50_EVALUATION_KIND=formal
mkdir -p "$OUTPUT_ROOT"
RESULTS_JSONL="$OUTPUT_ROOT/results.jsonl"
if [[ -e "$RESULTS_JSONL" ]]; then
  [[ "${WORKSPACE50_RESUME_VALID:-false}" == "true" ]] || { echo "ERROR: refusing to overwrite existing evaluation" >&2; exit 2; }
  jq -se 'all(.[]; .evaluation_valid == true and .evaluation_kind == "formal" and
    .controller_max_arm_step_rad == 0.13 and .goal_time_tolerance_override_s == 0 and
    (if (.policy == "pi05" or .policy == "pap_moe") then .action_exchange_contract == "paired-v1" else true end))' \
    "$RESULTS_JSONL" >/dev/null || {
      echo "ERROR: existing results use another/unknown contract; start a new output directory" >&2
      exit 2
    }
else
  : >"$RESULTS_JSONL"
fi

while (( $# )); do
  POLICY="$1"
  if [[ "$POLICY" == "demonstration" ]]; then
    echo "ERROR: demonstration replay is not a model evaluation" >&2
    exit 2
  fi
  LABEL="$2"
  CHECKPOINT="$3"
  shift 3
  [[ -d "$CHECKPOINT" ]] || { echo "ERROR: missing checkpoint $CHECKPOINT" >&2; exit 2; }

  for episode in "${EPISODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      already_valid="$(jq -se --arg eval_label_value "$LABEL" --arg checkpoint "$CHECKPOINT" --argjson episode "$episode" --argjson seed "$seed" \
          'any(.[]; .eval_label==$eval_label_value and .checkpoint==$checkpoint and .requested_episode==$episode and .requested_seed==$seed and .evaluation_valid==true)' "$RESULTS_JSONL")" || {
        status=$?
        [[ "$status" == 1 ]] || exit "$status"
      }
      if [[ "$already_valid" == true ]]; then
        echo "SKIP already valid: $LABEL episode=$episode seed=$seed"
        continue
      fi
      record_video=false
      [[ "$seed" == "$RECORD_SEED" ]] && record_video=true
      run_log="$OUTPUT_ROOT/${LABEL}_ep$(printf '%04d' "$episode")_seed${seed}.log"
      echo "EVAL label=$LABEL policy=$POLICY episode=$episode seed=$seed video=$record_video"
      set +e
      WORKSPACE50_POLICY_SEED="$seed" \
      WORKSPACE50_RECORD_VIDEO="$record_video" \
      PAP_MOE_ROUTING_SOURCE="$PAP_ROUTING_SOURCE" \
        "$RUNNER" "$POLICY" "$CHECKPOINT" "$episode" false \
        2>&1 | tee "$run_log"
      status=${PIPESTATUS[0]}
      set -e

      artifact_dir="$(awk '/^  artifacts:/{print $2; exit}' "$run_log")"
      if [[ -n "$artifact_dir" && -f "$artifact_dir/result.json" ]]; then
        jq -c \
          --arg eval_label_value "$LABEL" \
          --argjson requested_episode "$episode" \
          --argjson requested_seed "$seed" \
          --argjson runner_exit_status "$status" \
          --arg artifact_dir "$artifact_dir" \
          '. + {
            eval_label: $eval_label_value,
            requested_episode: $requested_episode,
            requested_seed: $requested_seed,
            runner_exit_status: $runner_exit_status,
            artifact_dir: $artifact_dir
          }' "$artifact_dir/result.json" >>"$RESULTS_JSONL"
      else
        jq -cn \
          --arg eval_label_value "$LABEL" \
          --arg policy "$POLICY" \
          --arg checkpoint "$CHECKPOINT" \
          --argjson requested_episode "$episode" \
          --argjson requested_seed "$seed" \
          --argjson runner_exit_status "$status" \
          --arg artifact_dir "$artifact_dir" \
          '{eval_label:$eval_label_value,policy:$policy,checkpoint:$checkpoint,
            requested_episode:$requested_episode,requested_seed:$requested_seed,
            runner_exit_status:$runner_exit_status,artifact_dir:$artifact_dir,
            outcome:"runner_failure",evaluation_valid:false,success:false}' >>"$RESULTS_JSONL"
      fi
      if ! tail -n 1 "$RESULTS_JSONL" | jq -e '.evaluation_valid == true' >/dev/null; then
        echo "ERROR: invalid initialization; stopping batch, not counting as policy failure" >&2
        exit 3
      fi
    done
  done
done

jq -s '.' "$RESULTS_JSONL" >"$OUTPUT_ROOT/results.json"
jq -s '
  group_by(.eval_label)
  | map({
      label: .[0].eval_label,
      trials: length,
      successes: map(select(.success == true)) | length,
      success_rate: ((map(select(.success == true)) | length) / length),
      outcomes: (group_by(.outcome) | map({key:.[0].outcome,value:length}) | from_entries)
    })' "$RESULTS_JSONL" >"$OUTPUT_ROOT/summary.json"
cat "$OUTPUT_ROOT/summary.json"

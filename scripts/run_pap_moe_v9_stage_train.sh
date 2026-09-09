#!/usr/bin/env bash
# Train one checkpointed PAP-MoE v9 stage from the previous stage's model.
set -euo pipefail

WS_DIR="/home/ubuntu/ur3_ft300_ws"
LEROBOT_DIR="/home/ubuntu/lerobot"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/pi0-env/bin/python"
PAP_TEMPLATE="$WS_DIR/ai-models/pap_moe_v6/config.json"
STAGE="${1:-}"
INPUT_MODEL="${2:-}"
DATASET_ROOT="${3:-}"
STEPS="${4:-}"
BATCH_SIZE="${5:-2}"
SAVE_CHECKPOINT="${PAP_SAVE_CHECKPOINT:-true}"
DRY_RUN="${PAP_MOE_DRY_RUN:-false}"
LOG_FREQ=50

case "$STAGE" in
  expert_action_joint) DEFAULT_STEPS=30000 ;;
  physicsgate_action_joint) DEFAULT_STEPS=30000 ;;
  physicsgate)      DEFAULT_STEPS=5000 ;;
  expert)           DEFAULT_STEPS=10000 ;;
  gate_calibration) DEFAULT_STEPS=3000 ;;
  route_forecaster) DEFAULT_STEPS=15000 ;;
  physicsgate_sequence) DEFAULT_STEPS=15000 ;;
  temporal_gate)    DEFAULT_STEPS=15000 ;;
  conditioner)      DEFAULT_STEPS=3000 ;;
  action_adapter)   DEFAULT_STEPS=100 ;;
  *)
    echo "Usage: $0 {expert_action_joint|physicsgate_action_joint|expert|physicsgate|physicsgate_sequence|gate_calibration|route_forecaster|temporal_gate|conditioner|action_adapter} INPUT_MODEL ABSOLUTE_DATASET [steps] [batch_size]" >&2
    exit 2
    ;;
esac
STEPS="${STEPS:-$DEFAULT_STEPS}"
SAVE_FREQ="${PAP_MOE_SAVE_FREQ:-$STEPS}"

if [[ -z "$INPUT_MODEL" || -z "$DATASET_ROOT" ]]; then
  echo "ERROR: INPUT_MODEL and ABSOLUTE_DATASET are required." >&2
  echo "The former relative-action defaults were removed to prevent silently mixing action semantics." >&2
  exit 2
fi

if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || [[ ! "$SAVE_FREQ" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: steps, batch size, and PAP_MOE_SAVE_FREQ must be positive integers" >&2
  exit 2
fi
if (( STEPS < LOG_FREQ )); then
  LOG_FREQ="$STEPS"
fi
if [[ "$SAVE_CHECKPOINT" != "true" && "$SAVE_CHECKPOINT" != "false" ]]; then
  echo "ERROR: PAP_SAVE_CHECKPOINT must be true or false" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "false" ]]; then
  echo "ERROR: PAP_MOE_DRY_RUN must be true or false" >&2
  exit 2
fi
for required in \
  "$PAP_TEMPLATE" \
  "$INPUT_MODEL/config.json" \
  "$INPUT_MODEL/model.safetensors" \
  "$DATASET_ROOT/meta/info.json"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input is missing: $required" >&2
    exit 2
  fi
done

# The temporary model view lives under /tmp. Relative symlink targets would
# otherwise resolve below /tmp and silently make pretrained loading fall back
# to random initialization.
INPUT_MODEL="$(realpath "$INPUT_MODEL")"
DATASET_ROOT="$(realpath "$DATASET_ROOT")"
if [[ "$INPUT_MODEL" == "$WS_DIR/.cleanup_trash/"* ]]; then
  echo "ERROR: quarantined checkpoints are not valid training inputs: $INPUT_MODEL" >&2
  exit 2
fi

# PAP-MoE v9 is trained on next-frame absolute joint targets.  Both action
# conversion processors must therefore be disabled in every input checkpoint.
PREPROCESSOR_JSON="$INPUT_MODEL/policy_preprocessor.json"
POSTPROCESSOR_JSON="$INPUT_MODEL/policy_postprocessor.json"
for processor_file in "$PREPROCESSOR_JSON" "$POSTPROCESSOR_JSON"; do
  if [[ ! -e "$processor_file" ]]; then
    echo "ERROR: action processor metadata is missing: $processor_file" >&2
    exit 2
  fi
done
DELTA_ENABLED="$(jq -r '[.steps[]? | select(.registry_name == "delta_actions_processor") | .config.enabled] | if length == 0 then "missing" else .[0] end' "$PREPROCESSOR_JSON")"
ABSOLUTE_ENABLED="$(jq -r '[.steps[]? | select(.registry_name == "absolute_actions_processor") | .config.enabled] | if length == 0 then "missing" else .[0] end' "$POSTPROCESSOR_JSON")"
if [[ "$DELTA_ENABLED" != "false" || "$ABSOLUTE_ENABLED" != "false" ]]; then
  echo "ERROR: incompatible action processors in $INPUT_MODEL" >&2
  echo "  delta_actions_processor.enabled=$DELTA_ENABLED (expected false)" >&2
  echo "  absolute_actions_processor.enabled=$ABSOLUTE_ENABLED (expected false)" >&2
  exit 2
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_FAMILY="${PAP_MOE_RUN_FAMILY:-pap_moe_v9}"
OUTPUT_DIR="$WS_DIR/outputs/train/${RUN_FAMILY}_${STAGE}_${RUN_TAG}"
LOG_FILE="$WS_DIR/artifacts/${RUN_FAMILY}_${STAGE}_${RUN_TAG}.log"
MODEL_VIEW="$(mktemp -d /tmp/${RUN_FAMILY}_${STAGE}.XXXXXX)"
ACTION_ANCHOR_WEIGHT="${PAP_MOE_ACTION_ANCHOR_WEIGHT:-1.0}"
ACTION_ADAPTER_TRAIN_LORA="${PAP_MOE_ACTION_ADAPTER_TRAIN_LORA:-true}"
ACTION_ADAPTER_LR="${PAP_MOE_ACTION_ADAPTER_LR:-5e-6}"
RECOVERY_ANCHOR_SCALE="${PAP_MOE_RECOVERY_ANCHOR_SCALE:-0.1}"
RECOVERY_EPISODE_START="${PAP_MOE_RECOVERY_EPISODE_START:-24}"
ACTION_CONTINUITY_WEIGHT="${PAP_MOE_ACTION_CONTINUITY_WEIGHT:-0.5}"
ACTION_CONTINUITY_HORIZON="${PAP_MOE_ACTION_CONTINUITY_HORIZON:-10}"
GRIPPER_TRANSITION_WEIGHT="${PAP_MOE_GRIPPER_TRANSITION_SAMPLING_WEIGHT:-1.0}"
GRIPPER_TRANSITION_WINDOW="${PAP_MOE_GRIPPER_TRANSITION_SAMPLING_WINDOW:-0}"
RECOVERY_START_WEIGHT="${PAP_MOE_RECOVERY_START_SAMPLING_WEIGHT:-1.0}"
RECOVERY_START_COUNT="${PAP_MOE_RECOVERY_START_SAMPLING_COUNT:-0}"
SENSOR_MODE="${PAP_MOE_CONTROLLED_SENSOR_MODE:-vsf}"
VISUAL_DEGRADATION_PROBABILITY="${PAP_MOE_VISUAL_DEGRADATION_PROBABILITY:-0.5}"
# Stage 2 optimizes the complete PAP-conditioned action policy.  Keeping the
# action flow close to the Pi0.5 baseline is an optional diagnostic, not the
# PAP-MoE objective, so the baseline-action anchor is disabled by default.
EXPERT_ACTION_ANCHOR_WEIGHT="${PAP_MOE_EXPERT_ACTION_ANCHOR_WEIGHT:-0.0}"
JOINT_ACTION_EXPERT_LR_SCALE="${PAP_MOE_JOINT_ACTION_EXPERT_LR_SCALE:-1.0}"
ROBUST_PHYSICS="${PAP_MOE_ROBUST_PHYSICS:-false}"
FACTORIZED_GATE="${PAP_MOE_FACTORIZED_GATE:-$ROBUST_PHYSICS}"
BOUNDED_CONDITIONING="${PAP_MOE_BOUNDED_CONDITIONING:-$ROBUST_PHYSICS}"
ACTION_STEP_ROUTING="${PAP_MOE_ACTION_STEP_ROUTING:-false}"
CONDITION_DROPOUT="${PAP_MOE_CONDITION_DROPOUT:-0.1}"
ROUTE_JITTER_STD="${PAP_MOE_ROUTE_JITTER_STD:-0.03}"
RESIDUAL_MAX_NORM="${PAP_MOE_RESIDUAL_MAX_NORM:-1.0}"
CONFIDENCE_FLOOR="${PAP_MOE_CONFIDENCE_FLOOR:-0.0}"
NOMINAL_EXPERT_MULTIPLIER="${PAP_MOE_NOMINAL_EXPERT_MULTIPLIER:-0.15}"
CONDITION_GRIPPER_WITH_EXPERTS="${PAP_MOE_CONDITION_GRIPPER_WITH_EXPERTS:-false}"
GATE_CALIBRATION_ROUTING_WEIGHT="${PAP_MOE_GATE_CALIBRATION_ROUTING_WEIGHT:-0.1}"
PHYSICAL_FUSION_ARCHITECTURE="${PAP_MOE_PHYSICAL_FUSION_ARCHITECTURE:-late_output_v1}"
EXPERT_REPRESENTATION_LOSS_WEIGHT="${PAP_MOE_EXPERT_REPRESENTATION_LOSS_WEIGHT:-0.0}"
USE_VISUAL_MEMORY="${PAP_MOE_USE_VISUAL_MEMORY:-false}"
VISUAL_MEMORY_DISTILLATION_LOSS_WEIGHT="${PAP_MOE_VISUAL_MEMORY_DISTILLATION_LOSS_WEIGHT:-0.0}"
if [[ "$SENSOR_MODE" != "vs" && "$SENSOR_MODE" != "vsf" ]]; then
  echo "ERROR: PAP_MOE_CONTROLLED_SENSOR_MODE must be vs or vsf" >&2
  exit 2
fi
if ! awk -v value="$VISUAL_DEGRADATION_PROBABILITY" 'BEGIN { exit !(value >= 0 && value <= 1) }'; then
  echo "ERROR: PAP_MOE_VISUAL_DEGRADATION_PROBABILITY must be in [0, 1]" >&2
  exit 2
fi
if ! awk -v value="$JOINT_ACTION_EXPERT_LR_SCALE" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
  echo "ERROR: PAP_MOE_JOINT_ACTION_EXPERT_LR_SCALE must be in (0, 1]" >&2
  exit 2
fi
if [[ "$ROBUST_PHYSICS" != "true" && "$ROBUST_PHYSICS" != "false" ]]; then
  echo "ERROR: PAP_MOE_ROBUST_PHYSICS must be true or false" >&2
  exit 2
fi
if [[ "$FACTORIZED_GATE" != "true" && "$FACTORIZED_GATE" != "false" ]]; then
  echo "ERROR: PAP_MOE_FACTORIZED_GATE must be true or false" >&2
  exit 2
fi
if [[ "$BOUNDED_CONDITIONING" != "true" && "$BOUNDED_CONDITIONING" != "false" ]]; then
  echo "ERROR: PAP_MOE_BOUNDED_CONDITIONING must be true or false" >&2
  exit 2
fi
if [[ "$ACTION_STEP_ROUTING" != "true" && "$ACTION_STEP_ROUTING" != "false" ]]; then
  echo "ERROR: PAP_MOE_ACTION_STEP_ROUTING must be true or false" >&2
  exit 2
fi
if ! awk -v value="$CONDITION_DROPOUT" 'BEGIN { exit !(value >= 0 && value <= 1) }'; then
  echo "ERROR: PAP_MOE_CONDITION_DROPOUT must be in [0, 1]" >&2
  exit 2
fi
if ! awk -v value="$ROUTE_JITTER_STD" 'BEGIN { exit !(value >= 0) }'; then
  echo "ERROR: PAP_MOE_ROUTE_JITTER_STD must be non-negative" >&2
  exit 2
fi
if ! awk -v value="$RESIDUAL_MAX_NORM" 'BEGIN { exit !(value > 0) }'; then
  echo "ERROR: PAP_MOE_RESIDUAL_MAX_NORM must be positive" >&2
  exit 2
fi
if ! awk -v value="$CONFIDENCE_FLOOR" 'BEGIN { exit !(value >= 0 && value <= 1) }'; then
  echo "ERROR: PAP_MOE_CONFIDENCE_FLOOR must be in [0, 1]" >&2
  exit 2
fi
if ! awk -v value="$NOMINAL_EXPERT_MULTIPLIER" 'BEGIN { exit !(value >= 0 && value <= 1) }'; then
  echo "ERROR: PAP_MOE_NOMINAL_EXPERT_MULTIPLIER must be in [0, 1]" >&2
  exit 2
fi
if [[ "$CONDITION_GRIPPER_WITH_EXPERTS" != "true" && "$CONDITION_GRIPPER_WITH_EXPERTS" != "false" ]]; then
  echo "ERROR: PAP_MOE_CONDITION_GRIPPER_WITH_EXPERTS must be true or false" >&2
  exit 2
fi
if ! awk -v value="$GATE_CALIBRATION_ROUTING_WEIGHT" 'BEGIN { exit !(value >= 0) }'; then
  echo "ERROR: PAP_MOE_GATE_CALIBRATION_ROUTING_WEIGHT must be non-negative" >&2
  exit 2
fi
if [[ "$PHYSICAL_FUSION_ARCHITECTURE" != "late_output_v1" && "$PHYSICAL_FUSION_ARCHITECTURE" != "action_input_tokens_v2" ]]; then
  echo "ERROR: PAP_MOE_PHYSICAL_FUSION_ARCHITECTURE must be late_output_v1 or action_input_tokens_v2" >&2
  exit 2
fi
if ! awk -v value="$EXPERT_REPRESENTATION_LOSS_WEIGHT" 'BEGIN { exit !(value >= 0) }'; then
  echo "ERROR: PAP_MOE_EXPERT_REPRESENTATION_LOSS_WEIGHT must be non-negative" >&2
  exit 2
fi
if [[ "$USE_VISUAL_MEMORY" != "true" && "$USE_VISUAL_MEMORY" != "false" ]]; then
  echo "ERROR: PAP_MOE_USE_VISUAL_MEMORY must be true or false" >&2
  exit 2
fi
if ! awk -v value="$VISUAL_MEMORY_DISTILLATION_LOSS_WEIGHT" 'BEGIN { exit !(value >= 0) }'; then
  echo "ERROR: PAP_MOE_VISUAL_MEMORY_DISTILLATION_LOSS_WEIGHT must be non-negative" >&2
  exit 2
fi

cleanup() {
  rm -f \
    "$MODEL_VIEW/config.json" \
    "$MODEL_VIEW/model.safetensors" \
    "$MODEL_VIEW/policy_preprocessor.json" \
    "$MODEL_VIEW/policy_postprocessor.json" \
    "$MODEL_VIEW"/policy_*_processor.safetensors
  rmdir "$MODEL_VIEW" 2>/dev/null || true
}
trap cleanup EXIT

# A Pi0.5 baseline has a pi05 config; the first PAP stage needs the PAP
# architecture template. Later PAP stages inherit the complete prior config.
if [[ "$(jq -r '.type // empty' "$INPUT_MODEL/config.json")" == "pap_moe" ]]; then
  CONFIG_SOURCE="$INPUT_MODEL/config.json"
else
  CONFIG_SOURCE="$PAP_TEMPLATE"
fi

jq --arg stage "$STAGE" \
  --arg gate_arch_override "${PAP_MOE_GATE_ARCHITECTURE:-}" \
  --arg sensor_mode "$SENSOR_MODE" \
  --slurpfile input_config "$INPUT_MODEL/config.json" \
  --argjson steps "$STEPS" \
  --argjson action_anchor_weight "$ACTION_ANCHOR_WEIGHT" \
  --argjson action_adapter_train_lora "$ACTION_ADAPTER_TRAIN_LORA" \
  --argjson action_adapter_lr "$ACTION_ADAPTER_LR" \
  --argjson recovery_anchor_scale "$RECOVERY_ANCHOR_SCALE" \
  --argjson recovery_episode_start "$RECOVERY_EPISODE_START" \
  --argjson action_continuity_weight "$ACTION_CONTINUITY_WEIGHT" \
  --argjson visual_degradation_probability "$VISUAL_DEGRADATION_PROBABILITY" \
  --argjson expert_action_anchor_weight "$EXPERT_ACTION_ANCHOR_WEIGHT" \
  --argjson joint_action_expert_lr_scale "$JOINT_ACTION_EXPERT_LR_SCALE" \
  --argjson robust_physics "$ROBUST_PHYSICS" \
  --argjson factorized_gate "$FACTORIZED_GATE" \
  --argjson bounded_conditioning "$BOUNDED_CONDITIONING" \
  --argjson action_step_routing "$ACTION_STEP_ROUTING" \
  --argjson condition_dropout "$CONDITION_DROPOUT" \
  --argjson route_jitter_std "$ROUTE_JITTER_STD" \
  --argjson residual_max_norm "$RESIDUAL_MAX_NORM" \
  --argjson confidence_floor "$CONFIDENCE_FLOOR" \
  --argjson nominal_expert_multiplier "$NOMINAL_EXPERT_MULTIPLIER" \
  --argjson condition_gripper_with_experts "$CONDITION_GRIPPER_WITH_EXPERTS" \
  --argjson gate_calibration_routing_weight "$GATE_CALIBRATION_ROUTING_WEIGHT" \
  --arg physical_fusion_architecture "$PHYSICAL_FUSION_ARCHITECTURE" \
  --argjson expert_representation_loss_weight "$EXPERT_REPRESENTATION_LOSS_WEIGHT" \
  --argjson use_visual_memory "$USE_VISUAL_MEMORY" \
  --argjson visual_memory_distillation_loss_weight "$VISUAL_MEMORY_DISTILLATION_LOSS_WEIGHT" \
  --argjson action_continuity_horizon "$ACTION_CONTINUITY_HORIZON" '
  del(.train_stagegate_only)
  | .mask_invalid_prefix_tokens =
      (if $gate_arch_override == "physics_gate_v2" and $physical_fusion_architecture == "action_input_tokens_v2" then true else (.mask_invalid_prefix_tokens // false) end)
  | .mask_invalid_history_cameras =
      (if $gate_arch_override == "physics_gate_v2" and $physical_fusion_architecture == "action_input_tokens_v2" then true else (.mask_invalid_history_cameras // false) end)
  | .train_expert_action_joint = ($stage == "expert_action_joint")
  | .train_physicsgate_action_joint = ($stage == "physicsgate_action_joint")
  | .train_physicsgate_only = ($stage == "physicsgate" or $stage == "physicsgate_sequence" or $stage == "temporal_gate")
  | .train_expert_only = ($stage == "expert")
  | .train_gate_calibration_only = ($stage == "gate_calibration")
  | .train_route_forecaster_only = ($stage == "route_forecaster")
  | .train_conditioner_only = ($stage == "conditioner")
  | .train_action_adapter_only = ($stage == "action_adapter")
  | .train_pap_moe_joint = false
  | .action_step_routing = ($stage == "physicsgate_action_joint" or $stage == "route_forecaster" or $stage == "physicsgate_sequence" or $stage == "temporal_gate" or $action_step_routing or (.action_step_routing // false))
  # Historical two-pass experiment only; the production PhysicsGate has no
  # draft action and performs exactly one action-chunk generation.
  | .temporal_gate_two_pass_inference = ($stage == "temporal_gate")
  | .route_sequence_loss_weight =
      (if $stage == "gate_calibration" then $gate_calibration_routing_weight else 1.0 end)
  # Physical experts are heterogeneous state specialists, not exchangeable
  # capacity shards. Their utilization must follow the physical labels; do
  # not bias routing toward any batch-level load distribution (especially
  # because the production training contract uses batch_size=1).
  | .balance_loss_weight = 0.0
  # The first PAP stage is initialized from the qualified Pi0.5 baseline.
  # Preserve the baseline visual/token/action interface instead of silently
  # falling back to stale values in the PAP architecture template.  PAP-only
  # force/history/routing input features remain supplied by the template and
  # are completed from dataset metadata when processors are rebuilt.
  | .empty_cameras = ($input_config[0].empty_cameras // .empty_cameras)
  | .tokenizer_max_length =
      ($input_config[0].tokenizer_max_length // .tokenizer_max_length)
  | .action_feature_names =
      ($input_config[0].action_feature_names // .action_feature_names)
  | .visual_degradation_training_probability =
      (if ($stage == "expert_action_joint" or $stage == "physicsgate_action_joint" or $stage == "expert" or $stage == "physicsgate" or $stage == "physicsgate_sequence" or $stage == "gate_calibration" or $stage == "route_forecaster" or $stage == "temporal_gate")
       then $visual_degradation_probability else 0.0 end)
  | .visual_degradation_dropout_fraction = 0.5
  | .visual_degradation_glare_gain_min = 2.0
  | .visual_degradation_glare_gain_max = 6.0
  | .expert_action_anchor_weight =
      (if ($stage == "expert_action_joint" or $stage == "expert") then $expert_action_anchor_weight else 0.0 end)
  # Fair baseline/PAP ablation: the complete action expert uses the same LR as
  # the qualified Pi0.5 baseline. PAP-specific modules are additional
  # trainables; they do not alter the action expert optimizer contract.
  | .joint_action_expert_lr_scale =
      (if ($stage == "expert_action_joint" or $stage == "physicsgate_action_joint") then $joint_action_expert_lr_scale
       else (.joint_action_expert_lr_scale // 1.0) end)
  # Auxiliary deterministic heads have their own launchers.  A checkpoint
  # produced by one of those launchers may retain its training-mode flag;
  # clear all three before selecting a PAP stage so config exclusivity remains
  # valid while the learned head weights themselves stay loaded and frozen.
  | .train_arm_head_only = false
  | .train_gripper_head_only = false
  | .train_release_head_only = false
  # The seventh action dimension is the continuous 0..0.8-rad gripper joint.
  # Keep exactly the same loss weighting as the qualified Pi0.5 baseline so
  # PAP-MoE differs only by its physical experts and conditioning pathway.
  | .gripper_action_index = 6
  | .gripper_loss_weight = 1.0
  | .gripper_open_loss_weight = 1.0
  | .controlled_ablation_sensor_mode = $sensor_mode
  # Shared B/P stages must see the same sampling distribution. Stage-balanced
  # sampling is therefore confined to the two P-only routing stages.
  | .stage_balanced_sampling =
      ($stage == "physicsgate" or $stage == "physicsgate_sequence" or $stage == "gate_calibration" or $stage == "route_forecaster" or $stage == "temporal_gate")
  | .action_adapter_anchor_weight =
      (if $stage == "action_adapter" then $action_anchor_weight else 0.0 end)
  | .action_adapter_train_lora = $action_adapter_train_lora
  | .action_adapter_recovery_anchor_scale = $recovery_anchor_scale
  | .action_adapter_recovery_episode_start = $recovery_episode_start
  | .action_adapter_continuity_weight =
      (if $stage == "action_adapter" then $action_continuity_weight else 0.0 end)
  | .action_adapter_continuity_horizon = $action_continuity_horizon
  | .gate_calibration_action_loss_weight = 1.0
  | .gate_calibration_routing_loss_weight = $gate_calibration_routing_weight
  # Conditioner training relies on a frozen cached VLM prefix. Transformers
  # disables that cache when gradient checkpointing is active in train mode,
  # producing a prefix/suffix attention-length mismatch. Only the 6.3M
  # conditioner parameters are trainable here, so checkpointing is unnecessary.
  | .gradient_checkpointing =
      (if ($stage == "expert_action_joint" or $stage == "physicsgate_action_joint") then true
       elif ($stage == "physicsgate_sequence" or $stage == "gate_calibration" or $stage == "route_forecaster" or $stage == "temporal_gate" or $stage == "conditioner") then false
       elif $stage == "action_adapter" then true
       else .gradient_checkpointing end)
  # The conditioner is a final distribution-alignment stage, not a fresh PAP
  # module fit. Keep its peak LR at the base action-policy rate instead of the
  # 1e-4 rate used to learn experts and Gate encoders from scratch.
  | .pap_moe_optimizer_lr =
      (if ($stage == "expert_action_joint" or $stage == "physicsgate_action_joint") then 2.5e-5
       elif $stage == "conditioner" then 2.5e-5
       elif ($stage == "physicsgate_sequence" or $stage == "gate_calibration" or $stage == "route_forecaster" or $stage == "temporal_gate") then 1e-5
       elif $stage == "action_adapter" then $action_adapter_lr
       else .pap_moe_optimizer_lr end)
  # A deterministic-head checkpoint carries its own high-LR short scheduler.
  # Never inherit that optimizer contract into action-flow adaptation: a
  # decay LR above the adapter peak silently turns cosine decay into LR growth.
  | .optimizer_lr =
      (if $stage == "action_adapter" then $action_adapter_lr else .optimizer_lr end)
  | .scheduler_decay_lr =
      (if $stage == "action_adapter" then ($action_adapter_lr / 10.0)
       else .scheduler_decay_lr end)
  | .scheduler_warmup_steps =
      (if $stage == "action_adapter" then 1000
       else .scheduler_warmup_steps end)
  | .scheduler_decay_steps =
      (if $stage == "action_adapter" then 30000
       else .scheduler_decay_steps end)
  # Expert v2 restores the original structurally heterogeneous design.  Set
  # this explicitly because pre-v2 checkpoints intentionally default to the
  # legacy adapters so they remain usable as diagnostic baselines.
  | .physics_expert_architecture =
      (if ($stage == "expert_action_joint" or $stage == "physicsgate_action_joint" or $stage == "expert" or $stage == "gate_calibration" or $stage == "route_forecaster" or $stage == "conditioner" or $stage == "action_adapter") then
         "heterogeneous_v2"
       else (.physics_expert_architecture // "heterogeneous_v2") end)
  # Preserve continuous route magnitude by applying expert probabilities after
  # the condition LayerNorm/projection. Pre-fix checkpoints default to the
  # legacy behavior in Python and remain loadable for diagnostics.
  | .conditioner_routing_mode =
      (if ($stage == "expert_action_joint" or $stage == "physicsgate_action_joint" or $stage == "expert" or $stage == "gate_calibration" or $stage == "route_forecaster" or $stage == "conditioner" or $stage == "action_adapter") then
         "post_projection_v2"
       else (.conditioner_routing_mode // "post_projection_v2") end)
  # Opt-in robust PAP-MoE v10 path. Existing v9 commands remain byte-for-byte
  # compatible unless PAP_MOE_ROBUST_PHYSICS=true is explicitly supplied.
  | .physics_gate_architecture =
      (if $gate_arch_override != "" then $gate_arch_override
       elif $stage == "physicsgate_sequence" then "physics_gate_v2"
       elif $stage == "temporal_gate" then "temporal_bcm_v2"
       # Prediction-aligned joint training must retain the direct 50-step
       # PhysicsGate learned by the preceding sequence stage. Falling back to
       # factorized_bcm_v1 here silently discards the deploy-time route head.
       elif ($stage == "physicsgate_action_joint" and
             $input_config[0].physics_gate_architecture == "physics_gate_v2") then
         "physics_gate_v2"
       elif $factorized_gate then "factorized_bcm_v1"
       else (.physics_gate_architecture // "legacy_softmax") end)
  | .bounded_action_conditioning =
      (if $physical_fusion_architecture == "action_input_tokens_v2" then $bounded_conditioning
       elif $stage == "physicsgate_action_joint" then true
       elif $bounded_conditioning then true
       else (.bounded_action_conditioning // false) end)
  | .expert_conditioning_scale_init =
      (if $robust_physics then 0.0 else (.expert_conditioning_scale_init // 0.0) end)
  | .action_conditioning_residual_max_norm =
      (if $stage == "physicsgate_action_joint" then 1.0
       elif $robust_physics then $residual_max_norm
       else (.action_conditioning_residual_max_norm // 1.0) end)
  | .action_conditioning_confidence_floor =
      (if $robust_physics then $confidence_floor
       else (.action_conditioning_confidence_floor // 0.0) end)
  | .nominal_expert_conditioning_multiplier =
      (if $stage == "physicsgate_action_joint" then $nominal_expert_multiplier
       else (.nominal_expert_conditioning_multiplier // 1.0) end)
  | .condition_gripper_with_physical_experts =
      (if $physical_fusion_architecture == "action_input_tokens_v2" then true
       elif $stage == "physicsgate_action_joint" then $condition_gripper_with_experts
       else (.condition_gripper_with_physical_experts // true) end)
  # vNext moves the routed physical representation in front of the complete
  # Action Expert. The legacy mode stays available for controlled ablation.
  | .physical_fusion_architecture = $physical_fusion_architecture
  | .expert_representation_loss_weight =
      (if ($stage == "expert_action_joint" or $stage == "expert")
       then $expert_representation_loss_weight else 0.0 end)
  | .use_visual_memory = $use_visual_memory
  | .visual_memory_history_indices = [-30, -20, -10, 0]
  | .visual_memory_hidden_dim = 128
  | .visual_memory_distillation_loss_weight =
      (if ($stage == "expert_action_joint" or $stage == "expert")
       then $visual_memory_distillation_loss_weight else 0.0 end)
  | .physical_condition_dropout_probability =
      (if ($robust_physics and ($stage == "expert_action_joint" or $stage == "expert"))
       then $condition_dropout else 0.0 end)
  | .physical_route_jitter_std =
      (if ($robust_physics and ($stage == "expert_action_joint" or $stage == "expert"))
       then $route_jitter_std else 0.0 end)
  # In later stages the input checkpoint already carries its Gate
  # architecture.  Do not silently disable b/c/m supervision merely because
  # the launcher did not repeat PAP_MOE_FACTORIZED_GATE=true.
  | .factorized_gate_factor_loss_weight =
      (if (.physics_gate_architecture == "factorized_bcm_v1" or .physics_gate_architecture == "physics_gate_v2" or .physics_gate_architecture == "temporal_bcm_v2") then
         (if $stage == "gate_calibration" then $gate_calibration_routing_weight else 1.0 end)
       else 0.0 end)
  # Physical experts are identified under dataset soft routing only.  This
  # prevents a still-imperfect learned gate from contaminating expert action
  # specialization.  Final conditioner alignment alone uses predicted routes.
  | .teacher_forcing_start_steps =
      (if ($stage == "expert_action_joint" or $stage == "expert") then
         ($steps + 1)
       elif ($stage == "physicsgate_action_joint" or $stage == "gate_calibration" or $stage == "route_forecaster" or $stage == "conditioner" or $stage == "action_adapter") then 0
       else .teacher_forcing_start_steps end)
  | .teacher_forcing_end_steps =
      (if ($stage == "expert_action_joint" or $stage == "expert") then
         ($steps + 2)
       elif ($stage == "physicsgate_action_joint" or $stage == "gate_calibration" or $stage == "route_forecaster" or $stage == "conditioner" or $stage == "action_adapter") then 1
       else .teacher_forcing_end_steps end)
' "$CONFIG_SOURCE" > "$MODEL_VIEW/config.json"
ln -s "$INPUT_MODEL/model.safetensors" "$MODEL_VIEW/model.safetensors"
ln -s "$PREPROCESSOR_JSON" "$MODEL_VIEW/policy_preprocessor.json"
ln -s "$POSTPROCESSOR_JSON" "$MODEL_VIEW/policy_postprocessor.json"
for processor_state in "$INPUT_MODEL"/policy_*_processor.safetensors; do
  if [[ -e "$processor_state" ]]; then
    ln -s "$processor_state" "$MODEL_VIEW/$(basename "$processor_state")"
  fi
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo "PAP-MoE stage configuration dry run"
  echo "  stage: $STAGE"
  echo "  input: $INPUT_MODEL"
  echo "  dataset: $DATASET_ROOT"
  jq '{
    type,
    train_expert_action_joint,
    train_physicsgate_action_joint,
    train_physicsgate_only,
    train_gate_calibration_only,
    train_route_forecaster_only,
    teacher_forcing_start_steps,
    teacher_forcing_end_steps,
    action_step_routing,
    route_sequence_loss_weight,
    balance_loss_weight,
    gate_calibration_action_loss_weight,
    gate_calibration_routing_loss_weight,
    physics_expert_architecture,
    physics_gate_architecture,
    conditioner_routing_mode,
    bounded_action_conditioning,
    expert_conditioning_scale_init,
    action_conditioning_residual_max_norm,
    action_conditioning_confidence_floor,
    nominal_expert_conditioning_multiplier,
    condition_gripper_with_physical_experts,
    physical_condition_dropout_probability,
    physical_route_jitter_std,
    physical_fusion_architecture,
    expert_representation_loss_weight,
    use_visual_memory,
    visual_memory_history_indices,
    visual_memory_distillation_loss_weight,
    factorized_gate_factor_loss_weight,
    pap_moe_optimizer_lr,
    joint_action_expert_lr_scale
  }' "$MODEL_VIEW/config.json"
  exit 0
fi

export PYTHONPATH="$LEROBOT_DIR/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# A first PAP stage must rebuild processors to introduce the multi-scale
# observation contract. Incremental auxiliary-head training from an existing
# PAP checkpoint must instead preserve its saved statistics so frozen arm,
# gripper, routing and expert modules see exactly the same input scales.
if [[ "$(jq -r '.type // empty' "$INPUT_MODEL/config.json")" == "pap_moe" ]]; then
  DEFAULT_REBUILD_PROCESSORS=0
  DEFAULT_PRESERVE_PROCESSOR_STATS=1
else
  DEFAULT_REBUILD_PROCESSORS=1
  DEFAULT_PRESERVE_PROCESSOR_STATS=0
fi
export LEROBOT_REBUILD_PROCESSORS="${PAP_MOE_REBUILD_PROCESSORS:-$DEFAULT_REBUILD_PROCESSORS}"
export LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS="${PAP_MOE_PRESERVE_PRETRAINED_PROCESSOR_STATS:-$DEFAULT_PRESERVE_PROCESSOR_STATS}"

mkdir -p "$WS_DIR/artifacts" "$WS_DIR/outputs/train"
EXTRA_TRAIN_ARGS=()
EXTRA_TRAIN_ARGS+=(
  --gripper_transition_sampling_weight="$GRIPPER_TRANSITION_WEIGHT"
  --gripper_transition_sampling_window="$GRIPPER_TRANSITION_WINDOW"
)
# Keep the same early-trajectory sampling contract as the qualified Pi0.5
# baseline unless a controlled ablation explicitly overrides it.
INITIAL_FRAME_COUNT="${PAP_MOE_INITIAL_FRAME_SAMPLING_COUNT:-20}"
INITIAL_FRAME_WEIGHT="${PAP_MOE_INITIAL_FRAME_SAMPLING_WEIGHT:-5}"
EXTRA_TRAIN_ARGS+=(
  --initial_frame_sampling_count="$INITIAL_FRAME_COUNT"
  --initial_frame_sampling_weight="$INITIAL_FRAME_WEIGHT"
)
D1_RECEIPT="$DATASET_ROOT/meta/d1_materialization.json"
if [[ -f "$D1_RECEIPT" ]]; then
  D1_MANIFEST_DIGEST="$(jq -r '.manifest_sha256' "$D1_RECEIPT")"
  D1_TRAIN_EPISODES="$(jq -c '.split_episode_indices.train' "$D1_RECEIPT")"
  if (( $(jq -r '.independent_recovery_episodes' "$D1_RECEIPT") < 2 )); then
    echo "ERROR: weighted dataset receipt must contain at least two independent recoveries" >&2
    exit 2
  fi
  EXTRA_TRAIN_ARGS+=(
    --dataset.episodes="$D1_TRAIN_EPISODES"
    --dataset_sample_weight_key=d1.sample_weight
  )
  echo "  D1 manifest: $D1_MANIFEST_DIGEST"
  echo "  D1 split:    train only ($(jq '.split_episode_indices.train | length' "$D1_RECEIPT") episodes)"
fi
# D1 sample weights are authoritative. Do not multiply recovery weight again.
if [[ "${PAP_MOE_PAIRED_RECOVERY_SAMPLING:-false}" != "true" ]] \
  && (( RECOVERY_START_COUNT > 0 )); then
  EXTRA_TRAIN_ARGS+=(
    --semantic_episode_sampling_start_index="$RECOVERY_EPISODE_START"
    --semantic_episode_sampling_weight="$RECOVERY_START_WEIGHT"
    --semantic_recovery_start_sampling_count="$RECOVERY_START_COUNT"
  )
fi
if [[ "${PAP_MOE_PAIRED_RECOVERY_SAMPLING:-false}" == "true" ]]; then
  if [[ -z "${PAP_MOE_RECOVERY_EPISODE_START:-}" ]]; then
    echo "ERROR: paired recovery sampling requires PAP_MOE_RECOVERY_EPISODE_START" >&2
    exit 2
  fi
  if (( BATCH_SIZE % 2 != 0 )); then
    echo "ERROR: paired recovery sampling requires an even batch size" >&2
    exit 2
  fi
  EXTRA_TRAIN_ARGS+=(--semantic_paired_recovery_sampling=true)
  EXTRA_TRAIN_ARGS+=(
    --semantic_episode_sampling_start_index="$PAP_MOE_RECOVERY_EPISODE_START"
    --semantic_recovery_start_sampling_count="${PAP_MOE_RECOVERY_START_SAMPLING_COUNT:-10}"
  )
fi
echo "PAP-MoE v9 checkpointed stage training"
echo "  stage:       $STAGE"
echo "  input:       $INPUT_MODEL"
echo "  dataset:     $DATASET_ROOT"
echo "  steps:       $STEPS"
echo "  batch:       $BATCH_SIZE"
echo "  modalities:  $SENSOR_MODE"
echo "  visual slots: $(jq -r '.empty_cameras + 2' "$MODEL_VIEW/config.json") (2 real + $(jq -r '.empty_cameras' "$MODEL_VIEW/config.json") empty)"
echo "  tokenizer:    $(jq -r '.tokenizer_max_length' "$MODEL_VIEW/config.json")"
echo "  initial:      first $INITIAL_FRAME_COUNT frames x $INITIAL_FRAME_WEIGHT"
echo "  visual fail:  $(jq -r '.visual_degradation_training_probability' "$MODEL_VIEW/config.json") paired blackout/glare probability"
echo "  fusion:       $(jq -r '.physical_fusion_architecture' "$MODEL_VIEW/config.json")"
echo "  expert repr:  $(jq -r '.expert_representation_loss_weight' "$MODEL_VIEW/config.json")"
echo "  E2 memory:    $(jq -r '.use_visual_memory' "$MODEL_VIEW/config.json") history=$(jq -c '.visual_memory_history_indices' "$MODEL_VIEW/config.json")"
echo "  E2 distill:   $(jq -r '.visual_memory_distillation_loss_weight' "$MODEL_VIEW/config.json")"
echo "  output:      $OUTPUT_DIR"
echo "  checkpoint:  $SAVE_CHECKPOINT"
echo "  save freq:   $SAVE_FREQ"
if (( RECOVERY_START_COUNT > 0 )); then
  echo "  recovery start: first $RECOVERY_START_COUNT positive frames x $RECOVERY_START_WEIGHT"
fi
if [[ "$STAGE" == "expert" ]]; then
  echo "  routing TF:  1.0 for all steps (dataset soft labels; gate cannot contaminate experts)"
elif [[ "$STAGE" == "conditioner" ]]; then
  echo "  routing TF:  disabled (prediction-aligned final conditioning)"
elif [[ "$STAGE" == "action_adapter" ]]; then
  if [[ "$ACTION_ADAPTER_TRAIN_LORA" == "true" ]]; then
    echo "  trainable:   action-expert LoRA + action/time projections only"
  else
    echo "  trainable:   action/time projections only (expert LoRA frozen)"
  fi
  echo "  routing TF:  disabled (prediction-aligned action adaptation)"
fi
if [[ "$SAVE_CHECKPOINT" == "true" ]]; then
  echo "  final model: $OUTPUT_DIR/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
fi
echo "  log:         $LOG_FILE"

cd "$WS_DIR"
"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path="$MODEL_VIEW" \
  --dataset.repo_id=pap_moe/pap_moe_v9_stage_training \
  --dataset.root="$DATASET_ROOT" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers=0 \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=false \
  "${EXTRA_TRAIN_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

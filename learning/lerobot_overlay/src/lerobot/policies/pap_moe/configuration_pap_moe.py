#!/usr/bin/env python
"""
PAP-MoE Configuration — Physics-Aware Perceptual Mixture of Experts for PI0.5.
Inherits from PI05Config to preserve the Pi0.5 backbone and tokenizer.
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig

from ..pi05.configuration_pi05 import PI05Config

OBS_FORCE = "observation.force"
OBS_FORCE_FAST = "observation.force_fast"
OBS_FORCE_SLOW = "observation.force_slow"
OBS_STATE_HISTORY = "observation.state_history"
OBS_VISUAL_QUALITY = "observation.visual_quality"
OBS_PHYSICS_GATE_TARGET = "observation.physics_gate_target"
OBS_MODALITY_VALIDITY = "observation.modality_validity"
DEFAULT_GLOBAL_TASK = "pick up the peg and insert it into the hole"


@PreTrainedConfig.register_subclass("pap_moe")
@dataclass
class PAPMoEConfig(PI05Config):
    """Configuration for PAP-MoE policy based on PI0.5."""

    # ── PAP-MoE dimensions ──
    force_dim: int = 6
    robot_state_dim: int = 7
    pap_moe_feature_dim: int = 2048  # Must match VLM hidden dim (gemma_2b=2048, gemma_300m=1024)
    pap_moe_hidden_dim: int = 256  # Hidden dimension for encoders and PhysicsGate
    fast_force_window_size: int = 64
    slow_force_window_size: int = 50
    state_history_size: int = 10
    visual_quality_dim: int = 4
    # Observable probabilities/fractions already have meaningful physical
    # scales. They must bypass robot STATE quantile normalization (especially
    # visual-quality validity, which is constant 1 in clean Workspace50).
    identity_observation_feature_keys: tuple[str, ...] = (
        OBS_PHYSICS_GATE_TARGET,
        OBS_VISUAL_QUALITY,
    )
    # E2 visual-belief memory. Training reads episode-safe past frames through
    # feature delta indices; deployment maintains the same history online.
    use_visual_memory: bool = False
    # With the fixed 50-predict/10-execute contract at 10 Hz, policy queries
    # are about one second apart. These offsets match the online memory cadence.
    visual_memory_history_indices: tuple[int, ...] = (-30, -20, -10, 0)
    visual_memory_hidden_dim: int = 128
    # Opt-in fixes: absent fields preserve historical checkpoint inference.
    mask_invalid_prefix_tokens: bool = False
    mask_invalid_history_cameras: bool = False
    # Distill the episode-local history token toward the frozen clean current
    # VLM representation. Degraded current frames are excluded by E2 weight.
    visual_memory_distillation_loss_weight: float = 0.0
    require_multiscale_observations: bool = True
    # Training-only paired visual-failure augmentation. The trainer corrupts
    # both real policy cameras for a random subset of otherwise unchanged
    # state/action/force samples and recomputes the observable routing target.
    visual_degradation_training_probability: float = 0.0
    visual_degradation_dropout_fraction: float = 0.5
    visual_degradation_glare_gain_min: float = 2.0
    visual_degradation_glare_gain_max: float = 6.0

    # ── MoE configuration ──
    num_experts: int = 4
    moe_top_k: int = 4  # Legacy field; v6 uses four-expert cooperative soft routing.
    expert_transformer_layers: int = 0  # Legacy field; v6 uses heterogeneous lightweight adapters.
    # Missing in pre-v2 checkpoints, which must keep loading the old token-adapter experts.
    physics_expert_architecture: str = "legacy_token_adapters"
    e1_fusion_normalization: str = "joint"
    e1_fusion_mode: str = "early_sum"
    # Legacy checkpoints multiplied routing probabilities before a LayerNorm,
    # which erased almost all soft-routing magnitude. New runs weight projected
    # condition keys instead, while the legacy default preserves old behavior.
    conditioner_routing_mode: str = "pre_norm_legacy"
    # New checkpoints may use an interpretable factorized gate that predicts
    # visual blindness (b), contact (c), and contact mobility/compliance (m)
    # before mapping those factors to the four cooperative expert weights.
    # Keep the legacy default so existing stage-2/3 checkpoints load exactly.
    physics_gate_architecture: str = "legacy_softmax"
    # Where routed physical representations enter the Pi0.5 action path.
    # ``late_output_v1`` preserves historical checkpoints: condition the final
    # Action Expert hidden state immediately before action_out_proj.
    # ``action_input_tokens_v2`` conditions action/time input tokens before
    # the Action Expert transformer, so every transformer layer can reason
    # with the selected physical information.
    physical_fusion_architecture: str = "late_output_v1"
    # Experiment-3 opt-in: predict one b/c/m route for every action token
    # instead of holding the route from the first observation for the whole
    # 50-step action chunk. Disabled by default for checkpoint compatibility.
    action_step_routing: bool = False
    # Historical two-pass experiment compatibility only. The current
    # physics_gate_v2 predicts all 50 b/c/m triplets directly from observation.
    temporal_gate_two_pass_inference: bool = True
    route_sequence_loss_weight: float = 1.0
    # Deployment-time strength of the physical residual injected into the
    # pretrained Pi0.5 action tokens. Zero preserves the baseline exactly.
    action_conditioning_scale: float = 1.0
    # Isolated single-variable experiment. Old checkpoints retain exact defaults.
    action_conditioning_route_prior: str = "none"
    # Robust conditioning is opt-in for new runs. Legacy checkpoints retain
    # their exact conditioner path when this flag is false.
    bounded_action_conditioning: bool = False
    expert_conditioning_scale_init: float = 0.0
    # E1 is the nominal free-space path. New models can keep it close to the
    # Pi0.5 identity while allowing contact/degraded-vision experts to make
    # larger corrections. The compatibility default keeps old checkpoints.
    nominal_expert_conditioning_multiplier: float = 1.0
    # Physical experts should normally refine arm/contact behavior without
    # moving the continuous gripper decision boundary. Kept true by default
    # so historical checkpoints retain their exact behavior.
    condition_gripper_with_physical_experts: bool = True
    action_conditioning_residual_max_norm: float = 1.0
    action_conditioning_confidence_floor: float = 0.0
    physical_condition_dropout_probability: float = 0.0
    physical_route_jitter_std: float = 0.0
    # Checkpoint compatibility only. Single-PAP was removed; full PAP-MoE is
    # the sole supported architecture.
    controlled_ablation_architecture: str = "moe4"
    controlled_dense_expert_bottleneck: int | None = None  # ignored legacy field
    # ── Four physical experts (not semantic subtasks) ──
    physics_expert_names: list[str] = field(
        default_factory=lambda: [
            "free motion, no contact",
            "visual blind, force guided",
            "rigid surface contact, searching",
            "flexible insertion, compliant",
        ]
    )
    # Natural-language task instruction. PAP-MoE no longer predicts semantic
    # subtasks or task-internal progress; physical routing is observation-only.
    global_task: str = DEFAULT_GLOBAL_TASK

    # ── Loss weights ──
    stage_loss_weight: float = 0.5
    # PAP experts have distinct physical responsibilities and therefore must
    # follow the observed physical state rather than an artificial utilization
    # target. Keep the prior-matching term diagnostic-only by default.
    balance_loss_weight: float = 0.0
    # Final PhysicsGate/expert alignment: action flow is the main objective,
    # while a small routing term prevents a degenerate single-expert solution.
    gate_calibration_action_loss_weight: float = 1.0
    gate_calibration_routing_loss_weight: float = 0.1
    factorized_gate_factor_loss_weight: float = 1.0
    # Training-only supervision that makes each heterogeneous expert encode
    # its declared physical descriptors instead of relying on action loss
    # alone. Zero preserves historical checkpoints and training contracts.
    expert_representation_loss_weight: float = 0.0
    # Version supervision independently of weights: old checkpoints retain
    # their historical objective when reproduced, new runs opt in explicitly.
    expert_motion_target: str = "legacy_cross_normalized"
    visual_memory_teacher_policy: str = "legacy_soft_weight"

    # ── Stage loss warmup ──
    stage_loss_warmup_steps: int = 500
    stage_loss_warmup_end_steps: int = 2000

    # ── Scheduled sampling (teacher forcing for PhysicsGate routing) ──
    teacher_forcing_start_steps: int = 5000
    teacher_forcing_end_steps: int = 15000
    stage_balanced_sampling: bool = True
    stage_ema_gamma: float = 0.9

    # ── Class weights for balanced cross-entropy ──
    stage_class_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

    # ── LoRA for Action Expert ──
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05

    # ── Training ──
    pap_moe_optimizer_lr: float = 1e-4
    joint_action_expert_lr_scale: float = 0.1
    freeze_vision_encoder: bool = True
    # Primary PAP adaptation stage: physical experts condition the complete
    # Pi0.5 action expert, so both sides of that interface must learn jointly.
    # The original PaliGemma visual-language prefix remains frozen.
    train_expert_action_joint: bool = False
    # Training-only route source; inference always uses PhysicsGate unless
    # an explicit diagnostic override is supplied. No new model parameters.
    expert_action_training_route: str = "ground_truth"
    # Prediction-aligned end-to-end PAP adaptation. Unlike the historical
    # ``train_pap_moe_joint`` LoRA ablation, this mode trains PhysicsGate, all
    # routed experts/conditioning modules, and the complete Pi0.5 action
    # expert while keeping the original PaliGemma VLM frozen.
    train_physicsgate_action_joint: bool = False
    train_expert_only: bool = False
    train_physicsgate_only: bool = False
    train_gate_calibration_only: bool = False
    train_route_forecaster_only: bool = False
    train_conditioner_only: bool = False
    train_action_adapter_only: bool = False
    action_adapter_train_lora: bool = True
    train_pap_moe_joint: bool = True  # default: full joint training
    # Action-adapter safeguards. The anchor reuses the frozen source backbone
    # with a snapshot of the adapter parameters, so no second 4B model is kept.
    action_adapter_anchor_weight: float = 0.0
    action_adapter_recovery_anchor_scale: float = 1.0
    action_adapter_recovery_episode_start: int = 24
    action_adapter_continuity_weight: float = 0.0
    action_adapter_continuity_horizon: int = 10
    # Preserve the qualified Pi0.5 action field while identifying physical
    # experts. The frozen teacher is the zero-residual conditioner snapshot.
    expert_action_anchor_weight: float = 0.0

    # ── Backward-compatibility aliases (old checkpoints saved with sa_moe_* names) ──
    # These are intentionally NOT dataclass fields; they are injected via __post_init__.
    sa_moe_feature_dim: int | None = None  # remapped → pap_moe_feature_dim
    sa_moe_hidden_dim: int | None = None  # remapped → pap_moe_hidden_dim
    sa_moe_optimizer_lr: float | None = None  # remapped → pap_moe_optimizer_lr

    def __post_init__(self):
        super().__post_init__()

        legacy_physics_target = self.input_features.pop("observation.stage", None)
        if (
            legacy_physics_target is not None
            and OBS_PHYSICS_GATE_TARGET not in self.input_features
        ):
            self.input_features[OBS_PHYSICS_GATE_TARGET] = legacy_physics_target

        # ── Backward-compat: remap old sa_moe_* fields loaded from config.json ──
        if self.sa_moe_feature_dim is not None:
            self.pap_moe_feature_dim = self.sa_moe_feature_dim
            self.sa_moe_feature_dim = None
        if self.sa_moe_hidden_dim is not None:
            self.pap_moe_hidden_dim = self.sa_moe_hidden_dim
            self.sa_moe_hidden_dim = None
        if self.sa_moe_optimizer_lr is not None:
            self.pap_moe_optimizer_lr = self.sa_moe_optimizer_lr
            self.sa_moe_optimizer_lr = None

        # Adjust pap_moe_feature_dim based on the VLM variant
        if self.paligemma_variant == "gemma_300m":
            self.pap_moe_feature_dim = 1024
        elif self.paligemma_variant == "gemma_2b":
            self.pap_moe_feature_dim = 2048

        if self.num_experts < 1:
            raise ValueError(f"num_experts={self.num_experts} must be >= 1")
        if not 0.0 <= self.visual_degradation_training_probability <= 1.0:
            raise ValueError("visual_degradation_training_probability must be in [0, 1]")
        if not 0.0 <= self.visual_degradation_dropout_fraction <= 1.0:
            raise ValueError("visual_degradation_dropout_fraction must be in [0, 1]")
        if not 1.0 <= self.visual_degradation_glare_gain_min <= self.visual_degradation_glare_gain_max:
            raise ValueError(
                "visual degradation glare gains must satisfy "
                "1 <= gain_min <= gain_max"
            )
        if self.use_visual_memory:
            history = self.visual_memory_history_indices
            if len(history) < 2 or history[-1] != 0:
                raise ValueError(
                    "visual_memory_history_indices must contain past frames and end in 0"
                )
            if any(index > 0 for index in history) or tuple(sorted(history)) != history:
                raise ValueError(
                    "visual_memory_history_indices must be sorted and non-positive"
                )
            if self.visual_memory_hidden_dim <= 0:
                raise ValueError("visual_memory_hidden_dim must be positive")
        if self.visual_memory_distillation_loss_weight < 0:
            raise ValueError("visual_memory_distillation_loss_weight must be non-negative")
        if self.visual_memory_distillation_loss_weight > 0 and not self.use_visual_memory:
            raise ValueError(
                "visual_memory_distillation_loss_weight requires use_visual_memory=true"
            )
        if self.physics_expert_architecture not in {"legacy_token_adapters", "heterogeneous_v2"}:
            raise ValueError(
                "physics_expert_architecture must be 'legacy_token_adapters' or 'heterogeneous_v2', "
                f"received {self.physics_expert_architecture!r}"
            )
        if self.e1_fusion_normalization not in {"joint", "branchwise"}:
            raise ValueError("e1_fusion_normalization must be joint or branchwise")
        if self.e1_fusion_mode not in {"early_sum", "late_sum"}:
            raise ValueError("e1_fusion_mode must be early_sum or late_sum")
        if self.e1_fusion_mode != "early_sum" and self.physics_expert_architecture != "heterogeneous_v2":
            raise ValueError("late_sum requires heterogeneous_v2 experts")
        if self.e1_fusion_normalization == "branchwise" and self.physics_expert_architecture != "heterogeneous_v2":
            raise ValueError("branchwise E1 normalization requires heterogeneous_v2")
        if self.conditioner_routing_mode not in {"pre_norm_legacy", "post_projection_v2"}:
            raise ValueError(
                "conditioner_routing_mode must be 'pre_norm_legacy' or 'post_projection_v2', "
                f"received {self.conditioner_routing_mode!r}"
            )
        if self.physics_gate_architecture not in {
            "legacy_softmax",
            "factorized_bcm_v1",
            "physics_gate_v2",
            "temporal_bcm_v2",
        }:
            raise ValueError(
                "physics_gate_architecture must be 'legacy_softmax' or "
                "'factorized_bcm_v1', 'physics_gate_v2' or the historical "
                "'temporal_bcm_v2', received "
                f"{self.physics_gate_architecture!r}"
            )
        if self.action_step_routing and self.physics_gate_architecture not in {
            "factorized_bcm_v1",
            "physics_gate_v2",
            "temporal_bcm_v2",
        }:
            raise ValueError(
                "action_step_routing requires a b/c/m PhysicsGate"
            )
        if self.physics_gate_architecture in {"physics_gate_v2", "temporal_bcm_v2"} and not self.action_step_routing:
            raise ValueError(
                f"{self.physics_gate_architecture} requires action_step_routing=true"
            )
        if self.physical_fusion_architecture not in {
            "late_output_v1",
            "action_input_tokens_v2",
        }:
            raise ValueError(
                "physical_fusion_architecture must be 'late_output_v1' or "
                "'action_input_tokens_v2', received "
                f"{self.physical_fusion_architecture!r}"
            )
        if (
            self.physical_fusion_architecture == "action_input_tokens_v2"
            and not self.condition_gripper_with_physical_experts
        ):
            raise ValueError(
                "action_input_tokens_v2 conditions the unified Action Expert and "
                "cannot isolate the gripper dimension; set "
                "condition_gripper_with_physical_experts=true"
            )
        if (
            self.physical_fusion_architecture == "action_input_tokens_v2"
            and self.expert_action_anchor_weight > 0
        ):
            raise ValueError(
                "expert_action_anchor_weight is a legacy late-residual safeguard "
                "and must be zero for action_input_tokens_v2"
            )
        if self.route_sequence_loss_weight < 0:
            raise ValueError("route_sequence_loss_weight must be non-negative")
        if not 0.0 <= self.action_conditioning_scale <= 1.0:
            raise ValueError("action_conditioning_scale must be in [0, 1]")
        if self.action_conditioning_route_prior not in {"none", "log_probability"}:
            raise ValueError("action_conditioning_route_prior must be none or log_probability")
        if self.bounded_action_conditioning and self.conditioner_routing_mode != "post_projection_v2":
            raise ValueError(
                "bounded_action_conditioning requires conditioner_routing_mode='post_projection_v2'"
            )
        if self.action_conditioning_residual_max_norm <= 0:
            raise ValueError("action_conditioning_residual_max_norm must be positive")
        if not 0.0 <= self.nominal_expert_conditioning_multiplier <= 1.0:
            raise ValueError("nominal_expert_conditioning_multiplier must be in [0, 1]")
        if not 0.0 <= self.action_conditioning_confidence_floor <= 1.0:
            raise ValueError("action_conditioning_confidence_floor must be in [0, 1]")
        if not 0.0 <= self.physical_condition_dropout_probability <= 1.0:
            raise ValueError("physical_condition_dropout_probability must be in [0, 1]")
        if self.physical_route_jitter_std < 0:
            raise ValueError("physical_route_jitter_std must be non-negative")
        if self.controlled_ablation_architecture != "moe4":
            raise ValueError("Single-PAP was removed; controlled_ablation_architecture must be 'moe4'")
        if len(self.physics_expert_names) != self.num_experts:
            raise ValueError(
                "physics_expert_names length "
                f"({len(self.physics_expert_names)}) must match num_experts ({self.num_experts})"
            )
        if self.arm_head_semantic_context_dim:
            raise ValueError("PAP-MoE no longer supports semantic context in the arm head")
        if self.gripper_head_semantic_context_dim:
            raise ValueError("PAP-MoE no longer supports semantic context in the gripper head")
        enabled_stages = sum(
            bool(flag)
            for flag in (
                self.train_expert_action_joint,
                self.train_physicsgate_action_joint,
                self.train_expert_only,
                self.train_physicsgate_only,
                self.train_gate_calibration_only,
                self.train_route_forecaster_only,
                self.train_conditioner_only,
                self.train_action_adapter_only,
                self.train_arm_head_only,
                self.train_gripper_head_only,
                self.train_pap_moe_joint,
            )
        )
        if enabled_stages != 1:
            raise ValueError(f"Exactly one PAP-MoE training mode must be enabled; received {enabled_stages}")
        if self.action_adapter_anchor_weight < 0:
            raise ValueError("action_adapter_anchor_weight must be non-negative")
        if self.expert_action_anchor_weight < 0:
            raise ValueError("expert_action_anchor_weight must be non-negative")
        if not 0.0 < self.joint_action_expert_lr_scale <= 1.0:
            raise ValueError("joint_action_expert_lr_scale must be in (0, 1]")
        if not 0 <= self.action_adapter_recovery_anchor_scale <= 1:
            raise ValueError("action_adapter_recovery_anchor_scale must be in [0, 1]")
        if self.action_adapter_recovery_episode_start < 0:
            raise ValueError("action_adapter_recovery_episode_start must be non-negative")
        if self.action_adapter_continuity_weight < 0:
            raise ValueError("action_adapter_continuity_weight must be non-negative")
        if self.gate_calibration_action_loss_weight <= 0:
            raise ValueError("gate_calibration_action_loss_weight must be positive")
        if self.gate_calibration_routing_loss_weight < 0:
            raise ValueError("gate_calibration_routing_loss_weight must be non-negative")
        if self.factorized_gate_factor_loss_weight < 0:
            raise ValueError("factorized_gate_factor_loss_weight must be non-negative")
        if self.expert_representation_loss_weight < 0:
            raise ValueError("expert_representation_loss_weight must be non-negative")
        if self.expert_motion_target not in {"legacy_cross_normalized", "history_window_delta_v2"}:
            raise ValueError("Unknown expert_motion_target")
        if self.visual_memory_teacher_policy not in {"legacy_soft_weight", "clean_only"}:
            raise ValueError("Unknown visual_memory_teacher_policy")
        if (
            self.expert_representation_loss_weight > 0
            and self.physics_expert_architecture != "heterogeneous_v2"
        ):
            raise ValueError(
                "expert_representation_loss_weight requires "
                "physics_expert_architecture='heterogeneous_v2'"
            )
        if self.expert_representation_loss_weight > 0 and not self.require_multiscale_observations:
            raise ValueError(
                "expert representation targets require multiscale force and state history"
            )
        if not 2 <= self.action_adapter_continuity_horizon <= self.chunk_size:
            raise ValueError(
                "action_adapter_continuity_horizon must be between 2 and chunk_size"
            )

    def validate_features(self) -> None:
        super().validate_features()

        # Ensure force is in input features
        if OBS_FORCE not in self.input_features:
            self.input_features[OBS_FORCE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.force_dim,),
            )
        if self.require_multiscale_observations:
            self.input_features.setdefault(
                OBS_FORCE_FAST,
                PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(self.fast_force_window_size, self.force_dim),
                ),
            )
            self.input_features.setdefault(
                OBS_FORCE_SLOW,
                PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(self.slow_force_window_size, self.force_dim),
                ),
            )
            self.input_features.setdefault(
                OBS_STATE_HISTORY,
                PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(self.state_history_size, self.robot_state_dim),
                ),
            )
            self.input_features.setdefault(
                OBS_VISUAL_QUALITY,
                PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(self.visual_quality_dim,),
                ),
            )

        # Migrate old PAP checkpoints to the current PhysicsGate target key.
        self.input_features.pop("observation.stage", None)
        if OBS_PHYSICS_GATE_TARGET not in self.input_features:
            self.input_features[OBS_PHYSICS_GATE_TARGET] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.num_experts,),
            )

    @property
    def feature_delta_indices(self) -> dict[str, list[int]]:
        """Per-feature temporal sampling overrides used by LeRobotDataset.

        Routing targets span the action horizon. When E2 memory is enabled,
        real cameras additionally read episode-safe past frames; other sensor
        observations remain current-time inputs.
        """
        indices = {}
        if self.action_step_routing:
            indices[OBS_PHYSICS_GATE_TARGET] = list(range(self.chunk_size))
        if self.use_visual_memory:
            for key, feature in self.input_features.items():
                if feature.type == FeatureType.VISUAL and "empty_camera" not in key:
                    indices[key] = list(self.visual_memory_history_indices)
        return indices

    def get_optimizer_preset(self):
        """Use the PAP-MoE learning rate for lightweight staged training."""
        preset = super().get_optimizer_preset()
        preset.lr = self.pap_moe_optimizer_lr
        return preset

    def get_scheduler_preset(self):
        """Keep the inherited schedule but use the PAP-MoE peak learning rate."""
        preset = super().get_scheduler_preset()
        preset.peak_lr = self.pap_moe_optimizer_lr
        return preset

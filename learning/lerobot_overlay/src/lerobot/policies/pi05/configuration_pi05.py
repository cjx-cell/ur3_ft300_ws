#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224
OBS_FORCE = "observation.force"
OBS_FORCE_FAST = "observation.force_fast"
OBS_FORCE_SLOW = "observation.force_slow"
OBS_STATE_HISTORY = "observation.state_history"


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`
    tokenizer_name: str = "google/paligemma-3b-pt-224"
    # When set, use one deployment-compatible instruction for every sample
    # while retaining dataset task_index values for semantic phase sampling.
    global_task: str | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optional loss reweighting for a binary gripper action. The threshold is
    # expressed in the normalized action space seen by the policy (with
    # quantile normalization, binary open/closed targets are approximately
    # -1/+1, so zero separates them). Keeping the index unset preserves the
    # original equal-weight flow-matching objective exactly.
    gripper_action_index: int | None = None
    gripper_loss_weight: float = 1.0
    gripper_open_loss_weight: float = 1.0
    gripper_open_threshold_normalized: float = 0.0

    # Optional temporal reweighting of the flow-matching objective.  Pi0.5
    # predicts ``chunk_size`` actions during training, while receding-horizon
    # deployment may execute only the first ``n_action_steps`` actions.  A
    # weight above one emphasizes that actually executed prefix without
    # discarding the remainder of the planning chunk.  Defaults preserve the
    # original equal-weight objective exactly.
    action_prefix_loss_horizon: int | None = None
    action_prefix_loss_weight: float = 1.0

    # A binary gripper is poorly represented by stochastic continuous flow:
    # the sampled noise can select open/closed modes independently of the
    # observation. This optional head predicts the complete gripper chunk
    # deterministically from the contextualized visual-language-state prefix.
    use_deterministic_gripper_head: bool = False
    train_gripper_head_only: bool = False
    gripper_head_hidden_dim: int = 512
    gripper_head_pooling: str = "last"
    # The legacy deterministic head pools only the visual-language prefix and
    # can collapse to an almost constant open/closed prediction.  This mode
    # additionally consumes the same calibrated force/proprioceptive summary
    # used by the release and arm heads, so grasp closure is conditioned on
    # measured task progress and remains deployable online.
    gripper_head_use_sensor_context: bool = False
    # Optional predicted task/progress state. PAP-MoE uses this to distinguish
    # visually similar pre-grasp and post-grasp observations without exposing
    # privileged labels at deployment.
    gripper_head_semantic_context_dim: int = 0
    # Prevent a causal shortcut in which the classifier waits until the
    # measured gripper is already closed before predicting a close command.
    # Existing checkpoints keep the historical behavior unless explicitly
    # enabled during training.
    gripper_head_mask_state_gripper: bool = False
    gripper_head_training_horizon: int | None = None
    gripper_head_loss_weight: float = 1.0
    gripper_head_open_loss_weight: float = 2.0
    gripper_head_probability_threshold: float = 0.5
    # Binary grippers should not change state speculatively inside one RTC
    # execution block.  When enabled, the decision at the replan boundary is
    # held for the complete predicted chunk; the next observation/replan may
    # then update it.  Disabled by default for checkpoint compatibility.
    gripper_head_hold_first_action: bool = False
    gripper_head_open_normalized_value: float = -1.0
    gripper_head_closed_normalized_value: float = 1.0

    # A conservative release-only override for a binary gripper. Unlike the
    # complete deterministic head above, this branch leaves the action-flow
    # prediction untouched unless the current observation is classified as a
    # release/retract phase. It consumes only observations available online:
    # the VLM prefix, calibrated multi-rate force, and proprioceptive history.
    use_release_gripper_override: bool = False
    train_release_head_only: bool = False
    release_head_hidden_dim: int = 256
    release_head_positive_weight: float = 2.0
    release_head_probability_threshold: float = 0.8
    release_head_open_normalized_value: float = -1.0
    release_open_subtask_names: tuple[str, ...] = ("release the peg after verification",)
    release_force_dim: int = 6
    release_state_dim: int = 7
    release_fast_force_window_size: int = 64
    release_slow_force_window_size: int = 50
    release_state_history_size: int = 10

    # Deterministic residual servo for the six arm joints. The stochastic
    # flow remains responsible for the gripper, while this head maps the
    # contextualized visual prefix plus online force/proprioceptive history
    # to a complete normalized arm chunk. Predicting residuals from the
    # current state makes the small head explicitly state-feedback driven.
    use_deterministic_arm_head: bool = False
    train_arm_head_only: bool = False
    arm_head_hidden_dim: int = 512
    arm_head_action_dim: int = 6
    arm_head_loss_weight: float = 1.0
    # Limit supervision to the prefix that is actually executed before RTC
    # replans. None preserves the historical full-chunk objective.
    arm_head_training_horizon: int | None = None
    arm_head_state_noise_std: float = 0.05
    arm_head_history_noise_std: float = 0.0
    arm_head_force_noise_std: float = 0.0
    arm_head_residual_scale: float = 1.0
    # Optional policy-specific semantic context (for example PAP-MoE
    # subtask/progress probabilities). Zero preserves the plain Pi0.5 head.
    arm_head_semantic_context_dim: int = 0
    arm_head_train_semantic_only: bool = False
    # Quantile normalization is affine and intentionally permits values
    # outside [-1, 1] for the distribution tails.  Keep the historical clip
    # available for old deployments, but allow it to be disabled explicitly.
    arm_head_normalized_output_clip: float | None = 1.0
    arm_replay_adapter_num_prototypes: int = 0
    arm_replay_adapter_feature_dim: int = 70
    arm_replay_adapter_bandwidth: float = 0.5
    arm_replay_adapter_strength: float = 1.0
    # Controlled D1 ablation. "vs" keeps images/current joint state while
    # zeroing FT300 and temporal joint history; "vsf" exposes both. The same
    # model graph is retained so modality is the only within-family change.
    controlled_ablation_sensor_mode: str = "vsf"

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        if self.controlled_ablation_sensor_mode not in {"vs", "vsf"}:
            raise ValueError("controlled_ablation_sensor_mode must be 'vs' or 'vsf'")

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.gripper_action_index is not None and not (
            0 <= self.gripper_action_index < self.max_action_dim
        ):
            raise ValueError(
                "gripper_action_index must be within the padded action vector, "
                f"got {self.gripper_action_index} for max_action_dim={self.max_action_dim}"
            )
        if self.gripper_loss_weight <= 0:
            raise ValueError("gripper_loss_weight must be positive")
        if self.gripper_open_loss_weight <= 0:
            raise ValueError("gripper_open_loss_weight must be positive")
        if not math.isfinite(self.gripper_open_threshold_normalized):
            raise ValueError("gripper_open_threshold_normalized must be finite")
        if self.action_prefix_loss_horizon is not None and not (
            1 <= self.action_prefix_loss_horizon <= self.chunk_size
        ):
            raise ValueError("action_prefix_loss_horizon must be within the action chunk")
        if not math.isfinite(self.action_prefix_loss_weight) or self.action_prefix_loss_weight <= 0:
            raise ValueError("action_prefix_loss_weight must be finite and positive")
        if self.train_gripper_head_only and not self.use_deterministic_gripper_head:
            raise ValueError("train_gripper_head_only requires use_deterministic_gripper_head")
        if self.use_deterministic_gripper_head and self.gripper_action_index is None:
            raise ValueError("use_deterministic_gripper_head requires gripper_action_index")
        if self.gripper_head_hidden_dim < 1:
            raise ValueError("gripper_head_hidden_dim must be positive")
        if self.gripper_head_pooling not in {"last", "mean"}:
            raise ValueError("gripper_head_pooling must be 'last' or 'mean'")
        if self.gripper_head_training_horizon is not None and not (
            1 <= self.gripper_head_training_horizon <= self.chunk_size
        ):
            raise ValueError("gripper_head_training_horizon must be within the action chunk")
        if self.gripper_head_loss_weight <= 0:
            raise ValueError("gripper_head_loss_weight must be positive")
        if self.gripper_head_open_loss_weight <= 0:
            raise ValueError("gripper_head_open_loss_weight must be positive")
        if not 0 < self.gripper_head_probability_threshold < 1:
            raise ValueError("gripper_head_probability_threshold must be in (0, 1)")
        if not math.isfinite(self.gripper_head_open_normalized_value):
            raise ValueError("gripper_head_open_normalized_value must be finite")
        if not math.isfinite(self.gripper_head_closed_normalized_value):
            raise ValueError("gripper_head_closed_normalized_value must be finite")
        if self.train_release_head_only and not self.use_release_gripper_override:
            raise ValueError("train_release_head_only requires use_release_gripper_override")
        if self.use_release_gripper_override and self.gripper_action_index is None:
            raise ValueError("use_release_gripper_override requires gripper_action_index")
        if (
            self.use_deterministic_gripper_head
            and self.use_release_gripper_override
            and not self.gripper_head_use_sensor_context
        ):
            raise ValueError(
                "legacy deterministic gripper head and release-only override are mutually exclusive; "
                "enable gripper_head_use_sensor_context to compose grasp closure with conservative release"
            )
        if self.gripper_head_mask_state_gripper and (
            self.gripper_action_index is None
            or self.gripper_action_index >= self.release_state_dim
        ):
            raise ValueError(
                "gripper_head_mask_state_gripper requires a gripper_action_index "
                "within release_state_dim"
            )
        if self.train_gripper_head_only and self.train_release_head_only:
            raise ValueError("only one gripper-head training mode may be enabled")
        if self.train_arm_head_only and not self.use_deterministic_arm_head:
            raise ValueError("train_arm_head_only requires use_deterministic_arm_head")
        if (
            sum(
                (
                    self.train_gripper_head_only,
                    self.train_release_head_only,
                    self.train_arm_head_only,
                )
            )
            > 1
        ):
            raise ValueError("only one auxiliary-head training mode may be enabled")
        if self.arm_head_hidden_dim < 1:
            raise ValueError("arm_head_hidden_dim must be positive")
        if not 1 <= self.arm_head_action_dim <= self.max_state_dim:
            raise ValueError("arm_head_action_dim must be within the padded state width")
        if self.arm_head_loss_weight <= 0:
            raise ValueError("arm_head_loss_weight must be positive")
        if self.arm_head_semantic_context_dim < 0:
            raise ValueError("arm_head_semantic_context_dim cannot be negative")
        if self.gripper_head_semantic_context_dim < 0:
            raise ValueError("gripper_head_semantic_context_dim cannot be negative")
        if self.arm_head_train_semantic_only and self.arm_head_semantic_context_dim == 0:
            raise ValueError(
                "arm_head_train_semantic_only requires arm_head_semantic_context_dim"
            )
        if self.arm_head_training_horizon is not None and not (
            1 <= self.arm_head_training_horizon <= self.chunk_size
        ):
            raise ValueError("arm_head_training_horizon must be within the action chunk")
        if not math.isfinite(self.arm_head_state_noise_std) or self.arm_head_state_noise_std < 0:
            raise ValueError("arm_head_state_noise_std must be finite and non-negative")
        if not math.isfinite(self.arm_head_history_noise_std) or self.arm_head_history_noise_std < 0:
            raise ValueError("arm_head_history_noise_std must be finite and non-negative")
        if not math.isfinite(self.arm_head_force_noise_std) or self.arm_head_force_noise_std < 0:
            raise ValueError("arm_head_force_noise_std must be finite and non-negative")
        if self.arm_replay_adapter_num_prototypes < 0:
            raise ValueError("arm_replay_adapter_num_prototypes cannot be negative")
        if self.arm_replay_adapter_feature_dim != self.arm_head_action_dim + 64:
            raise ValueError(
                "arm_replay_adapter_feature_dim must equal arm_head_action_dim + 64 sensor features"
            )
        if not math.isfinite(self.arm_replay_adapter_bandwidth) or self.arm_replay_adapter_bandwidth <= 0:
            raise ValueError("arm_replay_adapter_bandwidth must be finite and positive")
        if not math.isfinite(self.arm_replay_adapter_strength) or not 0 <= self.arm_replay_adapter_strength <= 1:
            raise ValueError("arm_replay_adapter_strength must be finite and in [0, 1]")
        if not math.isfinite(self.arm_head_residual_scale) or not 0 < self.arm_head_residual_scale <= 1:
            raise ValueError("arm_head_residual_scale must be in (0, 1]")
        if self.arm_head_normalized_output_clip is not None and (
            not math.isfinite(self.arm_head_normalized_output_clip)
            or self.arm_head_normalized_output_clip <= 0
        ):
            raise ValueError("arm_head_normalized_output_clip must be positive or None")
        if self.release_head_hidden_dim < 1:
            raise ValueError("release_head_hidden_dim must be positive")
        if self.release_head_positive_weight <= 0:
            raise ValueError("release_head_positive_weight must be positive")
        if not 0 < self.release_head_probability_threshold < 1:
            raise ValueError("release_head_probability_threshold must be in (0, 1)")
        if not math.isfinite(self.release_head_open_normalized_value):
            raise ValueError("release_head_open_normalized_value must be finite")
        if not self.release_open_subtask_names:
            raise ValueError("release_open_subtask_names must not be empty")
        for name, value in (
            ("release_force_dim", self.release_force_dim),
            ("release_state_dim", self.release_state_dim),
            ("release_fast_force_window_size", self.release_fast_force_window_size),
            ("release_slow_force_window_size", self.release_slow_force_window_size),
            ("release_state_history_size", self.release_state_history_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        if self.use_deterministic_arm_head:
            if ACTION not in self.output_features:
                raise ValueError("Deterministic arm head requires an action feature")
            action_dim = self.output_features[ACTION].shape[0]
            if self.arm_head_action_dim > action_dim:
                raise ValueError(
                    "arm_head_action_dim exceeds the real action width: "
                    f"{self.arm_head_action_dim} > {action_dim}"
                )
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if (
            self.use_release_gripper_override
            or self.use_deterministic_arm_head
            or (self.use_deterministic_gripper_head and self.gripper_head_use_sensor_context)
        ):
            self.input_features.setdefault(
                OBS_FORCE,
                PolicyFeature(type=FeatureType.STATE, shape=(self.release_force_dim,)),
            )
            self.input_features.setdefault(
                OBS_FORCE_FAST,
                PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(
                        self.release_fast_force_window_size,
                        self.release_force_dim,
                    ),
                ),
            )
            self.input_features.setdefault(
                OBS_FORCE_SLOW,
                PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(
                        self.release_slow_force_window_size,
                        self.release_force_dim,
                    ),
                ),
            )
            self.input_features.setdefault(
                OBS_STATE_HISTORY,
                PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(self.release_state_history_size, self.release_state_dim),
                ),
            )

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

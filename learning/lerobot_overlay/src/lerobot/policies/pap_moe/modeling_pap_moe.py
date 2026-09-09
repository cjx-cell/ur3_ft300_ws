#!/usr/bin/env python
"""
PAP-MoE Policy — Physics-Aware Perceptual Mixture of Experts for PI0.5.
"""

import copy
import logging
from typing import Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import require_package

from ..pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    PI05Pytorch,
    _clip_normalized_arm_chunk,
    _compute_weighted_action_loss,
    make_att_2d_masks,
)
from .configuration_pap_moe import (
    OBS_FORCE,
    OBS_FORCE_FAST,
    OBS_FORCE_SLOW,
    OBS_MODALITY_VALIDITY,
    OBS_PHYSICS_GATE_TARGET,
    OBS_STATE_HISTORY,
    OBS_VISUAL_QUALITY,
    PAPMoEConfig,
)
from .pap_moe_modules import (
    ActionTokenConditioner,
    FactorizedPhysicsGate,
    FactorizedRouteForecaster,
    HeterogeneousPhysicsExperts,
    LegacyActionConditionedPhysicsGate,
    LegacyPhysicsGate,
    LegacyTokenPhysicsExperts,
    MultiScaleForceEncoder,
    PhysicsExpertAuxiliaryHeads,
    PhysicsGate,
    ProprioHistoryEncoder,
    SoftExpertRouter,
    VisualHistoryEncoder,
    compute_load_balancing_loss,
    soft_target_cross_entropy,
)


_FACTORIZED_GATE_OUTPUT_SUFFIXES = (
    "physics_gate.context_head.3.weight",
    "physics_gate.context_head.3.bias",
    "physics_gate.fast_force_head.3.weight",
    "physics_gate.fast_force_head.3.bias",
)


def _filter_legacy_gate_outputs_for_factorized_gate(
    state_dict: dict[str, Tensor], physics_gate_architecture: str
) -> tuple[dict[str, Tensor], list[str]]:
    """Remove only legacy 4-way Gate heads when migrating to a 3-factor Gate."""
    if physics_gate_architecture not in {
        "factorized_bcm_v1",
        "physics_gate_v2",
        "temporal_bcm_v2",
    }:
        return state_dict, []
    skipped = [
        key
        for key, value in state_dict.items()
        if any(key.endswith(suffix) for suffix in _FACTORIZED_GATE_OUTPUT_SUFFIXES)
        and value.ndim > 0
        and value.shape[0] == 4
    ]
    if not skipped:
        return state_dict, []
    return {key: value for key, value in state_dict.items() if key not in skipped}, skipped


def _masked_stage_losses(
    batch: dict[str, Tensor],
    stage_logits: Tensor,
    stage_probs: Tensor,
    stage_targets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Exclude rows whose FT/history target could not be reconstructed."""
    validity = batch.get(OBS_MODALITY_VALIDITY)
    if validity is None:
        return (
            soft_target_cross_entropy(stage_logits, stage_targets),
            compute_load_balancing_loss(stage_probs, stage_targets),
        )
    validity = validity.to(device=stage_logits.device)
    if validity.ndim != 2 or validity.shape[1] < 7:
        raise ValueError(
            f"{OBS_MODALITY_VALIDITY} must be [B,7], got {tuple(validity.shape)}"
        )
    mask = validity[:, 3:7].amin(dim=1) > 0.5
    if not bool(mask.any()):
        zero = stage_logits.sum() * 0.0
        return zero, zero
    per_sample = soft_target_cross_entropy(stage_logits, stage_targets, reduction="none")
    stage_loss = per_sample[mask].mean()
    balancing_loss = compute_load_balancing_loss(stage_probs[mask], stage_targets[mask])
    return stage_loss, balancing_loss


def _masked_factorized_gate_loss(
    batch: dict[str, Tensor],
    factor_probs: Tensor,
    expert_targets: Tensor,
) -> Tensor:
    """Supervise b/c/m factors reconstructed exactly from four-expert targets."""
    factor_targets = FactorizedPhysicsGate.expert_probs_to_factors(expert_targets).to(
        device=factor_probs.device, dtype=factor_probs.dtype
    )
    per_factor = F.binary_cross_entropy(
        factor_probs.float().clamp(1e-6, 1.0 - 1e-6),
        factor_targets.float(),
        reduction="none",
    )
    # Mobility/compliance is undefined without contact. Do not force arbitrary
    # E3/E4 behavior on free-space rows.
    weights = torch.ones_like(per_factor)
    weights[..., 2] = factor_targets[..., 1]
    per_sample = (per_factor * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    validity = batch.get(OBS_MODALITY_VALIDITY)
    if validity is None:
        sample_mask = torch.ones(
            factor_probs.shape[0], device=factor_probs.device, dtype=torch.bool
        )
    else:
        validity = validity.to(device=factor_probs.device)
        sample_mask = validity[:, 3:7].amin(dim=1) > 0.5
    if per_sample.ndim == 1:
        mask = sample_mask
    else:
        mask = sample_mask.unsqueeze(1).expand_as(per_sample).clone()
        route_pad = batch.get(f"{OBS_PHYSICS_GATE_TARGET}_is_pad")
        if route_pad is not None:
            mask &= ~route_pad.to(device=factor_probs.device, dtype=torch.bool)
    if not bool(mask.any()):
        return factor_probs.sum() * 0.0
    return per_sample[mask].mean()

# ══════════════════════════════════════════════════════════════════════════════
# LoRA wrapper
# ══════════════════════════════════════════════════════════════════════════════


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_f, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        for p in base.parameters():
            p.requires_grad = False

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        base_out = self.base(x)
        A = self.lora_A.to(dtype=x.dtype)
        B = self.lora_B.to(dtype=x.dtype)
        lora_out = self.dropout(x) @ A @ B
        return base_out + lora_out * (self.alpha / self.rank)


# ══════════════════════════════════════════════════════════════════════════════
# PAP-MoE Model
# ══════════════════════════════════════════════════════════════════════════════


class PAPMoEPi05Model(PI05Pytorch):
    """PAP-MoE Model based on PI0.5 backbone."""

    def __init__(self, config: PAPMoEConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

        # Freeze all base PI0.5 parameters first
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze projections (always trainable)
        for mod in [self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out]:
            for param in mod.parameters():
                param.requires_grad = True

        D = config.pap_moe_feature_dim

        self.force_encoder = MultiScaleForceEncoder(config.force_dim, config.pap_moe_hidden_dim, D)
        self.proprio_encoder = ProprioHistoryEncoder(config.robot_state_dim, config.pap_moe_hidden_dim, D)
        self.visual_quality_encoder = nn.Sequential(
            nn.LayerNorm(config.visual_quality_dim),
            nn.Linear(config.visual_quality_dim, config.pap_moe_hidden_dim),
            nn.GELU(),
            nn.Linear(config.pap_moe_hidden_dim, D),
        )
        self.visual_history_encoder = (
            VisualHistoryEncoder(
                D, config.visual_memory_hidden_dim,
                mask_invalid_cameras=config.mask_invalid_history_cameras,
            )
            if config.use_visual_memory
            else None
        )

        # PhysicsGate classifier over raw VLM prefix + force tokens
        if config.physics_gate_architecture == "physics_gate_v2":
            self.physics_gate = PhysicsGate(
                d_model=D,
                num_experts=config.num_experts,
                nhead=8,
                hidden_dim=config.pap_moe_hidden_dim,
                horizon=config.chunk_size,
            )
        elif config.physics_gate_architecture == "temporal_bcm_v2":
            self.physics_gate = LegacyActionConditionedPhysicsGate(
                d_model=D,
                num_experts=config.num_experts,
                nhead=8,
                hidden_dim=config.pap_moe_hidden_dim,
                action_dim=self.action_in_proj.out_features,
                horizon=config.chunk_size,
            )
        else:
            gate_class = (
                FactorizedPhysicsGate
                if config.physics_gate_architecture == "factorized_bcm_v1"
                else LegacyPhysicsGate
            )
            self.physics_gate = gate_class(
                d_model=D,
                num_experts=config.num_experts,
                nhead=8,
                hidden_dim=config.pap_moe_hidden_dim,
            )
        # Deprecated checkpoint-compatibility path. physics_gate_v2 predicts
        # the full route sequence directly and never creates this module.
        self.route_forecaster = (
            FactorizedRouteForecaster(
                d_model=D,
                hidden_dim=config.pap_moe_hidden_dim,
                horizon=config.chunk_size,
                nhead=8,
            )
            if config.action_step_routing
            and config.physics_gate_architecture == "factorized_bcm_v1"
            else None
        )

        if config.physics_expert_architecture == "heterogeneous_v2":
            self.expert_library = HeterogeneousPhysicsExperts(
                d_model=D,
                hidden_dim=config.pap_moe_hidden_dim,
                state_dim=config.robot_state_dim,
                force_dim=config.force_dim,
                quality_dim=config.visual_quality_dim,
                e1_fusion_normalization=config.e1_fusion_normalization,
                e1_fusion_mode=config.e1_fusion_mode,
            )
            self.expert_auxiliary_heads = PhysicsExpertAuxiliaryHeads(
                d_model=D,
                hidden_dim=config.pap_moe_hidden_dim,
                state_dim=config.robot_state_dim,
                force_dim=config.force_dim,
                quality_dim=config.visual_quality_dim,
            )
        else:
            self.expert_library = LegacyTokenPhysicsExperts(
                d_model=D, hidden_dim=config.pap_moe_hidden_dim
            )
            self.expert_auxiliary_heads = None
        self.soft_router = SoftExpertRouter()

        self._expert_width = self.action_in_proj.out_features
        self.action_conditioner = ActionTokenConditioner(
            condition_dim=D,
            action_dim=self._expert_width,
            nhead=8,
            zero_init_output=not config.bounded_action_conditioning,
            route_attention_prior=config.action_conditioning_route_prior,
        )
        if config.bounded_action_conditioning:
            self.expert_conditioning_scales = nn.Parameter(
                torch.full(
                    (config.num_experts,),
                    float(config.expert_conditioning_scale_init),
                )
            )

        # Inject LoRA
        lora_rank = config.lora_rank
        lora_alpha = config.lora_alpha
        lora_dropout = config.lora_dropout

        for model_layers in [
            self.paligemma_with_expert.paligemma.model.language_model.layers,
            self.paligemma_with_expert.gemma_expert.model.layers,
        ]:
            for layer in model_layers:
                attn = layer.self_attn
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    setattr(
                        attn,
                        proj_name,
                        LoRALinear(getattr(attn, proj_name), lora_rank, lora_alpha, lora_dropout),
                    )
                if hasattr(layer, "mlp"):
                    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                        if hasattr(layer.mlp, proj_name):
                            setattr(
                                layer.mlp,
                                proj_name,
                                LoRALinear(
                                    getattr(layer.mlp, proj_name), lora_rank, lora_alpha, lora_dropout
                                ),
                            )

        self.to(config.device)

        # ── Staged training: per-stage freeze/unfreeze ──
        self._apply_training_stage(config)

    def _apply_training_stage(self, config: PAPMoEConfig):
        """Apply per-stage parameter freezing based on config flags."""
        for parameter in self.parameters():
            parameter.requires_grad = False

        pap_moe_modules = [
            self.force_encoder,
            self.proprio_encoder,
            self.visual_quality_encoder,
            self.physics_gate,
            self.expert_library,
            self.action_conditioner,
        ]
        if self.visual_history_encoder is not None:
            pap_moe_modules.append(self.visual_history_encoder)
        if self.expert_auxiliary_heads is not None:
            pap_moe_modules.append(self.expert_auxiliary_heads)
        route_forecaster = getattr(self, "route_forecaster", None)
        if route_forecaster is not None:
            pap_moe_modules.append(route_forecaster)

        if getattr(config, "train_gripper_head_only", False):
            if self.gripper_head is None:
                raise ValueError("train_gripper_head_only requires use_deterministic_gripper_head")
            for parameter in self.gripper_head.parameters():
                parameter.requires_grad = True
        elif getattr(config, "train_arm_head_only", False):
            if self.arm_head is None:
                raise ValueError("train_arm_head_only requires use_deterministic_arm_head")
            if config.arm_head_train_semantic_only:
                if self.arm_head.semantic is None:
                    raise ValueError(
                        "arm_head_train_semantic_only requires a semantic arm branch"
                    )
                for parameter in self.arm_head.semantic.parameters():
                    parameter.requires_grad = True
            else:
                for parameter in self.arm_head.parameters():
                    parameter.requires_grad = True
        elif config.train_expert_action_joint:
            # Primary PAP adaptation: the physical experts are conditions for
            # the Pi0.5 action expert. Train both sides of the interface under
            # ground-truth soft routing while the original PaliGemma VLM stays
            # frozen. This matches Pi0.5's "full action expert" finetuning
            # contract instead of forcing a frozen action model to interpret
            # an unseen conditional-token distribution.
            for mod in [
                self.expert_library,
                self.expert_auxiliary_heads,
                self.visual_history_encoder,
                self.action_conditioner,
                self.paligemma_with_expert.gemma_expert.model,
                self.action_in_proj,
                self.action_out_proj,
                self.time_mlp_in,
                self.time_mlp_out,
            ]:
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True
            if hasattr(self, "expert_conditioning_scales"):
                self.expert_conditioning_scales.requires_grad = True
            # The causal-LM vocabulary head is not used by the action flow.
            for p in self.paligemma_with_expert.gemma_expert.lm_head.parameters():
                p.requires_grad = False
        elif config.train_physicsgate_action_joint:
            # Final prediction-aligned joint adaptation. The direct
            # PhysicsGate supplies the per-action-step routes used during
            # training, so the complete conditional action policy learns the
            # same route distribution it receives in closed loop.
            for mod in [
                self.force_encoder,
                self.proprio_encoder,
                self.visual_quality_encoder,
                self.physics_gate,
                self.expert_library,
                self.expert_auxiliary_heads,
                self.visual_history_encoder,
                self.action_conditioner,
                self.paligemma_with_expert.gemma_expert.model,
                self.action_in_proj,
                self.action_out_proj,
                self.time_mlp_in,
                self.time_mlp_out,
            ]:
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True
            if hasattr(self, "expert_conditioning_scales"):
                self.expert_conditioning_scales.requires_grad = True
            for p in self.paligemma_with_expert.gemma_expert.lm_head.parameters():
                p.requires_grad = False
        elif config.train_expert_only:
            # Legacy diagnostic mode retained for old checkpoints. It trains
            # only the added PAP modules and is not the primary PAP method.
            for mod in [
                self.expert_library,
                self.expert_auxiliary_heads,
                self.visual_history_encoder,
                self.action_conditioner,
            ]:
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True
            if hasattr(self, "expert_conditioning_scales"):
                self.expert_conditioning_scales.requires_grad = True
        elif config.train_physicsgate_only:
            # Stage 2: learn multi-scale representations and initial routing.
            for mod in [
                self.force_encoder,
                self.proprio_encoder,
                self.visual_quality_encoder,
                self.physics_gate,
            ]:
                for p in mod.parameters():
                    p.requires_grad = True
            if route_forecaster is not None:
                for p in route_forecaster.parameters():
                    p.requires_grad = True
        elif config.train_gate_calibration_only:
            # Stage 3: experts stay fixed while the complete routing stack learns.
            for mod in [
                self.force_encoder,
                self.proprio_encoder,
                self.visual_quality_encoder,
                self.physics_gate,
            ]:
                for p in mod.parameters():
                    p.requires_grad = True
            if route_forecaster is not None:
                for p in route_forecaster.parameters():
                    p.requires_grad = True
        elif getattr(config, "train_route_forecaster_only", False):
            if route_forecaster is None:
                raise ValueError("train_route_forecaster_only requires action_step_routing")
            for p in route_forecaster.parameters():
                p.requires_grad = True
        elif config.train_conditioner_only:
            # Stage 4: low-LR final alignment without changing perception.
            for p in self.action_conditioner.parameters():
                p.requires_grad = True
            if hasattr(self, "expert_conditioning_scales"):
                self.expert_conditioning_scales.requires_grad = True
        elif config.train_action_adapter_only:
            # Adapt the actual flow action generator without disturbing the
            # visual-language backbone or any learned PAP routing/perception.
            for mod in [
                self.action_in_proj,
                self.action_out_proj,
                self.time_mlp_in,
                self.time_mlp_out,
            ]:
                for p in mod.parameters():
                    p.requires_grad = True
            if getattr(config, "action_adapter_train_lora", True):
                for name, parameter in self.named_parameters():
                    if "gemma_expert" in name and ("lora_A" in name or "lora_B" in name):
                        parameter.requires_grad = True
        elif config.train_pap_moe_joint:
            # Optional ablation: all PAP modules plus backbone LoRA.
            for mod in pap_moe_modules:
                for p in mod.parameters():
                    p.requires_grad = True
            for mod in [
                self.action_in_proj,
                self.action_out_proj,
                self.time_mlp_in,
                self.time_mlp_out,
            ]:
                for p in mod.parameters():
                    p.requires_grad = True
            # Keep LoRA trainable on VLM + Action Expert
            for n, p in self.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
            if hasattr(self, "expert_conditioning_scales"):
                self.expert_conditioning_scales.requires_grad = True

    def _get_vlm_output(self, images, img_masks, lang_tokens, lang_masks, *, use_cache=True):
        """Run PaliGemma prefix forward pass to get E_VL hidden states."""
        language_model = self.paligemma_with_expert.paligemma.model.language_model
        checkpointing_was_enabled = bool(
            getattr(language_model, "gradient_checkpointing", False)
        )
        # This prefix is frozen and evaluated under no_grad. Keeping gradient
        # checkpointing enabled here has no memory benefit, but Transformers
        # then silently disables use_cache and returns no prefix KV state.
        if use_cache and checkpointing_was_enabled:
            language_model.gradient_checkpointing = False
        try:
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks
                )
                prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)
                language_model.config._attn_implementation = "eager"
                (prefix_out, _), past_kv = self.paligemma_with_expert.forward(
                    attention_mask=prefix_att_4d,
                    position_ids=prefix_pos,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=use_cache,
                )
        finally:
            if use_cache and checkpointing_was_enabled:
                language_model.gradient_checkpointing = True
        return prefix_out, past_kv, prefix_pad_masks

    def _encode_physics_inputs(
        self,
        force: Tensor,
        state: Tensor,
        force_fast: Tensor | None = None,
        force_slow: Tensor | None = None,
        state_history: Tensor | None = None,
        visual_quality: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        target_dtype = next(self.force_encoder.parameters()).dtype
        force_tokens = self.force_encoder(
            force.to(dtype=target_dtype),
            None if force_fast is None else force_fast.to(dtype=target_dtype),
            None if force_slow is None else force_slow.to(dtype=target_dtype),
        )
        proprio_token = self.proprio_encoder(
            state.to(dtype=target_dtype),
            None if state_history is None else state_history.to(dtype=target_dtype),
        )
        if visual_quality is None:
            visual_quality = torch.zeros(
                force.shape[0],
                self.config.visual_quality_dim,
                device=force.device,
                dtype=target_dtype,
            )
        quality_token = self.visual_quality_encoder(visual_quality.to(dtype=target_dtype))
        return force_tokens, proprio_token, quality_token

    def _encode_visual_memory(
        self,
        visual_history: list[Tensor] | None,
        visual_history_padding: list[Tensor] | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.visual_history_encoder is None:
            return None, None
        if not visual_history:
            raise ValueError("PAP-MoE visual memory is enabled but no history was provided")
        target_dtype = next(self.visual_history_encoder.parameters()).dtype
        histories = [history.to(dtype=target_dtype) for history in visual_history]
        return self.visual_history_encoder(histories, visual_history_padding)

    def _pool_physics_prefix(self, features: Tensor, valid_mask: Tensor | None) -> Tensor:
        if not getattr(self.config, "mask_invalid_prefix_tokens", False):
            return features.mean(dim=1)
        if valid_mask is None or valid_mask.shape != features.shape[:2]:
            raise ValueError("Masked PAP prefix pooling requires a matching [B,N] valid mask")
        valid = valid_mask.to(device=features.device, dtype=torch.bool).unsqueeze(-1)
        # Accumulate in FP32 and explicitly handle all-padding observations.
        pooled = features.float().masked_fill(~valid, 0.0).sum(dim=1)
        return (pooled / valid.sum(dim=1).clamp_min(1)).to(features.dtype)

    def _forward_pap_moe(
        self,
        force_tokens: dict[str, Tensor],
        E_VL: Tensor,
        proprio_token: Tensor,
        quality_token: Tensor,
        stage_labels_soft: Tensor | None = None,
        route_sequence_labels_soft: Tensor | None = None,
        tf_prob: float = 0.0,
        stage_override: Tensor | None = None,
        force: Tensor | None = None,
        force_fast: Tensor | None = None,
        force_slow: Tensor | None = None,
        state: Tensor | None = None,
        state_history: Tensor | None = None,
        visual_quality: Tensor | None = None,
        expert_mask: Tensor | None = None,
        route_action_tokens: Tensor | None = None,
        run_experts: bool = True,
        visual_memory: Tensor | None = None,
        memory_age: Tensor | None = None,
        prefix_valid_mask: Tensor | None = None,
    ) -> dict:
        target_dtype = next(self.force_encoder.parameters()).dtype
        E_VL = E_VL.to(dtype=target_dtype)
        visual_token = self._pool_physics_prefix(E_VL, prefix_valid_mask)
        context_tokens = torch.stack(
            [
                visual_token,
                proprio_token,
                force_tokens["current"],
                force_tokens["fast"],
                force_tokens["slow"],
                quality_token,
            ],
            dim=1,
        )
        gate_architecture = getattr(
            self.config, "physics_gate_architecture", "legacy_softmax"
        )
        if gate_architecture == "temporal_bcm_v2":
            gate_probs, gate_logits, _ = self.physics_gate(
                context_tokens,
                force_tokens["fast"],
                action_tokens=route_action_tokens,
            )
        else:
            gate_probs, gate_logits, _ = self.physics_gate(
                context_tokens, force_tokens["fast"]
            )

        predicted_route_sequence = None
        predicted_factor_sequence = None
        route_forecaster = getattr(self, "route_forecaster", None)
        if route_forecaster is not None:
            current_factors = getattr(self.physics_gate, "last_factor_probs", None)
            if current_factors is None:
                raise RuntimeError("stepwise routing requires a factorized PhysicsGate")
            predicted_factor_sequence, predicted_route_sequence = route_forecaster(
                context_tokens, current_factors
            )
        elif gate_architecture in {"physics_gate_v2", "temporal_bcm_v2"}:
            predicted_factor_sequence = getattr(
                self.physics_gate, "last_predicted_factor_sequence", None
            )
            predicted_route_sequence = getattr(
                self.physics_gate, "last_predicted_route_sequence", None
            )

        # Continuous teacher forcing blends dataset soft targets with
        # PhysicsGate probabilities.
        if stage_override is not None and stage_override.ndim <= 2:
            if stage_override.dtype in (torch.long, torch.int32, torch.int64):
                routing_probs = F.one_hot(stage_override, num_classes=self.config.num_experts).to(
                    dtype=gate_probs.dtype
                )
            else:
                routing_probs = stage_override.to(dtype=gate_probs.dtype)
        elif stage_labels_soft is not None and self.training and tf_prob > 0.0:
            soft_gt = stage_labels_soft.to(dtype=gate_probs.dtype)
            routing_probs = tf_prob * soft_gt + (1.0 - tf_prob) * gate_probs
        else:
            routing_probs = gate_probs

        conditioning_routing_probs = routing_probs
        if stage_override is not None and stage_override.ndim == 3:
            conditioning_routing_probs = stage_override.to(dtype=gate_probs.dtype)
        elif predicted_route_sequence is not None:
            if route_sequence_labels_soft is not None and self.training and tf_prob > 0.0:
                route_gt = route_sequence_labels_soft.to(dtype=gate_probs.dtype)
                conditioning_routing_probs = (
                    tf_prob * route_gt + (1.0 - tf_prob) * predicted_route_sequence
                )
            else:
                conditioning_routing_probs = predicted_route_sequence

        routing_float = conditioning_routing_probs.float().clamp_min(1e-8)
        normalized_entropy = -(
            routing_float * routing_float.log()
        ).sum(dim=-1) / torch.log(
            routing_float.new_tensor(float(self.config.num_experts))
        )
        routing_confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0)

        result = {
            "gate_probs": gate_probs,  # [B, E]
            "gate_logits": gate_logits,  # [B, E]
            "stage_probs": gate_probs,  # Backward compatibility alias
            "stage_logits": gate_logits,  # Backward compatibility alias
            "routing_probs": routing_probs,
            "routing_confidence": routing_confidence,
        }
        factor_probs = getattr(self.physics_gate, "last_factor_probs", None)
        if factor_probs is not None:
            result["factor_probs"] = factor_probs
        if predicted_route_sequence is not None:
            result["predicted_route_sequence"] = predicted_route_sequence
            result["predicted_factor_sequence"] = predicted_factor_sequence
        if run_experts:
            if self.config.physics_expert_architecture == "heterogeneous_v2":
                if force is None or state is None:
                    raise ValueError("heterogeneous_v2 experts require raw force and state inputs")
                expert_tokens = self.expert_library(
                    visual_token,
                    state.to(dtype=target_dtype),
                    force.to(dtype=target_dtype),
                    None if force_fast is None else force_fast.to(dtype=target_dtype),
                    None if force_slow is None else force_slow.to(dtype=target_dtype),
                    None if state_history is None else state_history.to(dtype=target_dtype),
                    None if visual_quality is None else visual_quality.to(dtype=target_dtype),
                    None if visual_memory is None else visual_memory.to(dtype=target_dtype),
                    None if memory_age is None else memory_age.to(dtype=target_dtype),
                )
            else:
                expert_tokens = self.expert_library(
                    visual_token,
                    proprio_token,
                    force_tokens,
                )
            effective_routing_probs = conditioning_routing_probs
            route_jitter_std = getattr(self.config, "physical_route_jitter_std", 0.0)
            is_training = bool(getattr(self, "training", False))
            if is_training and route_jitter_std > 0:
                route_logits = effective_routing_probs.float().clamp_min(1e-8).log()
                route_logits = route_logits + torch.randn_like(route_logits) * float(route_jitter_std)
                effective_routing_probs = route_logits.softmax(dim=-1).to(
                    dtype=routing_probs.dtype
                )
            condition_keep = None
            condition_dropout = getattr(
                self.config, "physical_condition_dropout_probability", 0.0
            )
            if is_training and condition_dropout > 0:
                keep_shape = (
                    (routing_probs.shape[0], 1, 1)
                    if effective_routing_probs.ndim == 3
                    else (routing_probs.shape[0], 1)
                )
                condition_keep = (
                    torch.rand(
                        *keep_shape,
                        device=routing_probs.device,
                    )
                    >= condition_dropout
                ).to(dtype=routing_probs.dtype)
                effective_routing_probs = effective_routing_probs * condition_keep
            if expert_mask is not None:
                mask = expert_mask.to(device=expert_tokens.device, dtype=expert_tokens.dtype)
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)
                if mask.shape[-1] != self.config.num_experts:
                    raise ValueError(
                        f"expert_mask must end in {self.config.num_experts}, got {tuple(mask.shape)}"
                    )
                if self.config.conditioner_routing_mode == "pre_norm_legacy":
                    expert_tokens = expert_tokens * mask.unsqueeze(-1)
                else:
                    route_mask = mask.unsqueeze(1) if effective_routing_probs.ndim == 3 else mask
                    effective_routing_probs = effective_routing_probs * route_mask
            current_effective_routing = (
                effective_routing_probs[:, 0]
                if effective_routing_probs.ndim == 3
                else effective_routing_probs
            )
            weighted_tokens, fused_token = self.soft_router(
                expert_tokens,
                current_effective_routing,
            )
            result["expert_tokens"] = expert_tokens
            result["fused_expert_token"] = fused_token
            result["conditioning_routing_probs"] = effective_routing_probs
            if condition_keep is not None:
                result["condition_keep"] = condition_keep.squeeze(-1)
            # The action flow receives only the four routed physical experts.
            if self.config.conditioner_routing_mode == "post_projection_v2":
                result["conditioning_tokens"] = expert_tokens
                result["conditioning_weights"] = effective_routing_probs
            else:
                result["conditioning_tokens"] = weighted_tokens
        return result

    def _apply_physical_conditioning(
        self,
        action_tokens: Tensor,
        pap_result: dict,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Fuse routed physical representations into action hidden tokens.

        The returned difference is reported only as a numerical diagnostic;
        it is not an independently generated correction action.
        """
        token_scales = None
        residual_max_norm = None
        if self.config.bounded_action_conditioning:
            token_scales = torch.tanh(self.expert_conditioning_scales)
            nominal_multiplier = self.config.nominal_expert_conditioning_multiplier
            if nominal_multiplier != 1.0:
                expert_multipliers = token_scales.new_ones(token_scales.shape)
                expert_multipliers[0] = nominal_multiplier
                token_scales = token_scales * expert_multipliers
            residual_max_norm = self.config.action_conditioning_residual_max_norm

        conditioned = self.action_conditioner(
            action_tokens,
            pap_result["conditioning_tokens"],
            pap_result.get("conditioning_weights"),
            condition_token_scales=token_scales,
            residual_max_norm=residual_max_norm,
        )
        residual = conditioned - action_tokens
        confidence = pap_result["routing_confidence"].to(
            device=residual.device, dtype=residual.dtype
        )
        if self.config.bounded_action_conditioning:
            floor = self.config.action_conditioning_confidence_floor
            confidence = floor + (1.0 - floor) * confidence
        else:
            confidence = torch.ones_like(confidence)
        applied_residual = (
            self.config.action_conditioning_scale
            * (confidence[:, None, None] if confidence.ndim == 1 else confidence.unsqueeze(-1))
            * residual
        )
        combined = action_tokens + applied_residual
        diagnostics = {
            "routing_confidence": confidence,
            "condition_residual_norm": residual.float().norm(dim=-1).mean(dim=-1),
            "applied_condition_residual_norm": applied_residual.float()
            .norm(dim=-1)
            .mean(dim=-1),
        }
        if token_scales is not None:
            diagnostics["expert_conditioning_scales"] = token_scales
        return combined, diagnostics

    def _uses_action_input_fusion(self) -> bool:
        """Whether physical representations condition Action Expert inputs."""
        return self.config.physical_fusion_architecture == "action_input_tokens_v2"

    def _fuse_physics_into_action_input(
        self,
        action_input_tokens: Tensor,
        pap_result: dict,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Shared training/inference entry point for vNext physical context."""
        if not self._uses_action_input_fusion():
            return action_input_tokens, {}
        return self._apply_physical_conditioning(action_input_tokens, pap_result)

    def _expert_motion_descriptor(self, state: Tensor, state_history: Tensor) -> Tensor:
        """Window motion in one normalization coordinate system, not radians."""
        if self.config.expert_motion_target == "history_window_delta_v2":
            return state_history[:, -1, : self.config.robot_state_dim] - state_history[
                :, 0, : self.config.robot_state_dim
            ]
        return state[..., : self.config.robot_state_dim] - state_history[
            :, 0, : self.config.robot_state_dim
        ]

    def _compute_expert_representation_loss(
        self,
        expert_tokens: Tensor,
        routing_targets: Tensor | None,
        state: Tensor,
        state_history: Tensor | None,
        force: Tensor,
        force_fast: Tensor | None,
        force_slow: Tensor | None,
        visual_quality: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Supervise auditable physical descriptors in each expert token."""
        zero = expert_tokens.sum() * 0.0
        if self.expert_auxiliary_heads is None or routing_targets is None:
            return zero, {}
        if force_fast is None or force_slow is None or state_history is None:
            raise ValueError("expert representation loss requires multiscale histories")
        if visual_quality is None:
            visual_quality = force.new_zeros(force.shape[0], self.config.visual_quality_dim)

        predictions = self.expert_auxiliary_heads(expert_tokens)
        factors = FactorizedPhysicsGate.expert_probs_to_factors(routing_targets)
        contact = factors[..., 1:2]
        mobility = factors[..., 2:3]
        state_delta = self._expert_motion_descriptor(state, state_history)
        fast_delta = force_fast[:, -1] - force_fast[:, 0]
        slow_delta = force_slow[:, -1] - force_slow[:, 0]
        targets = {
            # Descriptor supervision is semantic, not raw sensor regression.
            # Tanh keeps rare FT spikes and near-zero torque std dimensions
            # from overwhelming the action objective while preserving sign,
            # ordering, and saturation severity.
            "E1": torch.cat([state_delta.tanh(), force.tanh()], dim=-1),
            "E2": torch.cat(
                [state_delta.tanh(), slow_delta.tanh(), visual_quality.clamp(0.0, 1.0)],
                dim=-1,
            ),
            "E3": torch.cat(
                [
                    fast_delta.tanh(),
                    force_fast.std(dim=1, unbiased=False).tanh(),
                    force_fast.abs().amax(dim=1).tanh(),
                    contact,
                ],
                dim=-1,
            ),
            "E4": torch.cat(
                [
                    slow_delta.tanh(),
                    force_slow.std(dim=1, unbiased=False).tanh(),
                    force_slow.mean(dim=1).tanh(),
                    mobility,
                ],
                dim=-1,
            ),
        }

        losses = {}
        weighted_loss_sum = zero
        active_count = zero
        for expert_index, name in enumerate(("E1", "E2", "E3", "E4")):
            per_sample = F.mse_loss(
                predictions[name].float(), targets[name].detach().float(), reduction="none"
            ).mean(dim=-1)
            weight = routing_targets[:, expert_index].detach().float().clamp_min(0.0)
            mass = weight.sum()
            loss = (per_sample * weight).sum() / mass.clamp_min(1e-8)
            loss = torch.where(mass > 0, loss, zero)
            losses[name] = loss
            weighted_loss_sum = weighted_loss_sum + loss
            active_count = active_count + (mass > 0).to(dtype=loss.dtype)
        total = weighted_loss_sum / active_count.clamp_min(1.0)
        return total, losses

    @staticmethod
    def _compute_visual_memory_distillation_loss(
        visual_memory: Tensor,
        memory_age: Tensor,
        clean_visual_teacher: Tensor,
        route_labels_soft: Tensor | None,
        *,
        clean_only: bool = False,
    ) -> Tensor:
        """Teach past-frame memory to predict the current clean VLM context.

        clean_only excludes every sample with a nonzero labelled E2 weight,
        including partial glare. Legacy soft weighting does NOT guarantee a
        clean teacher. Neither mode supplies clean teachers for blind samples;
        that requires a separate clean-image encoding path.
        """
        per_sample = 1.0 - F.cosine_similarity(
            visual_memory.float(), clean_visual_teacher.detach().float(), dim=-1
        )
        weights = (memory_age.squeeze(-1) < 1.0).to(dtype=per_sample.dtype)
        if route_labels_soft is not None:
            if clean_only:
                weights = weights * (route_labels_soft[:, 1] == 0).to(weights.dtype)
            else:
                weights = weights * (1.0 - route_labels_soft[:, 1].float()).clamp(0.0, 1.0)
        elif clean_only:
            raise ValueError("Clean-only distillation requires labelled teacher validity")
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    def _preserve_unconditioned_gripper_velocity(
        self,
        baseline_velocity: Tensor,
        conditioned_velocity: Tensor,
    ) -> Tensor:
        """Optionally prevent PAP residuals from directly changing gripper flow."""
        if self.config.condition_gripper_with_physical_experts:
            return conditioned_velocity
        gripper_index = self.config.gripper_action_index
        if gripper_index is None or gripper_index >= conditioned_velocity.shape[-1]:
            return conditioned_velocity
        output = conditioned_velocity.clone()
        output[..., gripper_index] = baseline_velocity[..., gripper_index]
        return output

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        force=None,
        force_fast=None,
        force_slow=None,
        state=None,
        state_history=None,
        visual_quality=None,
        visual_history=None,
        visual_history_padding=None,
        stage_labels=None,
        stage_labels_soft=None,
        route_sequence_labels_soft=None,
        stage_override=None,
        tf_prob: float = 0.0,
        expert_mask=None,
        noise=None,
        time=None,
    ) -> dict:
        if force is None:
            raise ValueError("PAP-MoE requires force/torque input")
        if state is None:
            state = torch.zeros(
                force.shape[0],
                self.config.robot_state_dim,
                device=force.device,
                dtype=force.dtype,
            )
        force_tokens, proprio_token, quality_token = self._encode_physics_inputs(
            force, state, force_fast, force_slow, state_history, visual_quality
        )
        visual_memory, memory_age = self._encode_visual_memory(
            visual_history, visual_history_padding
        )
        route_action_tokens = None
        if self.config.physics_gate_architecture == "temporal_bcm_v2":
            # Future physical labels correspond to the demonstrated action
            # chunk. Keep route supervision from backpropagating into the
            # pretrained action projection through this conditioning input.
            route_action_tokens = self._apply_checkpoint(
                self.action_in_proj,
                actions.to(dtype=self.action_in_proj.weight.dtype),
            ).detach()

        # ── PhysicsGate supervision fast path: no experts or action flow ──
        # Skips flow matching & action expert, only produces stage classification logits
        if self.training and (
            self.config.train_physicsgate_only or self.config.train_route_forecaster_only
        ):
            # Run frozen VLM prefix, but train all lightweight gate inputs.
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                    images, img_masks, tokens, masks
                )
                prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)
                self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
                    "eager"
                )
                (E_VL, _), _ = self.paligemma_with_expert.forward(
                    attention_mask=prefix_att_4d,
                    position_ids=prefix_pos,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=False,
                )

            return self._forward_pap_moe(
                force_tokens,
                E_VL,
                proprio_token,
                quality_token,
                prefix_valid_mask=prefix_pad_masks,
                route_sequence_labels_soft=route_sequence_labels_soft,
                route_action_tokens=route_action_tokens,
                run_experts=False,
                visual_memory=visual_memory,
                memory_age=memory_age,
            )

        B, device = actions.shape[0], actions.device

        # 1. Flow matching noise & time
        dtype = self.action_in_proj.weight.dtype
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(B, device)
        x_t = (time[:, None, None] * noise + (1 - time[:, None, None]) * actions).to(dtype=dtype)
        u_t = (noise - actions).to(dtype=dtype)

        # 2. Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        # 3. Embed suffix
        suffix_embs, suffix_pad, suffix_att, adarms = self.embed_suffix(x_t, time.to(dtype=dtype))

        # 4. PAP-MoE execution
        td = next(self.force_encoder.parameters()).dtype

        if (
            self.config.train_expert_only
            or self.config.train_conditioner_only
            or self.config.train_gate_calibration_only
            or self.config.train_route_forecaster_only
            or self._uses_action_input_fusion()
        ):
            # Cached prefix forward pass
            E_VL, past_kv, prefix_pad = self._get_vlm_output(images, img_masks, tokens, masks)

            sa = self._forward_pap_moe(
                force_tokens,
                E_VL.to(dtype=td),
                proprio_token,
                quality_token,
                prefix_valid_mask=prefix_pad,
                stage_labels_soft=stage_labels_soft,
                route_sequence_labels_soft=route_sequence_labels_soft,
                tf_prob=tf_prob,
                stage_override=stage_override,
                force=force,
                force_fast=force_fast,
                force_slow=force_slow,
                state=state,
                state_history=state_history,
                visual_quality=visual_quality,
                expert_mask=expert_mask,
                route_action_tokens=route_action_tokens,
                visual_memory=visual_memory,
                memory_age=memory_age,
            )

            input_conditioning_diagnostics = {}
            if self._uses_action_input_fusion():
                suffix_embs, input_conditioning_diagnostics = self._fuse_physics_into_action_input(
                    suffix_embs, sa
                )

            s_len = suffix_pad.shape[1]
            p_len = prefix_pad.shape[1]
            prefix_2d = prefix_pad[:, None, :].expand(B, s_len, p_len)
            full_att = torch.cat([prefix_2d, make_att_2d_masks(suffix_pad, suffix_att)], dim=2)
            pos_ids = torch.sum(prefix_pad, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
            full_att_4d = self._prepare_attention_masks_4d(full_att)

            if (
                self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

            def sf(se, fa, pi, pk, ac):
                # Gemma appends suffix keys to DynamicCache in place even
                # with use_cache=False. A checkpointed function is replayed
                # during backward, so each invocation must receive a fresh
                # prefix cache or the key length grows by another 50 tokens.
                pk = copy.deepcopy(pk)
                return self.paligemma_with_expert.forward(
                    attention_mask=fa,
                    position_ids=pi,
                    past_key_values=pk,
                    inputs_embeds=[None, se],
                    use_cache=False,
                    adarms_cond=[None, ac],
                )[0][1]

            suffix_out = self._apply_checkpoint(sf, suffix_embs, full_att_4d, pos_ids, past_kv, adarms)
        else:
            # Full joint pass (VLM training)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att], dim=1)
            full_att_2d = make_att_2d_masks(pad_masks, att_masks)
            pos_ids = torch.cumsum(pad_masks, dim=1) - 1
            full_att_4d = self._prepare_attention_masks_4d(full_att_2d)

            if (
                self.paligemma_with_expert.paligemma.model.language_model.layers[
                    0
                ].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

            def ff(pe, se, fa, pi, ac):
                (prefix_out, suffix_out_raw), _ = self.paligemma_with_expert.forward(
                    attention_mask=fa,
                    position_ids=pi,
                    past_key_values=None,
                    inputs_embeds=[pe, se],
                    use_cache=False,
                    adarms_cond=[None, ac],
                )
                return prefix_out, suffix_out_raw

            prefix_out, suffix_out_raw = self._apply_checkpoint(
                ff, prefix_embs, suffix_embs, full_att_4d, pos_ids, adarms
            )

            sa = self._forward_pap_moe(
                force_tokens,
                prefix_out.to(dtype=td),
                proprio_token,
                quality_token,
                prefix_valid_mask=prefix_pad_masks,
                stage_labels_soft=stage_labels_soft,
                route_sequence_labels_soft=route_sequence_labels_soft,
                tf_prob=tf_prob,
                stage_override=stage_override,
                force=force,
                force_fast=force_fast,
                force_slow=force_slow,
                state=state,
                state_history=state_history,
                visual_quality=visual_quality,
                expert_mask=expert_mask,
                route_action_tokens=route_action_tokens,
                visual_memory=visual_memory,
                memory_age=memory_age,
            )
            suffix_out = suffix_out_raw

        act_dtype = self.action_out_proj.weight.dtype
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=act_dtype)

        expert_representation_loss, expert_representation_losses = (
            self._compute_expert_representation_loss(
                sa["expert_tokens"],
                stage_labels_soft,
                state,
                state_history,
                force,
                force_fast,
                force_slow,
                visual_quality,
            )
            if self.training and self.config.expert_representation_loss_weight > 0
            else (suffix_out.sum() * 0.0, {})
        )
        if (
            self.training
            and self.config.visual_memory_distillation_loss_weight > 0
            and visual_memory is not None
            and memory_age is not None
        ):
            clean_visual_teacher_features = (
                E_VL if self._uses_action_input_fusion() or self.config.train_expert_only
                or self.config.train_conditioner_only
                or self.config.train_gate_calibration_only
                or self.config.train_route_forecaster_only
                else prefix_out
            )
            clean_visual_teacher = self._pool_physics_prefix(
                clean_visual_teacher_features, prefix_pad_masks
            )
            visual_memory_distillation_loss = (
                self._compute_visual_memory_distillation_loss(
                    visual_memory,
                    memory_age,
                    clean_visual_teacher,
                    stage_labels_soft,
                    clean_only=self.config.visual_memory_teacher_policy == "clean_only",
                )
            )
        else:
            visual_memory_distillation_loss = suffix_out.sum() * 0.0

        # 5. Produce the complete action flow. vNext conditions the action
        # input tokens before the transformer; legacy checkpoints condition
        # the final hidden tokens immediately before the output projection.
        if self._uses_action_input_fusion():
            v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
            baseline_velocity = None
            conditioning_diagnostics = input_conditioning_diagnostics
        else:
            baseline_velocity = self._apply_checkpoint(self.action_out_proj, suffix_out)
            combined, conditioning_diagnostics = self._apply_physical_conditioning(suffix_out, sa)
            v_t = self._apply_checkpoint(self.action_out_proj, combined)
            v_t = self._preserve_unconditioned_gripper_velocity(baseline_velocity, v_t)

        # Compute Action Loss
        action_loss = F.mse_loss(u_t.to(dtype=torch.float32), v_t.to(dtype=torch.float32), reduction="none")
        result = {
            "action_loss": action_loss,
            "velocity": v_t,
            "stage_probs": sa["stage_probs"],
            "stage_logits": sa["stage_logits"],
            "expert_token_norms": sa["expert_tokens"].float().norm(dim=-1),
            "routing_probs": sa.get("conditioning_routing_probs", sa["routing_probs"]),
            **conditioning_diagnostics,
            "expert_representation_loss": expert_representation_loss,
            "expert_representation_losses": expert_representation_losses,
            "visual_memory_distillation_loss": visual_memory_distillation_loss,
        }
        if baseline_velocity is not None:
            result["baseline_action_loss"] = F.mse_loss(
                u_t.to(dtype=torch.float32),
                baseline_velocity.to(dtype=torch.float32),
                reduction="none",
            )
            result["baseline_velocity"] = baseline_velocity
        if "factor_probs" in sa:
            result["factor_probs"] = sa["factor_probs"]
        if "predicted_factor_sequence" in sa:
            result["predicted_factor_sequence"] = sa["predicted_factor_sequence"]
            result["predicted_route_sequence"] = sa["predicted_route_sequence"]
        return result

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        conditioning_tokens=None,
        conditioning_weights=None,
    ):
        dtype = self.action_in_proj.weight.dtype
        suffix_embs, suffix_pad, suffix_att, adarms = self.embed_suffix(
            x_t.to(dtype=dtype), timestep.to(dtype=dtype)
        )
        suffix_len = suffix_pad.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        pos_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
        full_att_4d = self._prepare_attention_masks_4d(full_att_2d)

        # Force eager attention for both models
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        input_conditioning_diagnostics = {}
        if conditioning_tokens is not None and self._uses_action_input_fusion():
            pap_result = {
                "conditioning_tokens": conditioning_tokens,
                "conditioning_weights": conditioning_weights,
                "routing_confidence": self._active_routing_confidence,
            }
            suffix_embs, input_conditioning_diagnostics = self._fuse_physics_into_action_input(
                suffix_embs, pap_result
            )

        lm_dtype = self.paligemma_with_expert.paligemma.model.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=lm_dtype)

        out, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_4d,
            position_ids=pos_ids,
            past_key_values=copy.deepcopy(past_key_values),
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms],
        )
        act_dtype = self.action_out_proj.weight.dtype
        suffix_out = out[1][:, -self.config.chunk_size :].to(dtype=act_dtype)

        baseline_velocity = None
        if (
            conditioning_tokens is not None
            and not self.config.condition_gripper_with_physical_experts
        ):
            baseline_velocity = self.action_out_proj(suffix_out).to(dtype=torch.float32)

        if conditioning_tokens is not None and not self._uses_action_input_fusion():
            pap_result = {
                "conditioning_tokens": conditioning_tokens,
                "conditioning_weights": conditioning_weights,
                "routing_confidence": self._active_routing_confidence,
            }
            suffix_out, diagnostics = self._apply_physical_conditioning(suffix_out, pap_result)
            self._last_conditioning_diagnostics = diagnostics
        elif conditioning_tokens is not None:
            self._last_conditioning_diagnostics = input_conditioning_diagnostics

        velocity = self.action_out_proj(suffix_out).to(dtype=torch.float32)
        if baseline_velocity is not None:
            velocity = self._preserve_unconditioned_gripper_velocity(
                baseline_velocity, velocity
            )
        return velocity

    def predict_gripper_logits(
        self,
        images,
        img_masks,
        tokens,
        masks,
        force=None,
        force_fast=None,
        force_slow=None,
        state_history=None,
        state=None,
    ) -> Tensor:
        """Train the gripper head from the shared observation prefix."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        (vlm_tokens, _), _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_att_2d),
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        return self._gripper_logits_from_prefix(
            vlm_tokens,
            prefix_pad_masks,
            force,
            force_fast,
            force_slow,
            state_history,
            None,
        )

    def predict_arm_chunk(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        state: Tensor,
        force: Tensor,
        force_fast: Tensor,
        force_slow: Tensor,
        state_history: Tensor,
    ) -> Tensor:
        """Train/evaluate the arm head from the shared observation prefix."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
            "eager"
        )
        (vlm_tokens, _), _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_att_2d),
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        return self._arm_chunk_from_prefix(
            vlm_tokens,
            prefix_pad_masks,
            state,
            force,
            force_fast,
            force_slow,
            state_history,
            None,
        )

    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        force=None,
        force_fast=None,
        force_slow=None,
        state=None,
        state_history=None,
        visual_quality=None,
        visual_history=None,
        visual_history_padding=None,
        noise=None,
        num_steps=None,
        stage_override=None,
        expert_mask=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> dict:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        B, device = tokens.shape[0], tokens.device
        if force is None:
            raise ValueError("PAP-MoE requires force/torque input")
        if state is None:
            state = torch.zeros(B, self.config.robot_state_dim, device=device, dtype=force.dtype)

        # VLM prefix  — must use eager attn to avoid sdpa dtype mismatch with bf16 mask
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        (E_VL, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        td = next(self.force_encoder.parameters()).dtype
        force_tokens, proprio_token, quality_token = self._encode_physics_inputs(
            force, state, force_fast, force_slow, state_history, visual_quality
        )
        visual_memory, memory_age = self._encode_visual_memory(
            visual_history, visual_history_padding
        )

        # PAP-MoE routing is determined only by observation-derived physics.
        sa = self._forward_pap_moe(
            force_tokens,
            E_VL.to(dtype=td),
            proprio_token,
            quality_token,
            prefix_valid_mask=prefix_pad_masks,
            stage_override=stage_override,
            force=force,
            force_fast=force_fast,
            force_slow=force_slow,
            state=state,
            state_history=state_history,
            visual_quality=visual_quality,
            expert_mask=expert_mask,
            visual_memory=visual_memory,
            memory_age=memory_age,
        )
        self._active_routing_confidence = sa["routing_confidence"]
        self._last_conditioning_diagnostics = {}

        if noise is None:
            noise = self.sample_noise((B, self.config.chunk_size, self.config.max_action_dim), device)
        dt = -1.0 / num_steps

        def denoise_chunk(active_sa: dict, *, use_rtc: bool) -> Tensor:
            candidate = noise.to(dtype=self.action_in_proj.weight.dtype).clone()
            for step in range(num_steps):
                tv = 1.0 + step * dt
                tt = torch.tensor(tv, dtype=torch.float32, device=device).expand(B)
                if use_rtc:
                    velocity = self._denoise_action_step(
                        x_t=candidate,
                        timestep=tt,
                        time=tv,
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        conditioning_tokens=active_sa["conditioning_tokens"],
                        conditioning_weights=active_sa.get("conditioning_weights"),
                        prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                        inference_delay=kwargs.get("inference_delay"),
                        execution_horizon=kwargs.get("execution_horizon"),
                        rtc_action_mask=kwargs.get("rtc_action_mask"),
                    )
                else:
                    # Draft generation is internal planning, not execution.
                    # Do not feed it through RTC or mutate RTC history.
                    velocity = self.denoise_step(
                        prefix_pad_masks,
                        past_key_values,
                        candidate,
                        tt,
                        conditioning_tokens=active_sa["conditioning_tokens"],
                        conditioning_weights=active_sa.get("conditioning_weights"),
                    )
                candidate = candidate + dt * velocity
                if (
                    use_rtc
                    and self.rtc_processor is not None
                    and self.rtc_processor.is_debug_enabled()
                ):
                    self.rtc_processor.track(time=tv, x_t=candidate, v_t=velocity)
            return candidate

        has_sequence_override = stage_override is not None and stage_override.ndim == 3
        if (
            self.config.physics_gate_architecture == "temporal_bcm_v2"
            and self.config.temporal_gate_two_pass_inference
            and not has_sequence_override
        ):
            draft_actions = denoise_chunk(sa, use_rtc=False)
            draft_tokens = self.action_in_proj(
                draft_actions.to(dtype=self.action_in_proj.weight.dtype)
            ).detach()
            sa = self._forward_pap_moe(
                force_tokens,
                E_VL.to(dtype=td),
                proprio_token,
                quality_token,
                prefix_valid_mask=prefix_pad_masks,
                stage_override=stage_override,
                force=force,
                force_fast=force_fast,
                force_slow=force_slow,
                state=state,
                state_history=state_history,
                visual_quality=visual_quality,
                expert_mask=expert_mask,
                route_action_tokens=draft_tokens,
                visual_memory=visual_memory,
                memory_age=memory_age,
            )
            self._active_routing_confidence = sa["routing_confidence"]

        x_t = denoise_chunk(sa, use_rtc=True)

        # Keep PAP perception/routing active, but replace the stochastic arm
        # dimensions with a learned state-feedback chunk. The gripper remains
        # generated by the PAP-conditioned flow head.
        if self.arm_head is not None:
            if force_fast is None or force_slow is None or state_history is None:
                raise ValueError("PAP deterministic arm head requires all multi-scale observations")
            arm_chunk = self._arm_chunk_from_prefix(
                E_VL,
                prefix_pad_masks,
                state,
                force,
                force_fast,
                force_slow,
                state_history,
                None,
            )
            arm_dim = self.config.arm_head_action_dim
            x_t[..., :arm_dim] = _clip_normalized_arm_chunk(
                arm_chunk, self.config.arm_head_normalized_output_clip
            )

        if self.gripper_head is not None:
            gripper_logits = self._gripper_logits_from_prefix(
                E_VL,
                prefix_pad_masks,
                force,
                force_fast,
                force_slow,
                state_history,
                None,
            )
            gripper_closed = (
                torch.sigmoid(gripper_logits)
                >= self.config.gripper_head_probability_threshold
            )
            if self.config.gripper_head_hold_first_action:
                gripper_closed = gripper_closed[:, :1].expand_as(gripper_closed)
            gripper_values = torch.where(
                gripper_closed,
                torch.full_like(gripper_logits, self.config.gripper_head_closed_normalized_value),
                torch.full_like(gripper_logits, self.config.gripper_head_open_normalized_value),
            )
            x_t[..., self.config.gripper_action_index] = gripper_values

        result = {
            "actions": x_t,
            "stage_probs": sa["stage_probs"],
            # This is the route that conditioned the generated action block.
            # It is [B,E] for fixed routing and [B,T,E] for temporal routing.
            "routing_probs": sa.get("conditioning_routing_probs", sa["routing_probs"]),
            "routing_confidence": sa["routing_confidence"],
            "expert_token_norms": sa["expert_tokens"].float().norm(dim=-1),
            **self._last_conditioning_diagnostics,
        }
        if "factor_probs" in sa:
            result["factor_probs"] = sa["factor_probs"]
        if "predicted_factor_sequence" in sa:
            result["predicted_factor_sequence"] = sa["predicted_factor_sequence"]
            result["predicted_route_sequence"] = sa["predicted_route_sequence"]
        return result

    def _denoise_action_step(
        self,
        *,
        x_t: Tensor,
        timestep: Tensor,
        time: float,
        prefix_pad_masks: Tensor,
        past_key_values,
        conditioning_tokens: Tensor,
        conditioning_weights: Tensor | None,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int | None,
        execution_horizon: int | None,
        rtc_action_mask: Tensor | None,
    ) -> Tensor:
        """Run one PAP denoising step, with RTC prefix guidance when enabled."""

        def denoise_step_partial_call(input_x_t: Tensor) -> Tensor:
            return self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                input_x_t,
                timestep,
                conditioning_tokens=conditioning_tokens,
                conditioning_weights=conditioning_weights,
            )

        if not self._rtc_enabled():
            return denoise_step_partial_call(x_t)

        return self.rtc_processor.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=0 if inference_delay is None else inference_delay,
            time=time,
            original_denoise_step_partial=denoise_step_partial_call,
            execution_horizon=execution_horizon,
            action_mask=rtc_action_mask,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAP-MoE Policy Wrapper
# ══════════════════════════════════════════════════════════════════════════════


class PAPMoEPolicy(PI05Policy):
    """PAP-MoE Policy for LeRobot based on PI0.5."""

    config_class = PAPMoEConfig
    name = "pap_moe"

    def reset(self):
        super().reset()
        self._online_visual_history: dict[str, list[tuple[Tensor, Tensor]]] = {}

    def _preprocess_pap_images(
        self, batch: dict[str, Tensor], *, update_online_memory: bool = False
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor] | None, list[Tensor] | None]:
        """Separate current Pi0.5 images from episode-local E2 history."""
        if not self.config.use_visual_memory:
            images, masks = self._preprocess_images(batch)
            return images, masks, None, None

        current_batch = dict(batch)
        histories = []
        history_padding = []
        history_steps = len(self.config.visual_memory_history_indices) - 1
        real_keys = [
            key
            for key in self.config.image_features
            if key in batch and "empty_camera" not in key
        ]
        for key in real_keys:
            image = batch[key]
            if image.ndim == 5:
                if image.shape[1] != history_steps + 1:
                    raise ValueError(
                        f"{key} history length={image.shape[1]}, expected={history_steps + 1}"
                    )
                histories.append(image[:, :-1])
                padding = batch.get(f"{key}_is_pad")
                history_padding.append(
                    torch.zeros(
                        image.shape[:2], device=image.device, dtype=torch.bool
                    )[:, :-1]
                    if padding is None
                    else padding[:, :-1].to(dtype=torch.bool)
                )
                current_batch[key] = image[:, -1]
                continue
            if image.ndim != 4:
                raise ValueError(f"{key} must be BCHW or BTCHW, got {tuple(image.shape)}")
            if not update_online_memory:
                raise ValueError(
                    "visual-memory training requires episode-safe camera history from the dataset"
                )

            entries = self._online_visual_history.setdefault(key, [])
            entries = entries[-history_steps:]
            missing = history_steps - len(entries)
            history_frames = [torch.zeros_like(image) for _ in range(missing)]
            history_valid = [
                torch.zeros(image.shape[0], device=image.device, dtype=torch.bool)
                for _ in range(missing)
            ]
            history_frames.extend(entry[0].to(image.device) for entry in entries)
            history_valid.extend(entry[1].to(image.device) for entry in entries)
            histories.append(torch.stack(history_frames, dim=1))
            history_padding.append(~torch.stack(history_valid, dim=1))

            gray = image.float().mean(dim=1)
            black = (gray <= 0.02).float().mean(dim=(1, 2))
            saturated = (gray >= 0.98).float().mean(dim=(1, 2))
            contrast = gray.std(dim=(1, 2), unbiased=False)
            valid = (black < 0.95) & (saturated < 0.95) & (contrast > 0.01)
            entries.append((image.detach(), valid.detach()))
            self._online_visual_history[key] = entries[-history_steps:]

        if len(histories) != 2:
            raise ValueError(
                f"E2 visual memory requires exactly two real cameras, found {real_keys}"
            )
        images, masks = self._preprocess_images(current_batch)
        return images, masks, histories, history_padding

    def _save_pretrained(self, save_directory) -> None:
        # cuDNN GRU setup can leave parameter views whose storage is larger
        # than the tensor. safetensors.save_model refuses such tensors even
        # though they are valid PyTorch parameters. Materialize only those
        # views before delegating to the standard LeRobot saver.
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.GRU):
                    for parameter in module.parameters(recurse=False):
                        # A cuDNN flat-weight view can report contiguous while
                        # covering only a slice of a larger storage.
                        parameter.data = parameter.detach().clone(
                            memory_format=torch.contiguous_format
                        )
            for parameter in self.parameters():
                if not parameter.is_contiguous():
                    parameter.data = parameter.data.contiguous()
            for module in self.modules():
                for name, buffer in module._buffers.items():
                    if buffer is not None and not buffer.is_contiguous():
                        module._buffers[name] = buffer.contiguous()
        super()._save_pretrained(save_directory)

    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        """Map standard Linear state keys to LoRALinear base keys.

        This keeps the inherited PI0.5 backbone loadable after LoRA injection.
        """
        fixed_state_dict = super()._fix_pytorch_state_dict_keys(state_dict, model_config)

        remapped = {}
        lora_projs = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        # A factorized Gate predicts b/c/m (3 outputs), while legacy PAP
        # checkpoints predict four expert logits directly.  PyTorch's
        # strict=False still raises on same-name tensors with different
        # shapes; that exception used to abort the *entire* checkpoint load
        # and could silently leave the action backbone and experts randomly
        # initialized.  Drop only the four incompatible final-layer tensors.
        # Every shared Gate layer and every non-Gate parameter remains loaded.
        fixed_state_dict, skipped_legacy_gate_outputs = (
            _filter_legacy_gate_outputs_for_factorized_gate(
                fixed_state_dict, model_config.physics_gate_architecture
            )
        )

        for key, value in fixed_state_dict.items():
            new_key = key
            # ── LoRA base weight remapping ──
            if "language_model" in new_key or "gemma_expert" in new_key:
                parts = new_key.split(".")
                if len(parts) >= 2 and parts[-2] in lora_projs:
                    if parts[-1] in ("weight", "bias") and not any(
                        p in ("base", "lora_A", "lora_B") for p in parts[-2:]
                    ):
                        parts.insert(-1, "base")
                        new_key = ".".join(parts)
            remapped[new_key] = value

        if skipped_legacy_gate_outputs:
            print(
                "Migrating legacy four-output PhysicsGate to factorized b/c/m Gate: "
                f"reinitializing {len(skipped_legacy_gate_outputs)} output tensors only"
            )

        return remapped

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        kwargs["strict"] = False
        # Keep the same mixed-precision contract as PI05Policy. In particular,
        # action/time projections and AdaRMS layers are intentionally FP32;
        # casting the whole policy to BF16 changes an otherwise frozen Pi0.5
        # rollout even when the PAP residual is disabled.
        return super().from_pretrained(pretrained_name_or_path, **kwargs)

    def __init__(self, config: PAPMoEConfig, **kwargs):
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()

        # Override self.model with our custom PAP-MoE Model
        self.model = PAPMoEPi05Model(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()
        self._action_adapter_source_parameters: dict[str, Tensor] | None = None

        # Track training steps for scheduling
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=True)

    def load_force_stats(self, dataset_stats: dict):
        """Compatibility hook; force normalization is owned by the processor."""
        logging.info("PAP-MoE v6 uses processor-side force normalization exactly once.")

    def get_optim_params(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not (
            self.config.train_expert_action_joint
            or getattr(self.config, "train_physicsgate_action_joint", False)
        ):
            return trainable

        # Both physical representations and the complete action generator are
        # jointly optimized. Their LR ratio is an explicit experiment setting,
        # not a baseline-output preservation constraint or a proven cause of
        # closed-loop performance differences.
        action_modules = [
            self.model.paligemma_with_expert.gemma_expert.model,
            self.model.action_in_proj,
            self.model.action_out_proj,
            self.model.time_mlp_in,
            self.model.time_mlp_out,
        ]
        action_parameter_ids = {
            id(parameter)
            for module in action_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        pap_parameters = [p for p in trainable if id(p) not in action_parameter_ids]
        action_parameters = [p for p in trainable if id(p) in action_parameter_ids]
        return [
            {"params": pap_parameters},
            {
                "params": action_parameters,
                "lr": self.config.pap_moe_optimizer_lr
                * self.config.joint_action_expert_lr_scale,
            },
        ]

    def _get_action_adapter_source_parameters(self) -> dict[str, Tensor]:
        """Lazily snapshot the loaded source adapter without copying the backbone."""
        if self._action_adapter_source_parameters is None:
            self._action_adapter_source_parameters = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
        return self._action_adapter_source_parameters

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """PAP-MoE Training forward pass computing both Action and Stage loss."""
        if self.config.train_arm_head_only or self.config.train_gripper_head_only:
            # Reuse the base Pi0.5 head objective. PAPMoEPi05Model inherits the
            # same prefix and sensor-conditioned arm-head implementation.
            return PI05Policy.forward(self, batch, reduction)

        # 1. Setup scheduling states
        step = self._step.item()
        if self.training:
            self._step += 1

        device = next(self.parameters()).device
        B = batch[OBS_FORCE].shape[0] if OBS_FORCE in batch else batch["observation.state"].shape[0]

        if OBS_FORCE not in batch:
            raise ValueError("PAP-MoE requires observation.force")
        force = self._pap_sensor(batch, OBS_FORCE, 4)
        force_fast = self._pap_sensor(batch, OBS_FORCE_FAST, 5)
        force_slow = self._pap_sensor(batch, OBS_FORCE_SLOW, 6)
        state = batch[OBS_STATE]
        state_history = self._pap_sensor(batch, OBS_STATE_HISTORY, 3)
        visual_quality = batch.get(OBS_VISUAL_QUALITY)
        assert force is not None
        if self.config.require_multiscale_observations:
            missing = [
                key
                for key, value in (
                    (OBS_FORCE_FAST, force_fast),
                    (OBS_FORCE_SLOW, force_slow),
                    (OBS_STATE_HISTORY, state_history),
                    (OBS_VISUAL_QUALITY, visual_quality),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"PAP-MoE v6 missing required observations: {missing}")

        # Extract both the discrete expert ID and four-dimensional soft
        # probabilities used for supervision and continuous teacher forcing.
        stage_labels = None
        stage_labels_soft = None
        route_sequence_labels_soft = None
        if OBS_PHYSICS_GATE_TARGET in batch:
            s = batch[OBS_PHYSICS_GATE_TARGET]
            if s.dim() in (2, 3) and s.shape[-1] > 1:
                # Reconstruct clean 0~1 continuous soft probability vector from normalized quantiles (E2=0.0)
                pos_s = F.relu(s.to(dtype=torch.float32))
                normalized = pos_s / (pos_s.sum(dim=-1, keepdim=True) + 1e-8)
                if s.dim() == 3:
                    route_sequence_labels_soft = normalized
                    stage_labels_soft = normalized[:, 0]
                    stage_labels = s[:, 0].argmax(dim=-1).to(dtype=torch.long)
                else:
                    stage_labels_soft = normalized
                    stage_labels = s.argmax(dim=-1).to(dtype=torch.long)
            else:
                stage_labels = s.squeeze(-1).to(dtype=torch.long)
                stage_labels_soft = F.one_hot(stage_labels, num_classes=self.config.num_experts).to(
                    dtype=torch.float32
                )

        # ── PhysicsGate supervision fast path (no experts or action flow) ──
        if self.training and (
            self.config.train_physicsgate_only or self.config.train_route_forecaster_only
        ):
            # Still need images + language tokens for VLM prefix (frozen) → E_VL
            images, img_masks, visual_history, visual_history_padding = (
                self._preprocess_pap_images(batch)
            )
            tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

            if stage_labels_soft is None:
                raise ValueError(
                    "PAP-MoE PhysicsGate-only training requires "
                    "observation.physics_gate_target soft expert targets"
                )
            # The unified temporal gate must see the demonstrated action
            # chunk while learning future b/c/m. Legacy gates ignore actions.
            routing_actions = (
                self.prepare_action(batch)
                if self.config.physics_gate_architecture == "temporal_bcm_v2"
                else torch.zeros(
                    B,
                    self.config.chunk_size,
                    self.config.max_action_dim,
                    device=device,
                )
            )
            outputs = self.model.forward(
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                masks=masks,
                actions=routing_actions,
                force=force,
                force_fast=force_fast,
                force_slow=force_slow,
                state=state,
                state_history=state_history,
                visual_quality=visual_quality,
                visual_history=visual_history,
                visual_history_padding=visual_history_padding,
                stage_labels=stage_labels,
                stage_labels_soft=stage_labels_soft,
                route_sequence_labels_soft=route_sequence_labels_soft,
            )
            stage_logits = outputs["stage_logits"]
            stage_probs = outputs["stage_probs"]

            stage_loss, balancing_loss = _masked_stage_losses(
                batch, stage_logits, stage_probs, stage_labels_soft
            )

            factor_loss = stage_loss.new_zeros(())
            if outputs.get("factor_probs") is not None:
                factor_loss = _masked_factorized_gate_loss(
                    batch, outputs["factor_probs"], stage_labels_soft
                )

            route_sequence_loss = stage_loss.new_zeros(())
            if (
                route_sequence_labels_soft is not None
                and outputs.get("predicted_factor_sequence") is not None
            ):
                route_sequence_loss = _masked_factorized_gate_loss(
                    batch,
                    outputs["predicted_factor_sequence"],
                    route_sequence_labels_soft,
                )

            if self.config.train_route_forecaster_only:
                total_loss = self.config.route_sequence_loss_weight * route_sequence_loss
            else:
                total_loss = (
                    stage_loss
                    + self.config.factorized_gate_factor_loss_weight * factor_loss
                    + self.config.route_sequence_loss_weight * route_sequence_loss
                    + self.config.balance_loss_weight * balancing_loss
                )

            loss_dict = {
                "loss": total_loss.item(),
                "loss_optimized": total_loss.item(),
                "loss_action": 0.0,
                "loss_stage": stage_loss.item(),
                "loss_balancing": balancing_loss.item(),
                "loss_factor_bcm": factor_loss.item(),
                "loss_route_sequence_bcm": route_sequence_loss.item(),
                "tf_prob": 0.0,
                "optimize_action": 0.0,
                "optimize_stage": 1.0,
            }
            mean_pred = stage_probs.detach().float().mean(dim=0)
            mean_target = stage_labels_soft.detach().float().mean(dim=0)
            for expert_index in range(min(self.config.num_experts, 4)):
                loss_dict[f"pred_E{expert_index + 1}"] = mean_pred[expert_index].item()
                loss_dict[f"gt_E{expert_index + 1}"] = mean_target[expert_index].item()
            return total_loss, loss_dict

        # ── Full forward path (Stage 2/3) ──
        # Stage loss weight warmup
        if step < self.config.stage_loss_warmup_steps:
            stage_weight = 0.05
        elif step < self.config.stage_loss_warmup_end_steps:
            frac = (step - self.config.stage_loss_warmup_steps) / (
                self.config.stage_loss_warmup_end_steps - self.config.stage_loss_warmup_steps
            )
            stage_weight = 0.05 + frac * (self.config.stage_loss_weight - 0.05)
        else:
            stage_weight = self.config.stage_loss_weight

        # Teacher forcing decay
        if step < self.config.teacher_forcing_start_steps:
            tf_prob = 1.0
        elif step < self.config.teacher_forcing_end_steps:
            frac = (step - self.config.teacher_forcing_start_steps) / (
                self.config.teacher_forcing_end_steps - self.config.teacher_forcing_start_steps
            )
            tf_prob = 1.0 - frac
        else:
            tf_prob = 0.0

        # Extract inputs
        images, img_masks, visual_history, visual_history_padding = self._preprocess_pap_images(
            batch
        )
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        stage_override = batch.get("stage_override")
        route_source = getattr(self.config, "expert_action_training_route", "ground_truth")
        if route_source not in {"ground_truth", "predicted"}:
            raise ValueError(f"Unknown expert_action_training_route: {route_source}")
        if self.config.train_expert_action_joint and route_source == "predicted":
            # Same observation-only 50-step PhysicsGate path as deployment.
            # Labels remain available for the unchanged auxiliary objectives,
            # but neither current nor future labels may route the action flow.
            stage_override = None
            tf_prob = 0.0
        elif self.config.train_expert_only or self.config.train_expert_action_joint:
            if stage_labels_soft is None:
                raise ValueError(
                    "PAP-MoE expert training requires observation.physics_gate_target "
                    "for deterministic physical-expert routing"
                )
            # A staged run normally starts from a prior checkpoint whose
            # persistent `_step` counter is already large.  Do not use that
            # counter's teacher-forcing schedule here: expert identification
            # and expert/action co-adaptation must always see the dataset's
            # physical soft-route target, independent of checkpoint history.
            stage_override = (
                route_sequence_labels_soft
                if self.config.action_step_routing and route_sequence_labels_soft is not None
                else stage_labels_soft
            )
            tf_prob = 1.0

        model_kwargs = dict(
            force=force,
            force_fast=force_fast,
            force_slow=force_slow,
            state=state,
            state_history=state_history,
            visual_quality=visual_quality,
            visual_history=visual_history,
            visual_history_padding=visual_history_padding,
            stage_labels=stage_labels,
            stage_labels_soft=stage_labels_soft,
            route_sequence_labels_soft=route_sequence_labels_soft,
            stage_override=stage_override,
            tf_prob=tf_prob,
            expert_mask=batch.get("expert_mask"),
        )
        noise = None
        time = None
        source_parameters = None
        needs_expert_anchor = (
            self.config.train_expert_only or self.config.train_expert_action_joint
        ) and self.config.expert_action_anchor_weight > 0
        if self.config.train_action_adapter_only and (
            self.config.action_adapter_anchor_weight > 0
            or self.config.action_adapter_continuity_weight > 0
        ):
            # Use identical flow noise/time for the optimized action loss,
            # continuity estimate, and frozen-source teacher.
            noise = self.model.sample_noise(actions.shape, device)
            time = self.model.sample_time(B, device)
            if self.config.action_adapter_anchor_weight > 0:
                source_parameters = self._get_action_adapter_source_parameters()

        # Model Forward
        outputs = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise=noise,
            time=time,
            **model_kwargs,
        )

        # Truncate actions loss to actual dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = outputs["action_loss"][:, :, :original_action_dim]

        normalized_actions = actions[:, :, :original_action_dim]
        per_sample_action_loss, action_loss_weights = _compute_weighted_action_loss(
            losses,
            normalized_actions,
            gripper_action_index=self.config.gripper_action_index,
            gripper_loss_weight=self.config.gripper_loss_weight,
            gripper_open_loss_weight=self.config.gripper_open_loss_weight,
            gripper_open_threshold_normalized=self.config.gripper_open_threshold_normalized,
        )
        action_loss = per_sample_action_loss if reduction == "none" else per_sample_action_loss.mean()

        # 5. Compute Stage Classification & Balancing Losses
        loss_dict = {
            "loss_action": action_loss.mean().item() if reduction == "none" else action_loss.item(),
            "effective_weight_per_dim": action_loss_weights.mean(dim=(0, 1)).detach().cpu().tolist(),
            "optimize_action": 1.0,
            "optimize_stage": (
                1.0
                if self.config.train_pap_moe_joint
                or self.config.train_physicsgate_action_joint
                or self.config.train_gate_calibration_only
                or self.config.train_route_forecaster_only
                else 0.0
            ),
        }

        if stage_labels is not None:
            stage_logits = outputs["stage_logits"]
            stage_probs = outputs["stage_probs"]

            stage_loss, balancing_loss = _masked_stage_losses(
                batch, stage_logits, stage_probs, stage_labels_soft
            )
            factor_loss = stage_loss.new_zeros(())
            if outputs.get("factor_probs") is not None and stage_labels_soft is not None:
                factor_loss = _masked_factorized_gate_loss(
                    batch, outputs["factor_probs"], stage_labels_soft
                )
            route_sequence_loss = stage_loss.new_zeros(())
            if (
                route_sequence_labels_soft is not None
                and outputs.get("predicted_factor_sequence") is not None
            ):
                route_sequence_loss = _masked_factorized_gate_loss(
                    batch,
                    outputs["predicted_factor_sequence"],
                    route_sequence_labels_soft,
                )

            # In isolated expert/conditioner stages these are diagnostics only.
            # Otherwise the frozen gate is kept diagnostic-only.
            if self.config.train_pap_moe_joint or self.config.train_physicsgate_action_joint:
                total_loss = (
                    action_loss
                    + stage_weight * stage_loss
                    + self.config.factorized_gate_factor_loss_weight * factor_loss
                    + self.config.route_sequence_loss_weight * route_sequence_loss
                    + self.config.balance_loss_weight * balancing_loss
                )
            elif self.config.train_gate_calibration_only:
                # Experts, action conditioner and Pi0.5 are frozen, but remain
                # in the differentiable path. The action-flow loss therefore
                # calibrates predicted routing to the experts' actual action
                # effect; the small routing term preserves physical identity.
                total_loss = (
                    self.config.gate_calibration_action_loss_weight * action_loss
                    + self.config.gate_calibration_routing_loss_weight * stage_loss
                    + self.config.factorized_gate_factor_loss_weight * factor_loss
                    + self.config.route_sequence_loss_weight * route_sequence_loss
                    + self.config.balance_loss_weight * balancing_loss
                )
            elif self.config.train_route_forecaster_only:
                # All existing Experiment-2 modules stay frozen. The new
                # temporal router learns both physical sequence supervision
                # and whether its per-token conditions improve action flow.
                total_loss = action_loss + self.config.route_sequence_loss_weight * route_sequence_loss
            else:
                total_loss = action_loss

            loss_dict["loss_stage"] = stage_loss.item()
            loss_dict["loss_balancing"] = balancing_loss.item()
            loss_dict["loss_factor_bcm"] = factor_loss.item()
            loss_dict["loss_route_sequence_bcm"] = route_sequence_loss.item()
            loss_dict["stage_loss_weight"] = stage_weight
            loss_dict["tf_prob"] = tf_prob

            # Record batch mean expert activations & GT probabilities
            mean_pred = stage_probs.detach().float().mean(dim=0).cpu().numpy()
            for idx_e in range(min(4, len(mean_pred))):
                loss_dict[f"pred_E{idx_e + 1}"] = float(mean_pred[idx_e])

            if stage_labels_soft is not None:
                gt_4d = stage_labels_soft.mean(dim=0).cpu().numpy()
                for idx_e in range(min(4, len(gt_4d))):
                    loss_dict[f"gt_E{idx_e + 1}"] = float(gt_4d[idx_e])
            elif stage_labels is not None:
                gt_4d = F.one_hot(stage_labels, num_classes=4).float().mean(dim=0).cpu().numpy()
                for idx_e in range(min(4, len(gt_4d))):
                    loss_dict[f"gt_E{idx_e + 1}"] = float(gt_4d[idx_e])
        else:
            total_loss = action_loss

        expert_representation_loss = outputs.get("expert_representation_loss")
        if (
            expert_representation_loss is not None
            and self.config.expert_representation_loss_weight > 0
        ):
            total_loss = (
                total_loss
                + self.config.expert_representation_loss_weight
                * expert_representation_loss
            )
            loss_dict["loss_expert_representation"] = (
                expert_representation_loss.detach().mean().item()
            )
            for name, value in outputs.get("expert_representation_losses", {}).items():
                loss_dict[f"loss_expert_representation_{name}"] = value.detach().item()

        visual_memory_distillation_loss = outputs.get("visual_memory_distillation_loss")
        if (
            visual_memory_distillation_loss is not None
            and self.config.visual_memory_distillation_loss_weight > 0
        ):
            total_loss = (
                total_loss
                + self.config.visual_memory_distillation_loss_weight
                * visual_memory_distillation_loss
            )
            loss_dict["loss_visual_memory_distillation"] = (
                visual_memory_distillation_loss.detach().mean().item()
            )

        if needs_expert_anchor:
            expert_anchor_per_sample = F.mse_loss(
                outputs["velocity"][:, :, :original_action_dim].float(),
                outputs["baseline_velocity"][:, :, :original_action_dim].detach().float(),
                reduction="none",
            ).mean(dim=(1, 2))
            expert_anchor_loss = (
                expert_anchor_per_sample
                if reduction == "none"
                else expert_anchor_per_sample.mean()
            )
            total_loss = total_loss + self.config.expert_action_anchor_weight * expert_anchor_loss
            loss_dict["loss_expert_action_anchor"] = expert_anchor_loss.mean().item()

        if self.config.train_action_adapter_only:
            auxiliary_loss = action_loss.new_zeros(action_loss.shape)
            if source_parameters is not None:
                was_training = self.model.training
                try:
                    self.model.eval()
                    with torch.no_grad():
                        source_outputs = functional_call(
                            self.model,
                            source_parameters,
                            args=(images, img_masks, tokens, masks, actions),
                            kwargs={"noise": noise, "time": time, **model_kwargs},
                            strict=False,
                        )
                finally:
                    self.model.train(was_training)
                x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
                student_actions = x_t - time[:, None, None] * outputs["velocity"].float()
                source_actions = x_t - time[:, None, None] * source_outputs["velocity"].float()
                anchor_per_sample = F.mse_loss(
                    student_actions[:, :, :original_action_dim],
                    source_actions[:, :, :original_action_dim],
                    reduction="none",
                ).mean(dim=(1, 2))
                episode_index = batch.get("episode_index")
                if episode_index is not None:
                    episode_index = episode_index.reshape(-1).to(device=anchor_per_sample.device)
                    recovery_scale = torch.where(
                        episode_index >= self.config.action_adapter_recovery_episode_start,
                        anchor_per_sample.new_tensor(
                            self.config.action_adapter_recovery_anchor_scale
                        ),
                        anchor_per_sample.new_tensor(1.0),
                    )
                    anchor_per_sample = anchor_per_sample * recovery_scale
                anchor_loss = anchor_per_sample if reduction == "none" else anchor_per_sample.mean()
                auxiliary_loss = auxiliary_loss + self.config.action_adapter_anchor_weight * anchor_loss
                loss_dict["loss_action_anchor"] = anchor_loss.mean().item()

            if self.config.action_adapter_continuity_weight > 0:
                x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
                student_actions = x_t - time[:, None, None] * outputs["velocity"].float()
                horizon = self.config.action_adapter_continuity_horizon
                arm_dim = min(6, original_action_dim)
                student_delta = torch.diff(student_actions[:, :horizon, :arm_dim], dim=1)
                target_delta = torch.diff(actions[:, :horizon, :arm_dim].float(), dim=1)
                continuity_per_sample = F.mse_loss(
                    student_delta, target_delta, reduction="none"
                ).mean(dim=(1, 2))
                continuity_loss = (
                    continuity_per_sample if reduction == "none" else continuity_per_sample.mean()
                )
                auxiliary_loss = (
                    auxiliary_loss
                    + self.config.action_adapter_continuity_weight * continuity_loss
                )
                loss_dict["loss_action_continuity"] = continuity_loss.mean().item()

            total_loss = total_loss + auxiliary_loss

        if outputs.get("expert_token_norms") is not None:
            mean_norms = outputs["expert_token_norms"].detach().mean(dim=0)
            for expert_index in range(min(self.config.num_experts, mean_norms.numel())):
                loss_dict[f"norm_E{expert_index + 1}"] = mean_norms[expert_index].item()

        loss_dict["loss"] = total_loss.mean().item() if reduction == "none" else total_loss.item()
        loss_dict["loss_optimized"] = loss_dict["loss"]

        return total_loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks, visual_history, visual_history_padding = self._preprocess_pap_images(
            batch, update_online_memory=True
        )
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

        if OBS_FORCE not in batch:
            raise ValueError("PAP-MoE requires observation.force")
        force = self._pap_sensor(batch, OBS_FORCE, 4)
        force_fast = self._pap_sensor(batch, OBS_FORCE_FAST, 5)
        force_slow = self._pap_sensor(batch, OBS_FORCE_SLOW, 6)
        state_history = self._pap_sensor(batch, OBS_STATE_HISTORY, 3)
        visual_quality = batch.get(OBS_VISUAL_QUALITY)
        assert force is not None
        if self.config.require_multiscale_observations and any(
            value is None for value in (force_fast, force_slow, state_history, visual_quality)
        ):
            raise ValueError("PAP-MoE v6 inference requires all multi-scale observations")

        # Stage override
        stage_override = None
        if "stage_override" in batch:
            stage_override = batch["stage_override"]
        expert_mask = batch.get("expert_mask")

        # Run sample actions
        outputs = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            force=force,
            force_fast=force_fast,
            force_slow=force_slow,
            state=batch[OBS_STATE],
            state_history=state_history,
            visual_quality=visual_quality,
            visual_history=visual_history,
            visual_history_padding=visual_history_padding,
            stage_override=stage_override,
            expert_mask=expert_mask,
            **kwargs,
        )
        actions = outputs["actions"]
        self.last_stage_probs = outputs["stage_probs"].detach().cpu()
        self.last_routing_probs = outputs["routing_probs"].detach().cpu()
        self.last_routing_confidence = outputs["routing_confidence"].detach().cpu()
        self.last_expert_token_norms = outputs["expert_token_norms"].detach().cpu()
        self.last_factor_probs = (
            None
            if outputs.get("factor_probs") is None
            else outputs["factor_probs"].detach().cpu()
        )
        self.last_predicted_factor_sequence = (
            None
            if outputs.get("predicted_factor_sequence") is None
            else outputs["predicted_factor_sequence"].detach().cpu()
        )
        self.last_predicted_route_sequence = (
            None
            if outputs.get("predicted_route_sequence") is None
            else outputs["predicted_route_sequence"].detach().cpu()
        )
        self.last_condition_residual_norm = (
            None
            if outputs.get("condition_residual_norm") is None
            else outputs["condition_residual_norm"].detach().cpu()
        )
        self.last_applied_condition_residual_norm = (
            None
            if outputs.get("applied_condition_residual_norm") is None
            else outputs["applied_condition_residual_norm"].detach().cpu()
        )
        self.last_expert_conditioning_scales = (
            None
            if outputs.get("expert_conditioning_scales") is None
            else outputs["expert_conditioning_scales"].detach().cpu()
        )

        # Unpad
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions
    def _pap_sensor(
        self, batch: dict[str, Tensor], key: str, validity_index: int
    ) -> Tensor | None:
        """Apply the VS ablation and erase unavailable legacy modalities after normalization."""
        value = self._controlled_sensor(batch, key)
        validity = batch.get(OBS_MODALITY_VALIDITY)
        if value is None or validity is None:
            return value
        if validity.ndim != 2 or validity.shape[1] < 7:
            raise ValueError(
                f"{OBS_MODALITY_VALIDITY} must be [B,7], got {tuple(validity.shape)}"
            )
        mask = validity[:, validity_index].to(device=value.device, dtype=value.dtype)
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        return value * mask

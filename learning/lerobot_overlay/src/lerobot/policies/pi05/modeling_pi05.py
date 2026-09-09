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

import builtins
import copy
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from ..pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)

from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_pi05 import (
    DEFAULT_IMAGE_SIZE,
    OBS_FORCE,
    OBS_FORCE_FAST,
    OBS_FORCE_SLOW,
    OBS_STATE_HISTORY,
    PI05Config,
)

SUBTASK_LABEL_KEY = "subtask_label"


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    rtc_action_mask: Tensor | None
    release_force: Tensor
    release_force_fast: Tensor
    release_force_slow: Tensor
    release_state_history: Tensor
    arm_state: Tensor
    arm_force: Tensor
    arm_force_fast: Tensor
    arm_force_slow: Tensor
    arm_state_history: Tensor
    gripper_force: Tensor
    gripper_force_fast: Tensor
    gripper_force_slow: Tensor
    gripper_state_history: Tensor


def _compute_weighted_action_loss(
    losses: Tensor,
    normalized_actions: Tensor,
    *,
    gripper_action_index: int | None,
    gripper_loss_weight: float,
    gripper_open_loss_weight: float,
    gripper_open_threshold_normalized: float,
    action_prefix_loss_horizon: int | None = None,
    action_prefix_loss_weight: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return per-sample flow loss and the effective element weights.

    Weight normalization is performed independently for every sample so the
    optimizer's overall loss scale does not grow when one action dimension is
    emphasized. ``normalized_actions`` must contain only real (unpadded)
    action dimensions.
    """
    if losses.shape != normalized_actions.shape:
        raise ValueError(
            f"Loss/action shape mismatch: {tuple(losses.shape)} != {tuple(normalized_actions.shape)}"
        )

    weights = torch.ones_like(losses)
    if action_prefix_loss_horizon is not None:
        if not 1 <= action_prefix_loss_horizon <= losses.shape[1]:
            raise ValueError(
                "action_prefix_loss_horizon must be within the loss time dimension, "
                f"got {action_prefix_loss_horizon} for horizon={losses.shape[1]}"
            )
        if not math.isfinite(action_prefix_loss_weight) or action_prefix_loss_weight <= 0:
            raise ValueError("action_prefix_loss_weight must be finite and positive")
        weights[:, :action_prefix_loss_horizon] *= action_prefix_loss_weight

    if gripper_action_index is not None:
        if not 0 <= gripper_action_index < losses.shape[-1]:
            raise ValueError(
                "gripper_action_index must reference a real action dimension, "
                f"got {gripper_action_index} for action_dim={losses.shape[-1]}"
            )

        gripper_targets = normalized_actions[..., gripper_action_index]
        open_targets = gripper_targets <= gripper_open_threshold_normalized
        gripper_weights = torch.full_like(gripper_targets, gripper_loss_weight)
        gripper_weights = torch.where(
            open_targets,
            gripper_weights * gripper_open_loss_weight,
            gripper_weights,
        )
        weights[..., gripper_action_index] *= gripper_weights

    weighted_sum = (losses * weights).sum(dim=(1, 2))
    return weighted_sum / weights.sum(dim=(1, 2)), weights


def _pool_last_valid_prefix(prefix_hidden: Tensor, prefix_pad_masks: Tensor) -> Tensor:
    """Select the last valid prefix token even when empty cameras add mask gaps."""
    if prefix_hidden.shape[:2] != prefix_pad_masks.shape:
        raise ValueError(
            "Prefix hidden/mask shape mismatch: "
            f"{tuple(prefix_hidden.shape)} vs {tuple(prefix_pad_masks.shape)}"
        )
    positions = torch.arange(prefix_pad_masks.shape[1], device=prefix_pad_masks.device)
    positions = positions.unsqueeze(0).expand_as(prefix_pad_masks)
    last_valid = positions.masked_fill(~prefix_pad_masks, -1).max(dim=1).values
    if torch.any(last_valid < 0):
        raise ValueError("Every prefix sample must contain at least one valid token")
    batch_indices = torch.arange(prefix_hidden.shape[0], device=prefix_hidden.device)
    return prefix_hidden[batch_indices, last_valid]


def _pool_mean_valid_prefix(prefix_hidden: Tensor, prefix_pad_masks: Tensor) -> Tensor:
    """Mean-pool every valid image/language/state prefix token."""
    if prefix_hidden.shape[:2] != prefix_pad_masks.shape:
        raise ValueError(
            "Prefix hidden/mask shape mismatch: "
            f"{tuple(prefix_hidden.shape)} vs {tuple(prefix_pad_masks.shape)}"
        )
    counts = prefix_pad_masks.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("Every prefix sample must contain at least one valid token")
    masked_hidden = prefix_hidden * prefix_pad_masks.unsqueeze(-1)
    return masked_hidden.sum(dim=1) / counts.to(dtype=prefix_hidden.dtype)


def _clip_normalized_arm_chunk(chunk: Tensor, limit: float | None) -> Tensor:
    """Optionally bound a normalized arm chunk without truncating quantile tails."""
    if limit is None:
        return chunk
    return chunk.clamp(min=-limit, max=limit)


def _compute_gripper_head_loss(
    logits: Tensor,
    normalized_gripper_targets: Tensor,
    *,
    open_threshold_normalized: float,
    open_loss_weight: float,
    loss_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return normalized per-sample BCE, target-closed labels and weights."""
    if logits.shape != normalized_gripper_targets.shape:
        raise ValueError(
            "Gripper logits/target shape mismatch: "
            f"{tuple(logits.shape)} != {tuple(normalized_gripper_targets.shape)}"
        )
    target_closed = normalized_gripper_targets > open_threshold_normalized
    element_loss = F.binary_cross_entropy_with_logits(
        logits, target_closed.to(dtype=logits.dtype), reduction="none"
    )
    weights = torch.where(
        target_closed,
        torch.ones_like(element_loss),
        torch.full_like(element_loss, open_loss_weight),
    )
    per_sample = (element_loss * weights).sum(dim=1) / weights.sum(dim=1)
    return per_sample * loss_weight, target_closed, weights


def _summarize_release_observations(
    force: Tensor,
    force_fast: Tensor,
    force_slow: Tensor,
    state_history: Tensor,
    *,
    force_dim: int,
    state_dim: int,
) -> Tensor:
    """Build compact online sensor statistics for release classification."""
    expected = {
        "force": (force, 2, force_dim),
        "force_fast": (force_fast, 3, force_dim),
        "force_slow": (force_slow, 3, force_dim),
        "state_history": (state_history, 3, state_dim),
    }
    batch_size = force.shape[0]
    for name, (value, ndim, width) in expected.items():
        if value.ndim != ndim or value.shape[-1] != width:
            raise ValueError(f"{name} must have rank {ndim} and width {width}, got {tuple(value.shape)}")
        if value.shape[0] != batch_size:
            raise ValueError(f"{name} batch dimension does not match force")

    force_summary = torch.cat(
        [
            force,
            force_fast.mean(dim=1),
            force_fast.std(dim=1, unbiased=False),
            force_fast.abs().amax(dim=1),
            force_slow.mean(dim=1),
            force_slow.std(dim=1, unbiased=False),
        ],
        dim=-1,
    )
    state_summary = torch.cat(
        [
            state_history[:, -1],
            state_history.mean(dim=1),
            state_history.std(dim=1, unbiased=False),
            state_history[:, -1] - state_history[:, 0],
        ],
        dim=-1,
    )
    return torch.cat([force_summary, state_summary], dim=-1)


def _mask_state_history_channel(state_history: Tensor, channel: int) -> Tensor:
    """Mask one proprioceptive channel without mutating the caller's tensor."""
    if state_history.ndim != 3 or not 0 <= channel < state_history.shape[-1]:
        raise ValueError(
            f"Cannot mask state-history channel {channel} for shape {tuple(state_history.shape)}"
        )
    masked = state_history.clone()
    masked[..., channel] = 0
    return masked


class ReleaseGripperHead(nn.Module):
    """Detect release/retract while preserving the base flow everywhere else."""

    def __init__(self, visual_dim: int, sensor_dim: int, hidden_dim: int):
        super().__init__()
        self.visual = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
        )
        self.sensor = nn.Sequential(
            nn.LayerNorm(sensor_dim),
            nn.Linear(sensor_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, visual: Tensor, sensors: Tensor) -> Tensor:
        fused = torch.cat([self.visual(visual), self.sensor(sensors)], dim=-1)
        return self.classifier(fused).squeeze(-1)


class ContextualGripperHead(nn.Module):
    """Predict a deterministic gripper chunk from visual and measured progress."""

    def __init__(
        self,
        visual_dim: int,
        sensor_dim: int,
        hidden_dim: int,
        chunk_size: int,
        semantic_context_dim: int = 0,
    ):
        super().__init__()
        self.visual = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
        )
        self.sensor = nn.Sequential(
            nn.LayerNorm(sensor_dim),
            nn.Linear(sensor_dim, hidden_dim),
            nn.GELU(),
        )
        self.semantic = None
        if semantic_context_dim > 0:
            self.semantic = nn.Sequential(
                nn.LayerNorm(semantic_context_dim),
                nn.Linear(semantic_context_dim, hidden_dim, bias=False),
            )
            nn.init.zeros_(self.semantic[-1].weight)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size),
        )

    def forward(
        self, visual: Tensor, sensors: Tensor, semantic_context: Tensor | None = None
    ) -> Tensor:
        sensor_features = self.sensor(sensors)
        if self.semantic is not None:
            if semantic_context is None:
                raise ValueError("Semantic-conditioned gripper head requires semantic_context")
            sensor_features = sensor_features + self.semantic(semantic_context.float())
        elif semantic_context is not None:
            raise ValueError("semantic_context was provided to a gripper head without a semantic branch")
        fused = torch.cat([self.visual(visual), sensor_features], dim=-1)
        return self.classifier(fused)


class DeterministicArmHead(nn.Module):
    """Predict a normalized state-residual chunk for deterministic arm servoing."""

    def __init__(
        self,
        visual_dim: int,
        sensor_dim: int,
        hidden_dim: int,
        chunk_size: int,
        action_dim: int,
        residual_scale: float,
        semantic_context_dim: int = 0,
        replay_adapter_num_prototypes: int = 0,
        replay_adapter_feature_dim: int = 70,
        replay_adapter_bandwidth: float = 0.5,
        replay_adapter_strength: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.residual_scale = residual_scale
        self.replay_adapter_bandwidth = replay_adapter_bandwidth
        self.replay_adapter_strength = replay_adapter_strength
        self.semantic_context_dim = semantic_context_dim
        self.visual = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
        )
        self.sensor = nn.Sequential(
            nn.LayerNorm(sensor_dim),
            nn.Linear(sensor_dim, hidden_dim),
            nn.GELU(),
        )
        self.semantic = None
        if semantic_context_dim > 0:
            self.semantic = nn.Sequential(
                nn.LayerNorm(semantic_context_dim),
                nn.Linear(semantic_context_dim, hidden_dim, bias=False),
            )
            # Enabling semantic conditioning on an existing checkpoint starts
            # as an exact no-op, then learns only the useful correction.
            nn.init.zeros_(self.semantic[-1].weight)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )
        # A newly attached servo head must initially preserve the current arm
        # state instead of emitting a large random residual before training.
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)
        if replay_adapter_num_prototypes > 0:
            self.configure_replay_adapter(
                torch.zeros(replay_adapter_num_prototypes, replay_adapter_feature_dim),
                torch.zeros(replay_adapter_num_prototypes, chunk_size, action_dim),
                torch.ones(replay_adapter_feature_dim),
                bandwidth=replay_adapter_bandwidth,
                strength=replay_adapter_strength,
            )

    def configure_replay_adapter(
        self,
        prototypes: Tensor,
        corrections: Tensor,
        feature_scale: Tensor,
        *,
        bandwidth: float,
        strength: float,
    ) -> None:
        """Install a local replay correction without modifying shared head weights."""
        expected_correction_shape = (prototypes.shape[0], self.chunk_size, self.action_dim)
        if prototypes.ndim != 2 or prototypes.shape[1] != self.action_dim + 64:
            raise ValueError(
                "Replay prototypes must be [count, arm_action_dim + 64], got "
                f"{tuple(prototypes.shape)}"
            )
        if corrections.shape != expected_correction_shape:
            raise ValueError(
                f"Replay corrections must be {expected_correction_shape}, got {tuple(corrections.shape)}"
            )
        if feature_scale.shape != (prototypes.shape[1],):
            raise ValueError(
                f"Replay feature scale must be {(prototypes.shape[1],)}, got {tuple(feature_scale.shape)}"
            )
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Replay adapter bandwidth must be finite and positive")
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Replay adapter strength must be finite and in [0, 1]")
        values = {
            "replay_prototypes": prototypes.detach().to(dtype=torch.float32),
            "replay_corrections": corrections.detach().to(dtype=torch.float32),
            "replay_feature_scale": feature_scale.detach().to(dtype=torch.float32).clamp_min(1e-6),
        }
        for name, value in values.items():
            if name in self._buffers:
                self._buffers[name] = value
            else:
                self.register_buffer(name, value, persistent=True)
        self.replay_adapter_bandwidth = bandwidth
        self.replay_adapter_strength = strength

    def replay_adapter_similarity(self, sensors: Tensor, state: Tensor) -> Tensor:
        """Return the maximum local-kernel activation for each observation."""
        if "replay_prototypes" not in self._buffers:
            return torch.zeros(state.shape[0], device=state.device, dtype=torch.float32)
        features = torch.cat([state[:, : self.action_dim], sensors], dim=-1).to(dtype=torch.float32)
        if features.shape[-1] != self.replay_prototypes.shape[-1]:
            raise ValueError(
                "Replay feature width changed: "
                f"{features.shape[-1]} != {self.replay_prototypes.shape[-1]}"
            )
        normalized_delta = (
            features[:, None, :] - self.replay_prototypes[None, :, :]
        ) / self.replay_feature_scale[None, None, :]
        distance_squared = normalized_delta.square().mean(dim=-1)
        similarities = torch.exp(
            -0.5 * distance_squared / (self.replay_adapter_bandwidth**2)
        )
        return similarities.amax(dim=1)

    def _apply_replay_adapter(self, base: Tensor, sensors: Tensor, state: Tensor) -> Tensor:
        if "replay_prototypes" not in self._buffers:
            return base
        features = torch.cat([state[:, : self.action_dim], sensors], dim=-1).to(dtype=torch.float32)
        normalized_delta = (
            features[:, None, :] - self.replay_prototypes[None, :, :]
        ) / self.replay_feature_scale[None, None, :]
        distance_squared = normalized_delta.square().mean(dim=-1)
        similarities = torch.exp(
            -0.5 * distance_squared / (self.replay_adapter_bandwidth**2)
        )
        normalized_weights = similarities / similarities.sum(dim=1, keepdim=True).clamp_min(1e-12)
        correction = torch.einsum("bp,ptd->btd", normalized_weights, self.replay_corrections)
        gate = similarities.amax(dim=1)[:, None, None]
        return base + self.replay_adapter_strength * gate * correction

    def forward(
        self,
        visual: Tensor,
        sensors: Tensor,
        state: Tensor,
        semantic_context: Tensor | None = None,
    ) -> Tensor:
        if state.ndim != 2 or state.shape[-1] < self.action_dim:
            raise ValueError(
                "Arm-head state must be [batch, state_dim] with at least "
                f"{self.action_dim} values, got {tuple(state.shape)}"
            )
        visual_features = self.visual(visual)
        if self.semantic is not None:
            if semantic_context is None:
                raise ValueError("Semantic-conditioned arm head requires semantic_context")
            if semantic_context.shape != (state.shape[0], self.semantic_context_dim):
                raise ValueError(
                    "Arm-head semantic context must be "
                    f"[batch, {self.semantic_context_dim}], got {tuple(semantic_context.shape)}"
                )
            visual_features = visual_features + self.semantic(
                semantic_context.to(dtype=visual_features.dtype)
            )
        fused = torch.cat([visual_features, self.sensor(sensors)], dim=-1)
        residual = self.regressor(fused).reshape(state.shape[0], self.chunk_size, self.action_dim)
        base = state[:, None, : self.action_dim] + self.residual_scale * residual
        return self._apply_replay_adapter(base, sensors, state)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # Construct the two large backbones in their requested final base
        # precision.  Building all parameters in float32 and only then calling
        # ``to(bfloat16)`` peaks around 40 GB of host RAM for Pi0.5 and can be
        # killed before checkpoint loading on 48 GB workstations.  The selected
        # vision/norm parameters are still promoted to float32 immediately
        # below, so the final parameter-dtype contract is unchanged.
        previous_default_dtype = torch.get_default_dtype()
        if precision == "bfloat16":
            torch.set_default_dtype(torch.bfloat16)
        try:
            self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
            self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        finally:
            torch.set_default_dtype(previous_default_dtype)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        # The action path consumes ``gemma_expert.model`` hidden states directly
        # and projects them with ``action_out_proj``.  The inherited causal-LM
        # vocabulary head is therefore never called and never receives a
        # gradient.  Freeze it explicitly so trainable-parameter/optimizer
        # accounting reflects the effective action backbone.
        for param in self.gemma_expert.lm_head.parameters():
            param.requires_grad = False
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            # PEFT ``modules_to_save`` normally wraps a module with a forward
            # signature that requires one positional tensor. GemmaModel is
            # called here with keyword-only ``inputs_embeds`` instead. Select
            # the active saved copy explicitly so a hybrid visual-LoRA/full-
            # expert checkpoint has the same inference interface before and
            # after serialization.
            expert_model = self.gemma_expert.model
            saved_modules = getattr(expert_model, "modules_to_save", None)
            active_adapters = getattr(expert_model, "active_adapters", [])
            if saved_modules is not None and active_adapters:
                expert_model = saved_modules[active_adapters[0]]
            suffix_output = expert_model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.gripper_head: nn.Module | None = None
        if config.use_deterministic_gripper_head:
            if config.gripper_head_use_sensor_context:
                sensor_dim = config.release_force_dim * 6 + config.release_state_dim * 4
                self.gripper_head = ContextualGripperHead(
                    visual_dim=paligemma_config.width,
                    sensor_dim=sensor_dim,
                    hidden_dim=config.gripper_head_hidden_dim,
                    chunk_size=config.chunk_size,
                    semantic_context_dim=config.gripper_head_semantic_context_dim,
                )
            else:
                self.gripper_head = nn.Sequential(
                    nn.LayerNorm(paligemma_config.width),
                    nn.Linear(paligemma_config.width, config.gripper_head_hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.gripper_head_hidden_dim, config.chunk_size),
                )

        self.release_head: ReleaseGripperHead | None = None
        if config.use_release_gripper_override:
            sensor_dim = config.release_force_dim * 6 + config.release_state_dim * 4
            self.release_head = ReleaseGripperHead(
                visual_dim=paligemma_config.width,
                sensor_dim=sensor_dim,
                hidden_dim=config.release_head_hidden_dim,
            )

        self.arm_head: DeterministicArmHead | None = None
        if config.use_deterministic_arm_head:
            sensor_dim = config.release_force_dim * 6 + config.release_state_dim * 4
            self.arm_head = DeterministicArmHead(
                visual_dim=paligemma_config.width,
                sensor_dim=sensor_dim,
                hidden_dim=config.arm_head_hidden_dim,
                chunk_size=config.chunk_size,
                action_dim=config.arm_head_action_dim,
                residual_scale=config.arm_head_residual_scale,
                semantic_context_dim=config.arm_head_semantic_context_dim,
                replay_adapter_num_prototypes=config.arm_replay_adapter_num_prototypes,
                replay_adapter_feature_dim=config.arm_replay_adapter_feature_dim,
                replay_adapter_bandwidth=config.arm_replay_adapter_bandwidth,
                replay_adapter_strength=config.arm_replay_adapter_strength,
            )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _gripper_logits_from_prefix(
        self,
        prefix_output: Tensor,
        prefix_pad_masks: Tensor,
        force: Tensor | None = None,
        force_fast: Tensor | None = None,
        force_slow: Tensor | None = None,
        state_history: Tensor | None = None,
        semantic_context: Tensor | None = None,
    ) -> Tensor:
        if self.gripper_head is None:
            raise RuntimeError("Deterministic gripper head is not enabled")
        if self.config.gripper_head_pooling == "mean":
            pooled_prefix = _pool_mean_valid_prefix(prefix_output, prefix_pad_masks)
        else:
            pooled_prefix = _pool_last_valid_prefix(prefix_output, prefix_pad_masks)
        if self.config.gripper_head_use_sensor_context:
            sensor_inputs = {
                "force": force,
                "force_fast": force_fast,
                "force_slow": force_slow,
                "state_history": state_history,
            }
            missing = [name for name, value in sensor_inputs.items() if value is None]
            if missing:
                raise ValueError(f"Contextual gripper head requires sensor inputs; missing={missing}")
            if self.config.gripper_head_mask_state_gripper:
                state_history = _mask_state_history_channel(
                    state_history, self.config.gripper_action_index
                )
            sensor_summary = _summarize_release_observations(
                force,
                force_fast,
                force_slow,
                state_history,
                force_dim=self.config.release_force_dim,
                state_dim=self.config.release_state_dim,
            )
            return self.gripper_head(
                pooled_prefix.to(dtype=torch.float32),
                sensor_summary.to(dtype=torch.float32),
                semantic_context,
            )
        if semantic_context is not None:
            raise ValueError("Semantic gripper context requires gripper_head_use_sensor_context")
        return self.gripper_head(pooled_prefix.to(dtype=torch.float32))

    def predict_gripper_logits(
        self,
        images,
        img_masks,
        tokens,
        masks,
        force: Tensor | None = None,
        force_fast: Tensor | None = None,
        force_slow: Tensor | None = None,
        state_history: Tensor | None = None,
    ) -> Tensor:
        """Predict a deterministic gripper chunk without running action flow."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (  # noqa: SLF001
            "eager"
        )
        prefix_outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_att_2d_masks),
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        return self._gripper_logits_from_prefix(
            prefix_outputs[0],
            prefix_pad_masks,
            force,
            force_fast,
            force_slow,
            state_history,
        )

    def _release_logits_from_prefix(
        self,
        prefix_output: Tensor,
        prefix_pad_masks: Tensor,
        force: Tensor,
        force_fast: Tensor,
        force_slow: Tensor,
        state_history: Tensor,
    ) -> Tensor:
        if self.release_head is None:
            raise RuntimeError("Release gripper override is not enabled")
        pooled_prefix = _pool_mean_valid_prefix(prefix_output, prefix_pad_masks)
        sensor_summary = _summarize_release_observations(
            force,
            force_fast,
            force_slow,
            state_history,
            force_dim=self.config.release_force_dim,
            state_dim=self.config.release_state_dim,
        )
        return self.release_head(
            pooled_prefix.to(dtype=torch.float32),
            sensor_summary.to(dtype=torch.float32),
        )

    def predict_release_logits(
        self,
        images,
        img_masks,
        tokens,
        masks,
        force: Tensor,
        force_fast: Tensor,
        force_slow: Tensor,
        state_history: Tensor,
    ) -> Tensor:
        """Predict release/retract from one prefix and online sensor snapshot."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (  # noqa: SLF001
            "eager"
        )
        prefix_outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_att_2d_masks),
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        return self._release_logits_from_prefix(
            prefix_outputs[0],
            prefix_pad_masks,
            force,
            force_fast,
            force_slow,
            state_history,
        )

    def _arm_chunk_from_prefix(
        self,
        prefix_output: Tensor,
        prefix_pad_masks: Tensor,
        state: Tensor,
        force: Tensor,
        force_fast: Tensor,
        force_slow: Tensor,
        state_history: Tensor,
        semantic_context: Tensor | None = None,
    ) -> Tensor:
        if self.arm_head is None:
            raise RuntimeError("Deterministic arm head is not enabled")
        pooled_prefix = _pool_mean_valid_prefix(prefix_output, prefix_pad_masks)
        sensor_summary = _summarize_release_observations(
            force,
            force_fast,
            force_slow,
            state_history,
            force_dim=self.config.release_force_dim,
            state_dim=self.config.release_state_dim,
        )
        return self.arm_head(
            pooled_prefix.to(dtype=torch.float32),
            sensor_summary.to(dtype=torch.float32),
            state.to(dtype=torch.float32),
            semantic_context,
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
        """Predict a deterministic normalized arm chunk without action flow."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (  # noqa: SLF001
            "eager"
        )
        prefix_outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_att_2d_masks),
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        return self._arm_chunk_from_prefix(
            prefix_outputs[0],
            prefix_pad_masks,
            state,
            force,
            force_fast,
            force_slow,
            state_history,
        )

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        prefix_outputs, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        gripper_logits = None
        if self.gripper_head is not None:
            required_gripper_inputs = {
                "gripper_force": kwargs.get("gripper_force"),
                "gripper_force_fast": kwargs.get("gripper_force_fast"),
                "gripper_force_slow": kwargs.get("gripper_force_slow"),
                "gripper_state_history": kwargs.get("gripper_state_history"),
            }
            if self.config.gripper_head_use_sensor_context:
                missing = [name for name, value in required_gripper_inputs.items() if value is None]
                if missing:
                    raise ValueError(f"Contextual gripper head requires online sensor inputs; missing={missing}")
            gripper_logits = self._gripper_logits_from_prefix(
                prefix_outputs[0],
                prefix_pad_masks,
                required_gripper_inputs["gripper_force"],
                required_gripper_inputs["gripper_force_fast"],
                required_gripper_inputs["gripper_force_slow"],
                required_gripper_inputs["gripper_state_history"],
            )
        release_logits = None
        if self.release_head is not None:
            required_release_inputs = {
                "release_force": kwargs.get("release_force"),
                "release_force_fast": kwargs.get("release_force_fast"),
                "release_force_slow": kwargs.get("release_force_slow"),
                "release_state_history": kwargs.get("release_state_history"),
            }
            missing = [name for name, value in required_release_inputs.items() if value is None]
            if missing:
                raise ValueError(f"Release gripper override requires online sensor inputs; missing={missing}")
            release_logits = self._release_logits_from_prefix(
                prefix_outputs[0],
                prefix_pad_masks,
                required_release_inputs["release_force"],
                required_release_inputs["release_force_fast"],
                required_release_inputs["release_force_slow"],
                required_release_inputs["release_state_history"],
            )
        arm_chunk = None
        if self.arm_head is not None:
            required_arm_inputs = {
                "arm_state": kwargs.get("arm_state"),
                "arm_force": kwargs.get("arm_force"),
                "arm_force_fast": kwargs.get("arm_force_fast"),
                "arm_force_slow": kwargs.get("arm_force_slow"),
                "arm_state_history": kwargs.get("arm_state_history"),
            }
            missing = [name for name, value in required_arm_inputs.items() if value is None]
            if missing:
                raise ValueError(f"Deterministic arm head requires online sensor inputs; missing={missing}")
            arm_chunk = self._arm_chunk_from_prefix(
                prefix_outputs[0],
                prefix_pad_masks,
                required_arm_inputs["arm_state"],
                required_arm_inputs["arm_force"],
                required_arm_inputs["arm_force_fast"],
                required_arm_inputs["arm_force_slow"],
                required_arm_inputs["arm_state_history"],
            )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")
                rtc_action_mask = kwargs.get("rtc_action_mask")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                    action_mask=rtc_action_mask,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        if arm_chunk is not None:
            arm_dim = self.config.arm_head_action_dim
            x_t[..., :arm_dim] = _clip_normalized_arm_chunk(
                arm_chunk, self.config.arm_head_normalized_output_clip
            )

        if gripper_logits is not None:
            gripper_closed = torch.sigmoid(gripper_logits) >= self.config.gripper_head_probability_threshold
            if self.config.gripper_head_hold_first_action:
                gripper_closed = gripper_closed[:, :1].expand_as(gripper_closed)
            gripper_values = torch.where(
                gripper_closed,
                torch.full_like(
                    gripper_logits,
                    self.config.gripper_head_closed_normalized_value,
                ),
                torch.full_like(
                    gripper_logits,
                    self.config.gripper_head_open_normalized_value,
                ),
            )
            x_t[..., self.config.gripper_action_index] = gripper_values

        if release_logits is not None:
            release_detected = torch.sigmoid(release_logits) >= self.config.release_head_probability_threshold
            current_gripper = x_t[..., self.config.gripper_action_index]
            x_t[..., self.config.gripper_action_index] = torch.where(
                release_detected[:, None],
                torch.full_like(
                    current_gripper,
                    self.config.release_head_open_normalized_value,
                ),
                current_gripper,
            )

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        if config.train_gripper_head_only:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            if self.model.gripper_head is None:
                raise RuntimeError("train_gripper_head_only requires an initialized gripper head")
            for parameter in self.model.gripper_head.parameters():
                parameter.requires_grad = True
            logging.info("Training only the deterministic Pi0.5 gripper head")
        elif config.train_release_head_only:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            if self.model.release_head is None:
                raise RuntimeError("train_release_head_only requires an initialized release head")
            for parameter in self.model.release_head.parameters():
                parameter.requires_grad = True
            logging.info("Training only the Pi0.5 release gripper override head")
        elif config.train_arm_head_only:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            if self.model.arm_head is None:
                raise RuntimeError("train_arm_head_only requires an initialized deterministic arm head")
            for parameter in self.model.arm_head.parameters():
                parameter.requires_grad = True
            logging.info("Training only the deterministic Pi0.5 arm servo head")

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Native LeRobot checkpoints saved by ``PreTrainedPolicy`` already use
        # the exact in-memory parameter names.  Load those tensors directly
        # into the model instead of materializing three 8.8 GB dictionaries
        # (original/fixed/remapped).  Besides being equivalent under strict
        # loading, this avoids a ~40 GB CPU-memory peak during evaluation.
        local_model_file = Path(pretrained_name_or_path) / "model.safetensors"
        # Direct device loading is useful on CPU, but on a 16 GB CUDA device
        # safetensors needs a transient device tensor and can fail near the
        # model's capacity.  Falling back after such a failure leaves too
        # little cache for warmup, so CUDA keeps the proven CPU-load/copy path.
        if local_model_file.is_file() and str(config.device).startswith("cpu"):
            try:
                from safetensors.torch import load_model

                missing_keys, unexpected_keys = load_model(
                    model,
                    str(local_model_file),
                    strict=strict,
                    device=str(config.device),
                )
                if not missing_keys and not unexpected_keys:
                    print("✓ Direct-loaded exact native LeRobot safetensors")
                    print("All keys loaded successfully!")
                    return model
            except (RuntimeError, TypeError, ValueError) as direct_error:
                print(
                    "Direct native load was not exact; falling back to legacy "
                    f"Pi0.5 key remapping: {direct_error}"
                )

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model.") and key != "_step":
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Auxiliary gripper heads can be introduced on a legacy action
            # checkpoint. In those controlled cases, load every shared tensor
            # strictly and leave only the explicitly new head initialized.
            auxiliary_head_prefixes = []
            for enabled, prefix in (
                (model.config.use_deterministic_gripper_head, "model.gripper_head."),
                (model.config.use_release_gripper_override, "model.release_head."),
                (model.config.use_deterministic_arm_head, "model.arm_head."),
            ):
                if enabled and not any(key.startswith(prefix) for key in remapped_state_dict):
                    auxiliary_head_prefixes.append(prefix)
            missing_keys, unexpected_keys = model.load_state_dict(
                remapped_state_dict,
                strict=strict and not auxiliary_head_prefixes,
            )
            if auxiliary_head_prefixes and strict:
                disallowed_missing = [
                    key
                    for key in missing_keys
                    if not any(key.startswith(prefix) for prefix in auxiliary_head_prefixes)
                ]
                if disallowed_missing or unexpected_keys:
                    raise RuntimeError(
                        "Only explicitly introduced auxiliary-head "
                        "parameters may be missing when upgrading a legacy checkpoint; "
                        f"missing={disallowed_missing}, unexpected={unexpected_keys}"
                    )

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            if strict:
                raise RuntimeError(
                    "Failed to load the requested PI0.5 pretrained weights. "
                    "Continuing with a randomly initialized model would invalidate "
                    "fine-tuning and evaluation."
                ) from e
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Transformers has used both ``vision_tower.<...>`` and
            # ``vision_tower.vision_model.<...>`` for the same SigLIP module.
            # Select the spelling used by the model instantiated in the
            # current environment. Blindly removing ``vision_model`` can
            # silently leave the whole vision tower randomly initialized
            # when loading with ``strict=False``.
            expected_keys = self.state_dict().keys()

            def is_expected(candidate: str) -> bool:
                """Account for the outer PI05Policy ``model.`` namespace."""
                return candidate in expected_keys or (
                    not candidate.startswith("model.")
                    and f"model.{candidate}" in expected_keys
                )

            if ".vision_tower.vision_model." in new_key:
                legacy_key = new_key.replace(
                    ".vision_tower.vision_model.", ".vision_tower."
                )
                if not is_expected(new_key) and is_expected(legacy_key):
                    new_key = legacy_key
            elif ".vision_tower." in new_key:
                wrapped_key = new_key.replace(
                    ".vision_tower.", ".vision_tower.vision_model.", 1
                )
                if not is_expected(new_key) and is_expected(wrapped_key):
                    new_key = wrapped_key

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _controlled_sensor(self, batch: dict[str, Tensor], key: str) -> Tensor | None:
        value = batch.get(key)
        if value is not None and self.config.controlled_ablation_sensor_mode == "vs":
            return torch.zeros_like(value)
        return value

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        if self.config.use_release_gripper_override:
            release_inputs = {
                "release_force": self._controlled_sensor(batch, OBS_FORCE),
                "release_force_fast": self._controlled_sensor(batch, OBS_FORCE_FAST),
                "release_force_slow": self._controlled_sensor(batch, OBS_FORCE_SLOW),
                "release_state_history": self._controlled_sensor(batch, OBS_STATE_HISTORY),
            }
            missing = [name for name, value in release_inputs.items() if value is None]
            if missing:
                raise ValueError(
                    f"Release gripper override checkpoint requires sensor observations; missing={missing}"
                )
            kwargs = {**kwargs, **release_inputs}
        if self.config.use_deterministic_gripper_head and self.config.gripper_head_use_sensor_context:
            gripper_inputs = {
                "gripper_force": self._controlled_sensor(batch, OBS_FORCE),
                "gripper_force_fast": self._controlled_sensor(batch, OBS_FORCE_FAST),
                "gripper_force_slow": self._controlled_sensor(batch, OBS_FORCE_SLOW),
                "gripper_state_history": self._controlled_sensor(batch, OBS_STATE_HISTORY),
            }
            missing = [name for name, value in gripper_inputs.items() if value is None]
            if missing:
                raise ValueError(
                    f"Contextual gripper-head checkpoint requires sensor observations; missing={missing}"
                )
            kwargs = {**kwargs, **gripper_inputs}
        if self.config.use_deterministic_arm_head:
            arm_inputs = {
                "arm_state": batch.get(OBS_STATE),
                "arm_force": self._controlled_sensor(batch, OBS_FORCE),
                "arm_force_fast": self._controlled_sensor(batch, OBS_FORCE_FAST),
                "arm_force_slow": self._controlled_sensor(batch, OBS_FORCE_SLOW),
                "arm_state_history": self._controlled_sensor(batch, OBS_STATE_HISTORY),
            }
            missing = [name for name, value in arm_inputs.items() if value is None]
            if missing:
                raise ValueError(
                    f"Deterministic arm-head checkpoint requires sensor observations; missing={missing}"
                )
            kwargs = {**kwargs, **arm_inputs}
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        if self.config.train_arm_head_only:
            arm_dim = self.config.arm_head_action_dim
            arm_state = batch[OBS_STATE]
            arm_history = self._controlled_sensor(batch, OBS_STATE_HISTORY)
            assert arm_history is not None
            noise_std = self.config.arm_head_state_noise_std
            history_noise_std = self.config.arm_head_history_noise_std
            force_noise_std = self.config.arm_head_force_noise_std
            if noise_std > 0:
                state_offset = torch.randn_like(arm_state[:, :arm_dim]) * noise_std
                arm_state = arm_state.clone()
                arm_state[:, :arm_dim] = _clip_normalized_arm_chunk(
                    arm_state[:, :arm_dim] + state_offset,
                    self.config.arm_head_normalized_output_clip,
                )
                arm_history = arm_history.clone()
                arm_history[..., :arm_dim] = _clip_normalized_arm_chunk(
                    arm_history[..., :arm_dim] + state_offset[:, None, :],
                    self.config.arm_head_normalized_output_clip,
                )
            if history_noise_std > 0:
                if noise_std <= 0:
                    arm_history = arm_history.clone()
                history_offset = torch.randn_like(arm_history[..., :arm_dim]) * history_noise_std
                arm_history[..., :arm_dim] = _clip_normalized_arm_chunk(
                    arm_history[..., :arm_dim] + history_offset,
                    self.config.arm_head_normalized_output_clip,
                )
            force = self._controlled_sensor(batch, OBS_FORCE)
            force_fast = self._controlled_sensor(batch, OBS_FORCE_FAST)
            force_slow = self._controlled_sensor(batch, OBS_FORCE_SLOW)
            assert force is not None and force_fast is not None and force_slow is not None
            if force_noise_std > 0:
                force = force + torch.randn_like(force) * force_noise_std
                force_fast = force_fast + torch.randn_like(force_fast) * force_noise_std
                force_slow = force_slow + torch.randn_like(force_slow) * force_noise_std
            predicted_arm = self.model.predict_arm_chunk(
                images,
                img_masks,
                tokens,
                masks,
                arm_state,
                force,
                force_fast,
                force_slow,
                arm_history,
            )
            target_arm = actions[..., :arm_dim]
            training_horizon = self.config.arm_head_training_horizon or self.config.chunk_size
            predicted_arm = predicted_arm[:, :training_horizon]
            target_arm = target_arm[:, :training_horizon]
            element_loss = (
                F.smooth_l1_loss(
                    predicted_arm,
                    target_arm,
                    reduction="none",
                    beta=0.1,
                ).mean(dim=(1, 2))
                * self.config.arm_head_loss_weight
            )
            loss = element_loss.mean()
            loss_dict = {
                "loss": loss.item(),
                "arm_head_loss": loss.item(),
                "arm_head_mae_normalized": (predicted_arm - target_arm).abs().mean().item(),
                "arm_head_state_noise_std": noise_std,
                "arm_head_history_noise_std": history_noise_std,
                "arm_head_force_noise_std": force_noise_std,
                "arm_head_training_horizon": training_horizon,
            }
            if reduction == "none":
                return element_loss, loss_dict
            return loss, loss_dict

        if self.config.train_release_head_only:
            labels = batch.get(SUBTASK_LABEL_KEY)
            if labels is None:
                raise ValueError(f"{SUBTASK_LABEL_KEY!r} is required for release-head training")
            if isinstance(labels, str):
                labels = [labels]
            if len(labels) != actions.shape[0]:
                raise ValueError(
                    "Release label batch size does not match action batch: "
                    f"{len(labels)} != {actions.shape[0]}"
                )
            release_logits = self.model.predict_release_logits(
                images,
                img_masks,
                tokens,
                masks,
                self._controlled_sensor(batch, OBS_FORCE),
                self._controlled_sensor(batch, OBS_FORCE_FAST),
                self._controlled_sensor(batch, OBS_FORCE_SLOW),
                self._controlled_sensor(batch, OBS_STATE_HISTORY),
            )
            open_names = set(self.config.release_open_subtask_names)
            release_targets = torch.tensor(
                [str(label).strip() in open_names for label in labels],
                dtype=release_logits.dtype,
                device=release_logits.device,
            )
            element_loss = F.binary_cross_entropy_with_logits(
                release_logits,
                release_targets,
                reduction="none",
                pos_weight=torch.as_tensor(
                    self.config.release_head_positive_weight,
                    dtype=release_logits.dtype,
                    device=release_logits.device,
                ),
            )
            predicted_release = (
                torch.sigmoid(release_logits) >= self.config.release_head_probability_threshold
            )
            loss = element_loss.mean()
            loss_dict = {
                "loss": loss.item(),
                "release_head_loss": loss.item(),
                "release_head_accuracy": (predicted_release == release_targets.bool()).float().mean().item(),
                "release_head_positive_fraction": release_targets.mean().item(),
                "release_head_mean_probability": torch.sigmoid(release_logits).mean().item(),
            }
            if reduction == "none":
                return element_loss, loss_dict
            return loss, loss_dict

        if self.config.train_gripper_head_only:
            original_action_dim = self.config.output_features[ACTION].shape[0]
            gripper_index = self.config.gripper_action_index
            if gripper_index is None or not 0 <= gripper_index < original_action_dim:
                raise ValueError(
                    "gripper_action_index must reference a real action dimension during gripper-head training"
                )
            gripper_logits = self.model.predict_gripper_logits(
                images,
                img_masks,
                tokens,
                masks,
                self._controlled_sensor(batch, OBS_FORCE),
                self._controlled_sensor(batch, OBS_FORCE_FAST),
                self._controlled_sensor(batch, OBS_FORCE_SLOW),
                self._controlled_sensor(batch, OBS_STATE_HISTORY),
            )
            normalized_targets = actions[:, :, gripper_index]
            training_horizon = self.config.gripper_head_training_horizon or self.config.n_action_steps
            gripper_logits = gripper_logits[:, :training_horizon]
            normalized_targets = normalized_targets[:, :training_horizon]
            per_sample_loss, target_closed, loss_weights = _compute_gripper_head_loss(
                gripper_logits,
                normalized_targets,
                open_threshold_normalized=self.config.gripper_open_threshold_normalized,
                open_loss_weight=self.config.gripper_head_open_loss_weight,
                loss_weight=self.config.gripper_head_loss_weight,
            )
            predicted_closed = torch.sigmoid(gripper_logits) >= self.config.gripper_head_probability_threshold
            loss = per_sample_loss.mean()
            loss_dict = {
                "loss": loss.item(),
                "gripper_head_loss": loss.item(),
                "gripper_head_accuracy": (predicted_closed == target_closed).float().mean().item(),
                "gripper_head_open_target_fraction": (~target_closed).float().mean().item(),
                "gripper_head_mean_effective_weight": loss_weights.mean().item(),
            }
            if reduction == "none":
                return per_sample_loss, loss_dict
            return loss, loss_dict

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        normalized_actions = actions[:, :, :original_action_dim]
        per_sample_loss, loss_weights = _compute_weighted_action_loss(
            losses,
            normalized_actions,
            gripper_action_index=self.config.gripper_action_index,
            gripper_loss_weight=self.config.gripper_loss_weight,
            gripper_open_loss_weight=self.config.gripper_open_loss_weight,
            gripper_open_threshold_normalized=self.config.gripper_open_threshold_normalized,
            action_prefix_loss_horizon=self.config.action_prefix_loss_horizon,
            action_prefix_loss_weight=self.config.action_prefix_loss_weight,
        )

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "weighted_loss_per_dim": (losses * loss_weights).mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "effective_weight_per_dim": loss_weights.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "effective_weight_per_timestep": loss_weights.mean(dim=[0, 2]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = per_sample_loss.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

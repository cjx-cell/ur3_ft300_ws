#!/usr/bin/env python
"""PAP-MoE v6 modules: multi-rate sensing and heterogeneous perceptual experts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvEncoder(nn.Module):
    """Encode a [B, T, C] signal with a small dilated temporal CNN."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError(f"Expected [B,T,C], got {tuple(signal.shape)}")
        x = self.net(signal.transpose(1, 2))
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.out(pooled)


class MultiScaleForceEncoder(nn.Module):
    """Encode current, fast-window, and slow-window 6D force/torque signals.

    Inputs are expected to be normalized exactly once by the policy processor.
    """

    def __init__(self, force_dim: int = 6, hidden_dim: int = 256, output_dim: int = 2048):
        super().__init__()
        self.force_dim = force_dim
        self.current = nn.Sequential(
            nn.LayerNorm(force_dim),
            nn.Linear(force_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.fast = TemporalConvEncoder(force_dim, hidden_dim, output_dim)
        self.slow_gru = nn.GRU(force_dim, hidden_dim, batch_first=True)
        self.slow_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.fused = nn.Sequential(
            nn.LayerNorm(output_dim * 3),
            nn.Linear(output_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for module in (self.current[-1], self.fast.out[-1], self.slow_out[-1], self.fused[-1]):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        current: torch.Tensor,
        fast_window: torch.Tensor | None = None,
        slow_window: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if current.ndim != 2 or current.shape[-1] != self.force_dim:
            raise ValueError(f"Expected current force [B,{self.force_dim}], got {tuple(current.shape)}")
        if fast_window is None:
            fast_window = current.unsqueeze(1)
        if slow_window is None:
            slow_window = current.unsqueeze(1)

        current_token = self.current(current)
        fast_token = self.fast(fast_window)
        _, slow_hidden = self.slow_gru(slow_window)
        slow_token = self.slow_out(slow_hidden[-1])
        fused_token = self.fused(torch.cat([current_token, fast_token, slow_token], dim=-1))
        return {
            "current": current_token,
            "fast": fast_token,
            "slow": slow_token,
            "fused": fused_token,
        }


class ProprioHistoryEncoder(nn.Module):
    """Encode joint/state history without assuming a fixed history length."""

    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

    def forward(
        self,
        current_state: torch.Tensor,
        state_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current_state = current_state[..., : self.state_dim]
        if state_history is None:
            state_history = current_state.unsqueeze(1)
        else:
            state_history = state_history[..., : self.state_dim]
        _, hidden = self.gru(state_history)
        return self.out(hidden[-1])


class LegacyPhysicsGate(nn.Module):
    """Explicit multi-modal gate; never infers token roles from sequence indices."""

    def __init__(
        self,
        d_model: int = 2048,
        num_experts: int = 4,
        hidden_dim: int = 256,
        num_queries: int = 4,
        nhead: int = 8,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.token_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU())
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.context_head = nn.Sequential(
            nn.LayerNorm(num_queries * hidden_dim),
            nn.Linear(num_queries * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.fast_force_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        fast_force_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context_tokens.ndim != 3:
            raise ValueError("context_tokens must have shape [B,N,D]")
        if fast_force_token.ndim != 2:
            raise ValueError("fast_force_token must have shape [B,D]")
        batch_size = context_tokens.shape[0]
        tokens = self.token_proj(context_tokens)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        pooled, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        logits = self.context_head(pooled.flatten(1)) + self.fast_force_head(fast_force_token)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities, logits, logits.argmax(dim=-1)


class FactorizedPhysicsGate(nn.Module):
    """Predict task-agnostic physical factors before four-expert routing.

    The factors are visual blindness ``b``, physical contact ``c``, and
    mobility/compliance under contact ``m``. The resulting expert weights are:

    ``E1=(1-b)(1-c), E2=b, E3=c(1-m), E4=cm`` (then normalized).

    E2 therefore remains able to cooperate with E3/E4 when vision is invalid
    during contact instead of competing in a flat mutually-exclusive class.
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_experts: int = 4,
        hidden_dim: int = 256,
        num_queries: int = 4,
        nhead: int = 8,
    ):
        super().__init__()
        if num_experts != 4:
            raise ValueError("FactorizedPhysicsGate requires exactly four physical experts")
        self.num_experts = num_experts
        self.token_proj = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU()
        )
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.context_head = nn.Sequential(
            nn.LayerNorm(num_queries * hidden_dim),
            nn.Linear(num_queries * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.fast_force_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.last_factor_probs: torch.Tensor | None = None
        self.last_factor_logits: torch.Tensor | None = None

    @staticmethod
    def factors_to_expert_probs(factors: torch.Tensor) -> torch.Tensor:
        if factors.shape[-1] != 3:
            raise ValueError(f"Expected [...,3] b/c/m factors, got {tuple(factors.shape)}")
        blindness, contact, mobility = factors.unbind(dim=-1)
        raw = torch.stack(
            [
                (1.0 - blindness) * (1.0 - contact),
                blindness,
                contact * (1.0 - mobility),
                contact * mobility,
            ],
            dim=-1,
        )
        return raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def expert_probs_to_factors(probabilities: torch.Tensor, *, stable: bool = True) -> torch.Tensor:
        """Invert normalized E1--E4 targets into b/c/m supervision.

        For x=E2 and y=E3+E4, the forward normalization is
        x=b/(1+bc), y=c/(1+bc). Solving z=1+bc gives
        ``xy*z^2-z+1=0``. The smaller root is the physical solution z in [1,2].
        """
        if probabilities.shape[-1] != 4:
            raise ValueError(
                f"Expected [...,4] expert probabilities, got {tuple(probabilities.shape)}"
            )
        probabilities = probabilities.float().clamp_min(0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        visual_blind = probabilities[..., 1]
        contact_mass = probabilities[..., 2] + probabilities[..., 3]
        product = visual_blind * contact_mass
        discriminant = (1.0 - 4.0 * product).clamp_min(0.0)
        if stable:
            # Rationalize the smaller quadratic root. Subtracting nearly equal
            # float32 values corrupts contact targets when blindness is tiny.
            z = 2.0 / (1.0 + discriminant.sqrt())
        else:
            # Explicit legacy path for reproducing historical supervision.
            denominator = 2.0 * product
            root = (1.0 - discriminant.sqrt()) / denominator.clamp_min(1e-8)
            z = torch.where(product > 1e-8, root, torch.ones_like(root))
        blindness = (visual_blind * z).clamp(0.0, 1.0)
        contact = (contact_mass * z).clamp(0.0, 1.0)
        mobility = torch.where(
            contact_mass > 1e-8,
            probabilities[..., 3] / contact_mass.clamp_min(1e-8),
            torch.zeros_like(contact_mass),
        ).clamp(0.0, 1.0)
        return torch.stack([blindness, contact, mobility], dim=-1)

    def forward(
        self,
        context_tokens: torch.Tensor,
        fast_force_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context_tokens.ndim != 3:
            raise ValueError("context_tokens must have shape [B,N,D]")
        if fast_force_token.ndim != 2:
            raise ValueError("fast_force_token must have shape [B,D]")
        batch_size = context_tokens.shape[0]
        tokens = self.token_proj(context_tokens)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        pooled, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        factor_logits = self.context_head(pooled.flatten(1)) + self.fast_force_head(
            fast_force_token
        )
        factor_probs = torch.sigmoid(factor_logits)
        probabilities = self.factors_to_expert_probs(factor_probs)
        # Existing soft-target loss consumes four logits. log(probability) is
        # an exact compatible representation because log_softmax(log(p))=log(p).
        expert_logits = probabilities.clamp_min(1e-8).log()
        self.last_factor_probs = factor_probs
        self.last_factor_logits = factor_logits
        return probabilities, expert_logits, probabilities.argmax(dim=-1)


class FactorizedRouteForecaster(nn.Module):
    """Forecast b/c/m factors for each token in the next action chunk.

    The current PhysicsGate prediction is kept exactly at token zero. Learned
    time queries forecast the remaining tokens from the same observation
    context, so inference needs no future sensor leakage.
    """

    def __init__(self, d_model: int, hidden_dim: int, horizon: int, nhead: int = 8):
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.token_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU())
        self.time_queries = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.factor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self, context_tokens: torch.Tensor, current_factor_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_tokens.ndim != 3:
            raise ValueError("context_tokens must have shape [B,N,D]")
        if current_factor_probs.shape != (context_tokens.shape[0], 3):
            raise ValueError("current_factor_probs must have shape [B,3]")
        tokens = self.token_proj(context_tokens)
        queries = self.time_queries.unsqueeze(0).expand(context_tokens.shape[0], -1, -1)
        future, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        predicted = torch.sigmoid(self.factor_head(future))
        factors = torch.cat([current_factor_probs.unsqueeze(1), predicted[:, 1:]], dim=1)
        routes = FactorizedPhysicsGate.factors_to_expert_probs(factors)
        return factors, routes


class PhysicsGate(FactorizedPhysicsGate):
    """Predict the current and next ``horizon`` b/c/m factors from observation.

    The gate directly emits one physical-factor triplet and one four-expert
    route per action-token position. It never consumes a draft action. Token
    zero is anchored to the factors measured from the current observation;
    tokens 1..T-1 are supervised by the corresponding future dataset labels.
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_experts: int = 4,
        hidden_dim: int = 256,
        num_queries: int = 4,
        nhead: int = 8,
        horizon: int = 50,
    ):
        super().__init__(d_model, num_experts, hidden_dim, num_queries, nhead)
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.sequence_context_proj = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU()
        )
        self.step_queries = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        self.sequence_cross_attn = nn.MultiheadAttention(
            hidden_dim, nhead, batch_first=True
        )
        self.sequence_factor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.last_predicted_factor_sequence: torch.Tensor | None = None
        self.last_predicted_route_sequence: torch.Tensor | None = None

    def forward(
        self,
        context_tokens: torch.Tensor,
        fast_force_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities, expert_logits, indices = super().forward(
            context_tokens, fast_force_token
        )
        context = self.sequence_context_proj(context_tokens)
        queries = self.step_queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        attended, _ = self.sequence_cross_attn(
            queries, context, context, need_weights=False
        )
        future_factors = torch.sigmoid(
            self.sequence_factor_head(attended + queries)
        )
        if self.last_factor_probs is None:
            raise RuntimeError("current factor head did not produce b/c/m probabilities")
        factors = torch.cat(
            [self.last_factor_probs.unsqueeze(1), future_factors[:, 1:]], dim=1
        )
        routes = self.factors_to_expert_probs(factors)
        self.last_predicted_factor_sequence = factors
        self.last_predicted_route_sequence = routes
        return probabilities, expert_logits, indices


class LegacyActionConditionedPhysicsGate(FactorizedPhysicsGate):
    """Historical two-pass gate retained only to load old experiment checkpoints."""

    def __init__(
        self,
        d_model: int = 2048,
        num_experts: int = 4,
        hidden_dim: int = 256,
        num_queries: int = 4,
        nhead: int = 8,
        action_dim: int = 1024,
        horizon: int = 50,
    ):
        super().__init__(d_model, num_experts, hidden_dim, num_queries, nhead)
        self.horizon = horizon
        self.temporal_context_proj = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU()
        )
        self.temporal_action_proj = nn.Sequential(
            nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden_dim), nn.GELU()
        )
        self.temporal_action_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_queries = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        self.temporal_cross_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.temporal_factor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.last_predicted_factor_sequence: torch.Tensor | None = None
        self.last_predicted_route_sequence: torch.Tensor | None = None

    def forward(
        self,
        context_tokens: torch.Tensor,
        fast_force_token: torch.Tensor,
        action_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities, expert_logits, indices = super().forward(context_tokens, fast_force_token)
        self.last_predicted_factor_sequence = None
        self.last_predicted_route_sequence = None
        if action_tokens is None:
            return probabilities, expert_logits, indices
        if action_tokens.ndim != 3 or action_tokens.shape[:2] != (
            context_tokens.shape[0],
            self.horizon,
        ):
            raise ValueError("action_tokens must have shape [B,horizon,D]")
        context = self.temporal_context_proj(context_tokens)
        action = self.temporal_action_proj(action_tokens)
        self.temporal_action_encoder.flatten_parameters()
        action_prefix, _ = self.temporal_action_encoder(action)
        queries = action_prefix + self.temporal_queries.unsqueeze(0)
        attended, _ = self.temporal_cross_attn(queries, context, context, need_weights=False)
        future_factors = torch.sigmoid(
            self.temporal_factor_head(torch.cat([action_prefix, attended], dim=-1))
        )
        if self.last_factor_probs is None:
            raise RuntimeError("current factor head did not produce b/c/m probabilities")
        factors = torch.cat([self.last_factor_probs.unsqueeze(1), future_factors[:, 1:]], dim=1)
        routes = self.factors_to_expert_probs(factors)
        self.last_predicted_factor_sequence = factors
        self.last_predicted_route_sequence = routes
        return probabilities, expert_logits, indices


class _FeatureExpert(nn.Module):
    """Small residual adapter over a declared set of physical feature tokens."""

    def __init__(self, d_model: int, hidden_dim: int, feature_count: int):
        super().__init__()
        self.feature_count = feature_count
        self.token_proj = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU())
                for _ in range(feature_count)
            ]
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * feature_count),
            nn.Linear(hidden_dim * feature_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        nn.init.xavier_uniform_(self.fuse[-1].weight, gain=0.02)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} features, received {len(features)}")
        projected = [
            projection(feature) for projection, feature in zip(self.token_proj, features, strict=True)
        ]
        return self.fuse(torch.cat(projected, dim=-1))


class LegacyTokenPhysicsExperts(nn.Module):
    """Legacy v6 experts: independent weights, but one shared adapter topology.

    This class is retained so the pre-v2 diagnostic checkpoints remain loadable.
    """

    def __init__(self, d_model: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.free_load = _FeatureExpert(d_model, hidden_dim, 3)
        self.visual_blind = _FeatureExpert(d_model, hidden_dim, 4)
        self.rigid_micro = _FeatureExpert(d_model, hidden_dim, 3)
        self.flexible = _FeatureExpert(d_model, hidden_dim, 3)

    def forward(
        self,
        visual_token: torch.Tensor,
        proprio_token: torch.Tensor,
        force_tokens: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        e1 = self.free_load([visual_token, proprio_token, force_tokens["current"]])
        e2 = self.visual_blind(
            [proprio_token, force_tokens["current"], force_tokens["fast"], force_tokens["slow"]]
        )
        e3 = self.rigid_micro([force_tokens["fast"], force_tokens["current"], visual_token])
        e4 = self.flexible([force_tokens["slow"], proprio_token, visual_token])
        return torch.stack([e1, e2, e3, e4], dim=1)


def _small_output_init(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight, gain=0.02)
    nn.init.zeros_(module.bias)


class VisualHistoryEncoder(nn.Module):
    """Compact episode-local visual memory for the E2 blind-view expert."""

    def __init__(self, d_model: int = 2048, hidden_dim: int = 128, *, mask_invalid_cameras: bool = False):
        super().__init__()
        self.mask_invalid_cameras = mask_invalid_cameras
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.frame_proj = nn.Linear(32, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, d_model),
        )

    def forward(
        self,
        camera_histories: list[torch.Tensor],
        history_padding: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not camera_histories:
            raise ValueError("camera_histories must contain at least one camera")
        batch_size, steps = camera_histories[0].shape[:2]
        per_camera = []
        camera_validities = []
        valid = None
        for camera_index, images in enumerate(camera_histories):
            if images.ndim != 5 or images.shape[2] != 3:
                raise ValueError("visual history must have shape [B,T,3,H,W]")
            if images.shape[:2] != (batch_size, steps):
                raise ValueError("all camera histories must share [B,T]")
            flat = images.reshape(batch_size * steps, *images.shape[2:]).float()
            encoded = self.frame_proj(self.frame_encoder(flat).flatten(1))
            per_camera.append(encoded.reshape(batch_size, steps, -1))
            if history_padding is not None:
                camera_valid = ~history_padding[camera_index].to(
                    device=images.device, dtype=torch.bool
                )
                valid = camera_valid if valid is None else valid | camera_valid
                camera_validities.append(camera_valid)

        if self.mask_invalid_cameras and history_padding is not None:
            weights = torch.stack(camera_validities, dim=0).unsqueeze(-1)
            encoded_cameras = torch.stack(per_camera, dim=0)
            sequence = encoded_cameras.masked_fill(~weights, 0.0).sum(dim=0)
            sequence = sequence / weights.sum(dim=0).clamp_min(1)
        else:
            sequence = torch.stack(per_camera, dim=0).mean(dim=0)
        if valid is None:
            valid = torch.ones(batch_size, steps, device=sequence.device, dtype=torch.bool)
        sequence = sequence.masked_fill(~valid.unsqueeze(-1), 0.0)
        outputs, _ = self.temporal(sequence)
        valid_count = valid.sum(dim=1)
        positions = torch.arange(steps, device=sequence.device).expand(batch_size, -1)
        last_index = positions.masked_fill(~valid, -1).amax(dim=1).clamp_min(0)
        memory = outputs[torch.arange(batch_size, device=outputs.device), last_index]
        has_memory = valid_count > 0
        memory = memory.masked_fill(~has_memory.unsqueeze(-1), 0.0)
        age = (steps - 1 - last_index).to(dtype=memory.dtype) / max(steps, 1)
        age = torch.where(has_memory, age, torch.ones_like(age)).unsqueeze(-1)
        token = self.out(torch.cat([memory, age], dim=-1))
        token = token.masked_fill(~has_memory.unsqueeze(-1), 0.0)
        return token, age


class _MotionHistoryEncoder(nn.Module):
    """Encode state history together with explicit velocity/acceleration statistics."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.history_gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        # current, mean/last/max velocity, and mean acceleration
        self.kinematic_stats = nn.Sequential(
            nn.LayerNorm(state_dim * 5),
            nn.Linear(state_dim * 5, hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

    def forward(self, state: torch.Tensor, history: torch.Tensor | None) -> torch.Tensor:
        state = state[..., : self.state_dim]
        history = state.unsqueeze(1) if history is None else history[..., : self.state_dim]
        _, hidden = self.history_gru(history)

        if history.shape[1] > 1:
            velocity = history[:, 1:] - history[:, :-1]
            mean_velocity = velocity.mean(dim=1)
            last_velocity = velocity[:, -1]
            max_velocity = velocity.abs().amax(dim=1)
        else:
            velocity = history.new_zeros(history.shape[0], 1, self.state_dim)
            mean_velocity = last_velocity = max_velocity = velocity[:, 0]
        if velocity.shape[1] > 1:
            mean_acceleration = (velocity[:, 1:] - velocity[:, :-1]).mean(dim=1)
        else:
            mean_acceleration = torch.zeros_like(mean_velocity)
        stats = torch.cat(
            [state, mean_velocity, last_velocity, max_velocity, mean_acceleration], dim=-1
        )
        return self.fuse(torch.cat([hidden[-1], self.kinematic_stats(stats)], dim=-1))


class E1FusionLayerNorm(nn.LayerNorm):
    """Same affine tensors as legacy LN; optionally isolate branch statistics."""

    def __init__(self, hidden_dim: int, mode: str = "joint"):
        super().__init__(hidden_dim * 3)
        if mode not in {"joint", "branchwise"}:
            raise ValueError(f"Unknown E1 normalization: {mode}")
        self.mode = mode
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "joint":
            return super().forward(x)
        branches = x.reshape(*x.shape[:-1], 3, self.hidden_dim)
        normalized = F.layer_norm(branches, (self.hidden_dim,), eps=self.eps).reshape_as(x)
        return (normalized * self.weight + self.bias).to(dtype=x.dtype)


class E1FusionMLP(nn.Sequential):
    """Parameter-identical early/late branch summation around the same GELU."""

    def __init__(self, hidden_dim: int, output_dim: int, normalization: str, mode: str):
        if mode not in {"early_sum", "late_sum"}:
            raise ValueError(f"Unknown E1 fusion mode: {mode}")
        super().__init__(E1FusionLayerNorm(hidden_dim, normalization),
                         nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
                         nn.Linear(hidden_dim, output_dim))
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "early_sum":
            return super().forward(x)
        branches = self[0](x).chunk(3, dim=-1)
        weights = self[1].weight.chunk(3, dim=-1)
        # Splitting the same bias preserves sum(z_i) == legacy preactivation.
        bias = self[1].bias / 3 if self[1].bias is not None else None
        parts = [self[2](F.linear(branch, weight, bias))
                 for branch, weight in zip(branches, weights, strict=True)]
        return self[3](torch.stack(parts, dim=0).sum(dim=0))


class FreeMotionExpert(nn.Module):
    """E1: visual-spatial and kinematic dynamics for free/load motion."""

    def __init__(self, d_model: int, hidden_dim: int, state_dim: int, force_dim: int,
                 fusion_normalization: str = "joint", fusion_mode: str = "early_sum"):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU())
        self.motion = _MotionHistoryEncoder(state_dim, hidden_dim)
        self.load = nn.Sequential(nn.LayerNorm(force_dim), nn.Linear(force_dim, hidden_dim), nn.GELU())
        self.out = E1FusionMLP(hidden_dim, d_model, fusion_normalization, fusion_mode)
        _small_output_init(self.out[-1])

    def forward(
        self,
        visual: torch.Tensor,
        state: torch.Tensor,
        state_history: torch.Tensor | None,
        current_force: torch.Tensor,
    ) -> torch.Tensor:
        features = [self.visual(visual), self.motion(state, state_history), self.load(current_force)]
        return self.out(torch.cat(features, dim=-1))


class VisualBlindForceExpert(nn.Module):
    """E2: multi-rate force/state temporal expert with no visual-content input."""

    def __init__(
        self, d_model: int, hidden_dim: int, state_dim: int, force_dim: int, quality_dim: int
    ):
        super().__init__()
        self.current = nn.Sequential(
            nn.LayerNorm(force_dim), nn.Linear(force_dim, hidden_dim), nn.GELU()
        )
        self.fast = TemporalConvEncoder(force_dim, hidden_dim, hidden_dim)
        self.slow_gru = nn.GRU(force_dim, hidden_dim, batch_first=True)
        self.motion = _MotionHistoryEncoder(state_dim, hidden_dim)
        self.quality = nn.Sequential(
            nn.LayerNorm(quality_dim), nn.Linear(quality_dim, hidden_dim), nn.GELU()
        )
        self.visual_memory = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU()
        )
        self.memory_decay = nn.Parameter(torch.tensor(1.0))
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, d_model),
        )
        _small_output_init(self.out[-1])

    def forward(
        self,
        state: torch.Tensor,
        state_history: torch.Tensor | None,
        current_force: torch.Tensor,
        fast_force: torch.Tensor,
        slow_force: torch.Tensor,
        visual_quality: torch.Tensor,
        visual_memory: torch.Tensor | None = None,
        memory_age: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, slow_hidden = self.slow_gru(slow_force)
        slow_context = slow_hidden[-1]
        if visual_memory is not None:
            if memory_age is None:
                memory_age = visual_memory.new_zeros(visual_memory.shape[0], 1)
            # Older memories are learned as lower-confidence additions. E2
            # still receives no current visual content.
            confidence = torch.exp(-F.softplus(self.memory_decay) * memory_age)
            slow_context = slow_context + self.visual_memory(visual_memory) * confidence
        features = [
            self.current(current_force),
            self.fast(fast_force),
            slow_context,
            self.motion(state, state_history),
            self.quality(visual_quality),
        ]
        return self.out(torch.cat(features, dim=-1))


class RigidContactExpert(nn.Module):
    """E3: fast TCN plus differential/spectral cues and motion direction."""

    def __init__(self, d_model: int, hidden_dim: int, state_dim: int, force_dim: int):
        super().__init__()
        self.fast_tcn = TemporalConvEncoder(force_dim, hidden_dim, hidden_dim)
        # current + first-difference mean/std/max + second-difference mean/std/max
        # + low/high relative-frequency energy = 9 groups of force_dim values.
        self.dynamic_stats = nn.Sequential(
            nn.LayerNorm(force_dim * 9),
            nn.Linear(force_dim * 9, hidden_dim),
            nn.GELU(),
        )
        self.motion = _MotionHistoryEncoder(state_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, d_model),
        )
        _small_output_init(self.out[-1])

    @staticmethod
    def _moments(signal: torch.Tensor) -> list[torch.Tensor]:
        return [
            signal.mean(dim=1),
            signal.std(dim=1, unbiased=False),
            signal.abs().amax(dim=1),
        ]

    def forward(
        self,
        state: torch.Tensor,
        state_history: torch.Tensor | None,
        current_force: torch.Tensor,
        fast_force: torch.Tensor,
    ) -> torch.Tensor:
        first = torch.diff(fast_force, dim=1)
        if first.shape[1] == 0:
            first = torch.zeros_like(fast_force[:, :1])
        second = torch.diff(first, dim=1)
        if second.shape[1] == 0:
            second = torch.zeros_like(first[:, :1])

        # Relative spectral bands avoid claiming a fixed Hz band when the sensor
        # sampling rate changes between Gazebo and the real FT300 stream.
        spectrum = torch.fft.rfft(fast_force.float(), dim=1).abs().square()
        split = max(1, spectrum.shape[1] // 3)
        low_energy = spectrum[:, :split].mean(dim=1)
        high_energy = spectrum[:, split:].mean(dim=1) if split < spectrum.shape[1] else low_energy
        stats = torch.cat(
            [current_force, *self._moments(first), *self._moments(second), low_energy, high_energy],
            dim=-1,
        ).to(dtype=fast_force.dtype)
        features = [
            self.fast_tcn(fast_force),
            self.dynamic_stats(stats),
            self.motion(state, state_history),
        ]
        return self.out(torch.cat(features, dim=-1))


class CompliantInsertionExpert(nn.Module):
    """E4: long-horizon force dynamics with a separate fast overload branch."""

    def __init__(self, d_model: int, hidden_dim: int, state_dim: int, force_dim: int):
        super().__init__()
        self.slow_gru = nn.GRU(force_dim, hidden_dim, batch_first=True)
        # current, slow mean/std/trend, and fast std/max = 6 force groups.
        self.force_trends = nn.Sequential(
            nn.LayerNorm(force_dim * 6),
            nn.Linear(force_dim * 6, hidden_dim),
            nn.GELU(),
        )
        self.motion = _MotionHistoryEncoder(state_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, d_model),
        )
        _small_output_init(self.out[-1])

    def forward(
        self,
        state: torch.Tensor,
        state_history: torch.Tensor | None,
        current_force: torch.Tensor,
        fast_force: torch.Tensor,
        slow_force: torch.Tensor,
    ) -> torch.Tensor:
        _, slow_hidden = self.slow_gru(slow_force)
        trends = torch.cat(
            [
                current_force,
                slow_force.mean(dim=1),
                slow_force.std(dim=1, unbiased=False),
                slow_force[:, -1] - slow_force[:, 0],
                fast_force.std(dim=1, unbiased=False),
                fast_force.abs().amax(dim=1),
            ],
            dim=-1,
        )
        features = [
            slow_hidden[-1],
            self.force_trends(trends),
            self.motion(state, state_history),
        ]
        return self.out(torch.cat(features, dim=-1))


class HeterogeneousPhysicsExperts(nn.Module):
    """Four structurally distinct physical experts with explicit modality boundaries."""

    def __init__(
        self,
        d_model: int = 2048,
        hidden_dim: int = 256,
        state_dim: int = 7,
        force_dim: int = 6,
        quality_dim: int = 4,
        e1_fusion_normalization: str = "joint",
        e1_fusion_mode: str = "early_sum",
    ):
        super().__init__()
        self.quality_dim = quality_dim
        self.free_load = FreeMotionExpert(d_model, hidden_dim, state_dim, force_dim,
                                         e1_fusion_normalization, e1_fusion_mode)
        self.visual_blind = VisualBlindForceExpert(
            d_model, hidden_dim, state_dim, force_dim, quality_dim
        )
        self.rigid_micro = RigidContactExpert(d_model, hidden_dim, state_dim, force_dim)
        self.compliant = CompliantInsertionExpert(d_model, hidden_dim, state_dim, force_dim)

    def forward(
        self,
        visual_token: torch.Tensor,
        state: torch.Tensor,
        current_force: torch.Tensor,
        fast_force: torch.Tensor | None = None,
        slow_force: torch.Tensor | None = None,
        state_history: torch.Tensor | None = None,
        visual_quality: torch.Tensor | None = None,
        visual_memory: torch.Tensor | None = None,
        memory_age: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fast_force is None:
            fast_force = current_force.unsqueeze(1)
        if slow_force is None:
            slow_force = current_force.unsqueeze(1)
        if visual_quality is None:
            visual_quality = current_force.new_zeros(current_force.shape[0], self.quality_dim)

        e1 = self.free_load(visual_token, state, state_history, current_force)
        # E2 deliberately receives no visual token. This is a code-level isolation boundary.
        e2 = self.visual_blind(
            state,
            state_history,
            current_force,
            fast_force,
            slow_force,
            visual_quality,
            visual_memory,
            memory_age,
        )
        e3 = self.rigid_micro(state, state_history, current_force, fast_force)
        e4 = self.compliant(state, state_history, current_force, fast_force, slow_force)
        return torch.stack([e1, e2, e3, e4], dim=1)


class PhysicsExpertAuxiliaryHeads(nn.Module):
    """Training-only probes for declared heterogeneous expert information.

    These heads never generate actions. Their losses make the expert tokens
    retain physical descriptors that can be audited independently of action
    imitation loss.
    """

    def __init__(
        self,
        d_model: int = 2048,
        hidden_dim: int = 256,
        state_dim: int = 7,
        force_dim: int = 6,
        quality_dim: int = 4,
    ):
        super().__init__()

        def probe(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.free_motion = probe(state_dim + force_dim)
        self.visual_blind = probe(state_dim + force_dim + quality_dim)
        self.rigid_contact = probe(force_dim * 3 + 1)
        self.compliant_contact = probe(force_dim * 3 + 1)

    def forward(self, expert_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if expert_tokens.ndim != 3 or expert_tokens.shape[1] != 4:
            raise ValueError("expert_tokens must have shape [B,4,D]")
        return {
            "E1": self.free_motion(expert_tokens[:, 0]),
            "E2": self.visual_blind(expert_tokens[:, 1]),
            "E3": self.rigid_contact(expert_tokens[:, 2]),
            "E4": self.compliant_contact(expert_tokens[:, 3]),
        }


class SoftExpertRouter(nn.Module):
    """Return weighted expert tokens plus their fused physical context."""

    def forward(
        self,
        expert_tokens: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if expert_tokens.shape[:2] != routing_probs.shape:
            shapes = (tuple(expert_tokens.shape), tuple(routing_probs.shape))
            raise ValueError(f"Expert-token and routing shapes disagree: {shapes}")
        weighted = expert_tokens * routing_probs.unsqueeze(-1).to(dtype=expert_tokens.dtype)
        return weighted, weighted.sum(dim=1)


class ActionTokenConditioner(nn.Module):
    """Query-dependent physical conditioning for every flow-matching action token."""

    def __init__(
        self,
        condition_dim: int,
        action_dim: int,
        nhead: int = 8,
        zero_init_output: bool = True,
        route_attention_prior: str = "none",
    ):
        super().__init__()
        if route_attention_prior not in {"none", "log_probability"}:
            raise ValueError("Unknown route attention prior")
        self.route_attention_prior = route_attention_prior
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, action_dim),
            nn.GELU(),
        )
        self.query_norm = nn.LayerNorm(action_dim)
        self.cross_attn = nn.MultiheadAttention(action_dim, nhead, batch_first=True)
        if zero_init_output:
            nn.init.zeros_(self.cross_attn.out_proj.weight)
        else:
            nn.init.xavier_uniform_(self.cross_attn.out_proj.weight, gain=0.02)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def _route_attention_mask(self, weights, query_length, dtype):
        if self.route_attention_prior == "none" or weights is None:
            return None
        weights = weights.reshape(-1, weights.shape[-1]).float()
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Route weights must be finite and nonnegative")
        inactive = weights.eq(0).all(dim=-1)
        # Avoid log(0)'s infinite derivative, including masked branches.
        sentinel = torch.zeros_like(weights, dtype=torch.bool)
        sentinel[:, 0] = inactive
        weights = torch.where(sentinel, torch.ones_like(weights), weights)
        bias = weights.clamp_min(1e-30).log().masked_fill(weights.eq(0), float("-inf"))
        bias = bias.to(dtype=dtype)[:, None, None, :]
        return bias.expand(-1, self.cross_attn.num_heads, query_length, -1).reshape(
            -1, query_length, weights.shape[-1]
        )

    def _project_condition_tokens(
        self,
        condition_tokens: torch.Tensor,
        condition_weights: torch.Tensor | None = None,
        condition_token_scales: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        keys = self.condition_proj(condition_tokens)
        if output_dtype is not None:
            keys = keys.to(dtype=output_dtype)
        key_padding_mask = None
        if condition_weights is not None:
            if condition_weights.shape != condition_tokens.shape[:2]:
                raise ValueError(
                    "condition_weights must match the first two condition-token dimensions, "
                    f"got {tuple(condition_weights.shape)} and {tuple(condition_tokens.shape)}"
                )
            weights = condition_weights.to(device=keys.device, dtype=keys.dtype)
            keys = keys * weights.unsqueeze(-1)
            key_padding_mask = weights <= 0
            # A fully masked row is a meaningful ablation: it represents the
            # baseline action path with no physical-expert contribution.
            # MultiheadAttention cannot consume a row whose every key is
            # masked, so expose one already-zero key as a numerical sentinel.
            inactive_rows = key_padding_mask.all(dim=1)
            if inactive_rows.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[inactive_rows, 0] = False
        if condition_token_scales is not None:
            scales = condition_token_scales.to(device=keys.device, dtype=keys.dtype)
            if scales.ndim == 1:
                scales = scales.unsqueeze(0)
            if scales.shape[-1] != condition_tokens.shape[1]:
                raise ValueError(
                    "condition_token_scales must match the condition-token count, "
                    f"got {tuple(scales.shape)} and {tuple(condition_tokens.shape)}"
                )
            if scales.shape[0] == 1 and keys.shape[0] != 1:
                scales = scales.expand(keys.shape[0], -1)
            if scales.shape != condition_tokens.shape[:2]:
                raise ValueError(
                    "condition_token_scales must broadcast to [B,E], "
                    f"got {tuple(scales.shape)} and {tuple(condition_tokens.shape)}"
                )
            keys = keys * scales.unsqueeze(-1)
        return keys, key_padding_mask

    def forward(
        self,
        action_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_weights: torch.Tensor | None = None,
        condition_token_scales: torch.Tensor | None = None,
        residual_max_norm: float | None = None,
    ) -> torch.Tensor:
        if condition_weights is not None and condition_weights.ndim == 3:
            batch_size, horizon, expert_count = condition_weights.shape
            if action_tokens.shape[:2] != (batch_size, horizon):
                raise ValueError("stepwise condition_weights must match action tokens [B,T]")
            if condition_tokens.shape[:2] != (batch_size, expert_count):
                raise ValueError("stepwise condition_weights expert count must match condition_tokens")
            keys, _ = self._project_condition_tokens(
                condition_tokens,
                None,
                condition_token_scales,
                output_dtype=action_tokens.dtype,
            )
            weights = condition_weights.to(device=keys.device, dtype=keys.dtype)
            keys = keys[:, None] * weights.unsqueeze(-1)
            keys = keys.reshape(batch_size * horizon, expert_count, -1)
            key_padding_mask = weights.le(0).reshape(batch_size * horizon, expert_count)
            inactive_rows = key_padding_mask.all(dim=1)
            if inactive_rows.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[inactive_rows, 0] = False
            queries = self.query_norm(action_tokens).reshape(batch_size * horizon, 1, -1)
            attention_mask = self._route_attention_mask(condition_weights, 1, queries.dtype)
            delta, _ = self.cross_attn(
                queries, keys, keys,
                key_padding_mask=key_padding_mask if attention_mask is None else None,
                attn_mask=attention_mask, need_weights=False
            )
            delta = delta.reshape(batch_size, horizon, -1)
            if inactive_rows.any():
                delta = delta.masked_fill(inactive_rows.reshape(batch_size, horizon, 1), 0.0)
        else:
            keys, key_padding_mask = self._project_condition_tokens(
                condition_tokens,
                condition_weights,
                condition_token_scales,
                output_dtype=action_tokens.dtype,
            )
            attention_mask = self._route_attention_mask(
                condition_weights, action_tokens.shape[1], action_tokens.dtype
            )
            delta, _ = self.cross_attn(
                self.query_norm(action_tokens),
                keys,
                keys,
                key_padding_mask=key_padding_mask if attention_mask is None else None,
                attn_mask=attention_mask,
                need_weights=False,
            )
            if condition_weights is not None:
                inactive_rows = condition_weights.to(device=delta.device).le(0).all(dim=1)
                if inactive_rows.any():
                    # Remove both attention output and its learned projection bias
                    # so the no-expert ablation is exactly the unconditioned path.
                    delta = delta.masked_fill(inactive_rows[:, None, None], 0.0)
        if residual_max_norm is not None:
            if residual_max_norm <= 0:
                raise ValueError("residual_max_norm must be positive")
            norm = delta.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
            multiplier = (float(residual_max_norm) / norm).clamp(max=1.0)
            delta = delta * multiplier.to(dtype=delta.dtype)
        return action_tokens + delta


def soft_target_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    targets = targets.float().clamp_min(0)
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    losses = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
    if reduction == "none":
        return losses
    if reduction != "mean":
        raise ValueError(f"unsupported reduction: {reduction}")
    return losses.mean()


def compute_load_balancing_loss(
    stage_probs: torch.Tensor,
    target_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match the observed expert prior instead of forcing an artificial uniform prior."""
    predicted_prior = stage_probs.float().mean(dim=0)
    if target_probs is None:
        return predicted_prior.new_zeros(())
    target_prior = target_probs.float().mean(dim=0)
    target_prior = target_prior / target_prior.sum().clamp_min(1e-8)
    return F.mse_loss(predicted_prior, target_prior)

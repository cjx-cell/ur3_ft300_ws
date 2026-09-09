#!/usr/bin/env python3
"""Shared validation and processing helpers for peg-in-hole policy inference."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


PREPROCESSOR_CONFIG_FILE = "policy_preprocessor.json"


class DemonstrationStateRecovery:
    """Closed-loop demonstration tracking using only the measured joint state."""

    def __init__(self, episode_npz: Path, horizon: int, execute_steps: int):
        with np.load(episode_npz, allow_pickle=True) as episode:
            self.states = np.asarray(episode["state"], dtype=np.float32)
            self.actions = np.asarray(episode["action"], dtype=np.float32)
        if self.states.ndim != 2 or self.actions.shape != self.states.shape:
            raise ValueError("demonstration state/action arrays must have matching [T, A] shape")
        self.actions[:, 6] = (self.actions[:, 6] >= 0.5).astype(np.float32)
        self.horizon = horizon
        self.execute_steps = execute_steps
        self.cursor: int | None = None

    def action_chunk(self, measured_state: np.ndarray) -> tuple[np.ndarray, int, float]:
        max_start = max(0, len(self.actions) - 1)
        if self.cursor is None:
            start, stop = 0, max_start + 1
        else:
            start = max(0, self.cursor - 5)
            stop = min(max_start + 1, self.cursor + 41)
        distances = np.linalg.norm(
            self.states[start:stop, :6] - measured_state[None, :6], axis=1
        )
        minimum = float(distances.min())
        # Start and final home postures can be nearly identical. Prefer the
        # earliest frame among practically equivalent matches so a fresh
        # episode does not jump directly to the end of the demonstration.
        nearest = start + int(np.flatnonzero(distances <= minimum + 0.01)[0])
        if self.cursor is not None:
            nearest = max(nearest, self.cursor - 2)
        self.cursor = min(nearest + self.execute_steps, max_start)
        indices = np.minimum(np.arange(nearest, nearest + self.horizon), max_start)
        return self.actions[indices].copy(), nearest, float(distances[nearest - start])


def validate_action_semantics(checkpoint: Path, policy_config: object) -> None:
    """Reject checkpoints whose action statistics contradict their action mode.

    Absolute joint actions have centers comparable to the observation-state
    centers. Relative arm actions are centered close to zero. Mixing relative
    statistics with ``use_relative_actions=False`` caused normalized training
    targets to be tens to thousands of times outside the intended range.
    """

    processor_path = checkpoint / PREPROCESSOR_CONFIG_FILE
    if not processor_path.is_file():
        raise FileNotFoundError(
            "Checkpoint is missing its saved preprocessor configuration"
        )

    required = ("observation.state.q50", "action.q50")
    stats = None
    for candidate in sorted(
        checkpoint.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")
    ):
        candidate_stats = load_file(str(candidate))
        if all(key in candidate_stats for key in required):
            stats = candidate_stats
            break
    if stats is None:
        raise FileNotFoundError(
            "Checkpoint has no saved normalizer containing observation.state and action statistics"
        )

    state_center = stats["observation.state.q50"][:6].abs().median().item()
    action_center = stats["action.q50"][:6].abs().median().item()
    if isinstance(policy_config, dict):
        relative_actions = bool(policy_config.get("use_relative_actions", False))
    else:
        relative_actions = bool(getattr(policy_config, "use_relative_actions", False))

    with processor_path.open(encoding="utf-8") as file:
        processor_config = json.load(file)
    step_names = {step.get("registry_name") for step in processor_config.get("steps", [])}
    has_delta_step = "delta_actions_processor" in step_names

    if relative_actions and not has_delta_step:
        raise ValueError(
            "Checkpoint config enables relative actions, but its saved preprocessor "
            "does not contain delta_actions_processor"
        )

    if not relative_actions and state_center > 0.25 and action_center < 0.10:
        raise ValueError(
            "UNSAFE CHECKPOINT: use_relative_actions=false, but action statistics "
            f"are delta-like (state |q50| median={state_center:.4f}, "
            f"action |q50| median={action_center:.6f}). "
            "The checkpoint was trained with incompatible absolute actions and "
            "relative-action statistics; retraining is required."
        )


def postprocess_action_chunk(
    normalized_chunk: torch.Tensor,
    postprocessor: object,
) -> torch.Tensor:
    """Apply a single-action postprocessor to a batched action chunk."""

    if normalized_chunk.ndim != 3:
        raise ValueError(
            f"Expected normalized action chunk [B, K, A], got {tuple(normalized_chunk.shape)}"
        )

    actions = [
        postprocessor(normalized_chunk[:, step_index, :])
        for step_index in range(normalized_chunk.shape[1])
    ]
    return torch.stack(actions, dim=1)

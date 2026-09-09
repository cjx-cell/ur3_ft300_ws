"""Learned joint/FT300 feedback adapter used only after contact-stage routing."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class InsertionFeedbackAdapter(nn.Module):
    INPUT_DIM = 13
    ARM_DIM = 6

    def __init__(self, hidden_dim: int = 128, max_delta_rad: float = 0.04):
        super().__init__()
        self.max_delta_rad = float(max_delta_rad)
        self.network = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.ARM_DIM),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.max_delta_rad * torch.tanh(self.network(features))


def build_feedback_chunk(
    model: InsertionFeedbackAdapter,
    current_arm: np.ndarray,
    terminal_arm: np.ndarray,
    *,
    contact_model: InsertionFeedbackAdapter | None = None,
    contact_force_start_n: float = 0.0,
    contact_force_full_n: float = 0.0,
    force_norm_n: float,
    force_baseline_n: float,
    force_full_scale_n: float,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    """Autoregressively produce an absolute semantic arm chunk."""
    if chunk_size < 1 or force_full_scale_n <= 0.0:
        raise ValueError("feedback chunk arguments must be positive")
    q = np.asarray(current_arm, dtype=np.float32).copy()
    terminal = np.asarray(terminal_arm, dtype=np.float32)
    if q.shape != (6,) or terminal.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("feedback arm inputs must be finite [6]")
    force_feature = np.float32(
        np.clip((force_norm_n - force_baseline_n) / force_full_scale_n, 0.0, 1.0)
    )
    contact_blend = 0.0
    if contact_model is not None:
        if contact_force_full_n <= contact_force_start_n:
            raise ValueError("contact force full scale must exceed its start")
        contact_blend = float(
            np.clip(
                (force_norm_n - contact_force_start_n)
                / (contact_force_full_n - contact_force_start_n),
                0.0,
                1.0,
            )
        )
    arm = np.empty((chunk_size, 6), dtype=np.float32)
    with torch.no_grad():
        for index in range(chunk_size):
            features = np.r_[q, terminal - q, force_feature][None].astype(np.float32)
            tensor = torch.from_numpy(features).to(device)
            delta = model(tensor)[0].cpu().numpy()
            if contact_model is not None and contact_blend > 0.0:
                contact_delta = contact_model(tensor)[0].cpu().numpy()
                delta = (1.0 - contact_blend) * delta + contact_blend * contact_delta
            q += delta
            arm[index] = q
    return arm

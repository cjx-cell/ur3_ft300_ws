#!/usr/bin/env python3
"""Pure NumPy helpers that align online PAP-MoE inputs with v6 training data."""

from __future__ import annotations

import numpy as np


FAST_FORCE_CLIP_N = 29.9
TRAINING_GRIPPER_BINARY_THRESHOLD_RAD = 0.12
TRAINING_GRIPPER_CLOSED_POSITION_RAD = 0.629


def estimate_wrench_bias(samples: list[np.ndarray]) -> np.ndarray:
    """Estimate a stable six-axis reference from filtered wrench samples."""
    if not samples:
        raise ValueError("Cannot estimate wrench bias without samples")
    values = np.stack(
        [np.asarray(sample, dtype=np.float32) for sample in samples], axis=0
    )
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected wrench samples with shape [N,6], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Wrench bias samples must be finite")
    return np.median(values, axis=0).astype(np.float32)


def is_settled_gripper_close(
    recent_positions: np.ndarray,
    current_position: float,
    *,
    min_closed_position: float = 0.20,
    stable_range: float = 0.01,
    min_samples: int = 5,
) -> bool:
    """Detect a stable close without assuming an object-specific stop angle."""
    positions = np.asarray(recent_positions, dtype=np.float32).reshape(-1)
    if min_samples < 1 or positions.size < min_samples:
        return False
    window = positions[-min_samples:]
    return bool(
        np.isfinite(window).all()
        and np.isfinite(current_position)
        and float(current_position) >= min_closed_position
        and float(np.ptp(window)) <= stable_range
        and abs(float(current_position) - float(window[-1])) <= stable_range
    )


def calibrate_wrench_window(
    values: np.ndarray,
    payload_flags: np.ndarray,
    empty_bias: np.ndarray,
    payload_bias: np.ndarray | None = None,
    *,
    clip_fast: bool = False,
) -> np.ndarray:
    """Apply the v6 per-sample empty/payload reference to a wrench window."""
    window = np.asarray(values, dtype=np.float32)
    flags = np.asarray(payload_flags, dtype=bool)
    empty = np.asarray(empty_bias, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError(f"Expected wrench window with shape [N,6], got {window.shape}")
    if flags.shape != (window.shape[0],):
        raise ValueError(
            f"Expected payload flags with shape {(window.shape[0],)}, got {flags.shape}"
        )
    if empty.shape != (6,):
        raise ValueError(f"Expected empty bias with shape (6,), got {empty.shape}")

    payload = empty if payload_bias is None else np.asarray(payload_bias, dtype=np.float32)
    if payload.shape != (6,):
        raise ValueError(f"Expected payload bias with shape (6,), got {payload.shape}")
    if not all(np.isfinite(array).all() for array in (window, empty, payload)):
        raise ValueError("Wrench samples and references must be finite")
    references = np.where(flags[:, None], payload[None, :], empty[None, :])
    calibrated = (window - references).astype(np.float32)

    if clip_fast:
        force_norm = np.linalg.norm(calibrated[:, :3], axis=1)
        mask = force_norm > FAST_FORCE_CLIP_N
        if np.any(mask):
            calibrated[mask] *= (FAST_FORCE_CLIP_N / force_norm[mask])[:, None]
    return calibrated


def map_gripper_history_to_training_units(
    state_history: np.ndarray,
    *,
    mode: str = "continuous_radians_0_0.8",
) -> np.ndarray:
    """Convert physical gripper angles to a checkpoint's history contract.

    ``state_history`` comes directly from the ROS joint-state callback, so its
    seventh coordinate is a physical Robotiq knuckle angle (about 0.100 rad
    open and 0.629 rad closed).  The materialized D2 datasets binarize that
    coordinate in both ``observation.state`` and
    ``observation.state_history``.  The baseline's learned release/insertion
    feedback modules predate that conversion and consume the original v6
    physical angles instead.  The caller must therefore select the contract
    belonging to the checkpoint it is serving.
    """
    history = np.asarray(state_history, dtype=np.float32).copy()
    if history.ndim != 2 or history.shape[1] != 7:
        raise ValueError(f"Expected state history with shape [N,7], got {history.shape}")
    if mode == "binary_0_open_1_closed_gt_0.12rad":
        history[:, 6] = (
            history[:, 6] > TRAINING_GRIPPER_BINARY_THRESHOLD_RAD
        ).astype(np.float32)
    elif mode == "v6_analog_0.100_open_0.629_closed":
        # JointState already contains the physical Robotiq knuckle angle.
        # Preserve it; applying a second semantic-to-physical conversion here
        # would corrupt both the open and closed values.
        pass
    elif mode == "continuous_radians_0_0.8":
        # The current v9 dataset stores the measured Robotiq knuckle joint in
        # physical radians for both observation.state and state_history.
        history[:, 6] = np.clip(history[:, 6], 0.0, 0.8)
    elif mode == "continuous_0_1_closed_0.629rad":
        history[:, 6] = np.clip(
            history[:, 6] / TRAINING_GRIPPER_CLOSED_POSITION_RAD, 0.0, 1.0
        )
    else:
        raise ValueError(f"Unsupported state-history gripper mode: {mode!r}")
    return history

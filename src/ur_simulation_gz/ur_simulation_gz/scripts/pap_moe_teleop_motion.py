#!/usr/bin/env python3
"""Pure motion helpers for the PAP-MoE keyboard/mouse teleoperator.

This module deliberately has no ROS or GUI dependency so command shaping can
be tested without starting Gazebo.  Commands are unitless MoveIt Servo inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


KEY_MOTIONS = {
    # Upright oblique global-camera axes.  Operators command what they see:
    # W/S are image up/down and A/D are image left/right.
    "w": (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "s": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "a": (0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    "d": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "r": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "f": (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
    "i": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "k": (0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "j": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "l": (0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    "q": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "e": (0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
}


@dataclass(frozen=True)
class SpeedMode:
    name: str
    scale: float
    acceleration: float
    deceleration: float


SPEED_MODES = (
    # Servo's configured free-space ceiling is 0.8 m/s.  These scales give
    # approximately 0.60, 0.24 and 0.08 m/s in simulation time before safety
    # scaling.  At the observed Gazebo RTF of 0.55 that is about 0.33, 0.13
    # and 0.044 m/s in wall time.  Proportional mouse input remains available
    # when less than the full selected speed is wanted.
    SpeedMode("coarse", 0.75, 6.0, 8.0),
    SpeedMode("normal", 0.30, 3.0, 5.0),
    SpeedMode("precision", 0.10, 1.2, 2.0),
)


def _normalize_group(values: list[float], start: int) -> None:
    norm = math.sqrt(sum(value * value for value in values[start : start + 3]))
    if norm > 1.0:
        for index in range(start, start + 3):
            values[index] /= norm


def compose_unit_command(
    held_keys: Iterable[str],
    mouse_translation: Sequence[float] = (0.0, 0.0, 0.0),
    mouse_rotation: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[float]:
    """Combine simultaneous keyboard and proportional mouse commands.

    Translation and rotation are normalized independently.  Consequently a
    diagonal input changes direction without exceeding the selected speed.
    """

    values = [0.0] * 6
    for key in held_keys:
        motion = KEY_MOTIONS.get(key)
        if motion is None:
            continue
        for index, value in enumerate(motion):
            values[index] += value
    for index, value in enumerate(mouse_translation[:3]):
        values[index] += float(value)
    for index, value in enumerate(mouse_rotation[:3], start=3):
        values[index] += float(value)
    _normalize_group(values, 0)
    _normalize_group(values, 3)
    return values


def _limit_vector_delta(
    current: list[float], target: Sequence[float], start: int, max_delta: float
) -> None:
    delta = [float(target[index]) - current[index] for index in range(start, start + 3)]
    norm = math.sqrt(sum(value * value for value in delta))
    factor = 1.0 if norm <= max_delta or norm == 0.0 else max_delta / norm
    for offset, value in enumerate(delta):
        current[start + offset] += factor * value


class MotionCommandShaper:
    """Acceleration-limited six-dimensional command generator."""

    def __init__(self) -> None:
        self.current = [0.0] * 6

    def reset(self) -> list[float]:
        self.current = [0.0] * 6
        return self.current.copy()

    def step(
        self,
        target: Sequence[float],
        dt: float,
        acceleration: float,
        deceleration: float,
    ) -> list[float]:
        if len(target) != 6:
            raise ValueError("target command must contain six values")
        dt = min(max(float(dt), 0.0), 0.1)
        if dt == 0.0:
            return self.current.copy()
        target_is_zero = math.sqrt(sum(float(v) ** 2 for v in target)) < 1e-9
        rate = deceleration if target_is_zero else acceleration
        max_delta = max(float(rate), 0.0) * dt
        _limit_vector_delta(self.current, target, 0, max_delta)
        _limit_vector_delta(self.current, target, 3, max_delta)
        if target_is_zero and math.sqrt(sum(v * v for v in self.current)) < 1e-6:
            self.current = [0.0] * 6
        return self.current.copy()

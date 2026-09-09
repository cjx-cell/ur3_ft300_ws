#!/usr/bin/env python3
"""Observable, factorized routing prior for the four PAP-MoE physics experts.

The prior estimates three observable factors:

* contact probability ``c`` from payload-referenced wrench and force impulse;
* visual reliability ``q`` from the controlled image degradation;
* constraint-dominance probability ``h`` from wrench history and generic
  robot-motion history.

The four expert targets are then

    E1 = q(1-c)
    E2 = 1-q
    E3 = c h
    E4 = c(1-h)

The values are normalized after construction.  Consequently E2 remains
active during visual failure both with and without contact, and can cooperate
with E3/E4 during blind contact.  No task stage, fixture geometry, end-effector
height, or semantic subtask is used by this prior.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
    value = np.clip(-steepness * (x - midpoint), -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(value)))


def _deadband_sigmoid(
    x: float,
    *,
    deadband: float,
    midpoint: float,
    steepness: float,
) -> float:
    """A sigmoid-like probability with an exact zero-noise anchor.

    A plain sigmoid has a non-zero value at x=0.  That created a permanent
    four-percent contact target in perfectly free space.  Values inside the
    calibrated sensor deadband are physical no-contact evidence; outside it,
    rescale the original sigmoid continuously into [0, 1].
    """
    value = max(float(x), 0.0)
    if value <= deadband:
        return 0.0
    base = _sigmoid(deadband, midpoint, steepness)
    raw = _sigmoid(value, midpoint, steepness)
    return float(np.clip((raw - base) / max(1.0 - base, 1e-8), 0.0, 1.0))


class PhysicsRoutingPrior:
    """Stateful, episode-scoped soft routing target generator."""

    def __init__(self, history_size: int = 6):
        self.history_size = history_size
        self.reset()

    def reset(self) -> None:
        self._force_history: deque[np.ndarray] = deque(maxlen=self.history_size)
        self._motion_history: deque[float] = deque(maxlen=self.history_size)
        self._contact_memory = 0.0
        self._last_force: np.ndarray | None = None

    @staticmethod
    def _visual_loss(
        cam_degraded: bool,
        degradation_type: str,
        glare_gain: float,
    ) -> float:
        if not cam_degraded or degradation_type == "normal":
            return 0.0
        if degradation_type == "dropout":
            return 1.0
        if degradation_type == "glare":
            return float(np.clip((glare_gain - 1.0) / 5.0, 0.0, 1.0))
        return 0.0

    def compute(
        self,
        force_6d: np.ndarray,
        force_fast_window: np.ndarray | None,
        tool0_z: float | None,
        cam_degraded: bool,
        cam_degradation_type: str = "normal",
        cam_glare_gain: float = 1.5,
        gripper_joint_val: float = 0.5,
        joint_vel_norm: float = 0.0,
        current_stage: int = 0,
        semantic_subtask: str | None = None,
    ) -> np.ndarray:
        wrench = np.asarray(force_6d, dtype=np.float64)
        force = wrench[:3]
        torque = wrench[3:]
        force_mag = float(np.linalg.norm(force))
        torque_mag = float(np.linalg.norm(torque))

        if force_fast_window is not None and len(force_fast_window) >= 4:
            fast = np.asarray(force_fast_window, dtype=np.float64)
            force_steps = np.linalg.norm(np.diff(fast[:, :3], axis=0), axis=1)
            df_mag = float(np.quantile(force_steps, 0.95))
            torque_variance = float(np.max(np.var(fast[:, 3:6], axis=0)))
            fast_force_norm = np.linalg.norm(fast[:, :3], axis=1)
            fast_peak_index = int(np.argmax(fast_force_norm))
            fast_peak_force = fast[fast_peak_index, :3]
            fast_peak_mag = float(fast_force_norm[fast_peak_index])
        else:
            df_mag = 0.0
            torque_variance = 0.0
            fast_peak_force = force
            fast_peak_mag = force_mag

        if self._last_force is not None:
            df_mag = max(df_mag, float(np.linalg.norm(force - self._last_force)))
        self._last_force = force.copy()

        # Sustained force must not be suppressed during a real contact motion.
        # Payload-referenced free-space noise is normally below 0.2 N. A
        # 0.8 N midpoint keeps a wide noise margin while making a deliberately
        # compliant 1--4 N seating plateau observable.
        force_contact = _deadband_sigmoid(
            force_mag, deadband=0.30, midpoint=0.8, steepness=5.0
        )
        torque_contact = _deadband_sigmoid(
            torque_mag, deadband=0.06, midpoint=0.35, steepness=8.0
        )
        sustained_contact = max(force_contact, 0.7 * torque_contact)

        # Only the impulse branch is velocity-suppressed: fast free-space arm
        # motion can create an inertial dF spike, whereas sustained contact
        # remains physical even while sliding along a rim.
        impulse_excess = max(df_mag - 0.50, 0.0)
        impulse_contact = 1.0 - float(
            np.exp(-(impulse_excess**2) / (2.0 * 2.0**2))
        )
        impulse_contact *= float(np.exp(-joint_vel_norm / 0.5))
        # force_fast has already replaced >30 N one-sample DART constraint
        # impulses.  A remaining bounded peak near the socket is therefore
        # useful contact evidence even while the arm is moving.  Do not apply
        # the inertial velocity suppression to this branch: doing so erased
        # the very rim contacts that recovery demonstrations must supervise.
        bounded_fast_contact = _deadband_sigmoid(
            fast_peak_mag,
            deadband=0.40,
            midpoint=0.8,
            steepness=5.0,
        )
        contact_now = float(
            np.clip(
                max(
                    sustained_contact,
                    impulse_contact,
                    bounded_fast_contact,
                ),
                0.0,
                1.0,
            )
        )

        self._contact_memory = max(contact_now, 0.82 * self._contact_memory)
        contact_probability = float(
            np.clip(max(contact_now, 0.55 * self._contact_memory), 0.0, 1.0)
        )

        self._force_history.append(force.copy())
        self._motion_history.append(max(float(joint_vel_norm), 0.0))

        effective_stiffness = 0.0
        if (
            len(self._force_history) == self.history_size
            and len(self._motion_history) == self.history_size
        ):
            delta_force = float(
                np.linalg.norm(self._force_history[-1] - self._force_history[0])
            )
            motion_exposure = float(np.mean(self._motion_history))
            effective_stiffness = delta_force / max(motion_exposure, 0.02)

        # Direction ratios are undefined near zero magnitude.  The old 1e-6
        # denominator turned tiny numerical XY noise into a ratio near one and
        # mislabeled the contact floor as E3.  Only classify contact geometry
        # from vectors that are themselves above the force deadband.
        lateral_candidates = []
        if force_mag > 0.30:
            lateral_candidates.append(
                float(np.linalg.norm(force[:2]) / force_mag)
            )
        if fast_peak_mag > 0.40:
            lateral_candidates.append(
                float(np.linalg.norm(fast_peak_force[:2]) / fast_peak_mag)
            )
        lateral_ratio = max(lateral_candidates, default=0.0)
        stiffness_signal = _sigmoid(
            effective_stiffness,
            midpoint=20.0,
            steepness=0.15,
        )
        stick_slip_signal = _sigmoid(
            torque_variance,
            midpoint=0.008,
            steepness=350.0,
        )
        lateral_constraint_signal = _sigmoid(
            lateral_ratio,
            midpoint=0.35,
            steepness=12.0,
        )
        low_motion_constraint = (
            force_contact
            * float(np.exp(-max(float(joint_vel_norm), 0.0) / 0.08))
        )
        constraint_probability = 1.0 - (
            (1.0 - stiffness_signal)
            * (1.0 - stick_slip_signal)
            * (1.0 - lateral_constraint_signal)
            * (1.0 - low_motion_constraint)
        )
        constraint_probability = float(
            np.clip(constraint_probability, 0.0, 1.0)
        )
        visual_loss = self._visual_loss(
            cam_degraded,
            cam_degradation_type,
            cam_glare_gain,
        )
        visual_reliability = 1.0 - visual_loss

        c = contact_probability
        q = visual_reliability
        h = constraint_probability
        weights = np.asarray(
            [
                q * (1.0 - c),
                1.0 - q,
                c * h,
                c * (1.0 - h),
            ],
            dtype=np.float32,
        )
        return weights / max(float(weights.sum()), 1e-8)


_DEFAULT_PRIOR = PhysicsRoutingPrior()


def compute_physical_state_vector(*args, **kwargs) -> np.ndarray:
    return _DEFAULT_PRIOR.compute(*args, **kwargs)


def reset_label_buffers() -> None:
    _DEFAULT_PRIOR.reset()

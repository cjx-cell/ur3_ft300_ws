"""Strict schema for rollout -> failure -> expert recovery episodes.

This module deliberately does not share the full expert-trajectory schema.
An episode is useful only when the policy command, expert command and command
actually sent to the robot are independently observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LEGACY_SCHEMA_VERSION = "pap_moe_rollout_recovery_v1"
LEGACY_TRAJECTORY_SCOPE = "rollout_failure_recovery_v1"
V2_SCHEMA_VERSION = "pap_moe_rollout_recovery_v2_full_modalities"
V2_TRAJECTORY_SCOPE = "rollout_failure_recovery_v2_full_modalities"
V3_SCHEMA_VERSION = "pap_moe_rollout_recovery_v3_multitask_full_modalities"
V3_TRAJECTORY_SCOPE = "rollout_failure_recovery_v3_multitask_full_modalities"
SCHEMA_VERSION = "pap_moe_rollout_recovery_v4_multi_handoff_full_episode"
TRAJECTORY_SCOPE = "rollout_expert_rejoin_full_episode_v4"
POLICY_MODE = 0
EXPERT_MODE = 1
ACTION_DIM = 7
RECOVERY_PHASES = (
    "grasp_lift", "transport", "align_precontact", "contact", "insertion", "full_task"
)
STATE_HISTORY_SAMPLES = 10
FAST_FORCE_SAMPLES = 64
SLOW_FORCE_SAMPLES = 50
VISUAL_QUALITY_DIM = 4
STAGE_DIM = 4
SKILL_PROGRESS_PHASES = (
    "enter", "approach", "align", "interact",
    "stabilize", "verify", "exit", "recover",
)
MODALITY_NAMES = (
    "camera0",
    "camera1",
    "state",
    "state_history",
    "force_current",
    "force_fast",
    "force_slow",
)


@dataclass(frozen=True)
class ValidationSummary:
    frames: int
    policy_frames: int
    expert_frames: int
    takeover_index: int
    duration_s: float
    max_camera_skew_s: float
    max_force_skew_s: float
    max_pose_skew_s: float
    recovery_phase: str
    recovery_success: bool


def _scalar_text(data: Mapping[str, Any], key: str) -> str:
    if key not in data:
        raise ValueError(f"missing scalar metadata: {key}")
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"{key} must be a scalar, got shape {value.shape}")
    return str(value.item())


def _array(data: Mapping[str, Any], key: str, *, ndim: int | None = None) -> np.ndarray:
    if key not in data:
        raise ValueError(f"missing array: {key}")
    value = np.asarray(data[key])
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{key} must have rank {ndim}, got shape {value.shape}")
    return value


def validate_episode(
    data: Mapping[str, Any],
    *,
    max_camera_skew_s: float = 0.15,
    max_pose_skew_s: float = 1.0,
    action_match_atol: float = 2e-3,
    require_success: bool = True,
    sustained_force_limit_n: float = 80.0,
    # Recovery frames arrive at roughly 8--10 Hz. Treat short exact-fit
    # insertion impulses as transients, while still rejecting >80 N contact
    # sustained for about two seconds or longer.
    sustained_force_frames: int = 20,
    allow_policy_failure_force_context: bool = False,
    require_full_modalities: bool = False,
) -> ValidationSummary:
    """Validate one raw recovery episode and return its transition summary."""

    schema_version = _scalar_text(data, "schema_version")
    is_multi_handoff = schema_version == SCHEMA_VERSION
    is_multitask = schema_version in (SCHEMA_VERSION, V3_SCHEMA_VERSION)
    is_full_modalities = schema_version in (
        SCHEMA_VERSION, V3_SCHEMA_VERSION, V2_SCHEMA_VERSION
    )
    if schema_version not in (
        SCHEMA_VERSION, V3_SCHEMA_VERSION, V2_SCHEMA_VERSION, LEGACY_SCHEMA_VERSION
    ):
        raise ValueError("schema_version does not identify a supported rollout recovery")
    expected_scope = (
        TRAJECTORY_SCOPE if is_multi_handoff else
        V3_TRAJECTORY_SCOPE if schema_version == V3_SCHEMA_VERSION else
        V2_TRAJECTORY_SCOPE if is_full_modalities else LEGACY_TRAJECTORY_SCOPE
    )
    if _scalar_text(data, "trajectory_scope") != expected_scope:
        raise ValueError(f"trajectory_scope does not match {schema_version}")
    if require_full_modalities and not is_full_modalities:
        raise ValueError("legacy recovery is missing the mandatory v2 full-modality contract")
    for key in ("source_policy_checkpoint", "recovery_trigger", "episode_id"):
        if not _scalar_text(data, key).strip():
            raise ValueError(f"{key} must be non-empty")
    recovery_phase = _scalar_text(data, "recovery_phase")
    if recovery_phase not in RECOVERY_PHASES:
        raise ValueError(
            f"recovery_phase must be one of {RECOVERY_PHASES}, got {recovery_phase!r}"
        )

    state = _array(data, "state", ndim=2)
    policy_action = _array(data, "policy_action", ndim=2)
    expert_action = _array(data, "expert_action", ndim=2)
    executed_action = _array(data, "executed_action", ndim=2)
    control_mode = _array(data, "control_mode", ndim=1).astype(np.int64)
    intervention = _array(data, "intervention_mask", ndim=1).astype(bool)
    timestamp = _array(data, "timestamp", ndim=1).astype(np.float64)
    policy_action_timestamp = _array(data, "policy_action_timestamp", ndim=1).astype(np.float64)
    expert_action_timestamp = _array(data, "expert_action_timestamp", ndim=1).astype(np.float64)
    executed_action_timestamp = _array(data, "executed_action_timestamp", ndim=1).astype(np.float64)
    camera0_timestamp = _array(data, "camera0_timestamp", ndim=1).astype(np.float64)
    camera1_timestamp = _array(data, "camera1_timestamp", ndim=1).astype(np.float64)
    force_timestamp = _array(data, "force_timestamp", ndim=1).astype(np.float64)
    pose_timestamp = _array(data, "pose_timestamp", ndim=1).astype(np.float64)
    camera0 = _array(data, "camera0", ndim=4)
    camera1 = _array(data, "camera1", ndim=4)
    force = _array(data, "force", ndim=2)
    attached = _array(data, "peg_attached", ndim=1).astype(bool)
    peg_position = _array(data, "peg_position", ndim=2)
    hole_position = _array(data, "hole_position", ndim=2)
    gripper_position = _array(data, "gripper_position", ndim=2)

    full_arrays: dict[str, np.ndarray] = {}
    if is_full_modalities:
        full_arrays = {
            "state_history": _array(data, "state_history", ndim=3),
            "state_history_timestamp": _array(data, "state_history_timestamp", ndim=2),
            "force_fast": _array(data, "force_fast", ndim=3),
            "force_fast_timestamp": _array(data, "force_fast_timestamp", ndim=2),
            "force_slow": _array(data, "force_slow", ndim=3),
            "force_slow_timestamp": _array(data, "force_slow_timestamp", ndim=2),
            "visual_quality": _array(data, "visual_quality", ndim=2),
            "stage": _array(data, "stage", ndim=2),
            "modality_validity": _array(data, "modality_validity", ndim=2),
            "semantic_subtask": _array(data, "semantic_subtask", ndim=1),
            "requested_action": _array(data, "requested_action", ndim=2),
            "controller_action": _array(data, "controller_action", ndim=2),
        }
        if is_multitask:
            full_arrays.update({
                "skill_progress_phase": _array(data, "skill_progress_phase", ndim=1),
                "skill_progress_phase_name": _array(data, "skill_progress_phase_name", ndim=1),
                "skill_progress": _array(data, "skill_progress", ndim=1),
                "transition_readiness": _array(data, "transition_readiness", ndim=1),
                "skill_progress_valid": _array(data, "skill_progress_valid", ndim=1),
                "skill_progress_label_source": _array(data, "skill_progress_label_source", ndim=1),
                "skill_progress_confidence": _array(data, "skill_progress_confidence", ndim=1),
            })
        else:
            full_arrays["grasp_progress_label"] = _array(data, "grasp_progress_label", ndim=1)

    frame_arrays = {
        "state": state,
        "policy_action": policy_action,
        "expert_action": expert_action,
        "executed_action": executed_action,
        "control_mode": control_mode,
        "intervention_mask": intervention,
        "timestamp": timestamp,
        "policy_action_timestamp": policy_action_timestamp,
        "expert_action_timestamp": expert_action_timestamp,
        "executed_action_timestamp": executed_action_timestamp,
        "camera0_timestamp": camera0_timestamp,
        "camera1_timestamp": camera1_timestamp,
        "force_timestamp": force_timestamp,
        "pose_timestamp": pose_timestamp,
        "camera0": camera0,
        "camera1": camera1,
        "force": force,
        "peg_attached": attached,
        "peg_position": peg_position,
        "hole_position": hole_position,
        "gripper_position": gripper_position,
        **full_arrays,
    }
    frames = len(timestamp)
    if frames < 10:
        raise ValueError(f"episode is too short: {frames} frames")
    for key, value in frame_arrays.items():
        if len(value) != frames:
            raise ValueError(f"{key} length {len(value)} != timestamp length {frames}")
    for key, value in {
        "state": state,
        "executed_action": executed_action,
        "timestamp": timestamp,
        "camera0_timestamp": camera0_timestamp,
        "camera1_timestamp": camera1_timestamp,
        "force_timestamp": force_timestamp,
        "pose_timestamp": pose_timestamp,
        "force": force,
        "peg_position": peg_position,
        "hole_position": hole_position,
        "gripper_position": gripper_position,
    }.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or Inf")
    for key, value in {
        "state": state,
        "policy_action": policy_action,
        "expert_action": expert_action,
        "executed_action": executed_action,
    }.items():
        if value.shape[1] != ACTION_DIM:
            raise ValueError(f"{key} must have {ACTION_DIM} columns, got {value.shape}")
    for key, value in {
        "peg_position": peg_position,
        "hole_position": hole_position,
        "gripper_position": gripper_position,
    }.items():
        if value.shape[1] != 3:
            raise ValueError(f"{key} must be [T,3], got {value.shape}")
    if force.shape[1] != 6:
        raise ValueError(f"force must be [T,6], got {force.shape}")
    if camera0.shape[1:] != camera1.shape[1:] or camera0.shape[-1] != 3:
        raise ValueError("both cameras must have equal [T,H,W,3] shapes")
    if np.any(np.diff(timestamp) <= 0.0):
        raise ValueError("timestamp must be strictly increasing")

    if is_full_modalities:
        expected_shapes = {
            "state_history": (frames, STATE_HISTORY_SAMPLES, ACTION_DIM),
            "state_history_timestamp": (frames, STATE_HISTORY_SAMPLES),
            "force_fast": (frames, FAST_FORCE_SAMPLES, 6),
            "force_fast_timestamp": (frames, FAST_FORCE_SAMPLES),
            "force_slow": (frames, SLOW_FORCE_SAMPLES, 6),
            "force_slow_timestamp": (frames, SLOW_FORCE_SAMPLES),
            "visual_quality": (frames, VISUAL_QUALITY_DIM),
            "stage": (frames, STAGE_DIM),
            "modality_validity": (frames, len(MODALITY_NAMES)),
            "requested_action": (frames, ACTION_DIM),
            "controller_action": (frames, ACTION_DIM),
        }
        for key, expected in expected_shapes.items():
            if full_arrays[key].shape != expected:
                raise ValueError(f"{key} must be {expected}, got {full_arrays[key].shape}")
        for key, value in full_arrays.items():
            if key in ("semantic_subtask", "skill_progress_phase_name", "skill_progress_label_source"):
                if any(not str(item).strip() for item in value):
                    raise ValueError("semantic_subtask contains an empty label")
                continue
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains NaN or Inf")
        validity = full_arrays["modality_validity"]
        if not np.array_equal(validity, np.ones_like(validity)):
            raise ValueError("v2 modality_validity must be all ones; incomplete episodes are rejected")
        if not np.allclose(full_arrays["visual_quality"][:, 3], 1.0):
            raise ValueError("both camera streams must pass the visual-quality gate on every frame")
        stage = full_arrays["stage"]
        if np.any(stage < 0.0) or not np.allclose(stage.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("stage must contain non-negative probability rows summing to one")
        if is_multitask:
            phase = full_arrays["skill_progress_phase"].astype(np.int64)
            if not np.array_equal(phase, full_arrays["skill_progress_phase"]):
                raise ValueError("skill_progress_phase must contain integer class ids")
            if np.any((phase < 0) | (phase >= len(SKILL_PROGRESS_PHASES))):
                raise ValueError("skill_progress_phase is outside the shared vocabulary")
            names = full_arrays["skill_progress_phase_name"]
            expected_names = np.asarray([SKILL_PROGRESS_PHASES[item] for item in phase])
            if not np.array_equal(names.astype(str), expected_names):
                raise ValueError("skill_progress_phase_name does not match its class id")
            for key in ("skill_progress", "transition_readiness", "skill_progress_confidence"):
                value = full_arrays[key]
                if np.any((value < 0.0) | (value > 1.0)):
                    raise ValueError(f"{key} must be in [0,1]")
            valid = full_arrays["skill_progress_valid"]
            if not np.array_equal(valid, valid.astype(bool)):
                raise ValueError("skill_progress_valid must be boolean")
            if np.any(full_arrays["skill_progress_confidence"][~valid.astype(bool)] != 0.0):
                raise ValueError("invalid skill-progress labels must have zero confidence")
        else:
            progress = full_arrays["grasp_progress_label"].astype(np.int64)
            if not np.array_equal(progress, full_arrays["grasp_progress_label"]):
                raise ValueError("grasp_progress_label must contain integer class ids")
            if np.any((progress < 0) | (progress > 5)):
                raise ValueError("grasp_progress_label must be in [0,5]")
        for key in ("state_history_timestamp", "force_fast_timestamp", "force_slow_timestamp"):
            history_ts = full_arrays[key]
            if np.any(np.diff(history_ts, axis=1) < 0.0):
                raise ValueError(f"{key} must be non-decreasing within each causal window")
            if np.any(history_ts > timestamp[:, None] + 1e-9):
                raise ValueError(f"{key} contains future sensor samples")
        required_spans = {
            "state_history_timestamp": 0.90,
            "force_fast_timestamp": 0.55,
            "force_slow_timestamp": 4.50,
        }
        for key, minimum_span in required_spans.items():
            spans = full_arrays[key][:, -1] - full_arrays[key][:, 0]
            if np.any(spans < minimum_span):
                raise ValueError(
                    f"{key} is padded/repeated and does not cover the native history window"
                )

    if not np.isin(control_mode, [POLICY_MODE, EXPERT_MODE]).all():
        raise ValueError("control_mode may contain only 0=policy or 1=expert")
    if not np.array_equal(intervention, control_mode == EXPERT_MODE):
        raise ValueError("intervention_mask must equal (control_mode == expert)")
    expert_indices = np.flatnonzero(intervention)
    policy_indices = np.flatnonzero(~intervention)
    if len(policy_indices) < 3 or len(expert_indices) < 3:
        raise ValueError("episode needs at least three policy and three expert frames")
    takeover_index = int(expert_indices[0])
    if takeover_index == 0 or np.any(intervention[:takeover_index]):
        raise ValueError("expert intervention must follow a policy roll-in")
    if is_multi_handoff:
        # A recovery may legitimately terminate in expert mode when the final
        # correction itself reaches the verified task-success geometry. In
        # that case, forcing a policy suffix commands motion after seating.
        # Failed/incomplete episodes must still return to policy control.
        terminal_expert_success = (
            bool(intervention[-1])
            and _scalar_text(data, "recovery_outcome") == "success"
        )
        if not terminal_expert_success:
            if not np.any(~intervention[takeover_index + 1 :]):
                raise ValueError("multi-handoff episode never returns control to policy")
            if len(np.flatnonzero(~intervention[np.flatnonzero(intervention)[-1] + 1 :])) < 3:
                raise ValueError(
                    "multi-handoff episode needs a policy suffix after the final correction"
                )
    elif not np.all(intervention[takeover_index:]):
        raise ValueError("only one monotonic policy-to-expert takeover is allowed")
    if is_multitask and not np.all(
        full_arrays["skill_progress_valid"][intervention].astype(bool)
    ):
        raise ValueError("every v3 expert frame needs a valid Skill-Progress annotation")

    if not np.isfinite(policy_action_timestamp[~intervention]).all():
        raise ValueError("policy action timestamps are missing in policy mode")
    if not np.isnan(policy_action_timestamp[intervention]).all():
        raise ValueError("policy action timestamps in expert mode must be NaN")
    if not np.isfinite(expert_action_timestamp[intervention]).all():
        raise ValueError("expert action timestamps are missing in expert mode")
    if not np.isnan(expert_action_timestamp[~intervention]).all():
        raise ValueError("expert action timestamps in policy mode must be NaN")
    if not np.isfinite(policy_action[~intervention]).all():
        raise ValueError("policy actions are missing in policy mode")
    if not np.isfinite(expert_action[intervention]).all():
        raise ValueError("expert actions are missing in expert mode")
    if not np.isnan(expert_action[~intervention]).all():
        raise ValueError("expert actions in policy mode must be NaN")
    if not np.isnan(policy_action[intervention]).all():
        raise ValueError("policy actions in expert mode must be NaN")
    if not np.isfinite(executed_action_timestamp).all():
        raise ValueError("executed action timestamps are incomplete")
    if not np.allclose(
        executed_action[~intervention], policy_action[~intervention], atol=action_match_atol
    ):
        raise ValueError("executed_action does not match policy_action in policy mode")
    if not np.allclose(
        executed_action[intervention], expert_action[intervention], atol=action_match_atol
    ):
        raise ValueError("executed_action does not match expert_action in expert mode")

    camera_skew = np.maximum(
        np.abs(camera0_timestamp - timestamp), np.abs(camera1_timestamp - timestamp)
    )
    observed_max_camera_skew = float(camera_skew.max())
    if observed_max_camera_skew > max_camera_skew_s:
        raise ValueError(
            f"camera/state skew {observed_max_camera_skew:.3f}s exceeds {max_camera_skew_s:.3f}s"
        )
    force_skew = np.abs(force_timestamp - timestamp)
    observed_max_force_skew = float(force_skew.max())
    if observed_max_force_skew > max_camera_skew_s:
        raise ValueError(
            f"force/state skew {observed_max_force_skew:.3f}s exceeds {max_camera_skew_s:.3f}s"
        )
    if sustained_force_limit_n <= 0.0 or sustained_force_frames < 1:
        raise ValueError("sustained force audit parameters must be positive")
    force_norm = np.linalg.norm(force[:, :3], axis=1)
    overload = force_norm > sustained_force_limit_n
    # A full DAgger episode deliberately contains the policy's failure state.
    # Those policy actions have zero imitation weight; rejecting their contact
    # context would remove the very observation from which recovery must be
    # learned.  Expert-controlled overload remains a hard failure.
    audited_overload = (
        overload & intervention if allow_policy_failure_force_context else overload
    )
    longest_overload = 0
    current_overload = 0
    longest_overload_end = -1
    for index, overloaded in enumerate(audited_overload):
        current_overload = current_overload + 1 if overloaded else 0
        if current_overload > longest_overload:
            longest_overload = current_overload
            longest_overload_end = index
    if longest_overload >= sustained_force_frames:
        longest_overload_start = longest_overload_end - longest_overload + 1
        overload_intervention = intervention[
            longest_overload_start : longest_overload_end + 1
        ]
        source = (
            "expert"
            if bool(np.all(overload_intervention))
            else "policy"
            if not bool(np.any(overload_intervention))
            else "handoff-mixed"
        )
        raise ValueError(
            f"sustained force overload: >{sustained_force_limit_n:.1f} N for "
            f"{longest_overload} consecutive frames "
            f"[{longest_overload_start}, {longest_overload_end}], "
            f"source={source}, peak={float(force_norm.max()):.2f} N"
        )
    pose_skew = np.abs(pose_timestamp - timestamp)
    observed_max_pose_skew = float(pose_skew.max())
    if observed_max_pose_skew > max_pose_skew_s:
        raise ValueError(
            f"pose/state skew {observed_max_pose_skew:.3f}s exceeds {max_pose_skew_s:.3f}s"
        )

    recovery_success = _scalar_text(data, "recovery_outcome") == "success"
    if require_success and not recovery_success:
        raise ValueError("recovery_outcome is not success")
    if recovery_success:
        if not np.any(attached[takeover_index:]):
            raise ValueError("successful recovery never has a pose-confirmed physical grasp")
        if not attached[-1]:
            raise ValueError("successful recovery does not retain the physical grasp at the end")

        peg_hole_xy = np.linalg.norm(peg_position[:, :2] - hole_position[:, :2], axis=1)
        if recovery_phase == "grasp_lift":
            lifted_z = float(
                np.max(peg_position[takeover_index:, 2]) - peg_position[takeover_index, 2]
            )
            if lifted_z < 0.05:
                raise ValueError(f"successful grasp recovery lifts peg only {lifted_z:.4f}m")
        elif recovery_phase == "transport":
            if float(np.min(peg_hole_xy[takeover_index:])) > 0.060:
                raise ValueError("transport recovery never reaches 60 mm from the hole")
            if float(peg_hole_xy[-1]) > 0.065:
                raise ValueError("transport recovery does not finish within 65 mm of the hole")
        elif recovery_phase == "align_precontact":
            if float(np.min(peg_hole_xy[takeover_index:])) > 0.010:
                raise ValueError("align recovery never enters the 10 mm peg-hole neighbourhood")
            if float(peg_hole_xy[-1]) > 0.012:
                raise ValueError("align recovery does not finish within 12 mm of the hole")
        elif recovery_phase == "contact":
            if float(peg_hole_xy[-1]) > 0.012:
                raise ValueError("contact recovery does not finish within 12 mm of the hole")
            baseline_end = max(1, takeover_index)
            baseline_force = float(np.median(force_norm[:baseline_end]))
            contact_force = float(np.max(force_norm[takeover_index:]))
            if contact_force - baseline_force < 0.5:
                raise ValueError("contact recovery has no >=0.5 N force increase")
        elif recovery_phase in ("insertion", "full_task"):
            # The current task contract requires a verified insertion, not a
            # bottomed-out peg: PAP pure evaluation declares insertion at
            # z<=0.905 m. Keep the same strict geometry here and allow at most
            # 1 mm of asynchronous settling/rebound in the final sample.
            # The expert success event and the recorder's final aligned pose are
            # asynchronous.  Require evidence of the original strict geometry
            # in the final one-second window, then allow at most 1 mm of sampled
            # settling/rebound in the last frame.  A trajectory that never
            # actually reaches the strict insertion depth remains invalid.
            # No minimum contact-force signature is part of task success.
            terminal_start = max(takeover_index, len(peg_hole_xy) - 10)
            terminal_strict_geometry = (
                (peg_hole_xy[terminal_start:] <= 0.008)
                & (peg_position[terminal_start:, 2] <= 0.905)
            )
            final_geometry_ok = (
                float(peg_hole_xy[-1]) <= 0.008
                and float(peg_position[-1, 2]) <= 0.906
            )
            if not bool(np.any(terminal_strict_geometry)) or not final_geometry_ok:
                raise ValueError(
                    "insertion recovery does not satisfy final Gazebo geometry: "
                    f"xy={float(peg_hole_xy[-1]):.6f}m "
                    f"peg_z={float(peg_position[-1, 2]):.6f}m "
                    f"terminal_strict={bool(np.any(terminal_strict_geometry))}"
                )

    return ValidationSummary(
        frames=frames,
        policy_frames=len(policy_indices),
        expert_frames=len(expert_indices),
        takeover_index=takeover_index,
        duration_s=float(timestamp[-1] - timestamp[0]),
        max_camera_skew_s=observed_max_camera_skew,
        max_force_skew_s=observed_max_force_skew,
        max_pose_skew_s=observed_max_pose_skew,
        recovery_phase=recovery_phase,
        recovery_success=recovery_success,
    )


def load_and_validate(path: Path, **kwargs: Any) -> ValidationSummary:
    with np.load(path, allow_pickle=False) as episode:
        return validate_episode(episode, **kwargs)

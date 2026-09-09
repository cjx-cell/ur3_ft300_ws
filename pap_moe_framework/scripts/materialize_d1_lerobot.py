#!/usr/bin/env python3
"""Materialize corrected D1 into one weighted, episode-safe LeRobot dataset.

Correction chunks are represented as 50-frame synthetic episodes. Only frame
zero has positive sampling weight; the remaining frames carry future absolute
targets so LeRobot's normal action-delta lookup returns the exact stored chunk.
No correction may cross its parent episode or become an independent recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
for path in (ROOT, LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
SCRIPTS_DIR = ROOT / "pap_moe_framework/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pap_moe_framework.rollout_recovery.schema import load_and_validate


CHUNK_SIZE = 50
FPS = 10
WEIGHT_KEY = "d1.sample_weight"
VALIDITY_KEY = "observation.modality_validity"
VALIDITY_NAMES = (
    "camera0", "camera1", "state", "state_history",
    "force_current", "force_fast", "force_slow",
)
FINE_GRASP = (
    "approach the peg",
    "descend onto the peg",
    "stabilize over the peg",
    "close the gripper on the peg",
    "lift the grasped peg",
)
PHASE_TASK = {
    "transport": "transport to the hole",
    "align_precontact": "approach and align with the hole",
    "contact": "recover contact and relocate the hole",
    "insertion": "insert the peg into the hole",
    "grasp_lift": "grasp the peg",
    "full_task": "pick up the peg and insert it into the hole",
}
GRIPPER_CONTRACTS = (
    "semantic_binary",
    "pi05_hybrid",
    "pi05_continuous_action",
    "pi05_official_continuous",
    "gazebo_physical",
)
# 0.629 rad is an observed contact-limited state in legacy demonstrations,
# never a command endpoint.  Current actuator actions are universally 0.0
# (fully open command) and 0.8 (fully close command).
LEGACY_MEASURED_GRIPPER_CLOSED = 0.629
GRIPPER_OPEN_COMMAND = 0.0
GRIPPER_CLOSE_COMMAND = 0.8
GLOBAL_TASK_PROMPT = "pick up the peg and insert it into the hole"
SUBTASK_CLASSES = (
    "grasp the peg", "transport to the hole", "approach and align with the hole",
    "recover contact and relocate the hole", "insert the peg into the hole",
    "verify insertion success", "release the peg after verification",
    "retract and go back to home",
)


def _subtask_id(task: str) -> int:
    task = str(task)
    if task in FINE_GRASP or task == "grasp the peg":
        return 0
    try:
        return SUBTASK_CLASSES.index(task)
    except ValueError as exc:
        raise ValueError(f"task has no coarse subtask mapping: {task!r}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace = Path(manifest["workspace"]).resolve()
    if manifest.get("schema_version") not in {
        "pap_moe_baseline_comparison_manifest_v3",
        "pap_moe_baseline_comparison_manifest_v4_full_modal_d2",
    }:
        raise ValueError("materializer requires a corrected D1 v3 or full-modal D2 manifest")
    for section in ("success_episodes", "recovery_episodes", "qualified_corrections"):
        for entry in manifest[section]:
            asset = entry["asset"]
            path = workspace / asset["path"]
            if path.stat().st_size != int(asset["bytes"]):
                raise ValueError(f"source size changed: {path}")
            if asset.get("sha256") and _sha256(path) != asset["sha256"]:
                raise ValueError(f"source hash changed: {path}")
    return manifest, workspace


def _uint8_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype == np.uint8:
        return value
    if not np.isfinite(value).all():
        raise ValueError("camera image contains non-finite values")
    if float(value.max(initial=0.0)) <= 1.5:
        value = value * 255.0
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def _binary_joint(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[..., 6] = (result[..., 6] > 0.12).astype(np.float32)
    return result


OPEN_GRIPPER_TASKS = {
    "release the peg after verification",
    "retract and go back to home",
}


def _semantic_binary_action(values: np.ndarray, tasks: np.ndarray) -> np.ndarray:
    """Convert physical targets to the shared semantic gripper action contract.

    Old successful demonstrations store a physical gripper trajectory, while
    v3 recovery episodes store an immediate binary expert command.  Force the
    known release/retract commands open after thresholding so both sources
    supervise the same action semantics.  Observation state remains measured
    and is intentionally not overridden.
    """
    result = _binary_joint(values)
    task_values = np.asarray([str(value) for value in np.asarray(tasks).reshape(-1)])
    if result.ndim == 1:
        if len(task_values) != 1:
            raise ValueError("one action requires exactly one semantic task")
        if task_values[0] in OPEN_GRIPPER_TASKS:
            result[6] = 0.0
        return result
    if len(task_values) != len(result):
        raise ValueError(f"action/task length mismatch: {len(result)} != {len(task_values)}")
    result[np.isin(task_values, list(OPEN_GRIPPER_TASKS)), 6] = 0.0
    return result


def _continuous_semantic_action(values: np.ndarray, tasks: np.ndarray) -> np.ndarray:
    """Preserve measured gripper ramps while retaining the semantic 0..1 API.

    Legacy successful demonstrations contain a continuous physical knuckle
    trajectory (roughly 0..0.629 rad). Recovery episodes already store binary
    semantic commands. Mapping only physical inputs into 0..1 avoids turning
    the measured close ramp into an artificial discontinuity.
    """
    result = np.asarray(values, dtype=np.float32).copy()
    gripper = result[..., 6]
    binary_input = bool(
        np.all(np.isclose(gripper, 0.0, atol=1e-6) | np.isclose(gripper, 1.0, atol=1e-6))
    )
    if not binary_input:
        result[..., 6] = np.clip(gripper / LEGACY_MEASURED_GRIPPER_CLOSED, 0.0, 1.0)
    task_values = np.asarray([str(value) for value in np.asarray(tasks).reshape(-1)])
    if result.ndim == 1:
        if len(task_values) != 1:
            raise ValueError("one action requires exactly one semantic task")
        if task_values[0] in OPEN_GRIPPER_TASKS:
            result[6] = 0.0
        return result
    if len(task_values) != len(result):
        raise ValueError(f"action/task length mismatch: {len(result)} != {len(task_values)}")
    result[np.isin(task_values, list(OPEN_GRIPPER_TASKS)), 6] = 0.0
    return result


def _smooth_binary_gripper_command(
    values: np.ndarray, *, half_window: int = 10
) -> np.ndarray:
    """Replace binary command edges by ramps crossing 0.5 at the original edge."""
    gripper = np.asarray(values, dtype=np.float32).copy()
    if gripper.ndim != 1:
        raise ValueError(f"expected a one-dimensional gripper trajectory, got {gripper.shape}")
    if len(gripper) < 2:
        return gripper
    closed = gripper >= 0.5
    transitions = np.flatnonzero(closed[1:] != closed[:-1]) + 1
    for transition in transitions:
        lower = max(0, int(transition) - half_window)
        upper = min(len(gripper) - 1, int(transition) + half_window)
        start_value = float(closed[transition - 1])
        end_value = float(closed[transition])
        ramp = np.linspace(start_value, end_value, upper - lower + 1, dtype=np.float32)
        # With a symmetric window the midpoint is exactly 0.5, preserving the
        # original >=0.5 semantic actuation time used by the ROS bridge.
        gripper[lower : upper + 1] = ramp
    return gripper


def _official_continuous_action(values: np.ndarray) -> np.ndarray:
    """Produce the continuous 0..1 gripper action used by the pure Pi0.5 flow."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.ndim == 1:
        result = result[None, :]
        squeeze = True
    else:
        squeeze = False
    gripper = result[:, 6]
    finite = np.isfinite(gripper)
    finite_gripper = gripper[finite]
    binary_input = bool(
        len(finite_gripper)
        and np.all(
            np.isclose(finite_gripper, 0.0, atol=1e-6)
            | np.isclose(finite_gripper, 1.0, atol=1e-6)
        )
    )
    if binary_input:
        finite_indices = np.flatnonzero(finite)
        if len(finite_indices) > 1 and not np.all(np.diff(finite_indices) == 1):
            raise ValueError("finite binary gripper commands must form one contiguous segment")
        result[finite_indices, 6] = _smooth_binary_gripper_command(finite_gripper)
    else:
        # Native success demonstrations store physical knuckle targets. Keep
        # the complete grasp and release trajectories instead of applying
        # task-label overrides that create new discontinuities.
        result[finite, 6] = np.clip(
            finite_gripper / LEGACY_MEASURED_GRIPPER_CLOSED, 0.0, 1.0
        )
    return result[0] if squeeze else result


def _official_continuous_state(values: np.ndarray) -> np.ndarray:
    """Map measured physical knuckle positions to the same continuous 0..1 domain."""
    result = np.asarray(values, dtype=np.float32).copy()
    result[..., 6] = np.clip(
        result[..., 6] / LEGACY_MEASURED_GRIPPER_CLOSED, 0.0, 1.0
    )
    return result


def _state_for_contract(values: np.ndarray, contract: str) -> np.ndarray:
    if contract in {"semantic_binary", "pi05_hybrid", "pi05_continuous_action"}:
        return _binary_joint(values)
    if contract == "pi05_official_continuous":
        return _official_continuous_state(values)
    if contract == "gazebo_physical":
        return np.asarray(values, dtype=np.float32).copy()
    raise ValueError(f"unsupported gripper contract: {contract}")


def _history_for_contract(values: np.ndarray, contract: str) -> np.ndarray:
    """Keep Pi0.5's auxiliary history in its recorded physical joint units."""
    if contract == "semantic_binary":
        return _binary_joint(values)
    if contract in {"pi05_hybrid", "pi05_continuous_action", "gazebo_physical"}:
        return np.asarray(values, dtype=np.float32).copy()
    if contract == "pi05_official_continuous":
        return _official_continuous_state(values)
    raise ValueError(f"unsupported gripper contract: {contract}")


def _action_for_contract(
    values: np.ndarray, tasks: np.ndarray, contract: str
) -> np.ndarray:
    if contract in {"semantic_binary", "pi05_hybrid"}:
        return _semantic_binary_action(values, tasks)
    if contract == "pi05_continuous_action":
        return _continuous_semantic_action(values, tasks)
    if contract == "pi05_official_continuous":
        return _official_continuous_action(values)
    if contract != "gazebo_physical":
        raise ValueError(f"unsupported gripper contract: {contract}")
    result = np.asarray(values, dtype=np.float32).copy()
    gripper = result[..., 6]
    binary_input = bool(
        np.all(np.isclose(gripper, 0.0, atol=1e-6) | np.isclose(gripper, 1.0, atol=1e-6))
    )
    if binary_input:
        result[..., 6] = np.where(
            gripper > 0.5, GRIPPER_CLOSE_COMMAND, GRIPPER_OPEN_COMMAND
        )
    task_values = np.asarray([str(value) for value in np.asarray(tasks).reshape(-1)])
    if result.ndim == 1:
        if len(task_values) != 1:
            raise ValueError("one action requires exactly one semantic task")
        if task_values[0] in OPEN_GRIPPER_TASKS:
            result[6] = GRIPPER_OPEN_COMMAND
    else:
        if len(task_values) != len(result):
            raise ValueError(
                f"action/task length mismatch: {len(result)} != {len(task_values)}"
            )
        result[np.isin(task_values, list(OPEN_GRIPPER_TASKS)), 6] = GRIPPER_OPEN_COMMAND
    return result


def _canonicalize_legacy_recovery_hold_commands(
    actions: np.ndarray,
    states: np.ndarray,
    intervention: np.ndarray,
    tasks: np.ndarray,
) -> np.ndarray:
    """Repair legacy recovery labels that copied a contact-limited state.

    The Robotiq controller is always commanded to the universal 0.8-rad
    close endpoint.  When an object blocks the fingers, the measured knuckle
    position can settle near 0.627 rad.  Older rejoin code copied that
    measured position into subsequent expert action chunks.  Preserve the
    physical state observation, but canonicalize those held-object action
    labels back to the full-close command.  Policy roll-in frames remain
    untouched and open/release tasks remain at the open endpoint.
    """
    result = np.asarray(actions, dtype=np.float32).copy()
    state_values = np.asarray(states, dtype=np.float32)
    expert = np.asarray(intervention, dtype=bool).reshape(-1)
    task_values = np.asarray([str(value) for value in np.asarray(tasks).reshape(-1)])
    if len(result) != len(state_values) or len(result) != len(expert):
        raise ValueError("legacy recovery gripper canonicalization length mismatch")
    if len(task_values) != len(result):
        raise ValueError("legacy recovery action/task length mismatch")
    measured_as_command = (
        expert
        & ~np.isin(task_values, list(OPEN_GRIPPER_TASKS))
        # This migration targets the known legacy Robotiq/peg settle band
        # (roughly 0.60--0.65 rad).  Lower values can be legitimate samples
        # on a continuous closing trajectory and must not be collapsed.
        & (state_values[:, 6] >= 0.55)
        & (result[:, 6] >= 0.55)
        & (result[:, 6] < GRIPPER_CLOSE_COMMAND - 1.0e-4)
        & (np.abs(result[:, 6] - state_values[:, 6]) <= 0.03)
    )
    result[measured_as_command, 6] = GRIPPER_CLOSE_COMMAND
    return result


def _recovery_skill_progress_phase(raw_phase: int, recovery_phase: str) -> int:
    """Migrate the legacy grasp-clearance label without mutating raw data.

    The privileged grasp expert historically emitted shared phase ``enter``
    while raising clearance immediately after takeover.  That observation is
    a recovery state and can visually match a normal episode's initial
    ``approach`` state, so treating it as ``enter`` creates contradictory
    supervision.  Other recovery phases and labels remain byte-for-byte
    equivalent to their recorded targets.
    """
    if recovery_phase == "grasp_lift" and int(raw_phase) == 0:
        return 7
    return int(raw_phase)


def _canonical_local_progress(phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use one task-agnostic progress contract for every contiguous phase.

    ``progress`` is completion within the *current* phase and
    ``readiness`` is readiness to leave that phase for the next one.  Both
    therefore start at zero when a phase is entered.  Legacy recovery
    experts used a mixture of global task progress and readiness-to-enter
    the current phase (for example interact started at progress=.72 and
    readiness=1), which contradicted the successful demonstrations.
    """
    phases = np.asarray(phases, dtype=np.int64).reshape(-1)
    progress = np.zeros(len(phases), dtype=np.float32)
    readiness = np.zeros(len(phases), dtype=np.float32)
    start = 0
    while start < len(phases):
        end = start + 1
        while end < len(phases) and phases[end] == phases[start]:
            end += 1
        local = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
        progress[start:end] = local
        readiness[start:end] = np.clip((local - 0.75) / 0.25, 0.0, 1.0)
        start = end
    return progress, readiness


def _visual_quality(camera0: np.ndarray, camera1: np.ndarray) -> np.ndarray:
    cameras = np.stack([camera0, camera1]).astype(np.float32) / 255.0
    gray = cameras.mean(axis=-1)
    finite = bool(np.isfinite(cameras).all())
    black = float(np.mean(gray <= 0.02)) if finite else 1.0
    saturated = float(np.mean(gray >= 0.98)) if finite else 1.0
    contrast = float(np.mean(np.std(gray, axis=(1, 2)))) if finite else 0.0
    return np.asarray([black, saturated, contrast, float(finite and contrast >= 0.01)], dtype=np.float32)


def _causal_resample(
    values: np.ndarray, timestamps: np.ndarray, count: int, window_s: float
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    result = np.empty((len(values), count, *values.shape[1:]), dtype=np.float32)
    offsets = np.linspace(-window_s, 0.0, count, dtype=np.float64)
    for index, current in enumerate(timestamps):
        positions = np.searchsorted(timestamps[: index + 1], current + offsets, side="right") - 1
        result[index] = values[np.clip(positions, 0, index)]
    return result


def _fine_grasp_tasks(data: dict[str, np.ndarray]) -> np.ndarray:
    # V8 keeps the language prompt global for fairness and stores coarse
    # supervision separately.  Older datasets used ``task`` for both roles.
    label_source = data.get("semantic_subtask", data["task"])
    tasks = np.asarray([str(value) for value in label_source], dtype=object)
    grasp = np.flatnonzero(tasks == "grasp the peg")
    if not len(grasp):
        return tasks
    if not np.array_equal(grasp, np.arange(grasp[0], grasp[-1] + 1)):
        raise ValueError("coarse grasp segment is not contiguous")
    start, end = int(grasp[0]), int(grasp[-1] + 1)
    z = np.asarray(data["tool0_z"])[start:end]
    action_gripper = np.asarray(data["action"])[start:end, 6]
    state_gripper = np.asarray(data["state"])[start:end, 6]
    close_candidates = np.flatnonzero(action_gripper > 0.12)
    if not len(close_candidates):
        raise ValueError("successful grasp has no close command")
    close = int(close_candidates[0])
    peak = int(np.argmax(z[: max(close, 1)]))
    bottom = peak + int(np.flatnonzero(z[peak : close + 1] <= z[peak : close + 1].min() + 0.002)[0])
    closed = np.flatnonzero(state_gripper[close:] > 0.60)
    if not len(closed):
        raise ValueError("successful grasp never closes")
    lift = close + int(closed[0])
    boundaries = (0, peak + 1, bottom, close, lift, end - start)
    for name, lower, upper in zip(FINE_GRASP, boundaries[:-1], boundaries[1:], strict=True):
        tasks[start + lower : start + upper] = name
    return tasks


def _skill_progress_from_tasks(tasks: np.ndarray) -> dict[str, np.ndarray]:
    """Derive task-agnostic local progress from contiguous expert task segments."""
    phase_for_task = {
        "approach the peg": 1,
        "descend onto the peg": 2,
        "stabilize over the peg": 4,
        "close the gripper on the peg": 3,
        "lift the grasped peg": 6,
        "transport to the hole": 1,
        "approach and align with the hole": 2,
        "recover contact and relocate the hole": 7,
        "insert the peg into the hole": 3,
        "verify insertion success": 5,
        "release the peg after verification": 6,
        "retract and go back to home": 6,
    }
    names = [str(value) for value in tasks]
    phase = np.asarray([phase_for_task.get(name, 0) for name in names], dtype=np.int64)
    progress = np.zeros(len(names), dtype=np.float32)
    readiness = np.zeros(len(names), dtype=np.float32)
    start = 0
    while start < len(names):
        end = start + 1
        while end < len(names) and names[end] == names[start]:
            end += 1
        local = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
        progress[start:end] = local
        readiness[start:end] = np.clip((local - 0.75) / 0.25, 0.0, 1.0)
        start = end
    return {
        "phase": phase,
        "progress": progress,
        "readiness": readiness,
        "valid": np.ones(len(names), dtype=bool),
        "confidence": np.full(len(names), 0.85, dtype=np.float32),
    }


def _recovery_modalities(
    raw: dict[str, np.ndarray], phase: str, gripper_contract: str
) -> dict[str, np.ndarray]:
    schema_version = str(np.asarray(raw["schema_version"]).item())
    if schema_version in {
        "pap_moe_rollout_recovery_v2_full_modalities",
        "pap_moe_rollout_recovery_v3_multitask_full_modalities",
        "pap_moe_rollout_recovery_v4_multi_handoff_full_episode",
    }:
        return {
            "state": _state_for_contract(raw["state"], gripper_contract),
            "force": np.asarray(raw["force"], dtype=np.float32),
            "force_fast": np.asarray(raw["force_fast"], dtype=np.float32),
            "force_slow": np.asarray(raw["force_slow"], dtype=np.float32),
            "state_history": _history_for_contract(raw["state_history"], gripper_contract),
            "camera0": np.asarray(raw["camera0"], dtype=np.uint8),
            "camera1": np.asarray(raw["camera1"], dtype=np.uint8),
            "visual_quality": np.asarray(raw["visual_quality"], dtype=np.float32),
            "stage": np.asarray(raw["stage"], dtype=np.float32),
        }
    timestamps = np.asarray(raw["timestamp"], dtype=np.float64)
    state = _state_for_contract(raw["state"], gripper_contract)
    force = np.asarray(raw["force"], dtype=np.float32)
    force_fast = _causal_resample(force, timestamps, 64, 0.64)
    force_slow = _causal_resample(force, timestamps, 50, 5.0)
    history_source = _history_for_contract(raw["state"], gripper_contract)
    state_history = _causal_resample(history_source, timestamps, 10, 1.0)
    camera0 = np.asarray(raw["camera0"], dtype=np.uint8)
    camera1 = np.asarray(raw["camera1"], dtype=np.uint8)
    quality = np.stack([_visual_quality(a, b) for a, b in zip(camera0, camera1, strict=True)])

    # Recovery files do not contain a robot FK z stream. The prior therefore
    # uses only force, motion and the coarse negative-feasibility phase guard.
    from pap_moe_routing_prior import PhysicsRoutingPrior

    prior = PhysicsRoutingPrior()
    stage_number = 1 if phase in {"transport", "align_precontact"} else 3
    task = PHASE_TASK[phase]
    velocity = np.zeros(len(state), dtype=np.float32)
    if len(state) > 1:
        dt = np.maximum(np.diff(timestamps), 1e-4)
        velocity[1:] = np.linalg.norm(np.diff(state[:, :6], axis=0), axis=1) / dt
    stage = np.stack([
        prior.compute(
            force[i], force_fast[i], None, False,
            gripper_joint_val=float(state[i, 6]),
            joint_vel_norm=float(velocity[i]),
            current_stage=stage_number,
            semantic_subtask=task,
        )
        for i in range(len(state))
    ])
    return {
        "state": state,
        "force": force,
        "force_fast": force_fast,
        "force_slow": force_slow,
        "state_history": state_history,
        "camera0": camera0,
        "camera1": camera1,
        "visual_quality": quality,
        "stage": stage,
    }


def _features() -> dict[str, dict[str, Any]]:
    joints = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        "robotiq_85_left_knuckle_joint",
    ]
    return {
        "action": {"dtype": "float32", "shape": (7,), "names": joints},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": joints},
        "observation.force": {"dtype": "float32", "shape": (6,), "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
        "observation.force_fast": {"dtype": "float32", "shape": (64, 6), "names": None},
        "observation.force_slow": {"dtype": "float32", "shape": (50, 6), "names": None},
        "observation.state_history": {"dtype": "float32", "shape": (10, 7), "names": None},
        "observation.visual_quality": {"dtype": "float32", "shape": (4,), "names": ["black_fraction", "saturated_fraction", "contrast", "valid"]},
        "observation.stage": {"dtype": "float32", "shape": (4,), "names": ["E1_free", "E2_blind", "E3_rigid", "E4_compliant"]},
        "observation.physics_gate_target": {"dtype": "float32", "shape": (4,), "names": ["E1_free", "E2_blind", "E3_rigid", "E4_compliant"]},
        VALIDITY_KEY: {"dtype": "float32", "shape": (7,), "names": list(VALIDITY_NAMES)},
        WEIGHT_KEY: {"dtype": "float32", "shape": (1,), "names": ["sampling_weight"]},
        "observation.images.camera0": {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"]},
        "observation.images.camera1": {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"]},
    }


def _frame(
    *, state: np.ndarray, action: np.ndarray, force: np.ndarray,
    force_fast: np.ndarray, force_slow: np.ndarray, state_history: np.ndarray,
    visual_quality: np.ndarray, stage: np.ndarray, validity: np.ndarray,
    weight: float, camera0: np.ndarray, camera1: np.ndarray, task: str,
    skill_progress_phase: int = 7, skill_progress: float = 0.0,
    transition_readiness: float = 0.0, skill_progress_valid: bool = False,
    skill_progress_confidence: float = 0.0,
    language_task: str | None = None,
) -> dict[str, Any]:
    return {
        "observation.state": np.asarray(state, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "observation.force": np.asarray(force, dtype=np.float32),
        "observation.force_fast": np.asarray(force_fast, dtype=np.float32),
        "observation.force_slow": np.asarray(force_slow, dtype=np.float32),
        "observation.state_history": np.asarray(state_history, dtype=np.float32),
        "observation.visual_quality": np.asarray(visual_quality, dtype=np.float32),
        "observation.stage": np.asarray(stage, dtype=np.float32),
        "observation.physics_gate_target": np.asarray(stage, dtype=np.float32),
        VALIDITY_KEY: np.asarray(validity, dtype=np.float32),
        WEIGHT_KEY: np.asarray([weight], dtype=np.float32),
        "observation.images.camera0": _uint8_image(camera0),
        "observation.images.camera1": _uint8_image(camera1),
        "task": str(task if language_task is None else language_task),
    }


def _add_episode(dataset: Any, frames: list[dict[str, Any]]) -> None:
    for frame in frames:
        dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=True)


def _load_npz(path: Path, *, allow_pickle: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_pickle) as archive:
        return {key: archive[key] for key in archive.files}


def _audit_counts(manifest: dict[str, Any], workspace: Path) -> dict[str, int]:
    success = sum(int(entry["frames"]) for entry in manifest["success_episodes"])
    recovery = 0
    for entry in manifest["recovery_episodes"]:
        path = workspace / entry["asset"]["path"]
        with np.load(path, allow_pickle=False) as episode:
            phase = str(np.asarray(episode["recovery_phase"]).item())
        summary = load_and_validate(
            path,
            allow_policy_failure_force_context=(phase == "full_task"),
        )
        recovery += (
            summary.expert_frames
            if entry.get("positive_weight_scope") == "expert_frames"
            else summary.expert_frames
        )
    corrections = sum(int(entry["samples"]) for entry in manifest["qualified_corrections"])
    return {
        "success_observations": success,
        "recovery_observations": recovery,
        "correction_observations": corrections,
        "matched_failure_observations": int(manifest["counts"]["matched_failure_state_samples"]),
        "positive_weight_observations": success + recovery + corrections + int(manifest["counts"]["matched_failure_state_samples"]),
        "independent_recovery_episodes": int(
            manifest["counts"]["independent_recovery_episodes"]
        ),
    }


def _rewrite_positive_observation_stats(output: Path, gripper_contract: str) -> None:
    """Exclude zero-weight action-chunk carriers from observation statistics."""
    import pandas as pd
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.io_utils import write_stats

    stats_path = output / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    frames = pd.concat(
        [pd.read_parquet(path) for path in sorted((output / "data").glob("*/*.parquet"))],
        ignore_index=True,
    )
    positive = np.asarray(
        [float(np.asarray(value).reshape(-1)[0]) for value in frames[WEIGHT_KEY]]
    ) > 0
    keys = (
        "action", "observation.state", "observation.force", "observation.force_fast",
        "observation.force_slow", "observation.state_history",
        "observation.visual_quality", "observation.stage",
        "observation.physics_gate_target", VALIDITY_KEY,
    )
    for key in keys:
        # Arrow-backed fixed-shape extension columns in pandas cannot always
        # be selected with a boolean .loc mask (pandas may compare extension
        # dtypes through a missing private attribute). Select the individual
        # array cells explicitly and stack them instead.
        values = np.stack(
            [np.asarray(value) for value, keep in zip(frames[key].tolist(), positive) if keep]
        ).astype(np.float32)
        feature_stats = get_feature_stats(values, axis=0, keepdims=False)
        stats[key] = {name: np.asarray(value).tolist() for name, value in feature_stats.items()}

    # The semantic gripper coordinates are exact bits. Episode-level
    # approximate quantile aggregation can otherwise produce values such as
    # action.q01=0.054 even though the column contains only 0 and 1. Keep the
    # normalization endpoints exact so 0 -> -1 and 1 -> +1 in every model.
    if gripper_contract in {
        "semantic_binary",
        "pi05_hybrid",
        "pi05_continuous_action",
    }:
        binary_keys = ["action", "observation.state"]
        if gripper_contract == "semantic_binary":
            binary_keys.append("observation.state_history")
        for key in binary_keys:
            for statistic, endpoint in (("min", 0.0), ("q01", 0.0), ("q99", 1.0), ("max", 1.0)):
                values = np.asarray(stats[key][statistic], dtype=np.float32)
                values[..., 6] = endpoint
                stats[key][statistic] = values.tolist()
    write_stats(stats, output)


def materialize(
    manifest_path: Path, output: Path, gripper_contract: str = "semantic_binary",
    *, global_task_prompt: bool = False,
    recovery_entry_anchor_weight: float | None = None,
    recovery_entry_anchor_xy_m: float = 0.003,
    recovery_entry_anchor_min_peg_z_m: float = 0.960,
    recovery_entry_anchor_min_future_arm_span_rad: float = 0.020,
    recovery_takeover_anchor_weight: float | None = None,
    recovery_takeover_anchor_window: int = 50,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest, workspace = _verify_manifest(manifest_path)
    if gripper_contract not in GRIPPER_CONTRACTS:
        raise ValueError(f"unsupported gripper contract: {gripper_contract}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=f"pap_moe/{manifest['dataset_id']}", root=output, fps=FPS,
        features=_features(), robot_type="ur3", use_videos=True, vcodec="h264",
    )
    receipt: dict[str, Any] = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "weight_key": WEIGHT_KEY,
        "gripper_contract": gripper_contract,
        "episodes": [],
        "split_episode_indices": {"train": [], "validation": []},
        "recovery_entry_anchor_frames": 0,
        "recovery_takeover_anchor_frames": 0,
    }

    def record(source_id: str, parent: str, split: str, kind: str, positive: int) -> None:
        index = len(receipt["episodes"])
        receipt["episodes"].append({
            "dataset_episode_index": index, "source_id": source_id,
            "parent_episode_id": parent, "split": split, "kind": kind,
            "positive_weight_frames": positive,
        })
        receipt["split_episode_indices"][split].append(index)

    for entry in manifest["success_episodes"]:
        data = _load_npz(workspace / entry["asset"]["path"], allow_pickle=True)
        tasks = _fine_grasp_tasks(data)
        skill = _skill_progress_from_tasks(tasks)
        states = _state_for_contract(data["state"], gripper_contract)
        actions = _action_for_contract(data["action"], tasks, gripper_contract)
        histories = _history_for_contract(data["state_history"], gripper_contract)
        frames = [
            _frame(
                state=states[i], action=actions[i], force=data["force"][i],
                force_fast=data["force_fast"][i], force_slow=data["force_slow"][i],
                state_history=histories[i], visual_quality=data["visual_quality"][i],
                stage=data["stage"][i], validity=np.ones(7), weight=1.0,
                camera0=data["camera0"][i], camera1=data["camera1"][i], task=tasks[i],
                skill_progress_phase=int(skill["phase"][i]),
                skill_progress=float(skill["progress"][i]),
                transition_readiness=float(skill["readiness"][i]),
                skill_progress_valid=bool(skill["valid"][i]),
                skill_progress_confidence=float(skill["confidence"][i]),
                language_task=GLOBAL_TASK_PROMPT if global_task_prompt else None,
            )
            for i in range(len(states))
        ]
        _add_episode(dataset, frames)
        record(entry["episode_id"], entry["episode_id"], entry["split"], entry["kind"], len(frames))

    recovery_cache: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, str]] = {}
    for entry in manifest["recovery_episodes"]:
        path = workspace / entry["asset"]["path"]
        raw = _load_npz(path, allow_pickle=False)
        phase = str(np.asarray(raw["recovery_phase"]).item())
        summary = load_and_validate(
            path,
            # Full-task DAgger data deliberately retain the policy's failed
            # contact context with zero imitation weight.  Match the recorder
            # and standalone validator: expert overload is still rejected.
            allow_policy_failure_force_context=(phase == "full_task"),
        )
        modalities = _recovery_modalities(raw, phase, gripper_contract)
        is_full_episode = (
            str(np.asarray(raw["schema_version"]).item())
            == "pap_moe_rollout_recovery_v4_multi_handoff_full_episode"
        )
        start = 0 if is_full_episode else summary.takeover_index
        if is_full_episode:
            recovery_tasks = np.asarray(raw["semantic_subtask"], dtype=str)
            action_source = raw["executed_action"]
        else:
            recovery_tasks = np.asarray(
                [PHASE_TASK[phase]] * len(raw["expert_action"]), dtype=object
            )
            action_source = raw["expert_action"]
        actions = _action_for_contract(
            action_source, recovery_tasks, gripper_contract
        )
        if is_full_episode and gripper_contract == "gazebo_physical":
            actions = _canonicalize_legacy_recovery_hold_commands(
                actions,
                raw["state"],
                raw["intervention_mask"],
                recovery_tasks,
            )
        if is_full_episode:
            recovery_phases = np.asarray(raw["skill_progress_phase"][start:], dtype=np.int64)
            recovery_progress = np.asarray(raw["skill_progress"][start:], dtype=np.float32)
            recovery_readiness = np.asarray(raw["transition_readiness"][start:], dtype=np.float32)
        else:
            recovery_phases = np.asarray(
                [
                    _recovery_skill_progress_phase(raw["skill_progress_phase"][i], phase)
                    if "skill_progress_phase" in raw else 7
                    for i in range(start, len(actions))
                ],
                dtype=np.int64,
            )
            recovery_progress, recovery_readiness = _canonical_local_progress(recovery_phases)
        sample_weight = float(entry.get("sample_weight", 1.0))
        intervention = np.asarray(raw["intervention_mask"], dtype=bool)
        frame_weights = np.where(intervention, sample_weight, 0.0).astype(np.float32)
        if is_full_episode and recovery_takeover_anchor_weight is not None:
            previous_intervention = np.concatenate(
                [np.asarray([False]), intervention[:-1]]
            )
            takeover_starts = np.flatnonzero(intervention & ~previous_intervention)
            takeover_anchor = np.zeros_like(intervention)
            for takeover_start in takeover_starts:
                takeover_stop = min(
                    len(intervention),
                    int(takeover_start) + recovery_takeover_anchor_window,
                )
                takeover_anchor[takeover_start:takeover_stop] = True
            # Never give policy roll-in frames a positive target.  The anchor
            # begins at the first expert command, whose observation is the
            # actual failure state handed over by the policy.
            takeover_anchor &= intervention
            frame_weights[takeover_anchor] = np.maximum(
                frame_weights[takeover_anchor], recovery_takeover_anchor_weight
            )
            receipt["recovery_takeover_anchor_frames"] += int(takeover_anchor.sum())
        if is_full_episode and recovery_entry_anchor_weight is not None:
            peg = np.asarray(raw["peg_position"], dtype=np.float32)
            hole = np.asarray(raw["hole_position"], dtype=np.float32)
            peg_hole_xy = np.linalg.norm(peg[:, :2] - hole[:, :2], axis=1)
            arm_actions = np.asarray(raw["executed_action"], dtype=np.float32)[:, :6]
            future = np.minimum(
                np.arange(len(arm_actions), dtype=np.int64) + CHUNK_SIZE - 1,
                len(arm_actions) - 1,
            )
            future_arm_span = np.linalg.norm(arm_actions[future] - arm_actions, axis=1)
            entry_anchor = (
                intervention
                & (peg_hole_xy <= recovery_entry_anchor_xy_m)
                & (peg[:, 2] >= recovery_entry_anchor_min_peg_z_m)
                & (future_arm_span >= recovery_entry_anchor_min_future_arm_span_rad)
            )
            frame_weights[entry_anchor] = np.maximum(
                frame_weights[entry_anchor], recovery_entry_anchor_weight
            )
            receipt["recovery_entry_anchor_frames"] += int(entry_anchor.sum())
        frames = [
            _frame(
                state=modalities["state"][i], action=actions[i], force=modalities["force"][i],
                force_fast=modalities["force_fast"][i], force_slow=modalities["force_slow"][i],
                state_history=modalities["state_history"][i], visual_quality=modalities["visual_quality"][i],
                stage=modalities["stage"][i], validity=np.ones(7),
                weight=(sample_weight if not is_full_episode else float(frame_weights[i])),
                camera0=modalities["camera0"][i], camera1=modalities["camera1"][i], task=str(recovery_tasks[i]),
                skill_progress_phase=int(recovery_phases[i - start]),
                skill_progress=float(recovery_progress[i - start]),
                transition_readiness=float(recovery_readiness[i - start]),
                skill_progress_valid=(bool(raw["skill_progress_valid"][i]) if "skill_progress_valid" in raw else False),
                skill_progress_confidence=(float(raw["skill_progress_confidence"][i]) if "skill_progress_confidence" in raw else 0.0),
                language_task=GLOBAL_TASK_PROMPT if global_task_prompt else None,
            )
            for i in range(start, len(actions))
        ]
        _add_episode(dataset, frames)
        recovery_cache[entry["episode_id"]] = (raw, modalities, start, phase)
        positive_frames = int(intervention[start:].sum()) if is_full_episode else len(frames)
        record(entry["episode_id"], entry["episode_id"], "train", entry["kind"], positive_frames)

    anchor_cache: dict[Path, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    for entry in manifest["qualified_corrections"]:
        pack_path = workspace / entry["asset"]["path"]
        pack = _load_npz(pack_path, allow_pickle=False)
        parent = entry["parent_episode_id"]
        native = parent.startswith("recovery:")
        if native:
            raw, modalities, _, phase = recovery_cache[parent]
            raw_frames = np.asarray(pack["raw_frame"], dtype=np.int64)
            tasks = np.asarray([PHASE_TASK[phase]] * len(raw_frames), dtype=object)
        else:
            sidecar = json.loads((workspace / entry["sidecar"]["path"]).read_text())
            anchor_path = Path(sidecar["episode"]).resolve()
            if anchor_path not in anchor_cache:
                anchor = _load_npz(anchor_path, allow_pickle=True)
                anchor_cache[anchor_path] = (anchor, _fine_grasp_tasks(anchor))
            anchor, anchor_tasks = anchor_cache[anchor_path]
            demo = np.asarray(pack["demo_frame"], dtype=np.int64)
            tasks = anchor_tasks[demo]
        for sample in range(len(pack["state"])):
            camera0 = _uint8_image(pack["camera0"][sample])
            camera1 = _uint8_image(pack["camera1"][sample])
            state = _state_for_contract(pack["state"][sample], gripper_contract)
            target = _action_for_contract(
                pack["target_action"][sample],
                np.asarray([str(tasks[sample])] * len(pack["target_action"][sample]), dtype=object),
                gripper_contract,
            )
            if native:
                raw_index = int(raw_frames[sample])
                force = modalities["force"][raw_index]
                force_fast = modalities["force_fast"][raw_index]
                force_slow = modalities["force_slow"][raw_index]
                history = modalities["state_history"][raw_index]
                quality = modalities["visual_quality"][raw_index]
                stage = modalities["stage"][raw_index]
                validity = np.ones(7, dtype=np.float32)
            else:
                force = np.zeros(6, dtype=np.float32)
                force_fast = np.zeros((64, 6), dtype=np.float32)
                force_slow = np.zeros((50, 6), dtype=np.float32)
                history = np.zeros((10, 7), dtype=np.float32)
                quality = _visual_quality(camera0, camera1)
                stage = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                validity = np.asarray([1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
            frames = [
                _frame(
                    state=state, action=target[offset], force=force, force_fast=force_fast,
                    force_slow=force_slow, state_history=history, visual_quality=quality,
                    stage=stage, validity=validity, weight=4.0 if offset == 0 else 0.0,
                    camera0=camera0, camera1=camera1, task=str(tasks[sample]),
                    language_task=GLOBAL_TASK_PROMPT if global_task_prompt else None,
                )
                for offset in range(CHUNK_SIZE)
            ]
            _add_episode(dataset, frames)
            record(entry["source_id"], parent, "train", entry["kind"], 1)

    dataset.finalize()
    _rewrite_positive_observation_stats(output, gripper_contract)
    receipt["counts"] = _audit_counts(manifest, workspace)
    receipt["independent_recovery_episodes"] = int(
        manifest["counts"]["independent_recovery_episodes"]
    )
    receipt["normalization_stats"] = (
        "observation features use only positive-weight rows; action keeps all chunk targets"
    )
    receipt["recovery_entry_anchor_contract"] = (
        None
        if recovery_entry_anchor_weight is None
        else {
            "weight": recovery_entry_anchor_weight,
            "max_peg_hole_xy_m": recovery_entry_anchor_xy_m,
            "min_peg_z_m": recovery_entry_anchor_min_peg_z_m,
            "min_future_50_arm_span_rad": recovery_entry_anchor_min_future_arm_span_rad,
            "scope": "offline sampling metadata only; never a policy input",
        }
    )
    receipt["recovery_takeover_anchor_contract"] = (
        None
        if recovery_takeover_anchor_weight is None
        else {
            "weight": recovery_takeover_anchor_weight,
            "expert_frames_after_each_takeover": recovery_takeover_anchor_window,
            "source": "logged intervention-mask transition only",
            "scope": "offline sampling metadata only; never a policy input",
        }
    )
    receipt["semantic_head_contract"] = {
        "subtask_head": "removed",
        "skill_progress_head": "removed",
        "physics_routing_only": True,
    }
    receipt_path = output / "meta/d1_materialization.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "pap_moe_framework/datasets/manifests/pap_moe_d1_v3.json")
    parser.add_argument("--output", type=Path, default=ROOT / "pap_moe_framework/datasets/lerobot_d1_v3")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--gripper-contract", choices=GRIPPER_CONTRACTS, default="semantic_binary"
    )
    parser.add_argument(
        "--global-task-prompt",
        action="store_true",
        help="Keep one global language instruction; store stages only as auxiliary labels",
    )
    parser.add_argument("--recovery-entry-anchor-weight", type=float, default=None)
    parser.add_argument("--recovery-entry-anchor-xy-m", type=float, default=0.003)
    parser.add_argument("--recovery-entry-anchor-min-peg-z-m", type=float, default=0.960)
    parser.add_argument(
        "--recovery-entry-anchor-min-future-arm-span-rad", type=float, default=0.020
    )
    parser.add_argument("--recovery-takeover-anchor-weight", type=float, default=None)
    parser.add_argument("--recovery-takeover-anchor-window", type=int, default=50)
    args = parser.parse_args()
    if args.recovery_entry_anchor_weight is not None and args.recovery_entry_anchor_weight <= 0:
        parser.error("--recovery-entry-anchor-weight must be positive")
    if args.recovery_entry_anchor_xy_m <= 0:
        parser.error("--recovery-entry-anchor-xy-m must be positive")
    if args.recovery_entry_anchor_min_future_arm_span_rad <= 0:
        parser.error("--recovery-entry-anchor-min-future-arm-span-rad must be positive")
    if args.recovery_takeover_anchor_weight is not None and args.recovery_takeover_anchor_weight <= 0:
        parser.error("--recovery-takeover-anchor-weight must be positive")
    if args.recovery_takeover_anchor_window <= 0:
        parser.error("--recovery-takeover-anchor-window must be positive")
    manifest, workspace = _verify_manifest(args.manifest)
    if args.audit_only:
        print(json.dumps(_audit_counts(manifest, workspace), indent=2))
        return 0
    receipt = materialize(
        args.manifest, args.output.resolve(), gripper_contract=args.gripper_contract,
        global_task_prompt=args.global_task_prompt,
        recovery_entry_anchor_weight=args.recovery_entry_anchor_weight,
        recovery_entry_anchor_xy_m=args.recovery_entry_anchor_xy_m,
        recovery_entry_anchor_min_peg_z_m=args.recovery_entry_anchor_min_peg_z_m,
        recovery_entry_anchor_min_future_arm_span_rad=(
            args.recovery_entry_anchor_min_future_arm_span_rad
        ),
        recovery_takeover_anchor_weight=args.recovery_takeover_anchor_weight,
        recovery_takeover_anchor_window=args.recovery_takeover_anchor_window,
    )
    print(json.dumps(receipt["counts"], indent=2))
    print(f"Materialized D1: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

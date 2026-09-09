"""Atomic IPC protocol for policy rollout to expert recovery.

The normal inference/controller files remain the deployment contract.  This
module is used only when an explicit recovery session directory is configured.
The ROS execution side writes authoritative action events after applying its
safety clamp; an independent expert process writes chunks into ``input/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np


PROTOCOL_VERSION = "pap_moe_rollout_recovery_ipc_v2_multi_handoff"
ACTION_DIM = 7
POLICY_MODE = 0
EXPERT_MODE = 1
SKILL_PROGRESS_PHASES = (
    "enter",
    "approach",
    "align",
    "interact",
    "stabilize",
    "verify",
    "exit",
    "recover",
)


@dataclass(frozen=True)
class ExpertChunk:
    sequence: int
    timestamp: float
    action_chunk: np.ndarray
    publisher: str
    skill_progress_phase: int
    skill_progress: float
    transition_readiness: float
    label_confidence: float


@dataclass(frozen=True)
class ActionEvent:
    sequence: int
    mode: int
    source_sequence: int
    dispatch_timestamp: float
    action_timestamp: np.ndarray
    policy_action: np.ndarray
    expert_action: np.ndarray
    executed_action: np.ndarray
    requested_action: np.ndarray
    controller_action: np.ndarray


@dataclass(frozen=True)
class ExpertExecutionAck:
    """Authoritative acknowledgement emitted after a trajectory has finished."""

    expert_sequence: int
    action_event_sequence: int
    completed_timestamp: float


@dataclass(frozen=True)
class ActionCompletion:
    action_event_sequence: int
    mode: int
    source_sequence: int
    completed_timestamp: float


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_savez(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def validate_semantic_action_chunk(action_chunk: np.ndarray) -> np.ndarray:
    """Return a finite float32 [K,7] chunk with a strictly binary gripper."""
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != ACTION_DIM:
        raise ValueError(f"action chunk must have shape [K,{ACTION_DIM}], got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk contains NaN or Inf")
    if not np.all(np.logical_or(np.isclose(chunk[:, 6], 0.0), np.isclose(chunk[:, 6], 1.0))):
        raise ValueError("semantic gripper commands must be exactly 0=open or 1=closed")
    return chunk.copy()


def validate_recorded_action_chunk(action_chunk: np.ndarray) -> np.ndarray:
    """Validate recorded actions while preserving continuous gripper values."""
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != ACTION_DIM:
        raise ValueError(f"action chunk must have shape [K,{ACTION_DIM}], got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk contains NaN or Inf")
    # Pi0.5 uses physical 0..0.8 rad; legacy/PAP events may use semantic 0/1.
    if np.any(chunk[:, 6] < 0.0) or np.any(chunk[:, 6] > 1.0):
        raise ValueError("recorded gripper commands must stay in [0,1]")
    return chunk.copy()


def clamp_for_controller(
    requested_chunk: np.ndarray,
    current_state: np.ndarray,
    *,
    max_arm_step_rad: float,
    gripper_open_rad: float,
    gripper_closed_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the execution-side causal clamp.

    Returns ``(executed_semantic, controller_physical)``.  The former retains
    binary dataset gripper semantics and is used by the recovery schema; the
    latter is the exact target sent to FollowJointTrajectory.
    """
    requested = validate_semantic_action_chunk(requested_chunk)
    current = np.asarray(current_state, dtype=np.float32)
    if current.shape != (ACTION_DIM,) or not np.isfinite(current).all():
        raise ValueError(f"current_state must be finite [{ACTION_DIM}], got {current.shape}")
    if max_arm_step_rad <= 0.0:
        raise ValueError("max_arm_step_rad must be positive")

    executed = requested.copy()
    previous = current[:6].copy()
    for index in range(len(executed)):
        delta = np.clip(
            executed[index, :6] - previous,
            -max_arm_step_rad,
            max_arm_step_rad,
        )
        executed[index, :6] = previous + delta
        previous = executed[index, :6]

    controller = executed.copy()
    controller[:, 6] = np.where(
        executed[:, 6] >= 0.5,
        np.float32(gripper_closed_rad),
        np.float32(gripper_open_rad),
    )
    return executed, controller


def clamp_continuous_for_controller(
    requested_chunk: np.ndarray,
    current_state: np.ndarray,
    *,
    max_arm_step_rad: float,
    gripper_open_rad: float,
    gripper_closed_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clamp a physical-radian action chunk without changing its semantics."""
    requested = np.asarray(requested_chunk, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    if requested.ndim != 2 or requested.shape[0] < 1 or requested.shape[1] != ACTION_DIM:
        raise ValueError(f"action chunk must have shape [K,{ACTION_DIM}], got {requested.shape}")
    if not np.isfinite(requested).all():
        raise ValueError("continuous action chunk contains NaN or Inf")
    if current.shape != (ACTION_DIM,) or not np.isfinite(current).all():
        raise ValueError(f"current_state must be finite [{ACTION_DIM}], got {current.shape}")
    if max_arm_step_rad <= 0.0:
        raise ValueError("max_arm_step_rad must be positive")
    if not 0.0 <= gripper_open_rad < gripper_closed_rad:
        raise ValueError("invalid physical gripper interval")

    executed = requested.copy()
    previous = current[:6].copy()
    for index in range(len(executed)):
        delta = np.clip(
            executed[index, :6] - previous,
            -max_arm_step_rad,
            max_arm_step_rad,
        )
        executed[index, :6] = previous + delta
        previous = executed[index, :6]
    executed[:, 6] = np.clip(
        executed[:, 6], gripper_open_rad, gripper_closed_rad
    )
    # The dataset/action representation already is the actuator representation.
    return executed, executed.copy()


def initialize_session(session_dir: Path, metadata: Mapping[str, Any]) -> None:
    """Initialize a new session and refuse to reuse an event-bearing directory."""
    session_dir = Path(session_dir)
    event_dir = session_dir / "action_events"
    if event_dir.exists() and any(event_dir.glob("event_*.npz")):
        raise FileExistsError(f"recovery session already contains action events: {session_dir}")
    ack_path = session_dir / "output" / "expert_execution_ack.json"
    if ack_path.exists():
        raise FileExistsError(f"recovery session already contains an execution ACK: {session_dir}")
    completion_dir = session_dir / "completion_events"
    if completion_dir.exists() and any(completion_dir.glob("completion_*.json")):
        raise FileExistsError(f"recovery session already contains completion events: {session_dir}")
    control_dir = session_dir / "control_events"
    if control_dir.exists() and any(control_dir.glob("control_*.json")):
        raise FileExistsError(f"recovery session already contains control events: {session_dir}")
    event_dir.mkdir(parents=True, exist_ok=True)
    completion_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "input").mkdir(parents=True, exist_ok=True)
    (session_dir / "output").mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        session_dir / "session.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "created_timestamp": time.time(),
            **dict(metadata),
        },
    )


def request_control_mode(
    session_dir: Path,
    *,
    mode: int,
    sequence: int,
    reason: str,
    requester: str,
) -> None:
    """Atomically request one auditable policy/expert handoff.

    ``control_events`` is immutable history; ``input/control.json`` is only the
    latest-value mailbox consumed by the ROS execution side.
    """
    if mode not in (POLICY_MODE, EXPERT_MODE):
        raise ValueError("control mode must be policy or expert")
    if sequence < 0 or not reason.strip() or not requester.strip():
        raise ValueError("control sequence, reason and requester must be valid")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "sequence": int(sequence),
        "mode": int(mode),
        "active": mode == EXPERT_MODE,
        "timestamp": time.time(),
        "trigger": reason,
        "reason": reason,
        "requester": requester,
    }
    event_path = Path(session_dir) / "control_events" / f"control_{sequence:08d}.json"
    if event_path.exists():
        raise FileExistsError(f"control event already exists: {event_path}")
    existing = sorted((Path(session_dir) / "control_events").glob("control_*.json"))
    if existing:
        previous = int(existing[-1].stem.split("_")[-1])
        if sequence != previous + 1:
            raise ValueError("control sequence must advance contiguously")
    elif sequence != 0:
        raise ValueError("first control sequence must be zero")
    _atomic_write_json(event_path, payload)
    _atomic_write_json(Path(session_dir) / "input" / "control.json", payload)
    # Compatibility mailbox for older recorder/operator tools.
    _atomic_write_json(
        Path(session_dir) / "input" / "takeover.json",
        payload,
    )


def request_takeover(
    session_dir: Path, *, trigger: str, requester: str, sequence: int = 0
) -> None:
    request_control_mode(
        session_dir,
        mode=EXPERT_MODE,
        sequence=sequence,
        reason=trigger,
        requester=requester,
    )


def release_to_policy(
    session_dir: Path, *, reason: str, requester: str, sequence: int
) -> None:
    request_control_mode(
        session_dir,
        mode=POLICY_MODE,
        sequence=sequence,
        reason=reason,
        requester=requester,
    )


def read_control_mode(session_dir: Path) -> dict[str, Any] | None:
    path = Path(session_dir) / "input" / "control.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("invalid recovery control request version")
    if int(payload.get("mode", -1)) not in (POLICY_MODE, EXPERT_MODE):
        raise ValueError("invalid recovery control mode")
    if int(payload.get("sequence", -1)) < 0 or not str(payload.get("reason", "")).strip():
        raise ValueError("invalid recovery control request")
    return payload


def read_takeover(session_dir: Path) -> dict[str, Any] | None:
    payload = read_control_mode(session_dir)
    if payload is None or int(payload["mode"]) != EXPERT_MODE:
        return None
    return payload


def publish_expert_chunk(
    session_dir: Path,
    action_chunk: np.ndarray,
    *,
    sequence: int,
    publisher: str,
    timestamp: float | None = None,
    skill_progress_phase: int = 7,
    skill_progress: float = 0.0,
    transition_readiness: float = 0.0,
    label_confidence: float = 0.0,
) -> Path:
    if sequence < 0 or not publisher.strip():
        raise ValueError("expert sequence must be non-negative and publisher non-empty")
    # Recovery actions use the same continuous physical-radian gripper domain
    # as Pi0.5 and the scripted success demonstrations (0.0 open, 0.8 full
    # close command).  Keeping the expert mailbox binary would make it
    # impossible to record the gradual close-and-lift trajectory required for
    # a stable physical grasp.
    chunk = validate_recorded_action_chunk(action_chunk)
    if not 0 <= int(skill_progress_phase) < len(SKILL_PROGRESS_PHASES):
        raise ValueError("skill_progress_phase is outside the shared phase vocabulary")
    for name, value in {
        "skill_progress": skill_progress,
        "transition_readiness": transition_readiness,
        "label_confidence": label_confidence,
    }.items():
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1]")
    published_timestamp = time.time() if timestamp is None else timestamp
    path = Path(session_dir) / "input" / "expert_chunk.npz"
    payload = dict(
        protocol_version=np.asarray(PROTOCOL_VERSION),
        sequence=np.asarray(sequence, dtype=np.int64),
        timestamp=np.asarray(published_timestamp, dtype=np.float64),
        publisher=np.asarray(publisher),
        action_chunk=chunk,
        skill_progress_phase=np.asarray(skill_progress_phase, dtype=np.int64),
        skill_progress_phase_name=np.asarray(SKILL_PROGRESS_PHASES[skill_progress_phase]),
        skill_progress=np.asarray(skill_progress, dtype=np.float32),
        transition_readiness=np.asarray(transition_readiness, dtype=np.float32),
        label_confidence=np.asarray(label_confidence, dtype=np.float32),
    )
    # Immutable annotations let the recorder recover the label belonging to an
    # executed source_sequence even after the latest-value mailbox advances.
    annotation_path = (
        Path(session_dir) / "input" / "expert_annotations" / f"annotation_{sequence:08d}.npz"
    )
    if annotation_path.exists():
        raise FileExistsError(f"expert annotation already exists: {annotation_path}")
    _atomic_savez(annotation_path, **payload)
    _atomic_savez(path, **payload)
    return path


def read_expert_chunk(
    session_dir: Path,
    *,
    after_sequence: int,
    max_age_s: float,
    now: float | None = None,
) -> ExpertChunk | None:
    path = Path(session_dir) / "input" / "expert_chunk.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["protocol_version"]).item()) != PROTOCOL_VERSION:
                raise ValueError("expert chunk protocol version mismatch")
            sequence = int(np.asarray(data["sequence"]).item())
            timestamp = float(np.asarray(data["timestamp"]).item())
            publisher = str(np.asarray(data["publisher"]).item())
            chunk = validate_recorded_action_chunk(data["action_chunk"])
            phase = int(np.asarray(data["skill_progress_phase"]).item())
            progress = float(np.asarray(data["skill_progress"]).item())
            readiness = float(np.asarray(data["transition_readiness"]).item())
            confidence = float(np.asarray(data["label_confidence"]).item())
    except (OSError, EOFError, KeyError, ValueError):
        return None
    if sequence <= after_sequence:
        return None
    current_time = time.time() if now is None else now
    if timestamp > current_time + 0.25 or current_time - timestamp > max_age_s:
        return None
    if not publisher.strip():
        return None
    if not 0 <= phase < len(SKILL_PROGRESS_PHASES):
        return None
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in (progress, readiness, confidence)):
        return None
    return ExpertChunk(sequence, timestamp, chunk, publisher, phase, progress, readiness, confidence)


def load_expert_annotation(session_dir: Path, sequence: int) -> ExpertChunk:
    """Load the immutable semantic annotation for an executed expert chunk."""
    path = Path(session_dir) / "input" / "expert_annotations" / f"annotation_{sequence:08d}.npz"
    with np.load(path, allow_pickle=False) as data:
        if str(np.asarray(data["protocol_version"]).item()) != PROTOCOL_VERSION:
            raise ValueError("expert annotation protocol version mismatch")
        result = ExpertChunk(
            sequence=int(np.asarray(data["sequence"]).item()),
            timestamp=float(np.asarray(data["timestamp"]).item()),
            action_chunk=validate_recorded_action_chunk(data["action_chunk"]),
            publisher=str(np.asarray(data["publisher"]).item()),
            skill_progress_phase=int(np.asarray(data["skill_progress_phase"]).item()),
            skill_progress=float(np.asarray(data["skill_progress"]).item()),
            transition_readiness=float(np.asarray(data["transition_readiness"]).item()),
            label_confidence=float(np.asarray(data["label_confidence"]).item()),
        )
    if result.sequence != sequence or not 0 <= result.skill_progress_phase < len(SKILL_PROGRESS_PHASES):
        raise ValueError("invalid expert annotation sequence or phase")
    return result


def write_expert_execution_ack(
    session_dir: Path,
    *,
    expert_sequence: int,
    action_event_sequence: int,
    completed_timestamp: float | None = None,
) -> Path:
    """Acknowledge one expert chunk only after its ROS trajectory succeeded.

    The file is a latest-value mailbox.  Sequence monotonicity is enforced
    against an existing ACK so a delayed writer cannot move it backwards.
    """
    if expert_sequence < 0 or action_event_sequence < 0:
        raise ValueError("execution ACK sequences must be non-negative")
    path = Path(session_dir) / "output" / "expert_execution_ack.json"
    previous = read_expert_execution_ack(session_dir)
    if previous is not None:
        if expert_sequence <= previous.expert_sequence:
            raise ValueError(
                "expert execution ACK must advance monotonically: "
                f"previous={previous.expert_sequence}, requested={expert_sequence}"
            )
        if action_event_sequence <= previous.action_event_sequence:
            raise ValueError("action-event ACK sequence must advance monotonically")
    _atomic_write_json(
        path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "expert_sequence": expert_sequence,
            "action_event_sequence": action_event_sequence,
            "completed_timestamp": (
                time.time() if completed_timestamp is None else float(completed_timestamp)
            ),
            "status": "succeeded",
        },
    )
    return path


def read_expert_execution_ack(session_dir: Path) -> ExpertExecutionAck | None:
    path = Path(session_dir) / "output" / "expert_execution_ack.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("execution ACK protocol version mismatch")
        if payload.get("status") != "succeeded":
            raise ValueError("execution ACK status is not succeeded")
        ack = ExpertExecutionAck(
            expert_sequence=int(payload["expert_sequence"]),
            action_event_sequence=int(payload["action_event_sequence"]),
            completed_timestamp=float(payload["completed_timestamp"]),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if (
        ack.expert_sequence < 0
        or ack.action_event_sequence < 0
        or not np.isfinite(ack.completed_timestamp)
    ):
        return None
    return ack


def wait_for_expert_execution_ack(
    session_dir: Path,
    *,
    expert_sequence: int,
    timeout_s: float,
    poll_interval_s: float = 0.01,
) -> ExpertExecutionAck | None:
    """Wait until the exact expert sequence has completed.

    Seeing a later sequence is a protocol violation: an expert must never
    have more than one unacknowledged chunk in flight.
    """
    if expert_sequence < 0 or timeout_s <= 0.0 or poll_interval_s <= 0.0:
        raise ValueError("invalid execution ACK wait arguments")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ack = read_expert_execution_ack(session_dir)
        if ack is not None:
            if ack.expert_sequence == expert_sequence:
                return ack
            if ack.expert_sequence > expert_sequence:
                raise RuntimeError(
                    "execution ACK skipped requested expert sequence: "
                    f"requested={expert_sequence}, observed={ack.expert_sequence}"
                )
        time.sleep(poll_interval_s)
    return None


def write_action_completion(
    session_dir: Path,
    *,
    action_event_sequence: int,
    mode: int,
    source_sequence: int,
    completed_timestamp: float | None = None,
) -> Path:
    """Persist the wall-clock instant at which one full trajectory succeeded."""
    if action_event_sequence < 0 or source_sequence < 0:
        raise ValueError("action completion sequences must be non-negative")
    if mode not in (POLICY_MODE, EXPERT_MODE):
        raise ValueError("action completion mode is invalid")
    path = (
        Path(session_dir)
        / "completion_events"
        / f"completion_{action_event_sequence:08d}.json"
    )
    if path.exists():
        raise FileExistsError(f"action completion already exists: {path}")
    _atomic_write_json(
        path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "action_event_sequence": action_event_sequence,
            "mode": mode,
            "source_sequence": source_sequence,
            "completed_timestamp": (
                time.time() if completed_timestamp is None else float(completed_timestamp)
            ),
            "status": "succeeded",
        },
    )
    return path


def load_action_completion(path: Path) -> ActionCompletion:
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("action-completion protocol version mismatch")
    if payload.get("status") != "succeeded":
        raise ValueError("action-completion status is not succeeded")
    completion = ActionCompletion(
        action_event_sequence=int(payload["action_event_sequence"]),
        mode=int(payload["mode"]),
        source_sequence=int(payload["source_sequence"]),
        completed_timestamp=float(payload["completed_timestamp"]),
    )
    if (
        completion.action_event_sequence < 0
        or completion.mode not in (POLICY_MODE, EXPERT_MODE)
        or completion.source_sequence < 0
        or not np.isfinite(completion.completed_timestamp)
    ):
        raise ValueError("invalid action-completion payload")
    return completion


def write_action_event(
    session_dir: Path,
    *,
    sequence: int,
    mode: int,
    source_sequence: int,
    requested_action: np.ndarray,
    executed_action: np.ndarray,
    controller_action: np.ndarray,
    action_dt_s: float,
    dispatch_timestamp: float | None = None,
) -> Path:
    if mode not in (POLICY_MODE, EXPERT_MODE):
        raise ValueError("mode must be POLICY_MODE or EXPERT_MODE")
    requested = validate_recorded_action_chunk(requested_action)
    executed = validate_recorded_action_chunk(executed_action)
    controller = np.asarray(controller_action, dtype=np.float32)
    if executed.shape != requested.shape or controller.shape != requested.shape:
        raise ValueError("requested, executed and controller chunks must have equal shape")
    if not np.isfinite(controller).all() or action_dt_s <= 0.0:
        raise ValueError("controller chunk must be finite and action_dt_s positive")

    dispatch = time.time() if dispatch_timestamp is None else float(dispatch_timestamp)
    timestamps = dispatch + action_dt_s * np.arange(1, len(executed) + 1, dtype=np.float64)
    inactive = np.full_like(executed, np.nan, dtype=np.float32)
    policy = executed if mode == POLICY_MODE else inactive
    expert = executed if mode == EXPERT_MODE else inactive
    path = Path(session_dir) / "action_events" / f"event_{sequence:08d}.npz"
    _atomic_savez(
        path,
        protocol_version=np.asarray(PROTOCOL_VERSION),
        sequence=np.asarray(sequence, dtype=np.int64),
        mode=np.asarray(mode, dtype=np.int8),
        source_sequence=np.asarray(source_sequence, dtype=np.int64),
        dispatch_timestamp=np.asarray(dispatch, dtype=np.float64),
        action_timestamp=timestamps,
        policy_action=policy,
        expert_action=expert,
        executed_action=executed,
        requested_action=requested,
        controller_action=controller,
    )
    return path


def load_action_event(path: Path) -> ActionEvent:
    with np.load(path, allow_pickle=False) as data:
        if str(np.asarray(data["protocol_version"]).item()) != PROTOCOL_VERSION:
            raise ValueError("action-event protocol version mismatch")
        event = ActionEvent(
            sequence=int(np.asarray(data["sequence"]).item()),
            mode=int(np.asarray(data["mode"]).item()),
            source_sequence=int(np.asarray(data["source_sequence"]).item()),
            dispatch_timestamp=float(np.asarray(data["dispatch_timestamp"]).item()),
            action_timestamp=np.asarray(data["action_timestamp"], dtype=np.float64),
            policy_action=np.asarray(data["policy_action"], dtype=np.float32),
            expert_action=np.asarray(data["expert_action"], dtype=np.float32),
            executed_action=np.asarray(data["executed_action"], dtype=np.float32),
            requested_action=np.asarray(data["requested_action"], dtype=np.float32),
            controller_action=np.asarray(data["controller_action"], dtype=np.float32),
        )
    if event.mode not in (POLICY_MODE, EXPERT_MODE):
        raise ValueError("invalid action-event mode")
    expected_shape = event.executed_action.shape
    if expected_shape[1:] != (ACTION_DIM,):
        raise ValueError(f"invalid action-event shape {expected_shape}")
    for value in (
        event.policy_action,
        event.expert_action,
        event.requested_action,
        event.controller_action,
    ):
        if value.shape != expected_shape:
            raise ValueError("action-event arrays have inconsistent shapes")
    if event.action_timestamp.shape != (expected_shape[0],):
        raise ValueError("action-event timestamps have inconsistent shape")
    active = event.policy_action if event.mode == POLICY_MODE else event.expert_action
    inactive = event.expert_action if event.mode == POLICY_MODE else event.policy_action
    if not np.isfinite(active).all() or not np.isnan(inactive).all():
        raise ValueError("action-event active/inactive source arrays are invalid")
    if not np.allclose(active, event.executed_action, atol=1e-7):
        raise ValueError("active action does not equal executed semantic action")
    validate_recorded_action_chunk(event.requested_action)
    validate_recorded_action_chunk(event.executed_action)
    return event

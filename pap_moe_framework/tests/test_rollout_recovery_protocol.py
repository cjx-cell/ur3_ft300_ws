from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pap_moe_framework.rollout_recovery.protocol import (
    clamp_continuous_for_controller,
    EXPERT_MODE,
    POLICY_MODE,
    clamp_for_controller,
    initialize_session,
    load_action_completion,
    load_action_event,
    load_expert_annotation,
    publish_expert_chunk,
    read_expert_chunk,
    read_expert_execution_ack,
    read_control_mode,
    read_takeover,
    release_to_policy,
    request_takeover,
    write_action_event,
    write_action_completion,
    write_expert_execution_ack,
)


def test_continuous_gripper_stays_in_physical_radians():
    current = np.zeros(7, dtype=np.float32)
    requested = np.zeros((3, 7), dtype=np.float32)
    requested[:, 0] = [0.20, 0.40, 0.45]
    requested[:, 6] = [0.10, 0.63, 0.90]
    executed, controller = clamp_continuous_for_controller(
        requested,
        current,
        max_arm_step_rad=0.12,
        gripper_open_rad=0.0,
        gripper_closed_rad=0.8,
    )
    np.testing.assert_allclose(executed[:, 0], [0.12, 0.24, 0.36])
    np.testing.assert_allclose(executed[:, 6], [0.10, 0.63, 0.8])
    np.testing.assert_allclose(controller, executed)
from pap_moe_framework.rollout_recovery.schema import (
    LEGACY_SCHEMA_VERSION,
    LEGACY_TRAJECTORY_SCOPE,
    SCHEMA_VERSION,
    TRAJECTORY_SCOPE,
    V3_SCHEMA_VERSION,
    V3_TRAJECTORY_SCOPE,
    validate_episode,
)


def _chunk(offset: float, gripper: float, length: int = 4) -> np.ndarray:
    values = np.zeros((length, 7), dtype=np.float32)
    values[:, :6] = offset
    values[:, 6] = gripper
    return values


def test_protocol_takeover_and_action_events_are_monotonic(tmp_path: Path) -> None:
    initialize_session(tmp_path, {"source_policy_checkpoint": "checkpoint"})
    with pytest.raises(FileExistsError):
        write_action_event(
            tmp_path,
            sequence=0,
            mode=POLICY_MODE,
            source_sequence=0,
            requested_action=_chunk(0.01, 0.0),
            executed_action=_chunk(0.01, 0.0),
            controller_action=_chunk(0.01, 0.0),
            action_dt_s=0.1,
        )
        initialize_session(tmp_path, {"source_policy_checkpoint": "checkpoint"})

    request_takeover(tmp_path, trigger="peg drift", requester="operator")
    takeover = read_takeover(tmp_path)
    assert takeover is not None and takeover["trigger"] == "peg drift"

    release_to_policy(
        tmp_path, reason="back on demonstration manifold", requester="expert", sequence=1
    )
    assert read_takeover(tmp_path) is None
    control = read_control_mode(tmp_path)
    assert control is not None and control["mode"] == POLICY_MODE
    assert len(list((tmp_path / "control_events").glob("control_*.json"))) == 2

    publish_expert_chunk(
        tmp_path,
        _chunk(0.02, 1.0),
        sequence=3,
        publisher="moveit_recovery",
        timestamp=100.0,
        skill_progress_phase=2,
        skill_progress=0.55,
        transition_readiness=0.8,
        label_confidence=1.0,
    )
    assert read_expert_chunk(tmp_path, after_sequence=3, max_age_s=2.0, now=100.1) is None
    expert = read_expert_chunk(tmp_path, after_sequence=2, max_age_s=2.0, now=100.1)
    assert expert is not None and expert.sequence == 3
    assert expert.skill_progress_phase == 2
    assert expert.skill_progress == pytest.approx(0.55)
    assert expert.transition_readiness == pytest.approx(0.8)
    assert (tmp_path / "input/expert_annotations/annotation_00000003.npz").exists()

    event_path = write_action_event(
        tmp_path,
        sequence=1,
        mode=EXPERT_MODE,
        source_sequence=3,
        requested_action=expert.action_chunk,
        executed_action=expert.action_chunk,
        controller_action=expert.action_chunk,
        action_dt_s=0.1,
        dispatch_timestamp=100.0,
    )
    event = load_action_event(event_path)
    assert event.mode == EXPERT_MODE
    assert np.isnan(event.policy_action).all()
    np.testing.assert_allclose(event.expert_action, event.executed_action)


def test_expert_mailbox_and_annotation_preserve_continuous_close(tmp_path: Path) -> None:
    initialize_session(tmp_path, {"source_policy_checkpoint": "checkpoint"})
    chunk = _chunk(0.02, 0.0, length=4)
    chunk[:, 6] = [0.2, 0.4, 0.6, 0.8]
    publish_expert_chunk(
        tmp_path,
        chunk,
        sequence=0,
        publisher="continuous_recovery",
    )
    expert = read_expert_chunk(tmp_path, after_sequence=-1, max_age_s=2.0)
    assert expert is not None
    np.testing.assert_allclose(expert.action_chunk[:, 6], chunk[:, 6])
    annotation = load_expert_annotation(tmp_path, 0)
    assert annotation is not None
    np.testing.assert_allclose(annotation.action_chunk[:, 6], chunk[:, 6])


def test_controller_clamp_preserves_semantic_and_physical_domains() -> None:
    requested = _chunk(0.5, 1.0, length=3)
    executed, controller = clamp_for_controller(
        requested,
        np.zeros(7, dtype=np.float32),
        max_arm_step_rad=0.12,
        gripper_open_rad=0.1,
        gripper_closed_rad=0.629,
    )
    np.testing.assert_allclose(executed[:, 0], [0.12, 0.24, 0.36], atol=1e-6)
    np.testing.assert_allclose(executed[:, 6], 1.0)
    np.testing.assert_allclose(controller[:, 6], 0.629)


def test_policy_action_event_preserves_continuous_gripper(tmp_path: Path) -> None:
    initialize_session(tmp_path, {"source_policy_checkpoint": "checkpoint"})
    requested = _chunk(0.02, 0.0, length=3)
    requested[:, 6] = [0.03, 0.37, 0.8]
    path = write_action_event(
        tmp_path,
        sequence=0,
        mode=POLICY_MODE,
        source_sequence=0,
        requested_action=requested,
        executed_action=requested,
        controller_action=requested,
        action_dt_s=0.1,
    )
    event = load_action_event(path)
    np.testing.assert_allclose(event.policy_action[:, 6], [0.03, 0.37, 0.8])


def test_expert_action_event_preserves_continuous_gripper(tmp_path: Path) -> None:
    initialize_session(tmp_path, {"source_policy_checkpoint": "checkpoint"})
    continuous = _chunk(0.02, 0.37)
    path = write_action_event(
        tmp_path,
        sequence=0,
        mode=EXPERT_MODE,
        source_sequence=0,
        requested_action=continuous,
        executed_action=continuous,
        controller_action=continuous,
        action_dt_s=0.1,
    )
    event = load_action_event(path)
    np.testing.assert_allclose(event.expert_action[:, 6], 0.37)


def test_expert_semantic_request_records_physical_execution(tmp_path: Path) -> None:
    initialize_session(tmp_path, {"source_policy_checkpoint": "checkpoint"})
    requested = _chunk(0.02, 1.0)
    executed = _chunk(0.02, 0.8)
    path = write_action_event(
        tmp_path,
        sequence=0,
        mode=EXPERT_MODE,
        source_sequence=0,
        requested_action=requested,
        executed_action=executed,
        controller_action=executed,
        action_dt_s=0.1,
    )
    event = load_action_event(path)
    np.testing.assert_allclose(event.requested_action[:, 6], 1.0)
    np.testing.assert_allclose(event.expert_action[:, 6], 0.8)


def test_expert_execution_ack_is_exact_and_monotonic(tmp_path: Path) -> None:
    initialize_session(tmp_path, {"source_policy_checkpoint": "checkpoint"})
    assert read_expert_execution_ack(tmp_path) is None
    write_expert_execution_ack(
        tmp_path,
        expert_sequence=0,
        action_event_sequence=4,
        completed_timestamp=101.5,
    )
    ack = read_expert_execution_ack(tmp_path)
    assert ack is not None
    assert ack.expert_sequence == 0
    assert ack.action_event_sequence == 4
    assert ack.completed_timestamp == 101.5
    with pytest.raises(ValueError, match="advance monotonically"):
        write_expert_execution_ack(
            tmp_path,
            expert_sequence=0,
            action_event_sequence=5,
        )
    with pytest.raises(ValueError, match="action-event ACK sequence"):
        write_expert_execution_ack(
            tmp_path,
            expert_sequence=1,
            action_event_sequence=4,
        )
    write_expert_execution_ack(
        tmp_path,
        expert_sequence=1,
        action_event_sequence=6,
    )
    with pytest.raises(FileExistsError, match="execution ACK"):
        initialize_session(tmp_path, {"source_policy_checkpoint": "checkpoint"})


def test_action_completion_is_immutable_and_matches_source(tmp_path: Path) -> None:
    initialize_session(tmp_path, {"source_policy_checkpoint": "checkpoint"})
    path = write_action_completion(
        tmp_path,
        action_event_sequence=2,
        mode=EXPERT_MODE,
        source_sequence=7,
        completed_timestamp=104.0,
    )
    completion = load_action_completion(path)
    assert completion.action_event_sequence == 2
    assert completion.mode == EXPERT_MODE
    assert completion.source_sequence == 7
    assert completion.completed_timestamp == 104.0
    with pytest.raises(FileExistsError, match="already exists"):
        write_action_completion(
            tmp_path,
            action_event_sequence=2,
            mode=EXPERT_MODE,
            source_sequence=7,
        )


def _valid_episode(phase: str = "align_precontact") -> dict[str, np.ndarray]:
    frames = 12
    takeover = 6
    timestamp = np.arange(frames, dtype=np.float64) * 0.1 + 100.0
    policy = np.full((frames, 7), np.nan, dtype=np.float32)
    expert = np.full((frames, 7), np.nan, dtype=np.float32)
    policy[:takeover] = _chunk(0.01, 1.0, takeover)
    expert[takeover:] = _chunk(0.02, 1.0, frames - takeover)
    executed = np.where(np.arange(frames)[:, None] < takeover, policy, expert)
    peg = np.zeros((frames, 3), dtype=np.float32)
    peg[:, 2] = 0.97
    peg[:, 0] = np.linspace(0.02, 0.005, frames)
    hole = np.zeros((frames, 3), dtype=np.float32)
    hole[:, 2] = 0.775
    gripper = peg.copy()
    gripper[:, 2] += 0.095
    force = np.zeros((frames, 6), dtype=np.float32)
    if phase == "contact":
        force[takeover:, 2] = np.linspace(0.0, 1.0, frames - takeover)
    if phase == "insertion":
        peg[-1, 2] = 0.87
    if phase == "grasp_lift":
        peg[takeover:, 2] = np.linspace(0.865, 0.925, frames - takeover)
        gripper[takeover:, 2] = peg[takeover:, 2] + 0.095

    image = np.tile(np.arange(8, dtype=np.uint8)[None, :, None], (8, 1, 3)) * 20
    state = np.zeros((frames, 7), dtype=np.float32)
    stage_index = {
        "grasp_lift": 0,
        "transport": 0,
        "align_precontact": 0,
        "contact": 2,
        "insertion": 3,
    }[phase]
    stage = np.zeros((frames, 4), dtype=np.float32)
    stage[:, stage_index] = 1.0
    return {
        "schema_version": np.asarray(V3_SCHEMA_VERSION),
        "trajectory_scope": np.asarray(V3_TRAJECTORY_SCOPE),
        "source_policy_checkpoint": np.asarray("checkpoint"),
        "recovery_trigger": np.asarray("drift"),
        "episode_id": np.asarray("test"),
        "recovery_phase": np.asarray(phase),
        "recovery_outcome": np.asarray("success"),
        "state": state,
        "state_history": np.repeat(state[:, None, :], 10, axis=1),
        "state_history_timestamp": timestamp[:, None] + np.linspace(-1.0, 0.0, 10),
        "policy_action": policy,
        "expert_action": expert,
        "executed_action": executed,
        "requested_action": executed.copy(),
        "controller_action": executed.copy(),
        "control_mode": np.asarray([POLICY_MODE] * takeover + [EXPERT_MODE] * (frames - takeover)),
        "intervention_mask": np.asarray([False] * takeover + [True] * (frames - takeover)),
        "timestamp": timestamp,
        "policy_action_timestamp": np.where(np.arange(frames) < takeover, timestamp, np.nan),
        "expert_action_timestamp": np.where(np.arange(frames) >= takeover, timestamp, np.nan),
        "executed_action_timestamp": timestamp.copy(),
        "camera0_timestamp": timestamp.copy(),
        "camera1_timestamp": timestamp.copy(),
        "force_timestamp": timestamp.copy(),
        "pose_timestamp": timestamp.copy(),
        "camera0": np.repeat(image[None, ...], frames, axis=0),
        "camera1": np.repeat(image[None, ...], frames, axis=0),
        "force": force,
        "force_fast": np.repeat(force[:, None, :], 64, axis=1),
        "force_fast_timestamp": timestamp[:, None] + np.linspace(-0.64, 0.0, 64),
        "force_slow": np.repeat(force[:, None, :], 50, axis=1),
        "force_slow_timestamp": timestamp[:, None] + np.linspace(-5.0, 0.0, 50),
        "visual_quality": np.tile(
            np.asarray([0.0, 0.0, 0.2, 1.0], dtype=np.float32), (frames, 1)
        ),
        "stage": stage,
        "modality_validity": np.ones((frames, 7), dtype=np.float32),
        "semantic_subtask": np.asarray([phase] * frames),
        "skill_progress_phase": np.asarray([7] * takeover + [2] * (frames - takeover)),
        "skill_progress_phase_name": np.asarray(
            ["recover"] * takeover + ["align"] * (frames - takeover)
        ),
        "skill_progress": np.asarray([0.0] * takeover + [0.5] * (frames - takeover)),
        "transition_readiness": np.asarray([0.0] * takeover + [0.8] * (frames - takeover)),
        "skill_progress_valid": np.asarray([False] * takeover + [True] * (frames - takeover)),
        "skill_progress_label_source": np.asarray(
            ["unlabelled_policy_rollin"] * takeover + ["test_expert"] * (frames - takeover)
        ),
        "skill_progress_confidence": np.asarray(
            [0.0] * takeover + [1.0] * (frames - takeover), dtype=np.float32
        ),
        "peg_attached": np.ones(frames, dtype=bool),
        "peg_position": peg,
        "hole_position": hole,
        "gripper_position": gripper,
    }


@pytest.mark.parametrize(
    "phase", ["grasp_lift", "transport", "align_precontact", "contact", "insertion"]
)
def test_schema_accepts_phase_specific_success(phase: str) -> None:
    summary = validate_episode(_valid_episode(phase))
    assert summary.recovery_phase == phase
    assert summary.recovery_success


def test_schema_rejects_source_action_after_takeover() -> None:
    episode = _valid_episode()
    episode["policy_action"][7] = 0.0
    with pytest.raises(ValueError, match="expert mode must be NaN"):
        validate_episode(episode)


def test_schema_accepts_complete_episode_with_multiple_handoffs() -> None:
    episode = _valid_episode("insertion")
    frames = len(episode["control_mode"])
    intervention = np.asarray(
        [False, False, False, True, True, False, False, True, True, False, False, False]
    )
    policy = np.full((frames, 7), np.nan, dtype=np.float32)
    expert = np.full((frames, 7), np.nan, dtype=np.float32)
    policy[~intervention] = _chunk(0.01, 1.0, int((~intervention).sum()))
    expert[intervention] = _chunk(0.02, 1.0, int(intervention.sum()))
    episode["schema_version"] = np.asarray(SCHEMA_VERSION)
    episode["trajectory_scope"] = np.asarray(TRAJECTORY_SCOPE)
    episode["control_mode"] = intervention.astype(np.int8)
    episode["intervention_mask"] = intervention
    episode["policy_action"] = policy
    episode["expert_action"] = expert
    episode["executed_action"] = np.where(intervention[:, None], expert, policy)
    episode["requested_action"] = episode["executed_action"].copy()
    episode["controller_action"] = episode["executed_action"].copy()
    episode["policy_action_timestamp"] = np.where(
        ~intervention, episode["timestamp"], np.nan
    )
    episode["expert_action_timestamp"] = np.where(
        intervention, episode["timestamp"], np.nan
    )
    episode["skill_progress_valid"] = intervention.copy()
    episode["skill_progress_confidence"] = intervention.astype(np.float32)
    summary = validate_episode(episode)
    assert summary.policy_frames == 8
    assert summary.expert_frames == 4


def test_schema_accepts_successful_full_episode_ending_in_terminal_expert() -> None:
    episode = _valid_episode("insertion")
    frames = len(episode["control_mode"])
    intervention = np.asarray(
        [False, False, False, False, False, True, True, True, True, True, True, True]
    )
    policy = np.full((frames, 7), np.nan, dtype=np.float32)
    expert = np.full((frames, 7), np.nan, dtype=np.float32)
    policy[~intervention] = _chunk(0.01, 1.0, int((~intervention).sum()))
    expert[intervention] = _chunk(0.02, 1.0, int(intervention.sum()))
    episode["schema_version"] = np.asarray(SCHEMA_VERSION)
    episode["trajectory_scope"] = np.asarray(TRAJECTORY_SCOPE)
    episode["control_mode"] = intervention.astype(np.int8)
    episode["intervention_mask"] = intervention
    episode["policy_action"] = policy
    episode["expert_action"] = expert
    episode["executed_action"] = np.where(intervention[:, None], expert, policy)
    episode["requested_action"] = episode["executed_action"].copy()
    episode["controller_action"] = episode["executed_action"].copy()
    episode["policy_action_timestamp"] = np.where(
        ~intervention, episode["timestamp"], np.nan
    )
    episode["expert_action_timestamp"] = np.where(
        intervention, episode["timestamp"], np.nan
    )
    episode["skill_progress_valid"] = intervention.copy()
    episode["skill_progress_confidence"] = intervention.astype(np.float32)
    summary = validate_episode(episode)
    assert summary.policy_frames == 5
    assert summary.expert_frames == 7
    assert summary.recovery_success


def test_schema_accepts_small_final_pose_lag_after_strict_terminal_insertion() -> None:
    episode = _valid_episode("insertion")
    episode["peg_position"][-2, 2] = 0.8898
    episode["peg_position"][-1, 2] = 0.9052
    summary = validate_episode(episode)
    assert summary.recovery_success


def test_schema_rejects_terminal_pose_that_never_reaches_strict_insertion() -> None:
    episode = _valid_episode("insertion")
    episode["peg_position"][-10:, 2] = 0.9052
    with pytest.raises(ValueError, match="terminal_strict=False"):
        validate_episode(episode)


def test_schema_rejects_sustained_force_overload_but_allows_impulse() -> None:
    episode = _valid_episode("insertion")
    episode["force"][7, 2] = 120.0
    validate_episode(episode)
    episode["force"][7:10, 2] = 120.0
    with pytest.raises(ValueError, match="sustained force overload"):
        validate_episode(episode, sustained_force_frames=3)


def test_schema_can_keep_policy_failure_force_context_but_never_expert_overload() -> None:
    episode = _valid_episode("insertion")
    intervention = np.asarray(episode["intervention_mask"], dtype=bool)
    policy_indices = np.flatnonzero(~intervention)
    expert_indices = np.flatnonzero(intervention)
    episode["force"][policy_indices[:3], 2] = 120.0
    with pytest.raises(ValueError, match="sustained force overload"):
        validate_episode(episode, sustained_force_frames=3)
    validate_episode(
        episode,
        allow_policy_failure_force_context=True,
        sustained_force_frames=3,
    )

    episode["force"][expert_indices[:3], 2] = 120.0
    with pytest.raises(ValueError, match="sustained force overload"):
        validate_episode(
            episode,
            allow_policy_failure_force_context=True,
            sustained_force_frames=3,
        )


def test_schema_rejects_any_missing_new_collection_modality() -> None:
    episode = _valid_episode()
    del episode["force_slow"]
    with pytest.raises(ValueError, match="missing array: force_slow"):
        validate_episode(episode)


def test_schema_rejects_invalid_modality_bit() -> None:
    episode = _valid_episode()
    episode["modality_validity"][3, 5] = 0.0
    with pytest.raises(ValueError, match="must be all ones"):
        validate_episode(episode)


def test_schema_rejects_repeated_fake_force_history() -> None:
    episode = _valid_episode()
    episode["force_slow_timestamp"][:] = episode["timestamp"][:, None]
    with pytest.raises(ValueError, match="padded/repeated"):
        validate_episode(episode)


def test_schema_keeps_legacy_readable_but_blocks_it_from_new_ingestion() -> None:
    episode = _valid_episode()
    episode["schema_version"] = np.asarray(LEGACY_SCHEMA_VERSION)
    episode["trajectory_scope"] = np.asarray(LEGACY_TRAJECTORY_SCOPE)
    for key in (
        "state_history", "state_history_timestamp", "force_fast", "force_fast_timestamp",
        "force_slow", "force_slow_timestamp", "visual_quality", "stage",
        "modality_validity", "semantic_subtask", "skill_progress_phase",
        "skill_progress_phase_name", "skill_progress", "transition_readiness",
        "skill_progress_valid", "skill_progress_label_source", "skill_progress_confidence",
        "requested_action", "controller_action",
    ):
        del episode[key]
    validate_episode(episode)
    with pytest.raises(ValueError, match="missing the mandatory v2"):
        validate_episode(episode, require_full_modalities=True)

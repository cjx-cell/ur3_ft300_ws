from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_d1_lerobot.py"
SPEC = importlib.util.spec_from_file_location("materialize_d1_lerobot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_causal_resample_never_reads_future() -> None:
    values = np.arange(5, dtype=np.float32)[:, None]
    timestamps = np.asarray([0.0, 0.1, 0.25, 0.4, 0.8])
    history = MODULE._causal_resample(values, timestamps, 4, 0.3)
    assert history.shape == (5, 4, 1)
    for index in range(len(values)):
        assert np.all(history[index, :, 0] <= values[index, 0])
    np.testing.assert_array_equal(history[0, :, 0], np.zeros(4))
    assert history[-1, -1, 0] == 4


def test_binary_joint_applies_to_chunks_and_histories() -> None:
    values = np.zeros((2, 3, 7), dtype=np.float32)
    values[..., 6] = np.asarray([[0.1, 0.13, 0.12], [0.5, 0.0, 0.121]])
    result = MODULE._binary_joint(values)
    np.testing.assert_array_equal(
        result[..., 6], np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
    )


def test_gazebo_physical_state_contract_preserves_measured_joint() -> None:
    values = np.zeros((3, 7), dtype=np.float32)
    values[:, 6] = [0.1, 0.3145, 0.629]
    result = MODULE._state_for_contract(values, "gazebo_physical")
    np.testing.assert_array_equal(result, values)


def test_pi05_hybrid_uses_binary_frame_state_and_physical_history() -> None:
    values = np.zeros((3, 7), dtype=np.float32)
    values[:, 6] = [0.1, 0.3145, 0.629]
    state = MODULE._state_for_contract(values, "pi05_hybrid")
    history = MODULE._history_for_contract(values, "pi05_hybrid")
    np.testing.assert_array_equal(state[:, 6], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(history, values)


def test_pi05_hybrid_action_remains_semantic_binary() -> None:
    values = np.zeros((3, 7), dtype=np.float32)
    values[:, 6] = [0.1, 0.629, 0.629]
    tasks = np.asarray(
        ["grasp the peg", "grasp the peg", "release the peg after verification"]
    )
    result = MODULE._action_for_contract(values, tasks, "pi05_hybrid")
    np.testing.assert_array_equal(result[:, 6], [0.0, 1.0, 0.0])


def test_gazebo_physical_action_contract_uses_universal_command_endpoints() -> None:
    values = np.zeros((3, 7), dtype=np.float32)
    values[:, 6] = [0.0, 1.0, 1.0]
    tasks = np.asarray(
        ["grasp the peg", "grasp the peg", "release the peg after verification"]
    )
    result = MODULE._action_for_contract(values, tasks, "gazebo_physical")
    np.testing.assert_allclose(result[:, 6], [0.0, 0.8, 0.0])


def test_legacy_recovery_hold_measurement_is_not_used_as_action_command() -> None:
    actions = np.zeros((5, 7), dtype=np.float32)
    states = np.zeros((5, 7), dtype=np.float32)
    actions[:, 6] = [0.627, 0.627, 0.627, 0.0, 0.4]
    states[:, 6] = [0.627, 0.627, 0.627, 0.627, 0.4]
    intervention = np.asarray([True, False, True, True, True])
    tasks = np.asarray(
        [
            "insert the peg into the hole",
            "insert the peg into the hole",
            "transport to the hole",
            "release the peg after verification",
            "grasp the peg",
        ]
    )
    result = MODULE._canonicalize_legacy_recovery_hold_commands(
        actions, states, intervention, tasks
    )
    # Expert held-object labels are universal full-close commands.
    assert result[0, 6] == MODULE.GRIPPER_CLOSE_COMMAND
    assert result[2, 6] == MODULE.GRIPPER_CLOSE_COMMAND
    # Policy context, release, and a genuine continuous close transition stay intact.
    np.testing.assert_allclose(result[[1, 3, 4], 6], [0.627, 0.0, 0.4])


def test_grasp_clearance_enter_is_migrated_to_recover() -> None:
    assert MODULE._recovery_skill_progress_phase(0, "grasp_lift") == 7
    assert MODULE._recovery_skill_progress_phase(1, "grasp_lift") == 1
    assert MODULE._recovery_skill_progress_phase(0, "insertion") == 0


def test_canonical_local_progress_resets_at_every_phase_boundary() -> None:
    progress, readiness = MODULE._canonical_local_progress(
        np.asarray([2, 2, 2, 3, 3, 6], dtype=np.int64)
    )
    np.testing.assert_allclose(progress, [0.0, 0.5, 1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(readiness, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def test_visual_quality_accepts_uint8_and_is_observable() -> None:
    dark = np.zeros((224, 224, 3), dtype=np.uint8)
    gradient = np.tile(np.arange(224, dtype=np.uint8)[None, :, None], (224, 1, 3))
    quality = MODULE._visual_quality(dark, gradient)
    assert quality.shape == (4,)
    assert 0.0 <= quality[0] <= 1.0
    assert 0.0 <= quality[1] <= 1.0
    assert quality[2] > 0.0


def test_correction_carrier_contract_has_one_positive_anchor() -> None:
    weights = np.asarray([4.0 if offset == 0 else 0.0 for offset in range(MODULE.CHUNK_SIZE)])
    assert np.count_nonzero(weights) == 1
    assert weights.sum() == 4.0

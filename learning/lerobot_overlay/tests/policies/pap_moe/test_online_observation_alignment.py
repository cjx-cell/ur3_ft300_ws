from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


HELPER_PATH = Path(
    "/home/ubuntu/ur3_ft300_ws/src/ur_simulation_gz/ur_simulation_gz/"
    "scripts/peg_in_hole/pap_moe_online_observation.py"
)
SPEC = importlib.util.spec_from_file_location("pap_moe_online_observation", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
online_observation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(online_observation)


def test_estimate_wrench_bias_uses_axiswise_median() -> None:
    samples = [
        np.asarray([0.0, 1.0, 10.0, 0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([2.0, 3.0, 12.0, 0.3, 0.4, 0.5], dtype=np.float32),
        np.asarray([100.0, 2.0, 11.0, 0.2, 0.3, 0.4], dtype=np.float32),
    ]
    result = online_observation.estimate_wrench_bias(samples)
    np.testing.assert_allclose(result, [2.0, 2.0, 11.0, 0.2, 0.3, 0.4])


def test_calibrate_wrench_window_uses_per_sample_payload_reference() -> None:
    empty_bias = np.asarray([0.0, 0.0, 12.0, 0.0, 0.0, 0.0], dtype=np.float32)
    payload_bias = np.asarray([0.0, 0.0, 18.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw = np.asarray(
        [
            [1.0, 0.0, 13.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 20.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    result = online_observation.calibrate_wrench_window(
        raw,
        np.asarray([False, True]),
        empty_bias,
        payload_bias,
    )
    np.testing.assert_allclose(
        result,
        [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 2.0, 0.0, 0.0, 0.0]],
    )


def test_fast_force_clip_preserves_wrench_direction() -> None:
    raw = np.asarray([[40.0, 0.0, 0.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    result = online_observation.calibrate_wrench_window(
        raw,
        np.asarray([False]),
        np.zeros(6, dtype=np.float32),
        clip_fast=True,
    )
    np.testing.assert_allclose(result[0, 0], online_observation.FAST_FORCE_CLIP_N)
    np.testing.assert_allclose(
        result[0, 3], 4.0 * online_observation.FAST_FORCE_CLIP_N / 40.0
    )


def test_settled_gripper_close_is_object_width_independent() -> None:
    assert online_observation.is_settled_gripper_close(
        np.array([0.300, 0.301, 0.300, 0.302, 0.301]), 0.301
    )
    assert online_observation.is_settled_gripper_close(
        np.array([0.625, 0.626, 0.626, 0.627, 0.626]), 0.626
    )
    assert not online_observation.is_settled_gripper_close(
        np.array([0.20, 0.30, 0.40, 0.50, 0.60]), 0.60
    )
    assert not online_observation.is_settled_gripper_close(
        np.array([0.10, 0.10, 0.10, 0.10, 0.10]), 0.10
    )


def test_gripper_history_preserves_continuous_units_by_default() -> None:
    history = np.zeros((4, 7), dtype=np.float32)
    history[:, 6] = [0.100, 0.120, 0.121, 0.629]
    result = online_observation.map_gripper_history_to_training_units(history)
    np.testing.assert_array_equal(result, history)
    np.testing.assert_allclose(history[:, 6], [0.100, 0.120, 0.121, 0.629])


def test_gripper_history_can_preserve_v6_analog_units() -> None:
    history = np.zeros((3, 7), dtype=np.float32)
    history[:, 6] = [0.100, 0.3645, 0.629]
    result = online_observation.map_gripper_history_to_training_units(
        history,
        mode="v6_analog_0.100_open_0.629_closed",
    )
    np.testing.assert_array_equal(result, history)


def test_legacy_binary_history_requires_explicit_mode() -> None:
    history = np.zeros((4, 7), dtype=np.float32)
    history[:, 6] = [0.100, 0.120, 0.121, 0.629]
    result = online_observation.map_gripper_history_to_training_units(
        history, mode="binary_0_open_1_closed_gt_0.12rad"
    )
    np.testing.assert_array_equal(result[:, 6], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(history[:, 6], [0.100, 0.120, 0.121, 0.629])


def test_continuous_history_clips_only_gripper_and_does_not_mutate_input() -> None:
    history = np.full((3, 7), 2.0, dtype=np.float32)
    history[:, 6] = [-0.1, 0.4, 0.9]
    result = online_observation.map_gripper_history_to_training_units(history)
    np.testing.assert_allclose(result[:, 6], [0.0, 0.4, 0.8])
    np.testing.assert_array_equal(result[:, :6], history[:, :6])
    np.testing.assert_allclose(history[:, 6], [-0.1, 0.4, 0.9])

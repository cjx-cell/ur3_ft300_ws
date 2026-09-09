from pathlib import Path
import sys

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = WORKSPACE / "src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WORKSPACE))

from pap_moe_online_observation import map_gripper_history_to_training_units

from pap_moe_framework.scripts.materialize_d1_lerobot import _semantic_binary_action


def test_current_gripper_history_preserves_continuous_measured_radians() -> None:
    history = np.zeros((4, 7), dtype=np.float32)
    history[:, 6] = [0.100, 0.120, 0.121, 0.629]

    converted = map_gripper_history_to_training_units(history)

    np.testing.assert_array_equal(converted[:, 6], history[:, 6])
    np.testing.assert_array_equal(converted[:, :6], history[:, :6])


def test_gripper_history_conversion_does_not_mutate_callback_buffer() -> None:
    history = np.zeros((2, 7), dtype=np.float32)
    history[:, 6] = [0.100, 0.629]
    original = history.copy()

    map_gripper_history_to_training_units(history)

    np.testing.assert_array_equal(history, original)


def test_legacy_normalized_gripper_contract_requires_explicit_mode() -> None:
    history = np.zeros((4, 7), dtype=np.float32)
    history[:, 6] = [0.0, 0.100, 0.3145, 0.629]

    converted = map_gripper_history_to_training_units(
        history, mode="continuous_0_1_closed_0.629rad"
    )

    np.testing.assert_allclose(
        converted[:, 6], [0.0, 0.100 / 0.629, 0.5, 1.0], atol=1e-6
    )


def test_release_action_is_open_even_while_measured_joint_is_still_closing() -> None:
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, 6] = [0.629, 0.629, 0.100]
    tasks = np.asarray(
        [
            "verify insertion success",
            "release the peg after verification",
            "retract and go back to home",
        ],
        dtype=object,
    )

    converted = _semantic_binary_action(actions, tasks)

    np.testing.assert_array_equal(converted[:, 6], [1.0, 0.0, 0.0])

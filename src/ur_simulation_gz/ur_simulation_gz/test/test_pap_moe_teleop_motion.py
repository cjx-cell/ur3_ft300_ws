import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from pap_moe_teleop_motion import MotionCommandShaper, compose_unit_command


def test_diagonal_keyboard_command_is_normalized():
    command = compose_unit_command({"w", "a", "r"})
    assert math.isclose(math.sqrt(sum(v * v for v in command[:3])), 1.0)


def test_keyboard_and_mouse_are_composed():
    command = compose_unit_command({"w"}, (0.0, -0.5, 0.0))
    assert command[0] < 0.0
    assert command[1] < 0.0


def test_keyboard_translation_matches_global_camera_view():
    # Upright far-side camera: image up=-X and image right=+Y.
    assert compose_unit_command({"w"})[:3] == [-1.0, 0.0, 0.0]
    assert compose_unit_command({"d"})[:3] == [0.0, 1.0, 0.0]


def test_shaper_accelerates_and_decelerates_without_overshoot():
    shaper = MotionCommandShaper()
    first = shaper.step([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.02, 2.0, 4.0)
    second = shaper.step([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.02, 2.0, 4.0)
    stopped = shaper.step([0.0] * 6, 0.02, 2.0, 4.0)
    assert first[0] == 0.04
    assert second[0] == 0.08
    assert stopped[0] == 0.0


def test_shaper_limits_large_dt_after_gui_stall():
    shaper = MotionCommandShaper()
    command = shaper.step([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10.0, 2.0, 4.0)
    assert command[0] == 0.2

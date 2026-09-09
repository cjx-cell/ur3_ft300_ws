"""Engineering counterexamples; not claims of observed policy failures."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path('/home/ubuntu/ur3_ft300_ws/src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole')
spec = importlib.util.spec_from_file_location('wrench_audit_helper', ROOT / 'pap_moe_online_observation.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_bias_cannot_poison_reference(bad):
    value = np.zeros(6)
    value[0] = bad
    with pytest.raises(ValueError, match='finite'):
        helper.estimate_wrench_bias([value])
    with pytest.raises(ValueError, match='finite'):
        helper.calibrate_wrench_window(value[None], np.array([True]), np.zeros(6))


def test_current_close_must_agree_with_stable_history():
    assert not helper.is_settled_gripper_close(np.full(5, .3), .8)
    assert helper.is_settled_gripper_close(np.full(5, .3), .3)


def test_stable_external_force_is_absorbed_by_unqualified_bias_estimator():
    # A mathematical counterexample: stability alone cannot identify gravity.
    empty = np.array([0., 0., 12., 0., 0., 0.])
    external = np.array([20., 0., 0., 0., 0., 0.])
    measured = np.tile(empty + external, (100, 1))
    bias = helper.estimate_wrench_bias(list(measured))
    result = helper.calibrate_wrench_window(measured, np.ones(100, dtype=bool), empty, bias)
    np.testing.assert_array_equal(result, 0.)
    np.testing.assert_array_equal(bias - empty, external)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_invalid_force_callback_does_not_refresh_source_or_touch_cache(bad):
    source = ROOT / 'ur3_pap_moe_peg_in_hole_ros_side.py'
    method = next(n for n in ast.walk(ast.parse(source.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == '_wrench')
    scope = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    # No lock or caches: any access after invalid input would fail this test.
    node = NS(_message_timestamp=lambda msg: 1.)
    vector = NS(x=bad, y=0., z=0.)
    scope['_wrench'](node, NS(wrench=NS(force=vector, torque=NS(x=0., y=0., z=0.))))


@pytest.mark.parametrize('peg_z', [.887, .79, -.1])
def test_legacy_success_has_no_lower_height_or_orientation_check(peg_z):
    # Document the old scorer's accepted domain. Do not silently change scores.
    source = ROOT / 'ur3_peg_in_hole_ros_side_base.py'
    method = next(n for n in ast.walk(ast.parse(source.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == '_gazebo_task_success')
    scope = {'np': np, 'get_model_pose': lambda name: np.array([.3, .1, peg_z if name == 'peg' else .8])}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    node = NS(ENABLE_GAZEBO_SUCCESS_CHECK=True, SUCCESS_MAX_XY_M=.008, SUCCESS_MAX_PEG_Z_M=.890)
    success, _ = scope['_gazebo_task_success'](node)
    assert success

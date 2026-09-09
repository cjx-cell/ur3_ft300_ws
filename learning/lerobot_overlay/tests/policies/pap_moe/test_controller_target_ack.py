"""Test the actual bridge methods with no ROS runtime or robot motion."""
import ast
from pathlib import Path
from threading import Lock
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest


ROOT = Path('/home/ubuntu/ur3_ft300_ws')
SOURCE = ROOT / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_peg_in_hole_ros_side_base.py'


def methods():
    tree = ast.parse(SOURCE.read_text())
    selected = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name in ('_send_action_chunk', '_wait_for_action_chunk')]
    protocol = ROOT / 'pap_moe_framework/rollout_recovery/protocol.py'
    clamp = next(n for n in ast.walk(ast.parse(protocol.read_text()))
                 if isinstance(n, ast.FunctionDef) and n.name == 'clamp_continuous_for_controller')
    scope = dict(np=np, time=time, POLICY_MODE='policy', ACTION_DIM=7, ALL_JOINTS=list(range(7)),
                 rclpy=NS(ok=lambda: True), GoalStatus=NS(STATUS_SUCCEEDED=4),
                 diagnostic_trace=NS(enabled=lambda: False),
                 FollowJointTrajectory=NS(Goal=lambda: NS(trajectory=NS(points=[]), goal_time_tolerance=NS(sec=0, nanosec=0))),
                 JointTrajectoryPoint=lambda: NS(time_from_start=NS()))
    exec(compile(ast.Module(body=[clamp] + selected, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['_send_action_chunk'], scope['_wait_for_action_chunk']


def ready(value):
    return NS(done=lambda: True, result=lambda: value)


def fixture(status=6, error=-5, accepted=True):
    wrapped = NS(status=status, result=NS(error_code=error, error_string='test'))
    handle = NS(accepted=accepted, get_result_async=lambda: ready(wrapped))
    goals = []
    def send(goal):
        goals.append(goal)
        return ready(handle)
    node = NS(lock=Lock(), latest_pos=np.zeros(7),
              GRIPPER_ACTION_MODE='continuous_radians', ACTION_CHUNK_MAX_STEP_RAD=.025,
              GRIPPER_OPEN_POSITION_RAD=0., GRIPPER_CLOSED_POSITION_RAD=.8,
              ENABLE_DETACHABLE_JOINT=False, ACTION_DT_S=.1, ACTION_RESULT_TIMEOUT_S=1.,
              rollout_recovery_session_dir=None, _gripper_to_peg_distance=lambda: None,
              _action_client=NS(send_goal_async=send),
              get_logger=lambda: NS(info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None))
    return node, goals


@pytest.mark.parametrize('arm_offset,gripper,expected', [(0., .3, True), (.16, .3, False), (0., .15211, False)])
def test_ack_compares_sent_target_and_preserves_gripper_requirement(arm_offset, gripper, expected):
    send, wait = methods()
    node, goals = fixture()
    requested = np.full((10, 7), .8)
    future = send(node, requested)
    actual_target = np.array([p.positions for p in goals[0].trajectory.points])
    np.testing.assert_array_equal(future._pap_controller_target, actual_target)
    assert np.max(abs(requested[:, :6] - actual_target[:, :6])) > .5
    node.latest_pos = actual_target[-1].copy()
    node.latest_pos[0] += arm_offset
    node.latest_pos[6] = gripper
    assert wait(node, future) is expected


def test_goal_targets_are_independent_read_only_snapshots():
    send, wait = methods()
    node, _ = fixture()
    request = np.full((10, 7), .8)
    first = send(node, request)
    snapshot = first._pap_controller_target.copy()
    request[:] = 0
    second = send(node, request)
    np.testing.assert_array_equal(first._pap_controller_target, snapshot)
    assert not np.array_equal(first._pap_controller_target, second._pap_controller_target)
    assert not first._pap_controller_target.flags.writeable
    node.latest_pos = snapshot[-1].copy()
    node.latest_pos[6] = .3
    assert wait(node, first)


@pytest.mark.parametrize('status,error,accepted,expected', [(4, 0, True, True), (6, -4, True, False), (6, -5, False, False)])
def test_normal_result_and_rejection_unchanged(status, error, accepted, expected):
    send, wait = methods()
    node, _ = fixture(status, error, accepted)
    assert wait(node, send(node, np.full((10, 7), .8))) is expected


def test_missing_reference_and_nonfinite_feedback_cannot_grant_exception():
    send, wait = methods()
    node, _ = fixture()
    future = send(node, np.full((10, 7), .8))
    node.latest_pos = future._pap_controller_target[-1].copy()
    node.latest_pos[6] = np.nan
    assert not wait(node, future)
    del future._pap_controller_target
    node.latest_pos[6] = .3
    assert not wait(node, future)
    assert not wait(node, None)


def test_continuous_radians_above_pi_are_never_reinterpreted_as_degrees():
    send, _ = methods()
    node, _ = fixture()
    node.latest_pos[:6] = 3.3
    request = np.full((10, 7), 3.4)
    request[:, 6] = .8
    future = send(node, request)
    assert np.all(future._pap_controller_target[:, :6] > 3.3)
    np.testing.assert_allclose(future._pap_controller_target[-1, :6], 3.4)


def test_goal_grace_override_changes_only_time_not_joint_targets():
    send, _ = methods()
    node, goals = fixture()
    request = np.full((10, 7), .8)
    before = send(node, request)
    assert goals[-1].goal_time_tolerance.nanosec == 0
    node.GOAL_TIME_TOLERANCE_S = .2
    after = send(node, request)
    assert goals[-1].goal_time_tolerance.sec == 0
    assert goals[-1].goal_time_tolerance.nanosec == 200_000_000
    np.testing.assert_array_equal(before._pap_controller_target, after._pap_controller_target)


@pytest.mark.parametrize('advances,expected,checks', [(False, False, 1), (True, True, 5)])
def test_success_checks_require_simulation_progress(advances, expected, checks):
    send, wait = methods()
    node, _ = fixture(error=-4)
    node.SUCCESS_REQUIRED_CHECKS = 5
    count = {'ticks': 0, 'checks': 0}
    def now():
        count['ticks'] += 1
        return NS(nanoseconds=count['ticks'] * 110_000_000 if advances else 0)
    node.get_clock = lambda: NS(now=now)
    def task_success():
        count['checks'] += 1
        return True, (0., .88)
    node._gazebo_task_success = task_success
    future = send(node, np.full((10, 7), .8))
    handle = future.result()
    handle.cancel_goal_async = lambda: None
    wrapped = NS(status=6, result=NS(error_code=-4, error_string='test'))
    handle.get_result_async = lambda: NS(done=lambda: count['ticks'] >= 8, result=lambda: wrapped)
    assert wait(node, future) is expected
    assert count['checks'] == checks


@pytest.mark.parametrize('bad', ['missing', 'nan', 'none'])
def test_strict_joint_callback_never_fabricates_or_refreshes_invalid_state(bad):
    method = next(n for n in ast.walk(ast.parse(SOURCE.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == '_js')
    scope = dict(np=np, time=time, ALL_JOINTS=list(range(7)), action_exchange=NS(enabled=lambda: True))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), scope)
    node = NS(lock=Lock(), latest_pos=[.2] * 7, _source_stamps={})
    msg = NS(name=list(range(7)), position=[.3] * 7, header=NS(stamp=NS(sec=1, nanosec=0)))
    if bad == 'missing':
        msg.name.pop()
        msg.position.pop()
    elif bad == 'nan':
        msg.position[2] = float('nan')
    valid = scope['_js'](node, msg)
    assert valid is (bad == 'none')
    assert node.latest_pos == ([.3] * 7 if valid else [.2] * 7)
    assert ('joint' in node._source_stamps) == valid

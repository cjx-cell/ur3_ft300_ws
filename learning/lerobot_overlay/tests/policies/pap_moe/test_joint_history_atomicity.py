"""Deterministically reproduce the legacy two-lock callback interleaving."""
from collections import deque
from threading import Event, Lock, Thread

import numpy as np
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import time

ROOT = Path('/home/ubuntu/ur3_ft300_ws/src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole')


def production_node():
    base = ast.parse((ROOT/'ur3_peg_in_hole_ros_side_base.py').read_text())
    child = ast.parse((ROOT/'ur3_pap_moe_peg_in_hole_ros_side.py').read_text())
    names = {'_js', '_record_joint_history'}
    methods = [n for tree in (base, child) for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(np=np, time=time, ALL_JOINTS=list('abcdefg'), action_exchange=NS(enabled=lambda: True))
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<production callbacks>', 'exec'), scope)
    resets = []
    node = NS(lock=Lock(), latest_pos=None, _source_stamps={}, _last_state_timestamp=None,
              _state_history=deque(maxlen=10), _should_resample=lambda t,p: p is None or t-p>=.095,
              get_logger=lambda: NS(warning=resets.append),
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=0)))

    def hook(timestamp, position):
        assert node.lock.locked()
        np.testing.assert_array_equal(node.latest_pos, position)
        assert node._source_stamps['joint'][0] == timestamp
        scope['_record_joint_history'](node, timestamp, position)
    node._record_joint_history = hook
    def send(timestamp, position):
        msg = NS(name=list('abcdefg'), position=[position]*7,
                 header=NS(stamp=NS(sec=int(timestamp), nanosec=round((timestamp-int(timestamp))*1e9))))
        return scope['_js'](node, msg)
    return node, send, resets


def test_production_history_and_latest_are_one_transaction():
    node, send, resets = production_node()
    for i in range(20):
        assert send(i, i)
    assert len(node._state_history) == 10
    for timestamp, values in node._state_history:
        np.testing.assert_array_equal(values, np.full(7, timestamp))
    assert not resets
    node.latest_pos[0] = -100
    assert node._state_history[-1][1][0] == 19


def test_real_timestamp_reset_is_not_silently_padded():
    node, send, resets = production_node()
    for i in range(10):
        assert send(i, i)
    assert send(0., 99.)
    assert len(node._state_history) == 1
    assert len(resets) == 1 and 'JOINT HISTORY RESET' in resets[0]
    assert node._state_history[0][0] == 0.


def test_joint_subscription_is_mutually_exclusive():
    tree = ast.parse((ROOT/'ur3_peg_in_hole_ros_side_base.py').read_text())
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr == '_joint_callback_group' for t in n.targets)]
    assert len(assignments) == 1
    assert assignments[0].value.func.id == 'MutuallyExclusiveCallbackGroup'
    subscriptions = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'create_subscription'
                     and len(n.args)>1 and isinstance(n.args[1], ast.Constant) and n.args[1].value == '/joint_states']
    group = next(k.value for k in subscriptions[0].keywords if k.arg == 'callback_group')
    assert isinstance(group, ast.Attribute) and group.attr == '_joint_callback_group'


def test_legacy_interleaving_erases_warm_history_and_mislabels_sample():
    # Legacy _js: base writes latest_pos under one lock, subclass reacquires
    # it later to append history. Hold A in the gap and let B finish first.
    lock = Lock()
    paused, resume = Event(), Event()
    state = dict(latest=np.full(7, 9.), last=9., history=deque([(i, np.full(7, i)) for i in range(10)], maxlen=10))

    def callback(stamp, pause=False):
        with lock:
            state['latest'] = np.full(7, stamp)
        if pause:
            paused.set()
            assert resume.wait(2)
        with lock:
            if stamp < state['last']:
                state['history'].clear()
            state['history'].append((stamp, state['latest'].copy()))
            state['last'] = stamp

    worker = Thread(target=callback, args=(10., True))
    worker.start()
    try:
        assert paused.wait(2)
        callback(11.)
    finally:
        resume.set()
        worker.join(2)
    assert not worker.is_alive()
    assert len(state['history']) == 1
    timestamp, values = state['history'][0]
    assert timestamp == 10.
    np.testing.assert_array_equal(values, np.full(7, 11.))

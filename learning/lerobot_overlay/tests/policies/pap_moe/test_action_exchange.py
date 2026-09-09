"""Fault injection for paired action exchange, without ROS or a GPU."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
SOURCE = ROOT / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/policy_action_exchange.py'
spec = importlib.util.spec_from_file_location('action_exchange_test_module', SOURCE)
exchange = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exchange)


def test_reply_matches_only_its_observation(tmp_path):
    path = tmp_path / 'reply.npz'
    assert exchange.read_reply('new', path) is None
    action = np.ones((10, 7))
    exchange.publish_reply('old', action, path)
    assert exchange.read_reply('new', path) is None
    exchange.publish_reply('new', action * .2, path)
    np.testing.assert_allclose(exchange.read_reply('new', path), action * .2)
    assert exchange.read_reply('old', path) is None


def test_identical_joint_values_still_have_distinct_request_identity(tmp_path):
    path = tmp_path / 'joint.txt'
    path.write_text('0 0 0 0 0 0 0')
    first = exchange.observation_id(path)
    temporary = tmp_path / 'joint.next'
    temporary.write_text(path.read_text())
    temporary.replace(path)
    assert exchange.observation_id(path) != first


@pytest.mark.parametrize('bad', [np.zeros((0, 7)), np.zeros((10, 6)), np.full((10, 7), np.nan)])
def test_bad_publication_does_not_replace_good_reply(tmp_path, bad):
    path = tmp_path / 'reply.npz'
    exchange.publish_reply('good', np.zeros((10, 7)), path)
    with pytest.raises(ValueError):
        exchange.publish_reply('bad', bad, path)
    assert exchange.read_reply('good', path) is not None


def test_partial_or_bad_schema_reply_is_not_an_action(tmp_path):
    path = tmp_path / 'reply.npz'
    path.write_bytes(b'partial')
    with pytest.raises(ValueError):
        exchange.read_reply('request', path)
    np.savez(path, schema=2, request_id='request', action=np.zeros((10, 7)))
    with pytest.raises(ValueError, match='schema'):
        exchange.read_reply('request', path)


def test_infrastructure_fault_after_status_is_not_policy_failure(tmp_path):
    log = tmp_path / 'ros.log'
    log.write_text('EVAL status: t=0.0s, arm=[]\nINFRASTRUCTURE INVALID: paired action timeout\n')
    output = tmp_path / 'result.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/write_gazebo_eval_result.py'),
                             '--log', str(log), '--output', str(output), '--policy', 'pap_moe',
                             '--checkpoint', str(tmp_path), '--episode', '1', '--seed', '0'],
                            capture_output=True, text=True)
    assert result.returncode == 6
    data = json.loads(output.read_text())
    assert data['evaluation_valid'] is False
    assert data['success'] is False
    assert data['outcome'] == 'infrastructure_failure'
    assert data['infrastructure_invalid_reasons'] == ['paired action timeout']


def test_fresh_sources_pass_even_if_images_would_be_black():
    stamps = {key: (9.95, 19.9) for key in ('joint', 'camera0', 'camera1', 'force')}
    exchange.validate_source_stamps(stamps, 10., 20.)


@pytest.mark.parametrize('fault', ['missing', 'frozen_camera', 'frozen_force', 'paused_clock', 'skew', 'future'])
def test_source_faults_are_rejected(fault):
    stamps = {key: (9.95, 19.9) for key in ('joint', 'camera0', 'camera1', 'force')}
    if fault == 'missing':
        del stamps['joint']
    elif fault == 'frozen_camera':
        stamps['camera0'] = (9., 19.9)
    elif fault == 'frozen_force':
        stamps['force'] = (9., 19.9)
    elif fault == 'paused_clock':
        stamps['force'] = (9.95, 10.)
    elif fault == 'skew':
        stamps['camera1'] = (9.6, 19.9)
    elif fault == 'future':
        stamps['camera1'] = (11., 19.9)
    with pytest.raises(ValueError):
        exchange.validate_source_stamps(stamps, 10., 20.)


def test_formal_batch_refuses_legacy_resume(tmp_path):
    results = tmp_path / 'results.jsonl'
    original = json.dumps({'policy': 'pap_moe', 'evaluation_valid': True, 'success': False}) + '\n'
    results.write_text(original)
    env = dict(os.environ, WORKSPACE50_RESUME_VALID='true', WORKSPACE50_MAX_EPISODE_DURATION_S='120')
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_workspace50_multiseed_eval.sh'),
                             str(tmp_path), 'pap_moe', 'test', str(tmp_path)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'another/unknown contract' in result.stderr
    assert results.read_text() == original


def test_demonstration_replay_preserves_continuous_targets(monkeypatch):
    monkeypatch.setitem(sys.modules, 'policy_action_exchange', exchange)
    path = SOURCE.with_name('ur3_workspace50_demonstration_replay.py')
    replay_spec = importlib.util.spec_from_file_location('replay_test', path)
    replay = importlib.util.module_from_spec(replay_spec)
    replay_spec.loader.exec_module(replay)
    actions = np.arange(84, dtype=np.float32).reshape(12, 7) * .001
    actions[:, 6] = np.linspace(0, .8, 12)
    np.testing.assert_array_equal(replay.action_window(actions, 0), actions[:10])
    tail = replay.action_window(actions, 10)
    np.testing.assert_array_equal(tail[:2], actions[10:])
    np.testing.assert_array_equal(tail[2:], np.repeat(actions[-1:], 8, axis=0))
    with pytest.raises(ValueError, match='exhausted'):
        replay.action_window(actions, 12)


def test_source_change_invalidates_even_a_success_log(tmp_path):
    source = tmp_path / 'runtime.py'
    source.write_text('original')
    manifest = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    source.write_text('changed')
    log, output = tmp_path / 'ros.log', tmp_path / 'result.json'
    log.write_text('EVAL status: t=1.0s, arm=[]\nSUCCESS: peg is inserted (xy=0.0001m, z=0.88m).\n')
    env = dict(os.environ, WORKSPACE50_SOURCE_MANIFEST_JSON=json.dumps(manifest))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/write_gazebo_eval_result.py'),
                             '--log', str(log), '--output', str(output), '--policy', 'pap_moe',
                             '--checkpoint', str(tmp_path), '--episode', '1', '--seed', '0'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 6
    data = json.loads(output.read_text())
    assert not data['evaluation_valid'] and not data['success']
    assert 'Runtime source changed' in data['infrastructure_invalid_reasons'][0]

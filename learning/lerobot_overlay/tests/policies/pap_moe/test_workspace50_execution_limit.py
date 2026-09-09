"""Verify the data-derived Workspace50 limit without running Gazebo."""
import ast
import os
from pathlib import Path

import numpy as np


def test_default_controller_limit_preserves_all_canonical_action_sequences(monkeypatch):
    root = Path('/home/ubuntu/ur3_ft300_ws')
    path = root / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_workspace50_peg_in_hole_ros_side.py'
    tree = ast.parse(path.read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'ACTION_CHUNK_MAX_STEP_RAD' for t in n.targets))
    monkeypatch.delenv('POLICY_ACTION_CHUNK_MAX_STEP_RAD', raising=False)
    limit = eval(compile(ast.Expression(assignment.value), str(path), 'eval'), {'os': os})
    assert limit == .13
    protocol = root / 'pap_moe_framework/rollout_recovery/protocol.py'
    function = next(n for n in ast.walk(ast.parse(protocol.read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name == 'clamp_continuous_for_controller')
    scope = {'np': np, 'ACTION_DIM': 7}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(protocol), 'exec'), scope)
    paths = sorted((root / 'pap_moe_framework/datasets/workspace_50_v10_canonical').glob('*_success/data.npz'))
    assert len(paths) == 50
    for episode in paths:
        with np.load(episode, allow_pickle=False) as data:
            requested = data['action']
            executed, _ = scope['clamp_continuous_for_controller'](
                requested, data['state'][0], max_arm_step_rad=limit,
                gripper_open_rad=0., gripper_closed_rad=.8)
            np.testing.assert_allclose(executed, requested, atol=1e-7, rtol=0, err_msg=str(episode))

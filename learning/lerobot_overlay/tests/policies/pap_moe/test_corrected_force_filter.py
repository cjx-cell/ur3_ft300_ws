"""Exercise the actual recorder/bridge median statements without starting ROS."""
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _statements(path, owner, attribute):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        for _, block in ast.iter_fields(node):
            if not isinstance(block, list):
                continue
            for index, statement in enumerate(block[:-1]):
                if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                    continue
                if ast.unparse(statement.value.func) == f'{owner}.{attribute}.append':
                    following = block[index + 1]
                    if isinstance(following, ast.Assign) and ast.unparse(following.targets[0]) == 'filtered_value':
                        return compile(ast.Module(body=[statement, following], type_ignores=[]), str(path), 'exec')
    raise AssertionError('Unconditional median-window advance not found')


def test_collector_and_online_filter_match_sustained_contact_and_impulse():
    root = Path('/home/ubuntu/ur3_ft300_ws')
    collector = _statements(root / 'pap_moe_framework/scripts/pap_moe_peg_in_hole_record.py', 'buf', 'force_filter')
    online = _statements(root / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_pap_moe_peg_in_hole_ros_side.py', 'self', '_force_filter')
    for samples in ([0.] * 11 + [40.] * 20, [0.] * 11 + [400.] + [0.] * 20):
        windows = [deque(maxlen=11), deque(maxlen=11)]
        outputs = []
        for fz in samples:
            value = np.array([0, 0, fz, 0, 0, 0], dtype=np.float32)
            pair = []
            for code, name, field, window in zip([collector, online], ['buf', 'self'], ['force_filter', '_force_filter'], windows):
                scope = {name: SimpleNamespace(**{field: window}), 'value': value, 'np': np}
                exec(code, scope)
                pair.append(scope['filtered_value'])
            np.testing.assert_array_equal(*pair)
            outputs.append(pair[1][2])
        assert outputs[-1] == samples[-1]
        if samples[12] == 40:
            assert outputs[16] == 40
        else:
            assert max(outputs) == 0

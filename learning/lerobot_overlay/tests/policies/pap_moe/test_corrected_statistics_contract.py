import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.policies.pap_moe.experiment_contract import verify_contract


def test_contract_rejects_self_consistent_but_wrong_quantiles(tmp_path):
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'data').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps({'total_frames': 14418, 'total_episodes': 50}))
    keys = ['action', 'observation.state', 'observation.state_history', 'observation.force',
            'observation.force_fast', 'observation.force_slow']
    x = np.linspace(0, 1, 14418)[:, None]
    pq.write_table(pa.table({key: x.tolist() for key in keys}), tmp_path / 'data/test.parquet')
    stats = {key: {'q01': [.25], 'q99': [.75], 'mean': [.5], 'std': [float(x.std())]} for key in keys}
    (tmp_path / 'meta/stats.json').write_text(json.dumps(stats))
    step = SimpleNamespace(_tensor_stats={key: {name: torch.tensor(value) for name, value in row.items()}
                                         for key, row in stats.items()})
    processor = SimpleNamespace(steps=[step])
    cfg = SimpleNamespace(physical_fusion_architecture='action_input_tokens_v2', physics_gate_architecture='physics_gate_v2',
                          action_step_routing=True, temporal_gate_two_pass_inference=False,
                          balance_loss_weight=0, expert_action_anchor_weight=0, bounded_action_conditioning=False,
                          mask_invalid_prefix_tokens=True, mask_invalid_history_cameras=True)
    policy = SimpleNamespace(config=cfg, model=SimpleNamespace(route_forecaster=None))
    with pytest.raises(AssertionError):
        verify_contract(tmp_path, policy, processor, processor, tmp_path / 'result')

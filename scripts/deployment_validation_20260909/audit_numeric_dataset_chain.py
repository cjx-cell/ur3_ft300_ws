"""Read-only all-frame NPZ versus training Parquet numeric alignment audit."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
DATA = ROOT/'pap_moe_framework/datasets'
FIELDS = {'action': 'action', 'state': 'observation.state', 'force': 'observation.force',
          'force_fast': 'observation.force_fast', 'force_slow': 'observation.force_slow',
          'state_history': 'observation.state_history', 'visual_quality': 'observation.visual_quality',
          'stage': 'observation.physics_gate_target'}


def main():
    path = DATA/'lerobot_v3_workspace50_v10_full_clean_global_stats_v1/data/chunk-000/file-000.parquet'
    table = pq.read_table(path, columns=['episode_index', 'frame_index', *FIELDS.values()])
    eid_column = np.asarray(table['episode_index'])
    frame_column = np.asarray(table['frame_index'])
    rows = []
    for eid in range(1, 51):
        selected = np.flatnonzero(eid_column == eid-1)
        assert np.array_equal(frame_column[selected], np.arange(len(selected))), eid
        episode_table = table.take(selected)
        raw = next((DATA/'workspace_50_v10_canonical').glob(f'*{eid:04d}_success/data.npz'))
        fields = {}
        with np.load(raw, allow_pickle=True) as z:
            for source, dest in FIELDS.items():
                original = np.asarray(z[source], dtype=np.float32)
                converted = np.asarray(episode_table[dest].to_pylist(), dtype=np.float32)
                assert original.shape == converted.shape, (eid, source, original.shape, converted.shape)
                assert np.isfinite(original).all() and np.isfinite(converted).all(), (eid, source)
                assert np.array_equal(original, converted), (eid, source, np.abs(original-converted).max())
                fields[source] = dict(shape=list(original.shape), max_abs_diff=0.)
        rows.append(dict(episode=eid, frames=len(selected), fields=fields))
    result = dict(passed=True, episodes=50, frames=sum(r['frames'] for r in rows), fields=FIELDS, rows=rows,
                  limitation='Numeric conversion and frame-index alignment only. Not an independent verification of sensor timing, video codec fidelity, physical calibration, or semantic label accuracy.')
    out = ROOT/'artifacts/deployment_validation_20260909_interface_ab/numeric_dataset_chain_audit.json'
    with out.open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__': main()

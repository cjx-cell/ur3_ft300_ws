"""Verify saved demonstration routes equal training labels; describe coverage."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
DATA = ROOT / 'pap_moe_framework/datasets'


def main():
    table = pq.read_table(DATA / 'lerobot_v3_workspace50_v10_full_clean_global_stats_v1/data/chunk-000/file-000.parquet',
                          columns=['episode_index', 'frame_index', 'observation.physics_gate_target']).to_pandas()
    rows = []
    for eid in range(1, 51):
        path = next((DATA / 'workspace_50_v10_canonical').glob(f'*{eid:04d}_success/data.npz'))
        with np.load(path, allow_pickle=True) as z:
            actual = np.array(table[table.episode_index == eid - 1].sort_values('frame_index')['observation.physics_gate_target'].tolist())
            target = z['stage']
            assert np.array_equal(actual, target), eid
            insert = z['semantic_subtask'] == 'insert the peg into the hole'
            route = target[insert]
            rows.append(dict(episode=eid, frames=len(target), insertion_frames=int(insert.sum()),
                             insertion_route_mean=route.mean(0).tolist(),
                             insertion_contact_route_above_half=int(((route[:, 2] + route[:, 3]) >= .5).sum()),
                             all_frames_e2_positive=int((target[:, 1] > 0).sum())))
    anchors = {}
    folder = ROOT / 'artifacts/deployment_validation_20260909_fullflow_s4_v1'
    for phase in ('grasp', 'transport', 'insert'):
        routes = []
        for p in folder.glob(f'ep*_{phase}_*_seed1000.npz'):
            with np.load(p) as z:
                routes.append(z['old/true/route'][0, :10])
        routes = np.concatenate(routes)
        anchors[phase] = dict(first10_true_route_mean=routes.mean(0).tolist(),
                              frames=len(routes), contact_route_above_half=int(((routes[:, 2] + routes[:, 3]) >= .5).sum()))
    result = dict(all50_route_labels_exact=True, frames=sum(r['frames'] for r in rows), episodes=rows,
                  original_fullflow_anchor_coverage=anchors,
                  limitation='Semantic insertion includes free-space descent. Route labels are supervision, not independently measured contact ground truth. E2 training augmentation is not counted by raw-data coverage.')
    out = ROOT / 'artifacts/deployment_validation_20260909_interface_ab/route_coverage_audit.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(frames=result['frames'], all50_route_labels_exact=True, anchors=anchors,
                          insertion_frames=sum(r['insertion_frames'] for r in rows),
                          insertion_contact_route_above_half=sum(r['insertion_contact_route_above_half'] for r in rows)), indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Read-only trace audit; reconstructed filtered wrench is NOT raw FT truth."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(directory):
    rows = []
    previous = None
    changes = []
    for path in sorted(directory.glob('chunk_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            prefix = 'observation/metadata/'
            payload = bool(data[prefix + 'current_force_reference_payload'])
            valid = bool(data[prefix + 'payload_force_bias_valid'])
            name = 'payload_force_bias' if payload and valid else 'empty_force_bias'
            reference = np.array([float(data[f'{prefix}{name}/{i}']) for i in range(6)])
            force = data['observation/force'].astype(float)
            row = dict(chunk=int(data['metadata/chunk_id']),
                       simulation_s=float(data[prefix + 'simulation_time_s']),
                       payload_phase=payload, payload_bias_valid=valid,
                       reference=reference.tolist(), calibrated_force_norm_N=float(np.linalg.norm(force[:3])),
                       reconstructed_filtered_force_norm_N=float(np.linalg.norm((force + reference)[:3])),
                       gripper_rad=float(data['observation/state'][6]),
                       gate_probs=[float(data[f'metadata/gate_probs/{i}']) for i in range(4)])
            if previous is None or not np.array_equal(previous, reference):
                changes.append(dict(chunk=row['chunk'], reference=row['reference']))
            previous = reference
            rows.append(row)
    if not rows:
        raise ValueError('No saved observation chunks found')
    return dict(trace=str(directory.resolve()), chunks=len(rows), reference_changes=changes,
                caveat='Filtered wrench reconstructed algebraically; no independent external-force or contact truth.',
                rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.trace), indent=2, allow_nan=False))

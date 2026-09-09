#!/usr/bin/env python3
"""Reuse frozen feature extraction; group probes by physical XY, not stale IDs."""
import argparse
import json
from pathlib import Path

import numpy as np
from audit_e1_information_fusion import probes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with np.load(args.output/'features.npz') as archive:
        arrays={key:archive[key] for key in archive.files}
    positions,groups=np.unique(np.round(arrays['target_initial_xy'],6),axis=0,return_inverse=True)
    counts=[]
    for g,xy in enumerate(positions):
        selection=groups==g
        episodes=np.unique(arrays['episode'][selection])
        styles=np.unique(arrays['style'][selection])
        assert len(episodes)==10 and len(styles)==10, (g,episodes,styles)
        counts.append(dict(position=g,xy=xy.tolist(),episodes=episodes.tolist(),styles=styles.tolist()))
    assert len(positions)==5
    audit=dict(recorded_group_ids=np.unique(arrays['group']).tolist(),derived_groups=counts,
               rule='actual initial peg/hole XY rounded to1e-6; raw data and feature archive unchanged')
    (args.output/'probe_group_audit.json').write_text(json.dumps(audit,indent=2))
    result=probes(arrays)
    (args.output/'probes_by_xy.json').write_text(json.dumps(result,indent=2))
    summary=[]
    for target in sorted(set(r['target'] for r in result)):
        for kind in ['position','held_styles']:
            for feature in sorted(set(r['feature'] for r in result)):
                for alpha in [.001,.01,.1]:
                    rows=[r for r in result if r['target']==target and r['feature']==feature and r['alpha']==alpha
                          and r['fold'].startswith(kind)]
                    mse=float(np.mean([r['pooled_mse'] for r in rows]))
                    null=float(np.mean([np.mean(r['null_mse']) for r in rows]))
                    summary.append(dict(target=target,split=kind,feature=feature,alpha=alpha,mse=mse,
                                        relative_to_mean_predictor=mse/null if null else None))
    (args.output/'probe_summary.json').write_text(json.dumps(summary,indent=2))
    for row in summary:
        if row['alpha']==.01:print(json.dumps(row),flush=True)


if __name__=='__main__':main()

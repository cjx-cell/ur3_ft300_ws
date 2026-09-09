"""Describe brake events across the existing 45 trials, without reclassifying scores."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT=Path('/home/ubuntu/ur3_ft300_ws')
rows=json.loads((ROOT/'artifacts/paired_comparison_20260908_resident_v2_stable_read/results.json').read_text())
records=[];counts=Counter()
for row in rows:
    path=Path(row['artifact_dir'])/'gazebo.log'
    events=[dict(position=float(a),command=float(b)) for a,b in re.findall(
        r'contact stop latched at ([\d.e+\-]+) rad while endpoint command remains ([\d.e+\-]+) rad',path.read_text())]
    low=[e for e in events if e['position']<.2]
    records.append(dict(model=row['eval_label'],episode=row['episode'],seed=row['seed'],
                        outcome=row['outcome'],events=events,low_angle_events=low))
    counts[f'{row["eval_label"]}/trials']+=1
    if low:counts[f'{row["eval_label"]}/low_angle_trials/{row["outcome"]}']+=1
out=ROOT/'artifacts/deployment_validation_20260909_tolerance_v1/all45_latches.json'
with out.open('x') as f:json.dump(dict(counts=dict(counts),records=records,
    note='Low-angle lock events are diagnostic flags, not proof of contact absence and not retrospective outcome changes.'),f,indent=2)
print(json.dumps(dict(counts),indent=2))

"""Complete normalization -> physical model -> bounded execution -> controller chain."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/ubuntu/ur3_ft300_ws')
sys.path.insert(0,'/home/ubuntu/lerobot/src')
from lerobot.processor import PolicyProcessorPipeline, policy_action_to_transition, transition_to_policy_action
import lerobot.policies.pi05.processor_pi05


def main():
    out=ROOT/'artifacts/deployment_validation_20260909_tolerance_v1'
    rows=json.loads((out/'summary.json').read_text());results=[]
    for r in rows:
        artifact=Path(r['artifact']);meta=json.loads((artifact/'result.json').read_text())
        checkpoint=Path(meta['checkpoint'])
        post=PolicyProcessorPipeline.from_pretrained(pretrained_model_name_or_path=str(checkpoint),
            config_filename='policy_postprocessor.json',to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action)
        with np.load(r['worker_result'],allow_pickle=False) as z:
            normalized=torch.from_numpy(z['normalized_action'].copy())
            execution=z['execution_action'].copy()
        physical=torch.stack([post(normalized[:,k,:]) for k in range(normalized.shape[1])],dim=1)[0].float().cpu().numpy()
        bounded=physical[:10].copy();bounded[:,6]=np.clip(bounded[:,6],0.,.8)
        assert np.max(abs(bounded-execution))<1e-6
        record=dict(name=r['name'],checkpoint=str(checkpoint),
            raw_normalized_gripper=normalized[0,:10,6].tolist(),
            unbounded_physical_gripper=physical[:10,6].tolist(),
            bounded_execution_gripper=execution[:,6].tolist(),
            raw_to_bounded_gripper_max=float(abs(physical[:10,6]-execution[:,6]).max()),
            bounded_reconstruction_max=float(abs(bounded-execution).max()),
            clipped_points=int(np.count_nonzero((physical[:10,6]<0)|(physical[:10,6]>.8))))
        results.append(record)
        timeline=np.genfromtxt(out/f'{r["name"]}.csv',delimiter=',',names=True)
        fig,ax=plt.subplots(figsize=(8,3.5))
        t=np.arange(1,11)*.1
        ax.plot(t,physical[:10,6],'o--',label='model after unnormalization, before bound')
        ax.plot(t,execution[:,6],'x-',label='bounded execution = controller goal')
        ax.plot(timeline['trajectory_time_s'],timeline['desired_gripper_rad'],label='controller interpolated desired')
        ax.plot(timeline['trajectory_time_s'],timeline['actual_gripper_rad'],label='measured')
        ax.set(xlabel='simulation trajectory seconds',ylabel='gripper angle (rad)',title=r['name'])
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/f'{r["name"]}_full_chain.svg');plt.close(fig)
        print(r['name'],'clipped',record['clipped_points'],'max',record['raw_to_bounded_gripper_max'],flush=True)
    with (out/'raw_model_chain.json').open('x') as f:json.dump(results,f,indent=2)


if __name__=='__main__':main()

"""All dispatched goals: bounded model output -> requested -> sent -> feedback."""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results=json.loads((args.batch/'results.json').read_text())
    records=[]
    for row in results:
        path=Path(row['artifact_dir'])
        events=[json.loads(s) for s in (path/'diagnostic/controller.jsonl').read_text().splitlines()]
        requests={r['sequence']:r for r in [json.loads(s) for s in (path/'resident_requests.jsonl').read_text().splitlines()]}
        feedback={}
        for event in events:
            if event['kind']=='feedback':feedback[event['chunk_id']]=event
        goals=[]
        for dispatch in events:
            if dispatch['kind']!='dispatch':continue
            chunk=dispatch['chunk_id']
            assert chunk in requests, (path,chunk)
            req=np.asarray(dispatch['requested']);sent=np.asarray(dispatch['controller'])
            with np.load(requests[chunk]['result']) as z:model=z['execution_action']
            assert model.shape==req.shape
            mismatch=float(np.max(np.abs(model-req)))
            assert mismatch<1e-7,(path,chunk,mismatch)
            arm_clip=float(np.max(np.abs(req[:,:6]-sent[:,:6])))
            grip_clip=float(np.max(np.abs(req[:,6]-sent[:,6])))
            final=feedback.get(chunk)
            record=dict(chunk=chunk,bounded_model_to_requested_max=mismatch,
                        requested_to_sent_arm_max=arm_clip,requested_to_sent_gripper_max=grip_clip)
            if final:
                actual=np.asarray(final['actual']);desired=np.asarray(final['desired'])
                record.update(last_arm_desired_actual_max=float(np.max(np.abs(actual[:6]-desired[:6]))),
                              last_gripper_desired_actual_error=float(desired[6]-actual[6]),
                              last_gripper_actual=float(actual[6]),trajectory_time=float(final['trajectory_time']))
            goals.append(record)
        assert goals,path
        feedback_goals=[x for x in goals if 'last_arm_desired_actual_max' in x]
        locks=[dict(position=float(x),command=float(y)) for x,y in re.findall(
            r'contact stop latched at ([\d.e+\-]+) rad while endpoint command remains ([\d.e+\-]+) rad',
            (path/'gazebo.log').read_text())]
        records.append(dict(model=row['eval_label'],episode=row['episode'],outcome=row['outcome'],artifact=str(path),
            goals=len(goals),goals_without_feedback=len(goals)-len(feedback_goals),
            model_requested_max=max(x['bounded_model_to_requested_max'] for x in goals),
            arm_clipped_goals=sum(x['requested_to_sent_arm_max']>1e-7 for x in goals),
            arm_clip_max=max(x['requested_to_sent_arm_max'] for x in goals),
            gripper_clip_max=max(x['requested_to_sent_gripper_max'] for x in goals),
            last_feedback_arm_error_median=float(np.median([x['last_arm_desired_actual_max'] for x in feedback_goals])),
            last_feedback_arm_error_max=max(x['last_arm_desired_actual_max'] for x in feedback_goals),
            locks=locks,low_angle_locks=[x for x in locks if x['position']<.2],goal_records=goals))
    result=dict(trials=len(records),records=records,
        limitations=['execution_action is the bounded physical model output, not raw unnormalized output before bounds.',
                     'Last feedback may precede exact endpoint acceptance; it is a tracking diagnostic, not proof of zero dynamics error.',
                     'Low-angle locks are flags, not independent proof of contact absence. No outcomes are changed.'])
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('goal_records','locks','artifact')} for r in records],indent=2))


if __name__=='__main__':main()

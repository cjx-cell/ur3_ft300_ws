"""Fixed saved rollout observations, exact online history/noise/RTC reconstruction.

No oracle future route exists for these off-demonstration states. This measures
route sensitivity, not action correctness against a fabricated target.
"""
import argparse
import copy
import json
import sys
from pathlib import Path
from contextlib import nullcontext

ROOT=Path('/home/ubuntu/ur3_ft300_ws')
sys.path.insert(0,str(ROOT/'scripts/resident_policy'))
sys.path.insert(0,str(ROOT/'scripts'))
from backend import ResidentBackend, np, torch
from probe_route_continuity_flow import continuous_route_prior


def observation(path):
    with np.load(path,allow_pickle=False) as z:
        return {k.removeprefix('observation/'):z[k].copy() for k in z.files if k.startswith('observation/')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    batch=ROOT/'artifacts/paired_comparison_20260908_resident_v2_stable_read'
    rows=json.loads((batch/'results.json').read_text())
    specs=[(1,1,[(3,'grasp'),(87,'transport'),(92,'insert')]),
           (31,2,[(3,'grasp'),(10,'transport'),(19,'insert')]),
           (41,1,[(3,'grasp'),(87,'transport'),(91,'insert')]),
           (41,2,[(3,'grasp'),(10,'transport'),(14,'insert')]),
           (21,0,[(4,'grasp_tolerance_failure')])]
    selected=[]
    for eid,seed,anchors in specs:
        r=next(x for x in rows if x['eval_label']=='pap_s4' and x['episode']==eid and x['seed']==seed)
        selected.append(dict(episode=eid,seed=seed,artifact=r['artifact_dir'],anchors=anchors))
    checkpoint=ROOT/'outputs/train/pap_corrected_gate_calibration_20260907_104300/checkpoints/015000/pretrained_model'
    manifest=dict(checkpoint=str(checkpoint),trials=selected,
        anchor_basis='Saved controller and geometry timelines; grasp before closure, transport while lifted away from hole, insert after positive depth near hole.',
        contract='50/10; EXP arm-only RTC max10; exact recorded previous normalized chunk; original per-episode noise stream; warmup and image history reconstructed',
        limits='No action ground truth or true future routes for rollout observations. All changes are sensitivity, not accuracy metrics.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    torch.set_num_threads(4)
    backend=ResidentBackend('pap_moe',str(checkpoint)); policy=backend.policy
    records=[]
    for spec in selected:
        artifact=Path(spec['artifact']);requests=[json.loads(x) for x in (artifact/'resident_requests.jsonl').read_text().splitlines()]
        backend.reset_episode(spec['seed'])
        anchors=dict(spec['anchors'])
        for index in range(max(anchors)+1):
            req=requests[index];assert req['sequence']==index
            raw=observation(artifact/'resident_inputs'/f'chunk_{index:05d}.npz')
            base=backend.batch(raw)
            # Sampling is the only inference random draw. Save/restore around
            # interventions; matching the recorded full action validates this.
            noise=policy.model.sample_noise((1,50,policy.config.max_action_dim),backend.device)
            rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
            if index not in anchors:
                policy._preprocess_pap_images(base,update_online_memory=True)
                continue
            with np.load(req['result'],allow_pickle=False) as z:
                recorded=z['physical_action'].copy();recorded_route=z['route_sequence'].copy()
            previous=None
            if index:
                with np.load(requests[index-1]['result'],allow_pickle=False) as z:
                    previous=torch.from_numpy(z['normalized_action'][:,10:].copy()).cuda()
            memory=copy.deepcopy(policy._online_visual_history)
            routes={};outputs={};arrays={'noise':noise.cpu().numpy(),'recorded':recorded,'recorded_route':recorded_route}
            for rtc in [True,False]:
                policy.config.rtc_config.enabled=rtc
                for version in ['old','continuous']:
                    for mode in ['predicted','repeat','e2_zero','e2_1e-8','mix_e2_0.01','mix_e1_0.01']:
                        policy._online_visual_history=copy.deepcopy(memory)
                        call={k:(v.clone() if torch.is_tensor(v) else v) for k,v in base.items()}
                        route=None
                        if mode not in ('predicted','repeat'):
                            route=routes[(rtc,version,'predicted')].clone()
                            if mode in ('e2_zero','e2_1e-8'):
                                epsilon=0. if mode=='e2_zero' else 1e-8
                                route=route*((1.-epsilon)/(1.-route[...,1])).unsqueeze(-1)
                                route[...,1]=epsilon
                            else:
                                route*=.99;route[...,1 if 'e2' in mode else 0]+=.01
                            call['stage_override']=route
                        with continuous_route_prior(policy.model.action_conditioner) if version=='continuous' else nullcontext():
                            normalized=policy.predict_action_chunk(call,noise=noise.clone(),
                                prev_chunk_left_over=previous.clone() if previous is not None and rtc else None,
                                inference_delay=0,execution_horizon=10,
                                rtc_action_mask=torch.tensor([1.]*6+[0.],device='cuda'))
                        action=backend.postprocess_chunk(normalized,backend.post)[0].float().cpu().numpy()
                        key=(rtc,version,mode); outputs[key]=action
                        actual_route=route if route is not None else policy.last_predicted_route_sequence.cuda()
                        routes[key]=actual_route.clone()
                        diff=action-outputs[(rtc,version,'predicted')]
                        rec=dict(episode=spec['episode'],seed=spec['seed'],chunk=index,phase=anchors[index],rtc=rtc,
                            version=version,mode=mode,arm10_delta_rmse=float(np.sqrt(np.mean(diff[:10,:6]**2))),
                            action50_delta_max=float(abs(diff).max()),gripper10_delta_max=float(abs(diff[:10,6]).max()))
                        if mode=='e2_1e-8':
                            rec['zero_boundary_action50_max']=float(abs(action-outputs[(rtc,version,'e2_zero')]).max())
                        if mode=='repeat':
                            assert rec['action50_delta_max']<1e-6,rec
                        if rtc and version=='old' and mode=='predicted':
                            clipped=action.copy();clipped[:10,6]=np.clip(clipped[:10,6],0.,.8)
                            rec['recorded_replay_max']=float(abs(clipped-recorded).max())
                            rec['recorded_route_max']=float(abs(actual_route.cpu().numpy()[0]-recorded_route).max())
                            assert rec['recorded_replay_max']<1e-5,rec
                            assert rec['recorded_route_max']<1e-6,rec
                        records.append(rec)
                        name=f'rtc{int(rtc)}/{version}/{mode}'
                        arrays[name+'/action']=action;arrays[name+'/route']=actual_route.cpu().numpy()
                        with (args.output/'records.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                print(spec['episode'],spec['seed'],index,anchors[index],'RTC',rtc,'done',flush=True)
            with (args.output/f'ep{spec["episode"]:04d}_seed{spec["seed"]}_chunk{index:03d}.npz').open('xb') as f:np.savez(f,**arrays)
            policy._online_visual_history=copy.deepcopy(memory)
            policy._preprocess_pap_images(base,update_online_memory=True)
            policy.config.rtc_config.enabled=True
            torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
    (args.output/'completion.json').write_text(json.dumps(dict(completed=True,records=len(records))))


if __name__=='__main__':main()

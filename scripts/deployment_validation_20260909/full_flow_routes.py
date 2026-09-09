"""Paired full-Flow route interventions on preselected demonstration observations.

No finetuning, simulator, ground-truth actions in the model input, or RTC hints.
Future dataset routes are an oracle diagnostic ONLY, not a deployable input.
"""
import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
sys.path.insert(0, str(ROOT/'scripts'))
from replay_first_action_trace import PAPMoEPolicy, RTCConfig, make_pre_post_processors, np, postprocess_action_chunk, torch
from eval_pap_moe_expert_ablation import _load_episode_once, _raw_batch, _route_chunks
from probe_route_continuity_flow import continuous_route_prior

PHASES = {'grasp': 'grasp the peg', 'transport': 'transport to the hole',
          'insert': 'insert the peg into the hole'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()


def summary(records):
    output = {}
    for phase in PHASES:
        selected = [r for r in records if r['phase'] == phase]
        for variant in sorted({r['variant'] for r in selected}):
            rows = [r for r in selected if r['variant'] == variant]
            output[f'{phase}/{variant}'] = dict(n=len(rows), **{
                k:float(np.mean([r[k] for r in rows])) for k in (
                    'arm10_rmse', 'gripper10_rmse', 'arm50_rmse', 'gripper50_rmse',
                    'arm50_phase_masked_rmse', 'arm10_delta_vs_true_rmse',
                    'action50_delta_vs_true_max', 'route50_l1_vs_true')})
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--anchors-per-phase', type=int, default=2)
    p.add_argument('--seeds', type=int, nargs='+', default=[1000, 1001003])
    p.add_argument('--versions', nargs='+', choices=['old','continuous'], default=['old','continuous'])
    args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset=ROOT/'pap_moe_framework/datasets/workspace_50_v10_canonical'
    anchors=[]; paths={}
    for eid in [1,11,21,31,41]:
        path=next(dataset.glob(f'*{eid:04d}_success/data.npz')); paths[eid]=path
        with np.load(path,allow_pickle=True) as z:
            semantic=z['semantic_subtask']
        for name, label in PHASES.items():
            candidates=np.flatnonzero(semantic==label)
            candidates=candidates[candidates+9 <= np.flatnonzero(semantic==label)[-1]]
            assert len(candidates)
            fractions=np.linspace(.25,.75,args.anchors_per_phase) if args.anchors_per_phase>1 else [.5]
            for fraction in fractions:
                index=int(candidates[round(fraction*(len(candidates)-1))])
                anchors.append(dict(episode=eid,phase=name,index=index))
    manifest=dict(checkpoint=str(args.checkpoint), anchors=anchors,seeds=args.seeds,
        versions=args.versions, routes=['true','predicted','true_e2_1e-8','true_mix_e2_0.01','pred_mix_e2_0.01','pred_mix_e1_0.01'],
        checkpoint_config_sha256=sha(args.checkpoint/'config.json'),
        dataset_sha256={str(k):sha(v) for k,v in paths.items()},
        rtc=False, chunk_size=50, executed_prefix=10,
        metric='Complete denoised action joint-position RMSE in physical radians, not flow-matching MSE',
        primary='Mean paired first-10 arm and gripper errors, separated by anchor phase',
        limitations=['Demonstration observations are in-distribution, not rollout recovery targets.',
          'Ground-truth future routing is an oracle diagnostic, unavailable online.',
          'No-RTC isolates route effects; not a replacement for matched deployed RTC evaluation.',
          'Full50 can cross phases; phase-masked arm50 additionally reported.',
          'Continuous interface is an untrained counterfactual, not a validated policy.'])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    print('LOAD',args.checkpoint,flush=True)
    policy=PAPMoEPolicy.from_pretrained(str(args.checkpoint),strict=True).cuda().eval()
    policy.config.rtc_config=RTCConfig(enabled=False)
    policy.init_rtc_processor()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(args.checkpoint))
    records=[]; start=time.monotonic()
    for eid,path in paths.items():
        episode=_load_episode_once(path)
        with np.load(path,allow_pickle=True) as z:
            semantic=z['semantic_subtask'].copy()
        for anchor in [a for a in anchors if a['episode']==eid]:
            idx=anchor['index']; indices=np.array([idx])
            raw=_raw_batch(episode,indices,50,wrist_dropout=False,continuous_gripper=True,
                    visual_history_indices=tuple(policy.config.visual_memory_history_indices) if policy.config.use_visual_memory else None)
            target=raw.pop('action').numpy()[0].copy()
            raw.pop('observation.stage',None)  # oracle enters only explicit interventions
            base=pre(raw)
            truth=torch.from_numpy(_route_chunks(episode['stage'],indices,50,wrist_dropout=False,all_camera_dropout=False)).cuda()
            phase_mask=(semantic[np.minimum(idx+np.arange(50),len(semantic)-1)]==PHASES[anchor['phase']])
            for seed in args.seeds:
                gen=torch.Generator(device='cpu').manual_seed(seed+eid*1000+idx)
                noise=torch.randn((1,50,policy.config.max_action_dim),generator=gen).cuda()
                outputs={}; routes={}; arrays={'target':target,'initial_noise':noise.cpu().numpy(),'phase_mask':phase_mask}
                for version in args.versions:
                    for mode in manifest['routes']:
                        policy.reset()
                        batch={k:(v.clone() if torch.is_tensor(v) else v) for k,v in base.items()}
                        route=None
                        if mode=='true': route=truth.clone()
                        elif mode=='true_e2_1e-8':
                            route=truth*(1.-1e-8);route[...,1]+=1e-8
                        elif mode=='true_mix_e2_0.01':
                            route=truth*.99;route[...,1]+=.01
                        elif mode.startswith('pred_mix_'):
                            route=routes[f'{version}/predicted']*.99
                            route[...,1 if 'e2' in mode else 0]+=.01
                        if route is not None: batch['stage_override']=route
                        with continuous_route_prior(policy.model.action_conditioner) if version=='continuous' else nullcontext():
                            normalized=policy.predict_action_chunk(batch,noise=noise.clone())
                        physical=postprocess_action_chunk(normalized,post)[0].float().cpu().numpy()
                        key=f'{version}/{mode}'
                        actual_route=(route if route is not None else policy.last_predicted_route_sequence.cuda()).float()
                        assert actual_route.shape==(1,50,4),actual_route.shape
                        routes[key]=actual_route.detach().clone()
                        outputs[key]=physical
                        error=physical-target
                        difference=physical-outputs[f'{version}/true']
                        rec=dict(**anchor,seed=seed,variant=key,
                            arm10_rmse=float(np.sqrt(np.mean(error[:10,:6]**2))),
                            gripper10_rmse=float(np.sqrt(np.mean(error[:10,6]**2))),
                            arm50_rmse=float(np.sqrt(np.mean(error[:,:6]**2))),
                            gripper50_rmse=float(np.sqrt(np.mean(error[:,6]**2))),
                            arm50_phase_masked_rmse=float(np.sqrt(np.mean(error[phase_mask,:6]**2))),
                            arm10_delta_vs_true_rmse=float(np.sqrt(np.mean(difference[:10,:6]**2))),
                            action50_delta_vs_true_max=float(np.max(np.abs(difference))),
                            route50_l1_vs_true=float((actual_route-truth).abs().sum(-1).mean()),
                            condition_norm=policy.last_condition_residual_norm.tolist())
                        records.append(rec)
                        with (args.output/'records.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
                        arrays[key+'/action']=physical; arrays[key+'/route']=actual_route.cpu().numpy()
                        arrays[key+'/normalized']=normalized.float().cpu().numpy()
                        print(f"{len(records)} {eid:04d}/{anchor['phase']}/{idx}/{seed} {key} arm10={rec['arm10_rmse']:.6f} grip10={rec['gripper10_rmse']:.6f}",flush=True)
                with (args.output/f'ep{eid:04d}_{anchor["phase"]}_{idx}_seed{seed}.npz').open('xb') as f: np.savez(f,**arrays)
                (args.output/'progress.json').write_text(json.dumps(dict(completed=len(records),seconds=time.monotonic()-start,last_anchor=anchor)))
                (args.output/'summary.partial.json').write_text(json.dumps(summary(records),indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary(records),indent=2))
    (args.output/'completion.json').write_text(json.dumps(dict(completed=True,records=len(records),seconds=time.monotonic()-start)))


if __name__=='__main__':
    main()

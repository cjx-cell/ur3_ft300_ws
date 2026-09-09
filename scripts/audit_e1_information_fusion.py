#!/usr/bin/env python3
"""Frozen E1 probes and paired branch interventions; no policy optimization."""
import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from eval_pap_moe_expert_ablation import _load_episode_once, _raw_batch, _route_chunks
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy
import lerobot.policies.pap_moe.processor_pap_moe  # noqa: F401


def norm_match(donor, current):
    return donor * (current.norm(dim=-1, keepdim=True) /
                    donor.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def ridge_predict(x, y, train, test, alpha):
    # Fit every preprocessing statistic on the probe training fold only.
    mean, std = x[train].mean(0), x[train].std(0).clip(1e-6)
    a = (x[train]-mean)/std/np.sqrt(x.shape[1])
    b = (x[test]-mean)/std/np.sqrt(x.shape[1])
    ym = y[train].mean(0)
    return b @ a.T @ np.linalg.solve(a @ a.T + alpha*np.eye(len(a)), y[train]-ym) + ym


def probes(arrays):
    # Legacy capture metadata may contain a constant position_group_id. Derive
    # physical groups from actual recorded coordinates, not episode numbering.
    _, groups = np.unique(np.round(arrays['target_initial_xy'], 6), axis=0, return_inverse=True)
    styles = arrays['style'].ravel()
    folds = [('position_'+str(g), groups != g, groups == g) for g in np.unique(groups)]
    held_styles = np.unique(styles)[-2:]
    folds += [('held_styles', ~np.isin(styles, held_styles), np.isin(styles, held_styles))]
    features = {k: v.astype(np.float64) for k, v in arrays.items() if k.startswith('feature_')}
    features['feature_state_visual'] = np.concatenate([features['feature_state'], features['feature_visual']], -1)
    output = []
    for target in ['target_initial_xy', 'target_action_delta', 'target_tool_z']:
        y = arrays[target].astype(np.float64)
        for fold, train, test in folds:
            null = ((y[test]-y[train].mean(0))**2).mean(0)
            for feature, x in features.items():
                for alpha in [.001, .01, .1]:
                    predicted = ridge_predict(x, y, train, test, alpha)
                    mse = ((predicted-y[test])**2).mean(0)
                    output.append(dict(target=target, fold=fold, feature=feature, alpha=alpha,
                        train_n=int(train.sum()), test_n=int(test.sum()), mse=mse.tolist(),
                        null_mse=null.tolist(), pooled_mse=float(mse.mean())))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit-episodes', type=int, default=50)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted(args.dataset.glob('*_success/data.npz'))[:args.limit_episodes]
    assert len(files) == args.limit_episodes
    cfg_path = Path(args.checkpoint)/'config.json'
    meta = dict(checkpoint=args.checkpoint, config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        files=[str(x) for x in files], anchors=6, policy_updates=0,
        probe='Fixed ridge alphas .001/.01/.1; leave-one-position-out and last-two-style holdout; statistics fitted on train fold only',
        targets='initial peg/hole XY (NOT dynamic object pose), demonstrated next10 mean action minus current state, tool0_z',
        caveats='Policy itself trained on all50; probe holdout tests decoding only, NOT policy generalization. Geometry metadata used only as probe target. Within-episode donor may be future: offline intervention only.',
        inference='oracle50 routes; same noise per pair; RTC off; continuous absolute gripper; masks stay as checkpoint',
        interventions='branch output replaced with norm-matched another-anchor feature; E1 token norm/content separated; weight files never modified')
    (args.output/'contract.json').write_text(json.dumps(meta, indent=2))
    policy = PAPMoEPolicy.from_pretrained(args.checkpoint).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    cfg = policy.config
    if cfg.rtc_config:
        cfg.rtc_config.enabled = False
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.checkpoint)
    device = next(policy.parameters()).device
    e1 = policy.model.expert_library.free_load
    capture = {}
    hooks = [getattr(e1, name).register_forward_hook(
        lambda _m, _i, o, key=name: capture.__setitem__(key, o.detach().clone()))
        for name in ['visual', 'motion', 'load', 'out']]
    hooks.append(e1.register_forward_pre_hook(
        lambda _m, i: capture.__setitem__('prefix', i[0].detach().clone())))
    all_features, action_rows = [], []
    modes = ['full', 'drop_E1', 'visual_donor_normmatched', 'motion_donor_normmatched',
             'load_donor_normmatched', 'E1_donor_normmatched', 'E1_norm_only', 'full_repeat']
    with torch.inference_mode(), (args.output/'actions.jsonl').open('x') as log:
        for number, path in enumerate(files):
            data = _load_episode_once(path)
            with np.load(path, allow_pickle=True) as raw:
                group, style = int(raw['position_group_id']), int(raw['trajectory_style_id'])
                xy = np.array([float(raw[k]) for k in ['peg_x','peg_y','hole_x','hole_y']])
                tool_z = raw['tool0_z'].copy()
            cached = []
            indices = np.unique(np.linspace(0,len(data['action'])-cfg.chunk_size,6).astype(int))
            for index in indices:
                policy.reset()
                batch = pre(_raw_batch(data,np.array([index]),cfg.chunk_size,wrist_dropout=False,
                    continuous_gripper=True,visual_history_indices=tuple(cfg.visual_memory_history_indices)))
                images, masks, history, padding = policy._preprocess_pap_images(batch)
                route = torch.from_numpy(_route_chunks(data['stage'],np.array([index]),cfg.chunk_size,
                    wrist_dropout=False,all_camera_dropout=False)).to(device)
                common = dict(images=images,img_masks=masks,tokens=batch['observation.language.tokens'],
                    masks=batch['observation.language.attention_mask'],visual_history=history,
                    visual_history_padding=padding,stage_override=route)
                for key in ['state','state_history','force','force_fast','force_slow','visual_quality']:
                    common[key] = batch['observation.'+key]
                policy.model.sample_actions(**common,noise=torch.zeros(1,50,cfg.max_action_dim,device=device),num_steps=1)
                tokens = {k:v.clone() for k,v in capture.items()}
                all_features.append(dict(group=np.array([group]),style=np.array([style]),
                    episode=np.array([number]),frame=np.array([index]),
                    feature_state=data['state'][index],
                    feature_time=np.array([index/len(data['action'])]),
                    **{'feature_'+k:v.float().cpu().numpy()[0] for k,v in tokens.items()},
                    target_initial_xy=xy,target_tool_z=np.array([tool_z[index]]),
                    target_action_delta=data['action'][index:index+10].mean(0)-data['state'][index]))
                cached.append((index,common,tokens))
            # Full action interventions on same five episodes as previous audits.
            if number % 10 == 0:
                for anchor, (index, common, original) in enumerate(cached):
                    donor = cached[(anchor+3)%len(cached)][2]
                    target = data['action'][index:index+50]
                    for seed in range(3):
                        generator = torch.Generator().manual_seed(1000+(number//10)*10007+int(index)*101+seed*1000003)
                        noise = torch.randn(1,50,cfg.max_action_dim,generator=generator).to(device)
                        reference = None
                        for mode in modes:
                            intervention = None
                            if '_donor_normmatched' in mode:
                                branch = mode.split('_')[0]
                                key = 'out' if branch == 'E1' else branch
                                intervention = getattr(e1,key).register_forward_hook(
                                    lambda _m,_i,o,k=key: norm_match(donor[k],o))
                            elif mode == 'E1_norm_only':
                                intervention = e1.out.register_forward_hook(
                                    lambda _m,_i,o: norm_match(o,donor['out']))
                            traces = []
                            original_fuse = policy.model._apply_physical_conditioning
                            def trace_fusion(_self, action_tokens, pap_result):
                                combined, diagnostics = original_fuse(action_tokens,pap_result)
                                traces.append({k:v.detach().float().cpu().tolist() for k,v in diagnostics.items()})
                                return combined, diagnostics
                            policy.model._apply_physical_conditioning = MethodType(trace_fusion,policy.model)
                            try:
                                mask = torch.tensor([0.,1.,1.,1.] if mode=='drop_E1' else [1.,1.,1.,1.],device=device)
                                result = policy.model.sample_actions(**common,noise=noise.clone(),expert_mask=mask)
                                prediction = post(result['actions'][...,:7]).float().cpu().numpy()[0]
                                assert np.isfinite(prediction).all()
                                if mode=='full': reference=prediction.copy()
                                if mode=='full_repeat': assert np.array_equal(prediction,reference), 'intervention leaked into full'
                                row=dict(episode=number,group=group,style=style,frame=int(index),seed=seed,mode=mode,
                                    traces=traces,E1_route_mean=float(common['stage_override'][...,0].mean()))
                                for h in [10,50]:
                                    error=(prediction[:h]-target[:h])**2
                                    row[f'arm_mse_{h}']=float(error[:,:6].mean())
                                    row[f'gripper_mse_{h}']=float(error[:,6].mean())
                                action_rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                            finally:
                                policy.model._apply_physical_conditioning = original_fuse
                                if intervention is not None:intervention.remove()
            print(f'episode={number+1}/{len(files)} features={len(all_features)} actions={len(action_rows)}',flush=True)
            del cached,data
    for hook in hooks:hook.remove()
    arrays={k:np.stack([row[k] for row in all_features]) for k in all_features[0]}
    np.savez_compressed(args.output/'features.npz',**arrays)
    if len(np.unique(np.round(arrays['target_initial_xy'],6),axis=0))>=2:
        (args.output/'probes.json').write_text(json.dumps(probes(arrays),indent=2))
    summary={mode:{k:float(np.mean([r[k] for r in action_rows if r['mode']==mode]))
        for k in ['arm_mse_10','gripper_mse_10','arm_mse_50','gripper_mse_50']} for mode in modes}
    (args.output/'action_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

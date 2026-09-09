"""Diagnostic-only log-route attention prior; no deployment/config changes."""
import argparse
import copy
import json
from contextlib import contextmanager

from replay_first_action_trace import (
    ROOT, PAPMoEPolicy, RTCConfig, _raw_observation, batch_from,
    make_pre_post_processors, np, postprocess_action_chunk, torch,
)


@contextmanager
def continuous_route_prior(conditioner):
    """Keep K/V weighting intact; add log(w) ONLY to attention logits.

    Positive w tends continuously to an excluded key as w approaches zero.
    Original all-zero handling in ActionTokenConditioner still zeros the delta.
    """
    context = {}
    def remember_weights(module, args, kwargs):
        context['weights'] = args[2] if len(args) > 2 else kwargs.get('condition_weights')
    def prior_hook(module, args, kwargs):
        weights = context.get('weights')
        if weights is None:
            return
        weights = weights.to(device=args[0].device, dtype=torch.float32).reshape(-1, weights.shape[-1])
        assert bool(torch.isfinite(weights).all()) and not bool((weights < 0).any())
        inactive = weights.eq(0).all(dim=-1)
        if inactive.any():
            weights = weights.clone()
            weights[inactive, 0] = 1.  # numerical sentinel; caller erases its delta
        bias = weights.log().to(dtype=args[0].dtype)
        bias = bias[:, None, None, :].expand(-1, module.num_heads, args[0].shape[1], -1)
        bias = bias.reshape(-1, args[0].shape[1], weights.shape[-1])
        assert kwargs.get('attn_mask') is None
        kwargs = dict(kwargs, attn_mask=bias, key_padding_mask=None)
        return args, kwargs
    first = conditioner.register_forward_pre_hook(remember_weights, with_kwargs=True)
    second = conditioner.cross_attn.register_forward_pre_hook(prior_hook, with_kwargs=True)
    try:
        yield
    finally:
        first.remove()
        second.remove()


def unit_check():
    from lerobot.policies.pap_moe.pap_moe_modules import ActionTokenConditioner
    torch.manual_seed(7)
    c = ActionTokenConditioner(16,16,nhead=4,zero_init_output=False).eval()
    params = {k:v.clone() for k,v in c.state_dict().items()}
    action=torch.randn(1,50,16); experts=torch.randn(1,4,16)
    route=torch.tensor([.6,0.,.2,.2]).reshape(1,1,4).expand(1,50,4).clone()
    eps=route.clone();eps[...,1]=1e-8;eps/=eps.sum(-1,keepdim=True)
    one=torch.zeros_like(route);one[...,0]=1.
    old_one=c(action,experts,one)
    old_zero=c(action,experts,route);old_eps=c(action,experts,eps)
    with continuous_route_prior(c):
        new_zero=c(action,experts,route);new_eps=c(action,experts,eps)
        new_one=c(action,experts,one)
        empty=c(action,experts,torch.zeros_like(route))
    result=dict(old_boundary_max=float((old_zero-old_eps).abs().max()),
                new_boundary_max=float((new_zero-new_eps).abs().max()),
                one_hot_max=float((old_one-new_one).abs().max()),
                all_zero_max=float((empty-action).abs().max()))
    assert result['new_boundary_max'] < 1e-6, result
    assert result['one_hot_max'] < 1e-6, result
    assert result['all_zero_max'] == 0., result
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in params.items())
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--unit-only', action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    unit=unit_check();print('UNIT',json.dumps(unit),flush=True)
    if args.unit_only:
        return
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    refdir=ROOT/'artifacts/pi05_same_observation_chunk1_20260907_v3'
    meta=json.loads((refdir/'summary.json').read_text())
    reference=np.load(refdir/'actions.npz',allow_pickle=False)
    trace=ROOT/meta['trace'];chunk=meta['chunk']
    data=np.load(trace/f'chunk_{chunk:05d}.npz',allow_pickle=False)
    print('LOAD',meta['checkpoint'],flush=True)
    policy=PAPMoEPolicy.from_pretrained(meta['checkpoint'],strict=True).cuda().eval()
    policy.config.rtc_config=RTCConfig(enabled=False)
    pre,post=make_pre_post_processors(policy.config,pretrained_path=meta['checkpoint'])
    policy.reset()
    warm=dict(state=np.array([0,-1.5708,1.5708,-1.5708,-1.5708,0,0],np.float32),
              camera0=np.zeros((224,224,3),np.float32),camera1=np.zeros((224,224,3),np.float32),
              force=np.zeros(6,np.float32),force_fast=np.zeros((64,6),np.float32),
              force_slow=np.zeros((50,6),np.float32),
              state_history=np.tile(np.array([0,-1.5708,1.5708,-1.5708,-1.5708,0,.1],np.float32),(10,1)),
              visual_quality=np.array([1,0,0,0],np.float32),metadata={})
    policy._preprocess_pap_images(pre(_raw_observation(warm)),update_online_memory=True)
    for index in range(chunk):
        with np.load(trace/f'chunk_{index:05d}.npz',allow_pickle=False) as previous:
            policy._preprocess_pap_images(batch_from(previous),update_online_memory=True)
    memory=copy.deepcopy(policy._online_visual_history)
    base=batch_from(data)
    noise=torch.from_numpy(reference['shared_initial_noise'].copy()).cuda()
    route=torch.from_numpy(data['route_sequence'].copy()).cuda().unsqueeze(0)
    arrays={};results={}
    for version in ['old','continuous']:
        for name,epsilon in [('saved',None),('e2_zero',0.),('e2_1e-8',1e-8),('e2_1e-6',1e-6)]:
            policy._online_visual_history=copy.deepcopy(memory)
            batch={k:v.clone() for k,v in base.items()}
            active=route.clone()
            if epsilon is not None:
                # Valid normalized probability vector; preserve ratios of the other experts.
                active=active*((1.-epsilon)/(1.-active[...,1])).unsqueeze(-1)
                active[...,1]=epsilon
            assert torch.allclose(active.sum(-1),torch.ones_like(active[...,0]),atol=1e-6)
            batch['stage_override']=active
            torch.set_rng_state(torch.from_numpy(data['rng/cpu'].copy()))
            torch.cuda.set_rng_state_all([torch.from_numpy(data[f'rng/cuda/{i}'].copy())
                                         for i in range(torch.cuda.device_count())])
            from contextlib import nullcontext
            with continuous_route_prior(policy.model.action_conditioner) if version=='continuous' else nullcontext():
                normalized=policy.predict_action_chunk(batch,noise=noise.clone())
            physical=postprocess_action_chunk(normalized,post)[0].float().cpu().numpy()
            physical[:10,6]=np.clip(physical[:10,6],0.,.8)
            key=f'{version}_{name}'
            arrays[key]=physical
            arrays[key+'_normalized']=normalized.float().cpu().numpy()
            arrays[key+'_route']=active.cpu().numpy()
            results[key]=dict(first=physical[0].tolist(),last10=physical[9].tolist(),
                              arm10_mean_l2_vs_pi05=float(np.linalg.norm(physical[:10,:6]-reference['pi05_no_rtc'][:10,:6],axis=-1).mean()),
                              condition_norm=policy.last_condition_residual_norm.tolist())
            if key=='old_saved':
                diff=float(abs(physical-reference['full_no_rtc']).max())
                results[key]['saved_replay_max_abs_diff']=diff
                assert diff < 1e-5, diff
            print(key,json.dumps(results[key]),flush=True)
            with (out/f'{key}.npz').open('xb') as f:
                np.savez(f,action=physical,normalized=arrays[key+'_normalized'],route=arrays[key+'_route'])
    comparisons={}
    for version in ['old','continuous']:
        for name in ['saved','e2_1e-8','e2_1e-6']:
            delta=arrays[f'{version}_{name}']-arrays[f'{version}_e2_zero']
            comparisons[f'{version}_{name}_vs_zero']=dict(
                physical50_max_abs=float(abs(delta).max()),
                arm10_mean_l2=float(np.linalg.norm(delta[:10,:6],axis=-1).mean()),
                normalized50_max_abs=float(abs(arrays[f'{version}_{name}_normalized']-arrays[f'{version}_e2_zero_normalized']).max()))
    (out/'summary.json').write_text(json.dumps(dict(checkpoint=meta['checkpoint'],trace=str(trace),chunk=chunk,
        reference=str(refdir),unit=unit,rtc=False,explicit_shared_noise=True,
        change='Only log(route_weight) attention-logit prior; existing K/V scaling unchanged; hooks removed after each inference.',
        results=results,comparisons=comparisons,
        note='Full 50-step Flow output; no robot execution. Output difference to baseline is not correctness error.'),indent=2))
    print('COMPARISONS',json.dumps(comparisons),flush=True)


if __name__=='__main__':
    main()

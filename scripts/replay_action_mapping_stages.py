"""Offline stage comparison on identical saved observations and explicit noise."""
import argparse
import copy
import json
from pathlib import Path

from replay_first_action_trace import (
    ROOT, PAPMoEPolicy, RTCConfig, _raw_observation, batch_from,
    make_pre_post_processors, np, postprocess_action_chunk, torch,
)
from lerobot.configs.policies import PreTrainedConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--init-from', type=Path, help='Reconstruct initialization from baseline weights; not the historical RNG snapshot')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=ROOT/'artifacts/pi05_same_observation_chunk1_20260907_v3')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    reference_meta = json.loads((args.reference/'summary.json').read_text())
    trace = Path(reference_meta['trace'])
    chunk = reference_meta['chunk']
    data = np.load(trace/f'chunk_{chunk:05d}.npz', allow_pickle=False)
    reference = np.load(args.reference/'actions.npz', allow_pickle=False)
    config = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    pre, post = make_pre_post_processors(config, pretrained_path=str(args.checkpoint))
    raw = {k.removeprefix('observation/'): data[k].copy() for k in data.files
           if k.startswith('observation/') and '/' not in k.removeprefix('observation/')}
    batch = pre(_raw_observation(raw))
    saved_batch = batch_from(data)
    differences = {}
    for key, value in saved_batch.items():
        if key not in batch or not torch.is_tensor(batch[key]):
            continue
        assert value.shape == batch[key].shape, key
        differences[key] = float((value.double()-batch[key].to(value.device).double()).abs().max())
    assert not any(differences.values()), differences
    for key in ['observation.state','observation.images.camera0','observation.images.camera1',
                'observation.language.tokens','observation.language.attention_mask',
                'observation.force','observation.force_fast','observation.force_slow',
                'observation.state_history','observation.visual_quality']:
        assert key in differences, key
    ref_config = PreTrainedConfig.from_pretrained(reference_meta['checkpoint'])
    _, ref_post = make_pre_post_processors(ref_config, pretrained_path=reference_meta['checkpoint'])
    probe = torch.linspace(-1.5,1.5,350,device='cuda').reshape(1,50,7)
    post_diff = float((postprocess_action_chunk(probe,post)-postprocess_action_chunk(probe,ref_post)).abs().max())
    assert post_diff == 0., post_diff
    print(args.label, 'pre/post checks passed; loading checkpoint', flush=True)
    if args.init_from:
        torch.manual_seed(1000)
        torch.cuda.manual_seed_all(1000)
        policy = PAPMoEPolicy.from_pretrained(str(args.init_from), config=config, strict=False).cuda().eval()
    else:
        policy = PAPMoEPolicy.from_pretrained(str(args.checkpoint), strict=True).cuda().eval()
    policy.config.rtc_config = RTCConfig(enabled=False)
    warm = dict(state=np.array([0,-1.5708,1.5708,-1.5708,-1.5708,0,0],np.float32),
                camera0=np.zeros((224,224,3),np.float32), camera1=np.zeros((224,224,3),np.float32),
                force=np.zeros(6,np.float32), force_fast=np.zeros((64,6),np.float32),
                force_slow=np.zeros((50,6),np.float32),
                state_history=np.tile(np.array([0,-1.5708,1.5708,-1.5708,-1.5708,0,.1],np.float32),(10,1)),
                visual_quality=np.array([1,0,0,0],np.float32), metadata={})
    policy.reset()
    policy._preprocess_pap_images(pre(_raw_observation(warm)), update_online_memory=True)
    for previous_chunk in range(chunk):
        with np.load(trace/f'chunk_{previous_chunk:05d}.npz',allow_pickle=False) as previous:
            policy._preprocess_pap_images(batch_from(previous), update_online_memory=True)
    memory = copy.deepcopy(policy._online_visual_history)
    noise = torch.from_numpy(reference['shared_initial_noise'].copy()).cuda()
    assert noise.shape == (1, config.chunk_size, config.max_action_dim)
    scale = policy.config.action_conditioning_scale
    attention_capture = {}
    def capture_attention(module, positional, keyword):
        if not attention_capture:
            attention_capture.update(query=positional[0].detach().clone(),
                                     keys=positional[1].detach().clone(),
                                     values=positional[2].detach().clone(),
                                     mask=keyword['key_padding_mask'].detach().clone())
    hook = policy.model.action_conditioner.cross_attn.register_forward_pre_hook(capture_attention, with_kwargs=True)
    arrays, results = {}, {}
    for mode in ['native', 'common_route', 'zero']:
        attention_capture.clear()
        policy._online_visual_history = copy.deepcopy(memory)
        b = {k:v.clone() for k,v in saved_batch.items()}
        policy.config.action_conditioning_scale = 0. if mode == 'zero' else scale
        if mode == 'common_route':
            # A controlled route, not a ground-truth label for this off-trajectory state.
            b['stage_override'] = torch.from_numpy(data['route_sequence'].copy()).cuda().unsqueeze(0)
        torch.set_rng_state(torch.from_numpy(data['rng/cpu'].copy()))
        torch.cuda.set_rng_state_all([torch.from_numpy(data[f'rng/cuda/{i}'].copy())
                                     for i in range(torch.cuda.device_count())])
        action = policy.predict_action_chunk(b, noise=noise.clone())
        physical = postprocess_action_chunk(action,post)[0].float().cpu().numpy()
        physical[:10,6] = np.clip(physical[:10,6],0.,.8)
        arrays[mode] = physical
        arrays[mode+'_normalized'] = action.float().cpu().numpy()
        arrays[mode+'_route'] = policy.last_routing_probs.numpy()
        results[mode] = dict(first=physical[0].tolist(), last10=physical[9].tolist(),
                            elbow_delta10=float(physical[9,2]-data['observation/state'][2]),
                            arm10_mean_l2_vs_pi05=float(np.linalg.norm(physical[:10,:6]-reference['pi05_no_rtc'][:10,:6],axis=-1).mean()),
                            condition_norm=policy.last_condition_residual_norm.tolist(),
                            applied_norm=policy.last_applied_condition_residual_norm.tolist(),
                            route_mean10=arrays[mode+'_route'][0,:10].mean(axis=0).tolist())
        if args.label == 's4' and mode in ['native','zero']:
            ref_key = 'full_no_rtc' if mode == 'native' else 'zero_no_rtc'
            difference = float(np.abs(physical-reference[ref_key]).max())
            results[mode]['previous_replay_max_abs_diff'] = difference
            assert difference < 1e-5, difference
        print(args.label, mode, json.dumps(results[mode]), flush=True)
        if mode == 'common_route':
            with (args.output/'condition_attention_input.npz').open('xb') as stream:
                np.savez(stream, **{k:v.float().cpu().numpy() if k != 'mask' else v.cpu().numpy()
                                   for k,v in attention_capture.items()})
        with (args.output/f'{mode}.npz').open('xb') as stream:
            np.savez(stream, action=physical, normalized=arrays[mode+'_normalized'], route=arrays[mode+'_route'])
    with (args.output/'actions.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    (args.output/'summary.json').write_text(json.dumps(dict(
        label=args.label, checkpoint=str(args.checkpoint), reference=str(args.reference),
        reconstructed_init_from=None if args.init_from is None else str(args.init_from),
        trace=str(trace), chunk=chunk, preprocessing_differences=differences,
        postprocessing_difference=post_diff, rtc=False, explicit_shared_noise=True,
        note='S2 native gate is not supervised yet. common_route is saved S4 routing, NOT oracle. '
             'Offline output differences, not correctness errors or success rates.',
        results=results),indent=2))


if __name__ == '__main__':
    main()

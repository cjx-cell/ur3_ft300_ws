"""Paired, offline interventions on saved PAP input/RNG/RTC state."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
sys.path.insert(0, '/home/ubuntu/lerobot/src')
sys.path.insert(0, str(ROOT/'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole'))
import lerobot.policies.pi05.processor_pi05  # noqa
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from ur3_pap_moe_peg_in_hole_inference import _raw_observation, postprocess_action_chunk


def batch_from(data):
    return {k.removeprefix('processed/'): torch.from_numpy(data[k].copy()).cuda()
            for k in data.files if k.startswith('processed/') and data[k].dtype.kind not in 'OUS'
            and '/' not in k.removeprefix('processed/')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_grad_enabled(False)
    policy = PAPMoEPolicy.from_pretrained(str(args.checkpoint), strict=True).cuda().eval()
    policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=10,
        max_guidance_weight=10., prefix_attention_schedule=RTCAttentionSchedule.EXP)
    policy.init_rtc_processor()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))

    # Reconstruct precisely the online image memory, including the warmup frame.
    warm = dict(state=np.array([0,-1.5708,1.5708,-1.5708,-1.5708,0,0],np.float32),
                camera0=np.zeros((224,224,3),np.float32), camera1=np.zeros((224,224,3),np.float32),
                force=np.zeros(6,np.float32), force_fast=np.zeros((64,6),np.float32),
                force_slow=np.zeros((50,6),np.float32),
                state_history=np.tile(np.array([0,-1.5708,1.5708,-1.5708,-1.5708,0,.1],np.float32),(10,1)),
                visual_quality=np.array([1,0,0,0],np.float32), metadata={})
    policy.reset()
    policy._preprocess_pap_images(pre(_raw_observation(warm)), update_online_memory=True)
    for chunk in range(args.chunk):
        with np.load(args.trace/f'chunk_{chunk:05d}.npz',allow_pickle=False) as previous:
            policy._preprocess_pap_images(batch_from(previous), update_online_memory=True)
    memory = copy.deepcopy(policy._online_visual_history)
    data = np.load(args.trace/f'chunk_{args.chunk:05d}.npz',allow_pickle=False)
    base = batch_from(data)
    saved = data['physical_action']
    state = data['observation/state']
    results = {}
    arrays = {}
    scale = policy.config.action_conditioning_scale
    captured_noise = []
    original_sample_noise = policy.model.sample_noise
    def capture_noise(shape, device):
        value = original_sample_noise(shape, device)
        captured_noise.append(value.detach().clone())
        return value
    policy.model.sample_noise = capture_noise
    modes = ['full_rtc', 'full_rtc_repeat', 'recorded_route_rtc', 'zero_rtc', 'e1_rtc',
             'full_no_rtc', 'zero_no_rtc', 'e1_no_rtc']
    for mode in modes:
        policy._online_visual_history = copy.deepcopy(memory)
        batch = {k: v.clone() for k,v in base.items()}
        rtc = not mode.endswith('no_rtc')
        policy.config.rtc_config.enabled = rtc
        policy.config.action_conditioning_scale = 0. if mode.startswith('zero') else scale
        if mode.startswith('e1'):
            batch['stage_override'] = torch.zeros((1,50,4),device='cuda')
            batch['stage_override'][:,:,0] = 1.
        elif mode == 'recorded_route_rtc':
            batch['stage_override'] = torch.from_numpy(data['route_sequence'].copy()).cuda().unsqueeze(0)
        kwargs = dict(prev_chunk_left_over=torch.from_numpy(data['rng/previous_leftover'].copy()).cuda(),
                      inference_delay=0, execution_horizon=10,
                      rtc_action_mask=torch.tensor([1.]*6+[0.],device='cuda')) if rtc else {}
        torch.set_rng_state(torch.from_numpy(data['rng/cpu'].copy()))
        torch.cuda.set_rng_state_all([torch.from_numpy(data[f'rng/cuda/{i}'].copy())
                                     for i in range(torch.cuda.device_count())])
        normalized = policy.predict_action_chunk(batch, **kwargs)
        physical = postprocess_action_chunk(normalized,post)[0].float().cpu().numpy()
        # Online execution_chunk is a numpy view of physical_chunk; only its
        # published prefix has gripper clipping applied before trace saving.
        physical[:10,6] = np.clip(physical[:10,6],0.,.8)
        arrays[mode] = physical
        route = policy.last_routing_probs[0].numpy()
        arrays[mode+'_route'] = route
        results[mode] = dict(
            normalized_max_abs_diff_from_saved=float(abs(normalized.float().cpu().numpy()-data['normalized_action']).max()),
            max_abs_diff_from_saved=float(abs(physical-saved).max()),
            arm10_l2_change=float(np.linalg.norm(physical[:10,:6]-saved[:10,:6],axis=-1).mean()),
            first=physical[0].tolist(), last10=physical[9].tolist(),
            elbow_delta10=float(physical[9,2]-state[2]),
            gripper10=physical[:10,6].tolist(),
            condition_norm=policy.last_condition_residual_norm.tolist(),
            applied_norm=policy.last_applied_condition_residual_norm.tolist(),
        )
        print(mode, json.dumps(results[mode]),flush=True)
        if mode == 'full_rtc' and (results[mode]['max_abs_diff_from_saved'] > 1e-4
                                  or results[mode]['normalized_max_abs_diff_from_saved'] > 1e-5):
            raise RuntimeError('Replay mismatch >1e-4 rad; stop interventions until state reconstruction is fixed')
    comparison = None
    if args.baseline:
        import gc
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from ur3_baseline_peg_in_hole_inference import _raw_observation as baseline_raw, TASK
        # Every PAP intervention must have used exactly the same initial noise.
        assert len(captured_noise) == len(modes)
        assert all(torch.equal(captured_noise[0], n) for n in captured_noise)
        noise = captured_noise[0].clone()
        arrays['shared_initial_noise'] = noise.cpu().numpy()
        with (args.output/'pap_actions_and_noise.npz').open('xb') as stream:
            np.savez(stream, **arrays)
        print('PAP actions and identical noise saved; loading independent baseline', flush=True)
        del policy, original_sample_noise, captured_noise, pre, post
        gc.collect()
        torch.cuda.empty_cache()
        baseline = PI05Policy.from_pretrained(str(args.baseline), strict=True).cuda().eval()
        baseline.config.rtc_config = RTCConfig(enabled=False)
        pre, post = make_pre_post_processors(baseline.config, pretrained_path=str(args.baseline))
        raw = baseline_raw(data['observation/state'].copy(),
                           data['observation/camera0'].copy(), data['observation/camera1'].copy(),
                           task=TASK)  # saved processed/task already includes discretized state
        baseline_batch = pre(raw)
        keys = ['observation.state', 'observation.images.camera0', 'observation.images.camera1',
                'observation.language.tokens', 'observation.language.attention_mask']
        differences = {}
        for key in keys:
            actual = baseline_batch[key].detach().cpu().numpy()
            expected = data['processed/'+key]
            assert actual.shape == expected.shape, (key, actual.shape, expected.shape)
            differences[key] = float(np.abs(actual.astype(np.float64)-expected.astype(np.float64)).max())
        if any(differences.values()):
            raise RuntimeError(f'Shared preprocessing mismatch: {differences}')
        print('Shared state, cameras and tokens are exactly equal', flush=True)
        normalized = baseline.predict_action_chunk(baseline_batch, noise=noise)
        physical = postprocess_action_chunk(normalized, post)[0].float().cpu().numpy()
        physical[:10,6] = np.clip(physical[:10,6], 0., .8)
        arrays['pi05_no_rtc'] = physical
        with (args.output/'baseline_action.npz').open('xb') as stream:
            np.savez(stream, physical=physical, normalized=normalized.float().cpu().numpy())
        # Check the output normalization independently with identical test tensors.
        pap_config_path = args.checkpoint
        from lerobot.configs.policies import PreTrainedConfig
        pap_config = PreTrainedConfig.from_pretrained(str(pap_config_path))
        _, pap_post = make_pre_post_processors(pap_config, pretrained_path=str(pap_config_path))
        norm_diff = float((postprocess_action_chunk(normalized, post)-
                           postprocess_action_chunk(normalized, pap_post)).abs().max().item())
        assert norm_diff == 0., norm_diff
        comparison = dict(baseline=str(args.baseline), shared_preprocessing_max_abs_diff=differences,
                          postprocessing_max_abs_diff=norm_diff, noise_identical=True,
                          state=state.tolist(), first=physical[0].tolist(), last10=physical[9].tolist(),
                          gripper10=physical[:10,6].tolist(),
                          elbow_delta10=float(physical[9,2]-state[2]),
                          arm10_mean_l2_vs_pap={m:float(np.linalg.norm(physical[:10,:6]-arrays[m][:10,:6],axis=-1).mean())
                                               for m in ['full_no_rtc','zero_no_rtc']})
        print('pi05_same_observation', json.dumps(comparison), flush=True)
    with (args.output/'actions.npz').open('xb') as stream:
        np.savez(stream,**arrays)
    (args.output/'summary.json').write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint),trace=str(args.trace),chunk=args.chunk,
        note='Offline interventions, not task-success or correctness measurements. E1 is diagnostic only.',
        results=results, baseline_comparison=comparison),indent=2))


if __name__ == '__main__':
    main()

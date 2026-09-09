#!/usr/bin/env python3
"""Paired complete denoising audit; no weight changes or simulator execution."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/lerobot/src')
from eval_pap_moe_expert_ablation import _load_episode_once, _raw_batch, _route_chunks, MASKS
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy
import lerobot.policies.pap_moe.processor_pap_moe  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode-npz', action='append', required=True)
    parser.add_argument('--anchors', type=int, default=6)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--focused', action='store_true', help='Entropy, E1 content, and current-visual-failure paired audit')
    parser.add_argument('--mask-fixes', action='store_true', help='Pair old inference with each mask correction separately')
    args = parser.parse_args()
    if args.anchors < 2 or args.seeds < 1:
        parser.error('Need at least two anchors and one seed')
    if args.focused and args.mask_fixes:
        parser.error('Choose one audit suite')
    if (args.output / 'records.jsonl').exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    policy = PAPMoEPolicy.from_pretrained(args.checkpoint).eval()
    if policy.config.rtc_config is not None:
        policy.config.rtc_config.enabled = False
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)
    device = next(policy.parameters()).device
    modes = list(MASKS) + ['mismatched_tokens']
    if args.focused:
        modes = ['full', 'all_zero', 'no_entropy', 'E1_mean', 'E1_mismatch',
                 'drop_E1', 'drop_E3', 'drop_E4', 'blind_full', 'blind_drop_E2',
                 'blind_no_memory', 'blind_no_entropy']
    if args.mask_fixes:
        modes = ['full', 'masked_prefix', 'masked_history']
    records = []
    arrays = {}
    captured = {}

    def capture(_module, _inputs, output):
        captured['tokens'] = output.detach().clone()

    config = policy.config
    original_floor = config.action_conditioning_confidence_floor
    original_prefix_mask = config.mask_invalid_prefix_tokens
    original_history_mask = policy.model.visual_history_encoder.mask_invalid_cameras if config.use_visual_memory else False
    meta = dict(checkpoint=args.checkpoint, episodes=args.episode_npz,
                metric='complete 10-step denoising, decoded absolute joint target MSE in rad^2',
                rtc=False, reason='independent demonstration anchors, no previous chunk',
                route='oracle future labels; not deployable success rate',
                visual_history=config.visual_memory_history_indices,
                inference_steps=config.num_inference_steps,
                fusion=config.physical_fusion_architecture,
                scale=torch.tanh(policy.model.expert_conditioning_scales).detach().float().cpu().tolist(),
                limitations='Training episodes; zero/mismatched conditions are interventions, not separate Pi0.5 baseline. Focused mode adds synthetic immediate visual failure only.',
                modes=modes, focused=args.focused, mask_fixes=args.mask_fixes,
                original_confidence_floor=original_floor,
                interventions='no_entropy sets confidence_floor=1 only; E1_mean uses this episode anchor mean; E1_mismatch replaces E1 only; blind zeroes current cameras, retains history, sets oracle E2 weight to 1 and E1 to 0 then normalizes; blind_no_memory removes only E2 memory arguments with a pre-hook. Synthetic immediate outage, not sustained-outage validation.')
    (args.output / 'contract.json').write_text(json.dumps(meta, indent=2))
    with torch.inference_mode(), (args.output / 'records.jsonl').open('w') as log:
        for ep_number, path in enumerate(args.episode_npz):
            episode = _load_episode_once(Path(path))
            # Stay away from padded ends; include uniformly spaced full-chunk anchors.
            indices = np.unique(np.linspace(0, max(0, len(episode['action']) - config.chunk_size), args.anchors).astype(int))
            cached = []
            for index in indices:
                policy.reset()
                raw = _raw_batch(episode, np.array([index]), config.chunk_size,
                                 wrist_dropout=False, continuous_gripper=True,
                                 visual_history_indices=tuple(config.visual_memory_history_indices) if config.use_visual_memory else None)
                batch = pre(raw)
                images, masks, history, padding = policy._preprocess_pap_images(batch)
                route = torch.from_numpy(_route_chunks(episode['stage'], np.array([index]), config.chunk_size,
                                         wrist_dropout=False, all_camera_dropout=False)).to(device)
                common = dict(images=images, img_masks=masks,
                              tokens=batch['observation.language.tokens'], masks=batch['observation.language.attention_mask'],
                              visual_history=history, visual_history_padding=padding, stage_override=route)
                for key in ['force', 'force_fast', 'force_slow', 'state', 'state_history', 'visual_quality']:
                    common[key] = batch['observation.' + key]
                blind_common = None
                if args.focused:
                    blind_raw = _raw_batch(episode, np.array([index]), config.chunk_size,
                        wrist_dropout=False, continuous_gripper=True,
                        visual_history_indices=tuple(config.visual_memory_history_indices))
                    for key in ['observation.images.camera0', 'observation.images.camera1']:
                        assert blind_raw[key].ndim == 5
                        blind_raw[key][:, -1] = 0
                        assert torch.equal(blind_raw[key][:, :-1], raw[key][:, :-1])
                    blind_raw['observation.visual_quality'] = torch.tensor([[1., 0., 0., 0.]])
                    blind_batch = pre(blind_raw)
                    bi, bm, bh, bp = policy._preprocess_pap_images(blind_batch)
                    br = route.clone()
                    br[..., 0], br[..., 1] = 0, 1
                    br = br / br.sum(-1, keepdim=True)
                    blind_common = dict(common, images=bi, img_masks=bm, visual_history=bh,
                        visual_history_padding=bp, stage_override=br,
                        visual_quality=blind_batch['observation.visual_quality'])
                # Capture actual expert representations; inference only, no labels fed to experts.
                hook = policy.model.expert_library.register_forward_hook(capture)
                noise = torch.zeros((1, config.chunk_size, config.max_action_dim), device=device)
                policy.model.sample_actions(**common, noise=noise, num_steps=1)
                hook.remove()
                cached.append((int(index), common, raw['action'][0].numpy().copy(), captured['tokens'].clone(), blind_common))
            mean_tokens = torch.stack([item[3] for item in cached]).mean(0)
            for anchor, (index, common, target, tokens, blind_common) in enumerate(cached):
                for seed in range(args.seeds):
                    generator = torch.Generator(device='cpu').manual_seed(1000 + ep_number * 10007 + index * 101 + seed * 1000003)
                    noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator).to(device)
                    full = None
                    for mode in modes:
                        hook = None
                        if mode == 'masked_prefix':
                            config.mask_invalid_prefix_tokens = True
                        if mode == 'masked_history':
                            policy.model.visual_history_encoder.mask_invalid_cameras = True
                        active_common = common
                        mask_name = mode
                        if mode.startswith('blind_'):
                            active_common = dict(blind_common)
                            mask_name = 'drop_E2' if mode == 'blind_drop_E2' else 'full'
                            if mode == 'blind_no_memory':
                                hook = policy.model.expert_library.visual_blind.register_forward_pre_hook(
                                    lambda _m, inputs: inputs[:6] + (None, None))
                        if mode in ['no_entropy', 'blind_no_entropy']:
                            config.action_conditioning_confidence_floor = 1.0
                        if mode in ['E1_mean', 'E1_mismatch']:
                            donor = tokens.clone()
                            donor[:, 0] = (mean_tokens if mode == 'E1_mean' else cached[(anchor + max(1, len(cached)//2)) % len(cached)][3])[:, 0]
                            hook = policy.model.expert_library.register_forward_hook(lambda _m, _i, _o: donor)
                        if mode == 'mismatched_tokens':
                            donor = cached[(anchor + max(1, len(cached)//2)) % len(cached)][3]
                            hook = policy.model.expert_library.register_forward_hook(lambda _m, _i, _o: donor)
                        try:
                            output = policy.model.sample_actions(**active_common, noise=noise.clone(),
                                expert_mask=torch.tensor(MASKS.get(mask_name, MASKS['full']), device=device))
                        finally:
                            config.action_conditioning_confidence_floor = original_floor
                            config.mask_invalid_prefix_tokens = original_prefix_mask
                            if config.use_visual_memory:
                                policy.model.visual_history_encoder.mask_invalid_cameras = original_history_mask
                            if hook is not None:
                                hook.remove()
                        dim = config.output_features['action'].shape[0]
                        prediction = post(output['actions'][..., :dim]).detach().float().cpu().numpy()[0]
                        if not np.isfinite(prediction).all():
                            raise RuntimeError('Nonfinite generated actions')
                        if mode == 'full':
                            full = prediction.copy()
                        record = dict(episode=path, frame=index, seed=seed, mode=mode,
                                      donor_frame=cached[(anchor + max(1,len(cached)//2)) % len(cached)][0] if mode == 'mismatched_tokens' else None,
                                      route_mean=common['stage_override'].float().mean((0,1)).cpu().tolist(),
                                      token_norm=tokens.float().norm(dim=-1).cpu().tolist())
                        record['family'] = 'blind' if mode.startswith('blind_') else 'clean'
                        record['active_route_mean'] = active_common['stage_override'].float().mean((0,1)).cpu().tolist()
                        record['valid_history'] = index > 0
                        for h in [10, 50]:
                            diff = prediction[:h] - target[:h]
                            record[f'arm_mse_{h}'] = float(np.mean(diff[:, :6]**2))
                            record[f'gripper_mse_{h}'] = float(np.mean(diff[:, 6]**2))
                            record[f'arm_delta_full_{h}'] = float(np.sqrt(np.mean((prediction[:h,:6]-full[:h,:6])**2)))
                        # Per-action physical grouping, same labels across modes (not phase prompts).
                        physical_route = common['stage_override'][0].float().cpu().numpy()
                        group_weights = {'free': physical_route[:, 0], 'rigid': physical_route[:, 2],
                                         'movable': physical_route[:, 3]}
                        for group, weight in group_weights.items():
                            err = (prediction - target) ** 2
                            record[f'{group}_weight'] = float(weight.sum())
                            record[f'{group}_arm_sse'] = float((err[:, :6].mean(-1) * weight).sum())
                            record[f'{group}_gripper_sse'] = float((err[:, 6] * weight).sum())
                        for key in ['applied_condition_residual_norm', 'condition_residual_norm', 'expert_conditioning_scales']:
                            value = output.get(key)
                            record[key] = None if value is None else value.detach().float().cpu().tolist()
                        records.append(record)
                        log.write(json.dumps(record) + '\n')
                        log.flush()
                        arrays[f'e{ep_number}_f{index}_s{seed}_{mode}'] = prediction
                    arrays[f'e{ep_number}_f{index}_target'] = target
                    print(f'episode={ep_number+1} frame={index} seed={seed} complete', flush=True)
            del cached, episode
    summary = {}
    for mode in modes:
        selected = [r for r in records if r['mode'] == mode]
        summary[mode] = {key: float(np.mean([r[key] for r in selected]))
                         for key in ['arm_mse_10', 'arm_mse_50', 'gripper_mse_10', 'gripper_mse_50', 'arm_delta_full_10', 'arm_delta_full_50']}
    np.savez_compressed(args.output / 'generated_actions.npz', **arrays)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

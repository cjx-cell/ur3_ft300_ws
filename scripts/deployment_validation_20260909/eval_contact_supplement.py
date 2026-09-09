"""Predeclared contact-only supplement; does not replace primary A/B anchors.

Select two frames before AND at first E3+E4>=0.5 in semantic insertion,
without looking at A/B action errors. Targets are valid contact frames
only; padding beyond episode end never contributes. Complete Flow, no RTC.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
AB = ROOT / 'artifacts/deployment_validation_20260909_interface_ab'
sys.path.insert(0, str(AB))
from eval_native import PAPMoEPolicy, RTCConfig, _load_episode_once, _raw_batch, _route_chunks
from eval_native import make_pre_post_processors, postprocess_action_chunk, sha, np, torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = []
    data = ROOT / 'pap_moe_framework/datasets/workspace_50_v10_canonical'
    for eid in (1, 11, 21, 31, 41):
        path = next(data.glob(f'*{eid:04d}_success/data.npz'))
        with np.load(path, allow_pickle=True) as z:
            semantic = z['semantic_subtask']
            contact = (semantic == 'insert the peg into the hole') & (z['stage'][:, 2:].sum(-1) >= .5)
            onset = int(np.flatnonzero(contact)[0])
            before = max(onset - 2, int(np.flatnonzero(semantic == 'insert the peg into the hole')[0]))
            for anchor_kind, index in (('pre_contact', before), ('on_contact', onset)):
                future = index + np.arange(50)
                mask = (future < len(contact)) & contact[np.minimum(future, len(contact) - 1)]
                assert mask[:10].any()
                selected.append(dict(episode=eid, anchor_kind=anchor_kind, path=str(path), sha256=sha(path), index=index,
                                     contact_onset=onset, mask=mask.tolist()))
    modes = ['true', 'predicted', 'true_e2_1e-8', 'pred_mix_e2_0.01', 'e1_only', 'all_zero']
    manifest = dict(plan_version='v2_add_actual_contact_observations_before_any_supplement_results',
                    source_sha256=sha(Path(__file__)),
                    checkpoint=str(args.checkpoint), checkpoint_config_sha256=sha(args.checkpoint/'config.json'),
                    anchors=selected, seeds=[1000, 1001003], routes=modes, rtc=False,
                    native_route_prior=json.loads((args.checkpoint/'config.json').read_text())['action_conditioning_route_prior'],
                    interpretation='Contact-label-selected supplementary in-distribution action RMSE, not independent contact ground truth or held-out skill success; all_zero is an ablation of this trained PAP, not Pi0.5.')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    policy = PAPMoEPolicy.from_pretrained(str(args.checkpoint), strict=True).cuda().eval()
    policy.config.rtc_config = RTCConfig(enabled=False)
    policy.init_rtc_processor()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    rows = []
    for anchor in selected:
        episode = _load_episode_once(Path(anchor['path']))
        index = np.array([anchor['index']])
        raw = _raw_batch(episode, index, 50, wrist_dropout=False, continuous_gripper=True,
                         visual_history_indices=tuple(policy.config.visual_memory_history_indices) if policy.config.use_visual_memory else None)
        target = raw.pop('action').numpy()[0].copy()
        raw.pop('observation.stage', None)
        base = pre(raw)
        truth = torch.from_numpy(_route_chunks(episode['stage'], index, 50, wrist_dropout=False, all_camera_dropout=False)).cuda()
        mask = np.array(anchor['mask'])
        for seed in manifest['seeds']:
            gen = torch.Generator(device='cpu').manual_seed(seed + anchor['episode']*1000 + anchor['index'])
            noise = torch.randn((1, 50, policy.config.max_action_dim), generator=gen).cuda()
            arrays = dict(target=target, initial_noise=noise.cpu().numpy(), contact_mask=mask)
            predicted = None
            for mode in modes:
                policy.reset()
                batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in base.items()}
                route = None
                if mode == 'true': route = truth.clone()
                elif mode == 'true_e2_1e-8':
                    route = truth*(1.-1e-8)
                    route[..., 1] += 1e-8
                elif mode == 'pred_mix_e2_0.01':
                    route = predicted*.99
                    route[..., 1] += .01
                elif mode in ('e1_only', 'all_zero'):
                    route = torch.zeros_like(truth)
                    if mode == 'e1_only': route[..., 0] = 1.
                if route is not None: batch['stage_override'] = route
                output = policy.predict_action_chunk(batch, noise=noise.clone())
                action = postprocess_action_chunk(output, post)[0].float().cpu().numpy()
                if mode == 'predicted': predicted = policy.last_predicted_route_sequence.cuda().float().clone()
                actual = route if route is not None else predicted
                arrays[mode+'/action'] = action
                arrays[mode+'/route'] = actual.cpu().numpy()
                row = dict(episode=anchor['episode'], anchor_kind=anchor['anchor_kind'], index=anchor['index'], seed=seed, route=mode)
                for length in (10, 50):
                    delta = (action-target)[:length][mask[:length]]
                    row[f'contact_count{length}'] = len(delta)
                    row[f'arm{length}_contact_rmse'] = float(np.sqrt(np.mean(delta[:, :6]**2)))
                    row[f'gripper{length}_contact_rmse'] = float(np.sqrt(np.mean(delta[:, 6]**2)))
                rows.append(row)
                with (args.output/'records.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
            with (args.output/f"ep{anchor['episode']:04d}_{anchor['anchor_kind']}_seed{seed}.npz").open('xb') as f: np.savez(f, **arrays)
    result = {kind: {mode: {key: float(np.mean([row[key] for row in rows if row['route']==mode and row['anchor_kind']==kind]))
                     for key in ('arm10_contact_rmse','gripper10_contact_rmse','arm50_contact_rmse','gripper50_contact_rmse')}
              for mode in modes} for kind in ('pre_contact', 'on_contact')}
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    (args.output/'completion.json').write_text(json.dumps(dict(completed=True, records=len(rows))))


if __name__ == '__main__':
    main()

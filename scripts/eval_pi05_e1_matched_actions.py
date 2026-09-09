#!/usr/bin/env python3
"""Read-only Pi0.5 complete-action comparison at the existing E1 audit anchors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval_pap_moe_expert_ablation import _load_episode_once, _raw_batch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = [json.loads(line) for line in (args.reference / 'actions.jsonl').read_text().splitlines()]
    reference = [r for r in reference if r['mode'] == 'full']
    assert len(reference) == 90
    source = json.loads((args.reference / 'contract.json').read_text())
    files = [Path(x) for x in source['files']]
    args.output.mkdir(parents=True, exist_ok=False)
    policy = PI05Policy.from_pretrained(str(args.checkpoint), strict=True).eval()
    config = policy.config
    assert config.chunk_size == 50 and config.max_action_dim == 32
    for flag in ['use_deterministic_arm_head', 'use_deterministic_gripper_head', 'use_release_gripper_override']:
        assert not getattr(config, flag, False), flag
    if config.rtc_config:
        config.rtc_config.enabled = False
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    pre, post = make_pre_post_processors(config, pretrained_path=str(args.checkpoint))
    device = next(policy.parameters()).device
    contract = dict(checkpoint=str(args.checkpoint), reference=str(args.reference),
                    inference='RTC off, 50 actions, 10 denoising steps; continuous absolute gripper',
                    noise='CPU Generator seed=1000+(episode//10)*10007+frame*101+seed*1000003',
                    policy_updates=0, device=str(device),
                    caveats='Training-set independent-anchor open loop, NOT closed-loop success; PAP additionally uses force/history/oracle routes. Native checkpoint processors differ: see pi05_pap_statistics_audit.json; NOT normalization-matched architecture ablation.',
                    config_sha256=hashlib.sha256((args.checkpoint/'config.json').read_bytes()).hexdigest())
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2))
    rows = []
    with torch.inference_mode(), (args.output/'actions.jsonl').open('x') as log:
        for episode in sorted({r['episode'] for r in reference}):
            data = _load_episode_once(files[episode])
            for ref in [r for r in reference if r['episode'] == episode]:
                frame, seed = ref['frame'], ref['seed']
                policy.reset()
                raw = _raw_batch(data, np.array([frame]), 50, wrist_dropout=False, continuous_gripper=True)
                # No privileged routing, force or target actions enter Pi0.5.
                raw = {k: v for k, v in raw.items() if k in {
                    'observation.state', 'observation.images.camera0', 'observation.images.camera1', 'task'}}
                batch = pre(raw)
                generator = torch.Generator().manual_seed(1000+(episode//10)*10007+frame*101+seed*1000003)
                noise = torch.randn(1, 50, 32, generator=generator).to(device)
                normalized = policy.predict_action_chunk(batch, noise=noise, num_steps=10)
                prediction = post(normalized[..., :7]).float().cpu().numpy()[0]
                assert prediction.shape == (50, 7) and np.isfinite(prediction).all()
                target = data['action'][frame:frame+50]
                assert target.shape == prediction.shape
                row = dict(episode=episode, frame=frame, seed=seed, mode='full')
                for horizon in [10, 50]:
                    error = (prediction[:horizon] - target[:horizon]) ** 2
                    row[f'arm_mse_{horizon}'] = float(error[:, :6].mean())
                    row[f'gripper_mse_{horizon}'] = float(error[:, 6].mean())
                rows.append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(f'completed={len(rows)}/90 episode={episode+1} frame={frame} seed={seed}', flush=True)
            del data
    keys = ['arm_mse_10', 'gripper_mse_10', 'arm_mse_50', 'gripper_mse_50']
    assert {(r['episode'], r['frame'], r['seed']) for r in rows} == {
        (r['episode'], r['frame'], r['seed']) for r in reference}
    summary = {'full': {k: float(np.mean([r[k] for r in rows])) for k in keys},
               'by_episode': {str(ep+1): {k: float(np.mean([r[k] for r in rows if r['episode'] == ep]))
                                         for k in keys} for ep in sorted({r['episode'] for r in rows})}}
    (args.output/'action_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

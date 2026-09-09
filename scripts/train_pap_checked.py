#!/usr/bin/env python3
"""Run the unchanged trainer with a fail-fast real optimizer contract check."""
import json
import math
from pathlib import Path

from lerobot.scripts import lerobot_train as trainer

original_factory = trainer.make_optimizer_and_scheduler


def checked_factory(cfg, policy):
    if hasattr(cfg.policy, 'e1_fusion_normalization'):
        assert policy.model.expert_library.free_load.out.mode == cfg.policy.e1_fusion_mode
        assert policy.model.expert_library.free_load.out[0].mode == cfg.policy.e1_fusion_normalization
        assert policy.model.config.mask_invalid_prefix_tokens == cfg.policy.mask_invalid_prefix_tokens
        assert policy.model.visual_history_encoder.mask_invalid_cameras == cfg.policy.mask_invalid_history_cameras
    optimizer, scheduler = original_factory(cfg, policy)
    groups = optimizer.param_groups
    expected = [cfg.policy.pap_moe_optimizer_lr,
                cfg.policy.pap_moe_optimizer_lr * cfg.policy.joint_action_expert_lr_scale]
    actual = [g.get('initial_lr', g['lr']) for g in groups]
    assert len(groups) == 2, f'Expected two groups, got {actual}'
    assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(actual, expected)), (actual, expected)
    reference = policy.get_optim_params()
    for group, ref in zip(groups, reference):
        assert {id(p) for p in group['params']} == {id(p) for p in ref['params']}
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in policy.parameters() if p.requires_grad}
    result = dict(verified=True, peak_lrs=actual,
                  numel=[sum(p.numel() for p in g['params']) for g in groups],
                  initial_scheduled_lrs=[g['lr'] for g in groups],
                  steps=cfg.steps, source=str(cfg.policy.pretrained_path),
                  e1_normalization=getattr(cfg.policy, 'e1_fusion_normalization', None),
                  e1_fusion_mode=cfg.policy.e1_fusion_mode,
                  mask_invalid_prefix_tokens=cfg.policy.mask_invalid_prefix_tokens,
                  mask_invalid_history_cameras=cfg.policy.mask_invalid_history_cameras)
    path = Path(cfg.output_dir) / 'verified_optimizer_contract.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print('VERIFIED_OPTIMIZER_CONTRACT ' + json.dumps(result), flush=True)
    return optimizer, scheduler


if __name__ == '__main__':
    trainer.make_optimizer_and_scheduler = checked_factory
    trainer.main()

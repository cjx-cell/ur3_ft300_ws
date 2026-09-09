#!/usr/bin/env python3
"""Prepare matched continuation configs without launcher default drift."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
EXP = ROOT / 'artifacts/pap_lr_ab_10k_20260905'
SOURCE = ROOT / 'outputs/train/pap_moe_unified_v2_expert_action_joint_20260904_235835/checkpoints/030000/pretrained_model'


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    if (EXP / 'manifest.json').exists():
        raise FileExistsError(EXP)
    original = json.loads((SOURCE / 'train_config.json').read_text())
    configs = {}
    for label, scale in [('A', .1), ('B', 1.)]:
        cfg = copy.deepcopy(original)
        cfg.update(resume=False, steps=10000, save_freq=2500, seed=42,
                   use_policy_training_preset=False,
                   output_dir=str(ROOT / f'outputs/train/pap_lr_ab_{label}_10k_20260905'),
                   job_name=f'pap_lr_ab_{label}', eval_freq=20000)
        cfg['policy'].update(pretrained_path=str(SOURCE), joint_action_expert_lr_scale=scale,
                             mask_invalid_prefix_tokens=False, mask_invalid_history_cameras=False)
        assert cfg['policy']['train_expert_action_joint']
        assert cfg['policy']['expert_action_anchor_weight'] == 0
        assert cfg['batch_size'] == 1
        assert cfg['dataset']['episodes'] is None
        assert not Path(cfg['output_dir']).exists()
        configs[label] = cfg
        (EXP / f'{label}.json').write_text(json.dumps(cfg, indent=2))
    a, b = copy.deepcopy(configs['A']), copy.deepcopy(configs['B'])
    for c in (a,b):
        c.pop('output_dir'); c.pop('job_name')
        c['policy'].pop('joint_action_expert_lr_scale')
    assert a == b, 'A/B has an unintended difference'
    watched = [Path('/home/ubuntu/lerobot/src/lerobot/scripts/lerobot_train.py')]
    for folder in ['pap_moe','pi05']:
        watched += sorted(Path(f'/home/ubuntu/lerobot/src/lerobot/policies/{folder}').glob('*.py'))
    watched += [SOURCE/'config.json', SOURCE/'policy_preprocessor.json', SOURCE/'policy_postprocessor.json']
    watched += list(SOURCE.glob('policy_*processor.safetensors'))
    watched += [EXP/'A.json', EXP/'B.json']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    manifest = dict(source=str(SOURCE), configs={k:str(EXP/f'{k}.json') for k in configs},
                    hashes=hashes, expected_group_peak_lrs={'A':[2.5e-5,2.5e-6],'B':[2.5e-5,2.5e-5]},
                    normalizers='preserve source processor statistics, do not rebuild',
                    comparison='Only policy.joint_action_expert_lr_scale differs, excluding output/job labels',
                    evaluation='Final 10k complete decoded action audit, five positions; oracle routes; no closed-loop claim')
    (EXP/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='hashes'},indent=2))


if __name__ == '__main__':
    main()

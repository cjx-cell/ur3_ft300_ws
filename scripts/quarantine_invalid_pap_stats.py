"""Isolate only PAP run directories whose saved action quantiles are proven wrong."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path('/home/ubuntu/ur3_ft300_ws')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    reference = json.loads((ROOT / 'pap_moe_framework/datasets/lerobot_v3_workspace50_v10_full_clean_global_stats_v1/meta/stats.json').read_text())
    destination = ROOT / '.cleanup_trash/20260907_invalid_statistics'
    records = []
    for run in sorted((ROOT / 'outputs/train').glob('pap*')):
        if run.is_symlink() or not run.is_dir():
            continue
        if not any(date in run.name for date in ('20260904', '20260905', '20260906')):
            continue
        # Only the audited Workspace50 lineage, never a different dataset's statistics.
        configs = list(run.glob('checkpoints/[0-9]*/pretrained_model/train_config.json')) + list(run.glob('train_config.json'))
        if not any(json.loads(file.read_text()).get('dataset', {}).get('root', '').endswith(
            'lerobot_v3_workspace50_v10_full_clean') for file in configs):
            continue
        checkpoints = sorted((run / 'checkpoints').glob('[0-9]*/pretrained_model'))
        if not checkpoints:
            continue
        errors = []
        for ckpt in checkpoints:
            differences = []
            for file in ckpt.glob('*processor*.safetensors'):
                with safe_open(file, framework='np') as tensors:
                    for metric in ('q01', 'q99'):
                        key = f'action.{metric}'
                        if key in tensors.keys():
                            differences.append(float(np.max(np.abs(tensors.get_tensor(key) - reference['action'][metric]))))
            errors.append(max(differences) if differences else None)
        # Do not move mixed, unidentified, or correct-statistics runs.
        if not all(value is not None and value > 1e-3 for value in errors):
            continue
        target = destination / run.name
        if target.exists():
            raise FileExistsError(target)
        records.append(dict(source=str(run), target=str(target), checkpoint_quantile_maxabs=errors))
    print(json.dumps(records, indent=2), flush=True)
    if args.apply:
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / 'moves.jsonl'
        for record in records:
            shutil.move(record['source'], record['target'])
            with manifest.open('a') as handle:
                handle.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()

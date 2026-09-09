"""Compare frozen dataset values and checkpoint processor statistics; never replace them."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
MODELS = {
    'pi05': 'pi05_workspace50_global_stats_expert_only_30k_20260827/checkpoints/030000/pretrained_model',
    'pap': 'pap_e1_late_sum_s42_20260906/checkpoints/010000/pretrained_model',
}


def main():
    report, arrays, checkpoint_stats = {}, {}, {}
    for name, relative in MODELS.items():
        model = ROOT/'outputs/train'/relative
        config = json.loads((model/'train_config.json').read_text())
        dataset = Path(config['dataset']['root'])
        stats = json.loads((dataset/'meta/stats.json').read_text())
        rows = pq.read_table(sorted((dataset/'data').rglob('*.parquet')),
                             columns=['index', 'episode_index', 'frame_index', 'action', 'observation.state']).to_pydict()
        order = np.argsort(rows['index'])
        arrays[name] = {key: np.asarray(value)[order] for key, value in rows.items()}
        with safe_open(model/'policy_postprocessor_step_0_unnormalizer_processor.safetensors', framework='np') as sf:
            checkpoint_stats[name] = {f'{feature}.{q}': sf.get_tensor(f'{feature}.{q}')
                                      for feature in ['action', 'observation.state'] for q in ['q01', 'q99']}
        report[name] = dict(checkpoint=str(model), dataset=str(dataset), frames=len(order), stats={})
        for feature in ['action', 'observation.state']:
            for q, quantile in [('q01', .01), ('q99', .99)]:
                key = f'{feature}.{q}'
                saved = checkpoint_stats[name][key]
                global_q = np.quantile(arrays[name][feature], quantile, axis=0)
                report[name]['stats'][key] = dict(checkpoint=saved.tolist(),
                    dataset_meta=stats[feature][q], recomputed_global=global_q.tolist(),
                    checkpoint_vs_meta_maxabs=float(np.max(np.abs(saved-np.array(stats[feature][q])))),
                    checkpoint_vs_global_maxabs=float(np.max(np.abs(saved-global_q))))
    report['dataset_values'] = {key: dict(shape_a=list(arrays['pi05'][key].shape),
        shape_b=list(arrays['pap'][key].shape), equal=bool(np.array_equal(arrays['pi05'][key], arrays['pap'][key])))
        for key in arrays['pi05']}
    report['checkpoint_difference'] = {key: float(np.max(np.abs(checkpoint_stats['pi05'][key]-checkpoint_stats['pap'][key])))
                                       for key in checkpoint_stats['pi05']}
    report['interpretation'] = 'Native-checkpoint comparison only. Do not swap normalization of trained models. This audit does not establish causal impact on success.'
    output = ROOT/'artifacts/pap_e1_late_sum_20260906/pi05_pap_statistics_audit.json'
    with output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

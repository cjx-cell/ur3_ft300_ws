"""CPU-only paired audit of the completed interface A/B full-Flow experiment."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
AB = ROOT / 'artifacts/deployment_validation_20260909_interface_ab'


def main():
    folders = {name: AB / (name + '_fullflow') for name in ('A_legacy', 'B_continuous')}
    for folder in folders.values():
        completed = json.loads((folder / 'completion.json').read_text())
        assert completed['completed'] and completed['records'] == 480
    manifests = {k: json.loads((v / 'manifest.json').read_text()) for k, v in folders.items()}
    for key in ('anchors', 'seeds', 'dataset_sha256', 'rtc', 'chunk_size', 'executed_prefix'):
        assert manifests['A_legacy'][key] == manifests['B_continuous'][key], key
    names = sorted(p.name for p in folders['A_legacy'].glob('ep*.npz'))
    assert len(names) == 60
    assert names == sorted(p.name for p in folders['B_continuous'].glob('ep*.npz'))
    rows = []
    for name in names:
        with np.load(folders['A_legacy'] / name) as a, np.load(folders['B_continuous'] / name) as b:
            for key in ('target', 'initial_noise', 'phase_mask', 'native/true/route', 'native/predicted/route'):
                assert np.array_equal(a[key], b[key]), (name, key)
            with np.load(ROOT / 'artifacts/deployment_validation_20260909_fullflow_s4_v1' / name) as original:
                for key in ('target', 'initial_noise', 'phase_mask'):
                    assert np.array_equal(a[key], original[key]), (name, key, 'source S4')
                assert np.array_equal(a['native/predicted/route'], original['old/predicted/route']), (name, 'source route')
            phase = name.split('_')[1]
            for arm, z in (('A_legacy', a), ('B_continuous', b)):
                for route in ('true', 'predicted'):
                    repeat = 'true_repeat' if route == 'true' else 'pred_repeat'
                    assert np.array_equal(z[f'native/{route}/action'], z[f'native/{repeat}/action'])
                row = {'pair': name, 'phase': phase, 'arm': arm}
                target = z['target']
                for route in ('true', 'predicted'):
                    delta = z[f'native/{route}/action'] - target
                    for length in (10, 50):
                        row[f'{route}_arm{length}_rmse'] = float(np.sqrt(np.mean(delta[:length, :6] ** 2)))
                        row[f'{route}_gripper{length}_rmse'] = float(np.sqrt(np.mean(delta[:length, 6] ** 2)))
                for label, reference, perturbed in (
                    ('epsilon_e2', 'true', 'true_e2_1e-8'),
                    ('one_percent_true_e2', 'true', 'true_mix_e2_0.01'),
                    ('one_percent_pred_e2', 'predicted', 'pred_mix_e2_0.01'),
                    ('one_percent_pred_e1', 'predicted', 'pred_mix_e1_0.01'),
                ):
                    delta = z[f'native/{perturbed}/action'] - z[f'native/{reference}/action']
                    row[label + '_arm10_delta_rmse'] = float(np.sqrt(np.mean(delta[:10, :6] ** 2)))
                    row[label + '_gripper10_delta_max'] = float(np.max(np.abs(delta[:10, 6])))
                    row[label + '_action50_delta_max'] = float(np.max(np.abs(delta)))
                rows.append(row)
    metrics = [key for key in rows[0] if key not in ('pair', 'phase', 'arm')]
    summaries = {}
    for phase in ('grasp', 'transport', 'insert'):
        summaries[phase] = {}
        for arm in folders:
            group = [r for r in rows if r['phase'] == phase and r['arm'] == arm]
            assert len(group) == 20
            summaries[phase][arm] = {m: float(np.mean([r[m] for r in group])) for m in metrics}
        paired = {}
        for metric in metrics:
            aa = {r['pair']: r[metric] for r in rows if r['phase'] == phase and r['arm'] == 'A_legacy'}
            bb = {r['pair']: r[metric] for r in rows if r['phase'] == phase and r['arm'] == 'B_continuous'}
            delta = np.array([bb[k] - aa[k] for k in sorted(aa)])
            paired[metric] = dict(mean_B_minus_A=float(delta.mean()), median_B_minus_A=float(np.median(delta)),
                                  B_lower_pairs=int((delta < 0).sum()), total_pairs=len(delta))
        summaries[phase]['paired'] = paired
    result = dict(passed_input_identity=True, passed_exact_repeats=True, pairs=60,
                  predicted_routes_identical_between_arms=True, phases=summaries, rows=rows,
                  predicted_routes_identical_to_source_S4=True,
                  limitations=['20 pairs per phase share five scenes and two anchors; not 20 independent scenes.',
                               'Existing training demonstrations; not held-out accuracy or closed-loop success.',
                               'No RTC in this diagnostic; closed-loop uses the fixed original RTC contract.',
                               'Perturbation stability alone does not establish useful physical information.'])
    out = AB / 'paired_fullflow_analysis.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({phase: {arm: {k: v for k, v in values.items() if k.startswith('predicted_')}
                             for arm, values in data.items() if arm != 'paired'}
                      for phase, data in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()

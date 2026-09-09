"""Read-only evidence checks; writes a new final audit, never edits scores."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
AB = ROOT / 'artifacts/deployment_validation_20260909_interface_ab'


def main():
    batch = AB / 'closed_5pos_seed0_continued_v2'
    rows = json.loads((batch / 'results.json').read_text())
    expected = {(m, e, 0) for m in ('pi05_reference', 'A_legacy', 'B_continuous')
                for e in (1, 11, 21, 31, 41)}
    assert len(rows) == 15
    assert {(r['eval_label'], r['episode'], r['seed']) for r in rows} == expected
    sources = {}
    contract = json.loads((batch / 'contract.json').read_text())
    for r in rows:
        assert r['evaluation_valid'] and not r['infrastructure_invalid_reasons']
        assert r['runner_exit_code'] == (0 if r['success'] else 6)
        assert r['batch_contract_id'] == 'interface_ab_fresh_gripper_read_v1'
        assert r['execution_prefix'] == 10 and not r['ever_attached']
        assert r['checkpoint'] == contract['checkpoints'][r['eval_label']]
        original = json.loads((Path(r['artifact_dir']) / 'result.json').read_text())
        assert all(r[k] == v for k, v in original.items())
        for path, digest in r['runtime_source_manifest'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
            assert path not in sources or sources[path] == digest
            sources[path] = digest
    plugins = []
    for folder in ('closed_5pos_seed0', 'closed_5pos_seed0_continued_v2'):
        plugins += [json.loads(s) for s in (AB / folder / 'loaded_plugin_runtime.jsonl').read_text().splitlines()]
    plugin_keys = {(r['closed_state']['model'], r['closed_state']['episode'], 0) for r in plugins}
    assert plugin_keys == expected
    assert {r['sha256'] for r in plugins} == {'383681a9db7958b838f1a8938c17e694c4f419a7397fa46e6b726fd8aece0ee3'}
    assert all(len(r['loaded']) == 1 for r in plugins)
    evidence = {'closed_trials': len(rows), 'unique_keys': len(expected),
                'runtime_source_files_verified': len(sources),
                'plugin_trial_keys_verified': len(plugin_keys),
                'plugin_records': len(plugins), 'source_unchanged': True}
    contacts = {}
    folders = [AB / f'{name}_contact_supplement_v2' for name in ('A_legacy', 'B_continuous')]
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in folders]
    for key in ('anchors', 'seeds', 'routes', 'rtc', 'source_sha256'):
        assert manifests[0][key] == manifests[1][key], key
    for p in folders:
        assert json.loads((p / 'completion.json').read_text()) == {'completed': True, 'records': 120}
        contacts[p.name] = json.loads((p / 'summary.json').read_text())
    files = sorted(p.name for p in folders[0].glob('*.npz'))
    assert len(files) == 20 and files == sorted(p.name for p in folders[1].glob('*.npz'))
    for name in files:
        with np.load(folders[0] / name) as a, np.load(folders[1] / name) as b:
            for key in ('target', 'initial_noise', 'contact_mask'):
                assert np.array_equal(a[key], b[key]), (name, key)
            for mode in manifests[0]['routes']:
                assert np.array_equal(a[mode + '/route'], b[mode + '/route']), (name, mode)
                assert np.isfinite(a[mode + '/action']).all()
                assert np.isfinite(b[mode + '/action']).all()
    evidence.update(contact_pairs=len(files), contact_calls=240,
                    contact_target_noise_routes_exact=True, contact_summary=contacts)
    with (AB / 'final_evidence_audit.json').open('x') as f:
        json.dump(evidence, f, indent=2)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()

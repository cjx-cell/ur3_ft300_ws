import json
from pathlib import Path
import tempfile
import unittest

from formal_batch import CONTRACT, continuation_rows


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='resident-continuation-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'code').mkdir()
        self.checkpoints = {'pi05': Path('/checkpoint')}
        self.manifest = {'/checkpoint/model': {'sha256': 'same'}}
        self.write('checkpoint_manifest.json', self.manifest)
        self.write('runtime_sources.json', {})
        self.write('contract.json', dict(id=CONTRACT, episodes=[1,11,21,31,41], seeds=[0,1,2]))
        self.rows = []
        for seed, (outcome, valid) in enumerate([('success', True), ('timeout', True), ('infrastructure_failure', False)]):
            artifact = self.root / f'trial{seed}'
            artifact.mkdir()
            raw = dict(episode=1, seed=seed, checkpoint='/checkpoint', batch_contract_id=CONTRACT,
                       evaluation_kind='formal', evaluation_valid=valid, outcome=outcome)
            (artifact / 'result.json').write_text(json.dumps(raw))
            self.rows.append(dict(raw, eval_label='pi05', artifact_dir=str(artifact)))
        self.save_rows()

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def save_rows(self):
        (self.root / 'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def read(self):
        return continuation_rows(self.root, self.checkpoints, self.manifest)

    def test_carry_valid_failures_too_and_preserve_invalid_evidence(self):
        rows, excluded = self.read()
        self.assertEqual([r['outcome'] for r in rows], ['success', 'timeout'])
        self.assertEqual(excluded, [self.rows[2]['artifact_dir']])
        self.assertTrue((Path(excluded[0]) / 'result.json').exists())

    def test_changed_weights_rejected(self):
        self.write('checkpoint_manifest.json', {'/checkpoint/model': {'sha256': 'changed'}})
        with self.assertRaisesRegex(RuntimeError, 'contents changed'):
            self.read()

    def test_duplicate_rejected(self):
        self.rows.append(self.rows[0])
        self.save_rows()
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            self.read()

    def test_raw_result_tampering_rejected(self):
        self.rows[1]['outcome'] = 'success'
        self.save_rows()
        with self.assertRaisesRegex(RuntimeError, 'raw evidence'):
            self.read()

    def test_changed_contract_rejected(self):
        self.write('contract.json', dict(id='different', episodes=[1,11,21,31,41], seeds=[0,1,2]))
        with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
            self.read()


if __name__ == '__main__':
    unittest.main()

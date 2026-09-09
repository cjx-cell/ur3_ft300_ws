import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_canary


class RunnerContractTests(unittest.TestCase):
    def test_formal_resident_preserves_execution_and_clears_overrides(self):
        with tempfile.TemporaryDirectory(prefix='resident-contract-test-') as directory:
            output = Path(directory) / 'trial'
            argv = ['run_canary.py', '--kind', 'pi05', '--checkpoint', '/checkpoint',
                    '--episode', '11', '--seed', '2', '--socket', '/test/model.sock',
                    '--formal-contract', 'test-resident-contract', '--record-video', 'false',
                    '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(run_canary.subprocess, 'call', return_value=0) as launch:
                with patch.dict(run_canary.os.environ, {'PAP_MOE_ACTION_CONDITIONING_SCALE': '0',
                                                       'WORKSPACE50_MAX_EPISODE_DURATION_S': '5'}):
                    with self.assertRaises(SystemExit) as stopped:
                        run_canary.main()
            self.assertEqual(stopped.exception.code, 0)
            env = launch.call_args.kwargs['env']
            self.assertEqual(env['WORKSPACE50_EVALUATION_KIND'], 'formal')
            self.assertEqual(env['WORKSPACE50_BATCH_CONTRACT_ID'], 'test-resident-contract')
            self.assertEqual(env['WORKSPACE50_MAX_EPISODE_DURATION_S'], '120')
            self.assertEqual(env['POLICY_ACTION_CHUNK_MAX_STEP_RAD'], '0.13')
            self.assertEqual(env['POLICY_GOAL_TIME_TOLERANCE_S'], '0')
            self.assertEqual(env['WORKSPACE50_RECORD_VIDEO'], 'false')
            self.assertNotIn('PAP_MOE_ACTION_CONDITIONING_SCALE', env)
            contract = json.loads((output / 'contract.json').read_text())
            self.assertEqual((contract['prediction'], contract['execution']), (50, 10))
            self.assertTrue(contract['formal'])
            snapshot = (output / 'runner_snapshot.sh').read_text()
            self.assertIn('"$RESIDENT_BRIDGE" --kind "$POLICY"', snapshot)
            self.assertNotIn('INFERENCE_ARGS+=(--trace-dir', snapshot)

    def test_formal_requires_resident_socket(self):
        with patch.object(sys, 'argv', ['run_canary.py', '--kind', 'pi05',
                   '--checkpoint', '/checkpoint', '--output', '/not-created', '--formal-contract', 'test']):
            with self.assertRaises(ValueError):
                run_canary.main()


if __name__ == '__main__':
    unittest.main()

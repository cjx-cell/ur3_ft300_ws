#!/usr/bin/python3
"""Run an in-memory shell snapshot and record runtime source fingerprints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('/home/ubuntu/ur3_ft300_ws')
    runner = root / 'scripts/run_workspace50_lerobot_policy_gazebo.sh'
    source = runner.read_text()
    peg = root / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole'
    files = [runner, Path(__file__), root / 'scripts/write_gazebo_eval_result.py',
             root / 'scripts/observe_fixture_geometry.py', *peg.glob('*.py')]
    for policy in ('pap_moe', 'pi05', 'rtc'):
        files += list((Path('/home/ubuntu/lerobot/src/lerobot/policies') / policy).glob('*.py'))
    manifest = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    # Use exactly the source passed to bash, even if the on-disk script is
    # changed between its first read and the fingerprint traversal.
    manifest[str(runner)] = hashlib.sha256(source.encode()).hexdigest()
    frozen_path = os.environ.get('WORKSPACE50_BATCH_SOURCE_MANIFEST')
    if frozen_path:
        expected = json.loads(Path(frozen_path).read_text())
        if manifest != expected:
            raise RuntimeError('Batch runtime sources changed; refusing mixed-code evaluation')
    checkpoint_manifest = os.environ.get('WORKSPACE50_BATCH_CHECKPOINT_MANIFEST')
    if checkpoint_manifest:
        for filename, record in json.loads(Path(checkpoint_manifest).read_text()).items():
            stat = Path(filename).stat()
            if stat.st_size != record['size'] or stat.st_mtime_ns != record['mtime_ns']:
                raise RuntimeError('Checkpoint files changed during frozen comparison: ' + filename)
    env = dict(os.environ, WORKSPACE50_SOURCE_MANIFEST_JSON=json.dumps(manifest))
    print('Frozen launcher SHA256: ' + manifest[str(runner)], flush=True)
    return subprocess.call(['bash', '-c', source, str(runner), *sys.argv[1:]], env=env)


if __name__ == '__main__':
    sys.exit(main())

"""Wait for one existing trial to clean up while its batch scheduler is stopped.

Does not resume or replace formal evaluations. Uses saved traces for offline
equivalence checks only; historical trace results are never added to the batch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
HERE = Path(__file__).resolve().parent


def process_state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler', type=int, required=True)
    parser.add_argument('--runner', type=int, required=True)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def status(state, **extra):
        value = dict(state=state, wall_time=time.time(), scheduler_pid=args.scheduler,
                     runner_pid=args.runner, trial=str(args.trial), **extra)
        target = args.output / 'status.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(target)
        print(json.dumps(value), flush=True)

    try:
        status('waiting_for_CURRENT_TRIAL_only')
        while process_state(args.runner) not in (None, 'Z'):
            if process_state(args.scheduler) != 'T':
                raise RuntimeError('Scheduler is not paused; refusing GPU overlap')
            time.sleep(5)
        if process_state(args.scheduler) != 'T':
            raise RuntimeError('Scheduler pause lost')
        result = json.loads((args.trial / 'result.json').read_text())
        if not result.get('evaluation_valid'):
            raise RuntimeError('Current trial ended engineering-invalid; inspect before validation')
        for filename, expected in json.loads((args.batch / 'runtime_sources.json').read_text()).items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen source changed: {filename}')
        for filename, expected in json.loads((args.batch / 'checkpoint_manifest.json').read_text()).items():
            stat = Path(filename).stat()
            if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
                raise RuntimeError(f'Checkpoint changed: {filename}')
        s4 = ROOT / 'artifacts/gazebo_pap_moe_workspace50_20260908_170923_ep0031_seed2'
        s3 = ROOT / 'artifacts/gazebo_pap_moe_workspace50_20260908_134838_ep0001_seed0'
        baseline = ROOT / 'outputs/train/pi05_workspace50_global_stats_expert_only_30k_20260827/checkpoints/030000/pretrained_model'
        plan = [('pap_s4', 'pap_moe', s4, None), ('pap_s3', 'pap_moe', s3, None),
                ('pi05', 'pi05', s4, baseline)]
        for label, kind, artifact, override in plan:
            if process_state(args.scheduler) != 'T':
                raise RuntimeError('Scheduler resumed unexpectedly')
            source = json.loads((artifact / 'result.json').read_text())
            checkpoint = str(override or source['checkpoint'])
            status('validating_offline', label=label, checkpoint=checkpoint,
                   reference_trace=str(artifact / 'diagnostic'))
            with (args.output / f'{label}.log').open('x') as log:
                subprocess.run([sys.executable, str(HERE / 'validate.py'), '--kind', kind,
                    '--checkpoint', checkpoint, '--trace', str(artifact / 'diagnostic'),
                    '--seed', str(source['seed']), '--output', str(args.output / label)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        status('offline_checks_passed_FORMAL_BATCH_STILL_PAUSED',
               next_step='Live adapter and cold/resident closed-loop canary; do not mix formal modes')
    except Exception as exc:
        status('blocked_FORMAL_BATCH_STILL_PAUSED', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()

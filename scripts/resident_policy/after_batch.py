"""Wait for the frozen batch, then run OFFLINE resident equivalence checks.

Never start another simulator or change a formal evaluation launcher.
Fail closed on incomplete batch, changed runtime/checkpoint, or occupied GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / 'status.json'

    def status(state, **extra):
        data = dict(state=state, wall_time=time.time(), **extra)
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(status_path)
        print(json.dumps(data), flush=True)

    try:
        status('waiting_for_frozen_batch', batch=str(args.batch))
        completion_path = args.batch / 'completion.json'
        while not completion_path.exists():
            time.sleep(30)
        completion = json.loads(completion_path.read_text())
        if not completion.get('completed'):
            raise RuntimeError(f'Frozen batch did not finish normally: {completion}')
        rows = [json.loads(line) for line in (args.batch / 'results.jsonl').read_text().splitlines()]
        if len(rows) != 45 or not all(row.get('evaluation_valid') for row in rows):
            raise RuntimeError('Expected 45 valid completed evaluations')
        sources = json.loads((args.batch / 'runtime_sources.json').read_text())
        for filename, expected in sources.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen source changed: {filename}')
        checkpoints = json.loads((args.batch / 'checkpoint_manifest.json').read_text())
        for filename, expected in checkpoints.items():
            stat = Path(filename).stat()
            if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
                raise RuntimeError(f'Checkpoint changed: {filename}')
        # Let the batch's last process cleanup finish, then refuse other GPU owners.
        time.sleep(30)
        owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                         '--format=csv,noheader'], text=True).strip()
        if owners:
            raise RuntimeError(f'GPU still occupied; no validation launched: {owners}')
        selected = {}
        for row in rows:
            selected.setdefault(row['checkpoint'], row)
        if len(selected) != 3:
            raise RuntimeError('Expected exactly three checkpoint groups')
        pap_trace = next(Path(row['artifact_dir']) / 'diagnostic'
                         for row in selected.values() if row['policy'] == 'pap_moe')
        for index, (checkpoint, row) in enumerate(selected.items()):
            # Baseline has no per-chunk live trace in this batch; reuse identical
            # saved sensor inputs, but DO NOT compare it to PAP's action targets.
            trace = Path(row['artifact_dir']) / 'diagnostic' if row['policy'] == 'pap_moe' else pap_trace
            status('validating_offline', checkpoint=checkpoint, trace=str(trace))
            command = [sys.executable, str(HERE / 'validate.py'), '--kind', row['policy'],
                       '--checkpoint', checkpoint, '--trace', str(trace), '--seed', str(row['seed']),
                       '--output', str(args.output / f'model_{index}')]
            subprocess.run(command, check=True)
        status('offline_checks_passed_NOT_APPROVED_FOR_FORMAL_ROLLOUT',
               next_step='Implement live adapter and paired closed-loop canary; keep cold launcher unchanged')
    except Exception as exc:
        status('blocked_no_automatic_retry', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()

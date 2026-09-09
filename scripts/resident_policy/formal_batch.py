"""Fresh 45-trial comparison, same resident execution mode for all three models.

Requires explicit reviewed-canary evidence. Never resumes/mutates the old cold
batch or borrows its scores. Freezes candidate code before starting any model.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/ubuntu/ur3_ft300_ws')
CONTRACT = 'workspace50_resident_50_10_joint_atomic_v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def continuation_rows(previous, checkpoints, manifest):
    """Carry every valid trial, never select by outcome or overwrite evidence."""
    contract = json.loads((previous / 'contract.json').read_text())
    if contract['id'] != CONTRACT or contract['episodes'] != [1, 11, 21, 31, 41] or contract['seeds'] != [0, 1, 2]:
        raise RuntimeError('Continuation execution contract mismatch')
    old_manifest = json.loads((previous / 'checkpoint_manifest.json').read_text())
    if set(old_manifest) != set(manifest) or any(
            old_manifest[f]['sha256'] != manifest[f]['sha256'] for f in manifest):
        raise RuntimeError('Continuation checkpoint contents changed')
    allowed_changes = {'bridge.py', 'formal_batch.py'}
    for path in (previous / 'code').glob('*.py'):
        if path.name not in allowed_changes and sha(path) != sha(HERE / path.name):
            raise RuntimeError(f'Unreviewed continuation code change: {path.name}')
    old_runtime = json.loads((previous / 'runtime_sources.json').read_text())
    for filename, expected in old_runtime.items():
        if sha(Path(filename)) != expected:
            raise RuntimeError(f'Previous frozen/runtime source changed: {filename}')
    rows, excluded, seen = [], [], set()
    for line in (previous / 'results.jsonl').read_text().splitlines():
        row = json.loads(line)
        if not row.get('evaluation_valid'):
            excluded.append(row['artifact_dir'])
            continue
        key = (row['eval_label'], row['episode'], row['seed'])
        if key in seen or key[0] not in checkpoints or key[1] not in contract['episodes'] or key[2] not in contract['seeds']:
            raise RuntimeError(f'Duplicate/unexpected continuation key: {key}')
        if row['checkpoint'] != str(checkpoints[key[0]]) or row['batch_contract_id'] != CONTRACT or row['evaluation_kind'] != 'formal':
            raise RuntimeError(f'Continuation row contract mismatch: {key}')
        artifact = Path(row['artifact_dir'])
        raw = json.loads((artifact / 'result.json').read_text())
        if any(row.get(k) != value for k, value in raw.items()):
            raise RuntimeError(f'Continuation row differs from raw evidence: {key}')
        seen.add(key)
        rows.append(dict(row, carried_from=str(previous),
                         execution_revision='resident_v1_before_snapshot_retry'))
    return rows, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reviewed-canary-runs', nargs=2, type=Path, required=True)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--continue-from', type=Path,
                        help='Reviewed resident batch; preserve ALL valid trials and run missing keys only')
    args = parser.parse_args()
    v2, v3 = args.reviewed_canary_runs
    for label, folder in [('s4', v2), ('pi05', v3), ('s3', v3)]:
        evidence = json.loads((folder / f'{label}_summary.json').read_text())
        if not all(t.get('evaluation_valid') for t in evidence['trials']):
            raise RuntimeError(f'Invalid canary evidence: {label}')
        if len(evidence['trials']) != 3 or len(evidence['socket_replay']) < 2:
            raise RuntimeError(f'Incomplete canary evidence: {label}')
    if not args.frozen:
        args.output.mkdir(parents=True, exist_ok=False)
        target = args.output / 'code'
        target.mkdir()
        for path in HERE.glob('*.py'):
            copied = target / path.name
            shutil.copy2(path, copied)
            copied.chmod(0o444)
        with (args.output / 'batch.log').open('x') as log:
            command = [sys.executable, str(target / 'formal_batch.py'), '--frozen',
                '--output', str(args.output), '--reviewed-canary-runs', str(v2), str(v3)]
            if args.continue_from:
                command += ['--continue-from', str(args.continue_from.resolve())]
            code = subprocess.call(command,
                stdout=log, stderr=log)
        (args.output / 'completion.json').write_text(json.dumps(dict(exit_code=code, completed=code == 0)))
        raise SystemExit(code)

    entries = [('pi05', 'pi05', 'pi05_workspace50_global_stats_expert_only_30k_20260827', '030000'),
               ('pap_s3', 'pap_moe', 'pap_corrected_physicsgate_sequence_20260907_091425', '015000'),
               ('pap_s4', 'pap_moe', 'pap_corrected_gate_calibration_20260907_104300', '015000')]
    checkpoints = {label: ROOT / 'outputs/train' / name / 'checkpoints' / step / 'pretrained_model'
                   for label, _, name, step in entries}
    manifest = {}
    for checkpoint in checkpoints.values():
        for path in checkpoint.rglob('*'):
            if path.is_file():
                stat = path.stat()
                manifest[str(path)] = dict(sha256=sha(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    (args.output / 'checkpoint_manifest.json').write_text(json.dumps(manifest, indent=2))
    original_manifest = ROOT / 'artifacts/paired_comparison_20260908_joint_atomic_v2/runtime_sources.json'
    runtime = json.loads(original_manifest.read_text())
    for filename, expected in runtime.items():
        if sha(Path(filename)) != expected:
            raise RuntimeError(f'Original runtime differs from validated version: {filename}')
    runtime.update({str(path): sha(path) for path in HERE.glob('*.py')})
    (args.output / 'runtime_sources.json').write_text(json.dumps(runtime, indent=2))
    (args.output / 'contract.json').write_text(json.dumps(dict(id=CONTRACT, episodes=[1,11,21,31,41],
        seeds=[0,1,2], trials=45, prediction=50, execution=10, rtc='arm_only EXP max10',
        controller_max_step_rad=.13, duration_s=120, goal_grace_s=5,
        mode='resident model per checkpoint; fresh Gazebo every episode',
        original_partial_cold_batch='preserved separately; no scores imported',
        geometry='unchanged legacy termination plus cad_seating_shadow_v2',
        canary_evidence=[str(v2), str(v3)], run_order='model, position, seed',
        checkpoints={k:str(v) for k,v in checkpoints.items()},
        caveats=['finite canary evidence, not proof of equal success probabilities',
                 'no release stability proof; no same-information no-MoE baseline']), indent=2))
    rows, excluded = continuation_rows(args.continue_from, checkpoints, manifest) if args.continue_from else ([], [])
    (args.output / 'continuation.json').write_text(json.dumps(dict(
        previous=str(args.continue_from) if args.continue_from else None,
        carried_valid_trials=len(rows), excluded_invalid_artifacts=excluded,
        delta='pre-inference stable snapshot read only; backend/actions/RTC/checkpoints unchanged',
        note='Versioned continuation, not 45 newly rerun trials; no outcome-based selection'), indent=2))
    with (args.output / 'results.jsonl').open('x') as output:
        for row in rows:
            output.write(json.dumps(row) + '\n')
    completed_keys = {(r['eval_label'], r['episode'], r['seed']) for r in rows}

    def status(state, **extra):
        value = dict(state=state, completed=len(rows), wall_time=time.time(), **extra)
        path = args.output / 'status.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(path)
        print(json.dumps(value), flush=True)

    try:
        for label, kind, _, _ in entries:
            if all((label, e, s) in completed_keys for e in [1,11,21,31,41] for s in [0,1,2]):
                continue
            owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if owners:
                raise RuntimeError(f'GPU occupied: {owners}')
            checkpoint = checkpoints[label]
            socket_dir = Path(tempfile.mkdtemp(prefix='pap-formal-resident-'))
            sock = socket_dir / 'model.sock'
            daemon_dir = args.output / f'{label}_daemon'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
            status('loading_model_once', label=label)
            with (args.output / f'{label}_daemon.log').open('x') as log:
                daemon = subprocess.Popen([sys.executable, str(HERE / 'daemon.py'), '--kind', kind,
                    '--checkpoint', str(checkpoint), '--socket', str(sock), '--output', str(daemon_dir)],
                    env=env, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 900
                    while not (daemon_dir / 'ready.json').exists():
                        if daemon.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('Resident model loading failed')
                        time.sleep(2)
                    seen_tokens = set()
                    for episode in [1, 11, 21, 31, 41]:
                        for seed in [0, 1, 2]:
                            if (label, episode, seed) in completed_keys:
                                continue
                            for filename, expected in runtime.items():
                                if sha(Path(filename)) != expected:
                                    raise RuntimeError(f'Batch runtime changed: {filename}')
                            for filename, record in manifest.items():
                                stat = Path(filename).stat()
                                if stat.st_size != record['size'] or stat.st_mtime_ns != record['mtime_ns']:
                                    raise RuntimeError(f'Checkpoint changed: {filename}')
                            status('evaluating', label=label, episode=episode, seed=seed)
                            trial = args.output / f'{label}_ep{episode:04d}_seed{seed}'
                            code = subprocess.call([sys.executable, str(HERE / 'run_canary.py'),
                                '--kind', kind, '--checkpoint', str(checkpoint), '--episode', str(episode),
                                '--seed', str(seed), '--socket', str(sock), '--output', str(trial),
                                '--formal-contract', CONTRACT, '--record-video', 'true' if seed == 0 else 'false'])
                            text = (trial / 'run.log').read_text()
                            artifact = next(Path(line.split()[-1]) for line in text.splitlines()
                                            if line.strip().startswith('artifacts:'))
                            row = json.loads((artifact / 'result.json').read_text())
                            row.update(eval_label=label, artifact_dir=str(artifact), runner_exit_status=code)
                            row['execution_revision'] = 'resident_v2_stable_snapshot'
                            with (args.output / 'results.jsonl').open('a') as output:
                                output.write(json.dumps(row) + '\n')
                            rows.append(row)
                            if not row.get('evaluation_valid') or row.get('evaluation_kind') != 'formal':
                                raise RuntimeError(f'Invalid formal episode: {artifact}')
                            ready = json.loads((artifact / 'resident_ready.json').read_text())
                            token = ready['resident_episode_token']
                            if token in seen_tokens or daemon.poll() is not None:
                                raise RuntimeError('Resident session reuse/liveness failure')
                            seen_tokens.add(token)
                finally:
                    if daemon.poll() is None:
                        daemon.send_signal(signal.SIGTERM)
                        try:
                            daemon.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            daemon.kill()
                            daemon.wait()
                    if not list(socket_dir.iterdir()):
                        socket_dir.rmdir()
        (args.output / 'results.json').write_text(json.dumps(rows, indent=2))
        summary = {label: dict(trials=len(group), successes=sum(r['success'] for r in group),
                               outcomes=dict(Counter(r['outcome'] for r in group)))
                   for label, *_ in entries for group in [[r for r in rows if r['eval_label'] == label]]}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        status('complete', summary=summary)
    except Exception as exc:
        status('blocked_no_retry', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()

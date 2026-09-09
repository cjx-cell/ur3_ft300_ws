"""Serial engineering canaries for original-cold vs persistent-worker bridge.

Runs original cold, same-input socket replay, then two fresh Gazebo episodes
on ONE resident model. Reports evidence; never auto-approves a formal switch.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/ubuntu/ur3_ft300_ws')


def rpc(stream, request):
    stream.write(json.dumps(request) + '\n')
    stream.flush()
    response = json.loads(stream.readline())
    if 'error' in response:
        raise RuntimeError(response)
    return response


def trial_result(directory):
    path = next(line.split()[-1] for line in (directory / 'run.log').read_text().splitlines()
                if line.strip().startswith('artifacts:'))
    result = json.loads((Path(path) / 'result.json').read_text())
    if not result.get('evaluation_valid'):
        raise RuntimeError(f'Engineering-invalid canary: {path}')
    return Path(path), result


def socket_replay(sock, artifact, kind, seed, output):
    output.mkdir(exist_ok=False)
    errors = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(60)
        connection.connect(str(sock))
        with connection.makefile('rw') as stream:
            hello = json.loads(stream.readline())
            if hello['kind'] != kind:
                raise RuntimeError('Wrong socket policy')
            token = rpc(stream, dict(op='begin', seed=seed))['token']
            references = sorted((artifact / ('diagnostic' if kind == 'pap_moe' else 'cold_trace')).glob('chunk_*.npz'))
            if len(references) < 2:
                raise RuntimeError('Need at least two consecutive reference blocks for RTC')
            for sequence, reference in enumerate(references):
                if reference.name != f'chunk_{sequence:05d}.npz':
                    raise RuntimeError('Reference trace has a gap')
                trace = reference
                if kind == 'pi05':
                    trace = output / f'input_{sequence:05d}.npz'
                    with np.load(reference, allow_pickle=False) as data, trace.open('xb') as target:
                        np.savez(target, **{'observation/' + k: data[k] for k in ('state', 'camera0', 'camera1')})
                response = rpc(stream, dict(op='infer', token=token, sequence=sequence, trace=str(trace)))
                if response['token'] != token or response['sequence'] != sequence:
                    raise RuntimeError('Socket response identity mismatch')
                mappings = [('physical_action', 'physical_action' if kind == 'pap_moe' else 'predicted_action_chunk', 1e-4),
                            ('execution_action', 'published_action' if kind == 'pap_moe' else 'execution_chunk', 1e-4)]
                if kind == 'pap_moe':
                    mappings += [('normalized_action', 'normalized_action', 1e-5), ('route_sequence', 'route_sequence', 1e-6)]
                error = {}
                with np.load(reference, allow_pickle=False) as ref, np.load(response['result'], allow_pickle=False) as got:
                    for key, other, tolerance in mappings:
                        if got[key].shape != ref[other].shape:
                            raise RuntimeError(f'Original-main output shape mismatch: {key}')
                        difference = float(np.max(np.abs(got[key].astype(np.float64) - ref[other])))
                        if not np.isfinite(difference) or difference > tolerance:
                            raise RuntimeError(f'Original-main socket mismatch: {key} {difference}')
                        error[key] = difference
                errors.append(dict(sequence=sequence, errors=error, result=response['result']))
            rpc(stream, dict(op='end', token=token))
    (output / 'result.json').write_text(json.dumps(dict(passed=True, comparisons=errors), indent=2))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['s4', 'pi05', 's3'], default=['s4', 'pi05', 's3'])
    parser.add_argument('--reuse-cold-root', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = [
        ('s4', 'pap_moe', 'pap_corrected_gate_calibration_20260907_104300', '015000', 31, 2),
        ('pi05', 'pi05', 'pi05_workspace50_global_stats_expert_only_30k_20260827', '030000', 1, 1),
        ('s3', 'pap_moe', 'pap_corrected_physicsgate_sequence_20260907_091425', '015000', 1, 0),
    ]
    entries = [entry for entry in entries if entry[0] in args.models]

    def status(state, **extra):
        value = dict(state=state, wall_time=time.time(), **extra)
        path = args.output / 'status.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(path)
        print(json.dumps(value), flush=True)

    summaries = []
    try:
        for label, kind, name, step, episode, seed in entries:
            checkpoint = ROOT / 'outputs/train' / name / 'checkpoints' / step / 'pretrained_model'

            def run(mode, sock=None):
                if mode == 'cold' and args.reuse_cold_root:
                    previous = args.reuse_cold_root / f'{label}_cold'
                    if previous.exists():
                        contract = json.loads((previous / 'contract.json').read_text())
                        expected = dict(mode='original_cold', kind=kind, checkpoint=str(checkpoint.resolve()),
                                        episode=episode, seed=seed, prediction=50, execution=10, duration_s=120)
                        if any(contract.get(k) != v for k, v in expected.items()):
                            raise RuntimeError('Saved cold reference contract mismatch')
                        # Original model/controller must still match. Candidate transport
                        # source changes are the intervention under test, not a reason
                        # to regenerate a valid original-main reference.
                        import hashlib
                        for filename, digest in contract['sources'].items():
                            if ('/src/lerobot/policies/' in filename or '/scripts/peg_in_hole/' in filename
                                    or filename.endswith('run_workspace50_lerobot_policy_gazebo.sh')):
                                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                                    raise RuntimeError(f'Original runtime changed: {filename}')
                        artifact, result = trial_result(previous)
                        if Path(result['checkpoint']).resolve() != checkpoint.resolve():
                            raise RuntimeError('Saved reference checkpoint mismatch')
                        return artifact, dict(mode='cold_reused_reference', artifact=str(artifact),
                            outcome=result.get('outcome'), evaluation_valid=result.get('evaluation_valid'),
                            geometry=result.get('geometry_shadow'), reference_contract=str(previous / 'contract.json'))
                directory = args.output / f'{label}_{mode}'
                command = [sys.executable, str(HERE / 'run_canary.py'), '--kind', kind,
                           '--checkpoint', str(checkpoint), '--episode', str(episode),
                           '--seed', str(seed), '--output', str(directory)]
                if sock:
                    command += ['--socket', str(sock)]
                status('running_canary', label=label, mode=mode, episode=episode, seed=seed)
                started = time.monotonic()
                if directory.exists():
                    # Only the explicitly pre-started first S4 cold trial may exist.
                    if label != 's4' or mode != 'cold':
                        raise FileExistsError(directory)
                    while not (directory / 'completion.json').exists():
                        time.sleep(5)
                else:
                    subprocess.run(command, check=False)
                exit_code = json.loads((directory / 'completion.json').read_text())['exit_code']
                artifact, result = trial_result(directory)
                # A policy task failure is evidence, not an infrastructure retry.
                return artifact, dict(mode=mode, artifact=str(artifact), outcome=result.get('outcome'),
                    geometry=result.get('geometry_shadow'), evaluation_valid=result.get('evaluation_valid'),
                    runner_exit_code=exit_code,
                    observed_wait_or_run_seconds=time.monotonic() - started)

            cold, cold_result = run('cold')
            owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                             '--format=csv,noheader'], text=True).strip()
            if owners:
                raise RuntimeError(f'GPU not released after cold trial: {owners}')
            socket_dir = Path(tempfile.mkdtemp(prefix='pap-resident-'))
            sock = socket_dir / 'model.sock'
            daemon_dir = args.output / f'{label}_daemon'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
            status('loading_resident_daemon', label=label)
            with (args.output / f'{label}_daemon.log').open('x') as log:
                daemon = subprocess.Popen([sys.executable, str(HERE / 'daemon.py'), '--kind', kind,
                    '--checkpoint', str(checkpoint), '--socket', str(sock), '--output', str(daemon_dir)],
                    env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 600
                    while not sock.exists():
                        if daemon.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('Daemon failed or load timeout')
                        time.sleep(2)
                    worker_pid = json.loads((daemon_dir / 'ready.json').read_text())['worker_pid']
                    status('same_input_socket_replay', label=label, worker_pid=worker_pid)
                    replay = socket_replay(sock, cold, kind, seed, args.output / f'{label}_socket_replay')
                    trials = [cold_result]
                    session_tokens = set()
                    for mode in ('resident_1', 'resident_2'):
                        artifact, result = run(mode, sock)
                        if daemon.poll() is not None:
                            raise RuntimeError('Daemon died during Gazebo teardown')
                        ready = json.loads((artifact / 'resident_ready.json').read_text())
                        token = ready['resident_episode_token']
                        if token in session_tokens:
                            raise RuntimeError('Episode token reused')
                        session_tokens.add(token)
                        requests = [json.loads(line) for line in (artifact / 'resident_requests.jsonl').read_text().splitlines()]
                        if not requests or [r['sequence'] for r in requests] != list(range(len(requests))):
                            raise RuntimeError('Missing or duplicate sequence in bridge')
                        if any(r['token'] != token for r in requests):
                            raise RuntimeError('Cross-episode response contamination')
                        latencies = np.array([r['latency_ms'] for r in requests])
                        result.update(worker_pid=worker_pid, reset_seconds=ready['reset_seconds'],
                            chunks=len(requests), latency_ms_max=float(latencies.max()),
                            latency_ms_p95=float(np.quantile(latencies, .95)))
                        trials.append(result)
                    summary = dict(label=label, checkpoint=str(checkpoint), episode=episode, seed=seed,
                        socket_replay=replay, trials=trials, formal_approved=False,
                        note='Finite engineering canary, not a statistical policy comparison or equivalence proof')
                    (args.output / f'{label}_summary.json').write_text(json.dumps(summary, indent=2))
                    summaries.append(summary)
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
        (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
        status('canaries_completed_REVIEW_BEFORE_FORMAL_SWITCH')
    except Exception as exc:
        status('blocked_NO_RETRY_FORMAL_BATCH_PAUSED', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()

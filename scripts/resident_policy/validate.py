"""Compare fresh-process A against resident A -> B -> A, including RTC chunk 2.

No Gazebo launch. Run ONLY after the active evaluation releases the GPU.
PAP validation additionally checks outputs against the original live trace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def call(process, request):
    process.stdin.write(json.dumps(request) + '\n')
    process.stdin.flush()
    response = json.loads(process.stdout.readline())
    if 'error' in response:
        raise RuntimeError(response)
    return response


def run_worker(args, directory, seeds):
    owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                     '--format=csv,noheader'], text=True).strip()
    if owners:
        raise RuntimeError(f'GPU occupied; refusing concurrent validation: {owners}')
    log = directory.with_suffix('.log')
    all_results = []
    start = time.monotonic()
    with log.open('x') as stream:
        process = subprocess.Popen([sys.executable, str(HERE / 'worker.py'),
            '--kind', args.kind, '--checkpoint', str(args.checkpoint),
            '--output', str(directory)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stream, text=True)
        try:
            response = json.loads(process.stdout.readline())
            if not response.get('ready'):
                raise RuntimeError(response)
            load_seconds = time.monotonic() - start
            for seed in seeds:
                begin = time.monotonic()
                token = call(process, dict(op='begin', seed=seed))['token']
                reset_seconds = time.monotonic() - begin
                paths = []
                for sequence in range(args.chunks):
                    trace = args.trace / f'chunk_{sequence:05d}.npz'
                    response = call(process, dict(op='infer', token=token,
                                                 sequence=sequence, trace=str(trace)))
                    if response['token'] != token or response['sequence'] != sequence:
                        raise RuntimeError('Response session/sequence mismatch')
                    paths.append(response['result'])
                call(process, dict(op='end', token=token))
                all_results.append(dict(seed=seed, paths=paths, reset_seconds=reset_seconds))
            process.stdin.close()
            if process.wait(timeout=60) != 0:
                raise RuntimeError(f'Worker failed: {log}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return dict(load_seconds=load_seconds, episodes=all_results)


def compare(left, right, keys):
    result = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        for key, other, tolerance in keys:
            if a[key].shape != b[other].shape:
                raise RuntimeError(f'{key} shape mismatch')
            difference = float(np.max(np.abs(a[key].astype(np.float64) - b[other])))
            result[key] = difference
            if not np.isfinite(difference) or difference > tolerance:
                raise RuntimeError(f'{key} max error {difference} > {tolerance}: {left}, {right}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['pi05', 'pap_moe'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--chunks', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.chunks < 2:
        raise ValueError('At least two chunks are required to exercise RTC leftover')
    args.output.mkdir(parents=True, exist_ok=False)
    for i in range(args.chunks):
        if not (args.trace / f'chunk_{i:05d}.npz').is_file():
            raise FileNotFoundError(f'Missing trace chunk {i}')
    if args.kind == 'pap_moe':
        source = json.loads((args.trace.parent / 'result.json').read_text())
        if Path(source['checkpoint']).resolve() != args.checkpoint.resolve() or source['seed'] != args.seed:
            raise ValueError('Historical trace is from a different checkpoint/seed')
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(HERE.glob('*.py'))}
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        sources=manifest, checkpoint=str(args.checkpoint), trace=str(args.trace), seed=args.seed,
        contract='50/10; arm-only RTC EXP max10; exact checkpoint processors'), indent=2))
    cold = run_worker(args, args.output / 'cold', [args.seed])
    resident = run_worker(args, args.output / 'resident', [args.seed, args.seed + 1, args.seed])
    keys = [('normalized_action', 'normalized_action', 1e-5),
            ('physical_action', 'physical_action', 1e-4),
            ('execution_action', 'execution_action', 1e-4)]
    if args.kind == 'pap_moe':
        keys.append(('route_sequence', 'route_sequence', 1e-6))
    comparisons = []
    for i, reference in enumerate(cold['episodes'][0]['paths']):
        for j in (0, 2):
            comparisons.append(dict(chunk=i, resident_episode=j,
                errors=compare(reference, resident['episodes'][j]['paths'][i], keys)))
        if args.kind == 'pap_moe':
            historical_keys = [(key, 'published_action' if key == 'execution_action' else other, tol)
                               for key, other, tol in keys]
            comparisons.append(dict(chunk=i, original_live_trace=True,
                errors=compare(reference, args.trace / f'chunk_{i:05d}.npz', historical_keys)))
    report = dict(passed=True, cold=cold, resident=resident, comparisons=comparisons,
        formal_rollout_approved=False,
        outstanding=['Live paired-I/O adapter', 'Same-scene cold/resident closed-loop canary',
                     'Pi0.5 original-main numerical reference' if args.kind == 'pi05'
                     else 'Original-main reference passed only for tested trace chunks'])
    (args.output / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

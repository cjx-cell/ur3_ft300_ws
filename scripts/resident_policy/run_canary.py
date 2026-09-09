"""Engineering-only shell snapshot. Change inference process, not ROS control.

Original runtime files remain untouched; resulting trials are never appended
to the suspended formal batch. An immutable snapshot records the exact delta.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['pi05', 'pap_moe'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--socket', help='Omit for original cold inference')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--formal-contract', help='Reviewed resident batch ID; requires a resident socket')
    parser.add_argument('--record-video', choices=['true', 'false'], default='true')
    args = parser.parse_args()
    if args.formal_contract and not args.socket:
        raise ValueError('Formal resident runs require a persistent model socket')
    args.output.mkdir(parents=True, exist_ok=False)
    runner = ROOT / 'scripts/run_workspace50_lerobot_policy_gazebo.sh'
    original = runner.read_text()
    source = original
    if not args.socket and args.kind == 'pi05':
        marker = '"$PI_PYTHON" "$INFERENCE" "${INFERENCE_ARGS[@]}" >"$LOG_DIR/inference.log" 2>&1 &'
        if source.count(marker) != 1:
            raise RuntimeError('Cold inference launch signature changed')
        source = source.replace(marker,
            'INFERENCE_ARGS+=(--trace-dir "$LOG_DIR/cold_trace" --trace-first-chunks 200)\n' + marker)
    if args.socket:
        old = '"$PI_PYTHON" "$INFERENCE" "${INFERENCE_ARGS[@]}" >"$LOG_DIR/inference.log" 2>&1 &'
        new = ('"$PI_PYTHON" "$RESIDENT_BRIDGE" --kind "$POLICY" --checkpoint "$CHECKPOINT" '
               '--seed "$POLICY_SEED" --socket "$RESIDENT_SOCKET" >"$LOG_DIR/inference.log" 2>&1 &')
        if source.count(old) != 1:
            raise RuntimeError('Inference launch signature changed; manual review required')
        source = source.replace(old, new)
    # Fingerprint original and substituted source plus all runtime policy/ROS files.
    paths = [runner, ROOT / 'scripts/write_gazebo_eval_result.py',
             ROOT / 'scripts/observe_fixture_geometry.py',
             * (ROOT / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole').glob('*.py'),
             *HERE.glob('*.py')]
    for name in ('pap_moe', 'pi05', 'rtc'):
        paths.extend((Path('/home/ubuntu/lerobot/src/lerobot/policies') / name).glob('*.py'))
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('WORKSPACE50_', 'PAP_MOE_', 'POLICY_', 'RESIDENT_')):
            del env[key]
    env.update(WORKSPACE50_EVALUATION_KIND='formal' if args.formal_contract else 'engineering_resident_canary',
               WORKSPACE50_POLICY_SEED=str(args.seed), WORKSPACE50_RECORD_VIDEO=args.record_video,
               WORKSPACE50_DIAGNOSTIC_TRACE='true', WORKSPACE50_GEOMETRY_SHADOW='true',
               POLICY_ACTION_CHUNK_MAX_STEP_RAD='0.13', POLICY_GOAL_TIME_TOLERANCE_S='0',
               WORKSPACE50_MAX_EPISODE_DURATION_S='120', PAP_MOE_ROUTING_SOURCE='physicsgate',
               WORKSPACE50_SOURCE_MANIFEST_JSON=json.dumps(manifest),
               RESIDENT_BRIDGE=str(HERE / 'bridge.py'), RESIDENT_SOCKET=args.socket or '')
    if args.formal_contract:
        env['WORKSPACE50_BATCH_CONTRACT_ID'] = args.formal_contract
    (args.output / 'runner_snapshot.sh').write_text(source)
    (args.output / 'contract.json').write_text(json.dumps(dict(
        mode='resident' if args.socket else 'original_cold', kind=args.kind,
        checkpoint=str(args.checkpoint.resolve()), episode=args.episode, seed=args.seed,
        prediction=50, execution=10, rtc='arm_only EXP max10', duration_s=120,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        snapshot_sha256=hashlib.sha256(source.encode()).hexdigest(), sources=manifest,
        formal=bool(args.formal_contract)), indent=2))
    with (args.output / 'run.log').open('x') as log:
        code = subprocess.call(['bash', '-c', source, str(runner), args.kind,
                                str(args.checkpoint), str(args.episode), 'false'],
                               env=env, stdout=log, stderr=subprocess.STDOUT)
    (args.output / 'completion.json').write_text(json.dumps(dict(exit_code=code)))
    raise SystemExit(code)


if __name__ == '__main__':
    main()

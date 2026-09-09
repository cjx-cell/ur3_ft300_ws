#!/usr/bin/env python3
"""Freeze and execute the existing three-checkpoint ordinary-task comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/ubuntu/ur3_ft300_ws')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    entries = [
        ('pi05', 'pi05', 'pi05_workspace50_global_stats_expert_only_30k_20260827', '030000'),
        ('pap_moe', 'pap_s3', 'pap_corrected_physicsgate_sequence_20260907_091425', '015000'),
        ('pap_moe', 'pap_s4', 'pap_corrected_gate_calibration_20260907_104300', '015000'),
    ]
    triples, checkpoints = [], {}
    for policy, label, directory, step in entries:
        checkpoint = ROOT/'outputs/train'/directory/'checkpoints'/step/'pretrained_model'
        if not (checkpoint/'model.safetensors').is_file():
            raise FileNotFoundError(checkpoint)
        triples.extend([policy, label, str(checkpoint)])
        print('Fingerprint checkpoint: ' + label, flush=True)
        for path in sorted(checkpoint.rglob('*')):
            if path.is_file():
                stat = path.stat()
                checkpoints[str(path)] = dict(sha256=sha(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    peg = ROOT/'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole'
    files = [ROOT/'scripts/run_workspace50_lerobot_policy_gazebo.sh',
             ROOT/'scripts/run_frozen_workspace50_eval.py', ROOT/'scripts/write_gazebo_eval_result.py',
             ROOT/'scripts/observe_fixture_geometry.py', *peg.glob('*.py')]
    for policy in ('pap_moe', 'pi05', 'rtc'):
        files += list((Path('/home/ubuntu/lerobot/src/lerobot/policies')/policy).glob('*.py'))
    source_path = args.output/'runtime_sources.json'
    source_path.write_text(json.dumps({str(p): sha(p) for p in files}, indent=2))
    checkpoint_path = args.output/'checkpoint_manifest.json'
    checkpoint_path.write_text(json.dumps(checkpoints, indent=2))
    contract = dict(id='workspace50_20260908_controller013_geometryv2',
                    task='ordinary insertion, not special-regime superiority',
                    episodes=[1,11,21,31,41], seeds=[0,1,2], trials=45,
                    prediction=50, execution=10, rtc=True, controller_max_step_rad=.13,
                    goal_time_override_s=0, duration_s=120, routing='physicsgate',
                    legacy_termination=dict(xy_m=.008, peg_z_m=.890, checks=5),
                    geometry='cad_seating_shadow_v2; sampled inserted/seated, not release stability',
                    checkpoints=triples, run_order='model, position, seed',
                    remaining_caveats=['payload calibration', 'RTC elapsed-time causality',
                                       'no same-information no-MoE baseline', 'no dynamic release validation'])
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2))
    env = dict(os.environ)
    # Do not inherit exploratory condition masks, backbones, or execution overrides.
    for key in list(env):
        if key.startswith(('WORKSPACE50_', 'PAP_MOE_', 'POLICY_')):
            del env[key]
    env.update(WORKSPACE50_BATCH_CONTRACT_ID=contract['id'],
               WORKSPACE50_BATCH_SOURCE_MANIFEST=str(source_path),
               WORKSPACE50_BATCH_CHECKPOINT_MANIFEST=str(checkpoint_path),
               WORKSPACE50_GEOMETRY_SHADOW='true', WORKSPACE50_DIAGNOSTIC_TRACE='true',
               WORKSPACE50_EVAL_EPISODES='1 11 21 31 41', WORKSPACE50_EVAL_SEEDS='0 1 2',
               POLICY_ACTION_CHUNK_MAX_STEP_RAD='0.13', POLICY_GOAL_TIME_TOLERANCE_S='0',
               WORKSPACE50_MAX_EPISODE_DURATION_S='120', PAP_MOE_ROUTING_SOURCE='physicsgate')
    print('Frozen 45-trial comparison: ' + str(args.output), flush=True)
    with (args.output/'batch.log').open('x') as log:
        status = subprocess.call(['bash', str(ROOT/'scripts/run_workspace50_multiseed_eval.sh'),
                                  str(args.output), *triples], env=env, stdout=log, stderr=subprocess.STDOUT)
    (args.output/'completion.json').write_text(json.dumps(dict(exit_code=status, completed=status == 0)))
    raise SystemExit(status)


if __name__ == '__main__':
    main()

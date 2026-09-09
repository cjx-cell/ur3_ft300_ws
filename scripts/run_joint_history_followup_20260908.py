#!/usr/bin/env python3
"""Supplementary engineering runs; never append them to the old formal batch."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
OUTPUT = ROOT/'artifacts/joint_history_followup_20260908_v1'


def main():
    OUTPUT.mkdir(exist_ok=False)
    reference = ROOT/'artifacts/gazebo_pap_moe_workspace50_20260908_170923_ep0031_seed2/result.json'
    original = json.loads(reference.read_text())
    assert original['evaluation_valid'] and original['evaluation_kind'] == 'engineering_joint_history_v2'
    manifest = OUTPUT/'runtime_sources.json'
    manifest.write_text(json.dumps(original['runtime_source_manifest'], indent=2))
    baseline = ROOT/'outputs/train/pi05_workspace50_global_stats_expert_only_30k_20260827/checkpoints/030000/pretrained_model'
    pap = ROOT/'outputs/train/pap_corrected_gate_calibration_20260907_104300/checkpoints/015000/pretrained_model'
    runs = [('pi05', baseline, 31, 2), *[('pap_moe', pap, 41, seed) for seed in (0, 1, 2)]]
    (OUTPUT/'contract.json').write_text(json.dumps(dict(
        kind='engineering_joint_history_v2_supplement', reference=str(reference),
        runs=[[p,str(c),e,s] for p,c,e,s in runs],
        note='Not a complete new three-model comparison; do not merge into the old source revision.'),indent=2))
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('WORKSPACE50_', 'POLICY_', 'PAP_MOE_')):
            del env[key]
    env.update(WORKSPACE50_BATCH_SOURCE_MANIFEST=str(manifest),
               WORKSPACE50_BATCH_CHECKPOINT_MANIFEST=str(ROOT/'artifacts/paired_comparison_20260908_controller013_v1/checkpoint_manifest.json'),
               WORKSPACE50_EVALUATION_KIND='engineering_joint_history_v2_supplement',
               WORKSPACE50_GEOMETRY_SHADOW='true', WORKSPACE50_DIAGNOSTIC_TRACE='true',
               WORKSPACE50_MAX_EPISODE_DURATION_S='120', WORKSPACE50_RECORD_VIDEO='true',
               POLICY_ACTION_CHUNK_MAX_STEP_RAD='0.13', POLICY_GOAL_TIME_TOLERANCE_S='0')
    results = []
    for policy, checkpoint, episode, seed in runs:
        log = OUTPUT/f'{policy}_ep{episode:04d}_seed{seed}.log'
        print(f'RUN {policy} episode={episode} seed={seed}', flush=True)
        env['WORKSPACE50_POLICY_SEED'] = str(seed)
        with log.open('x') as stream:
            code = subprocess.call(['/usr/bin/python3',str(ROOT/'scripts/run_frozen_workspace50_eval.py'),
                                    policy,str(checkpoint),str(episode),'false'], env=env,
                                   stdout=stream,stderr=subprocess.STDOUT)
        paths = [line.split('artifacts:',1)[1].strip() for line in log.read_text().splitlines() if line.startswith('  artifacts:')]
        path = Path(paths[0])/'result.json' if paths else None
        result = json.loads(path.read_text()) if path and path.exists() else dict(evaluation_valid=False, outcome='runner_failure')
        result.update(runner_exit_code=code, source_result=str(path))
        results.append(result)
        (OUTPUT/'results.json').write_text(json.dumps(results,indent=2))
        print(f"RESULT {result['outcome']} valid={result['evaluation_valid']}",flush=True)
        if not result['evaluation_valid']:
            (OUTPUT/'completion.json').write_text(json.dumps(dict(completed=False,reason='engineering invalid')))
            raise SystemExit(3)
    (OUTPUT/'completion.json').write_text(json.dumps(dict(completed=True, trials=len(results))))


if __name__ == '__main__':
    main()

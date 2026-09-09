"""Re-evaluate fixed saved policies after environment-readiness repair, no training."""
import json
import argparse
import os
import shutil
import signal
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
PY = '/home/ubuntu/miniconda3/envs/pi0-env/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--carry-valid-from', type=Path)
    args = parser.parse_args()
    source = ROOT / 'artifacts/pap_corrected_pipeline_20260907_000738'
    output = ROOT / 'artifacts' / ('pap_corrected_reeval_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    (output / 'runner_snapshot').mkdir()
    for name in ['run_workspace50_lerobot_policy_gazebo.sh', 'run_workspace50_multiseed_eval.sh']:
        shutil.copy2(ROOT / 'scripts' / name, output / 'runner_snapshot' / name)
    for name in ['expert_action_joint_flow.json', 'physicsgate_sequence_flow.json', 'gate_calibration_flow.json']:
        shutil.copy2(source / name, output / name)
    old = json.loads((source / 'status.json').read_text())
    models = [('physicsgate_sequence', 'pap_moe', old['physicsgate_sequence_checkpoint']),
              ('gate_calibration', 'pap_moe', old['gate_calibration_checkpoint']),
              ('pi05', 'pi05', old['baseline'])]
    if args.carry_valid_from:
        for label, _, checkpoint in models:
            prior = args.carry_valid_from / (label+'_closed/results.jsonl')
            if not prior.exists():
                continue
            attempts = [json.loads(line) for line in prior.read_text().splitlines() if line.strip()]
            valid = [row for row in attempts if row.get('evaluation_valid') is True]
            assert all(row['checkpoint'] == checkpoint for row in valid)
            assert len({(row['requested_episode'], row['requested_seed']) for row in valid}) == len(valid)
            target = output / (label+'_closed')
            target.mkdir()
            (target/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in valid))
            shutil.copy2(prior, output / (label+'_previous_attempts.jsonl'))
    env = dict(os.environ, PYTHONPATH='/home/ubuntu/lerobot/src', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WORKSPACE50_EVAL_RUNNER=str(output / 'runner_snapshot/run_workspace50_lerobot_policy_gazebo.sh'),
               WORKSPACE50_EVAL_EPISODES='1 11 21 31 41', WORKSPACE50_EVAL_SEEDS='0 1 2',
               WORKSPACE50_ROS_DOMAIN_ID='77', PAP_MOE_ROUTING_SOURCE='physicsgate')
    env['WORKSPACE50_RESUME_VALID'] = 'true'
    state = {'status': 'running', 'completed': [], 'source': str(source), 'output': str(output), 'models': models}

    def save():
        state['updated'] = datetime.now().isoformat()
        temp = output / 'status.pending'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(output / 'status.json')

    def run(name, command, hours):
        state.update(task=name, command=list(map(str, command)))
        save()
        with (output / (name+'.log')).open('x') as log:
            proc = subprocess.Popen(list(map(str, command)), env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['child_pid'] = proc.pid
            save()
            try:
                rc = proc.wait(timeout=hours*3600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                raise RuntimeError(name + ': timeout')
        if rc:
            raise RuntimeError(f'{name}: exit {rc}; stopped for inspection, no blind retry')
        state['completed'].append(name)
        save()

    print(output, flush=True)
    save()
    subprocess.Popen([PY, ROOT/'scripts/write_corrected_pap_report.py', '--run', output], cwd=ROOT)
    try:
        for label, policy, checkpoint in models:
            run(label+'_closed', ['bash', output/'runner_snapshot/run_workspace50_multiseed_eval.sh',
                                 output/(label+'_closed'), policy, label, checkpoint], 12)
        episodes = []
        for ep in [1, 11, 21, 31, 41]:
            episodes += ['--episode-npz', ROOT/f'pap_moe_framework/datasets/workspace_50_v10_canonical/pick_up_the_peg_and_insert_it_into_the_hole_episode_{ep:04d}_success/data.npz']
        run('pi05_flow', [PY, ROOT/'scripts/eval_pi05_matched_flow_mse.py', '--checkpoint', old['baseline'],
                         *episodes, '--max-frames-per-expert', '8', '--batch-size', '1', '--seed', '1000',
                         '--num-seeds', '3', '--output', output/'pi05_flow.json'], 4)
        state['status'] = 'completed'
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        save()
        # Write immediately as well; watcher provides progress/failure resilience.
        import runpy
        report = runpy.run_path(str(ROOT/'scripts/write_corrected_pap_report.py'))
        report['update'].__globals__['RUN'] = output
        report['update'](state)


if __name__ == '__main__':
    main()

"""Sequential corrected-statistics training and matched evaluation; no crash retries."""
import argparse
import json
import os
import re
import subprocess
import shutil
import signal
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
PY = '/home/ubuntu/miniconda3/envs/pi0-env/bin/python'
DATA = ROOT / 'pap_moe_framework/datasets/lerobot_v3_workspace50_v10_full_clean_global_stats_v1'
BASE = ROOT / 'outputs/train/pi05_workspace50_global_stats_expert_only_30k_20260827/checkpoints/030000/pretrained_model'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-run', type=Path)
    args = parser.parse_args()
    out = args.resume_run or ROOT / 'artifacts' / ('pap_corrected_pipeline_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args.resume_run:
        previous = json.loads((out / 'status.json').read_text())
        assert previous['status'] == 'failed' and previous['task'] in ('expert_action_joint', 'expert_action_joint_flow')
        log = (out / 'expert_action_joint.log').read_text()
        assert 'End of training' in log and 'CORRECTED_PAP_CONTRACT_PASS' in log
        resume_checkpoint = Path(re.search(r'final model:\s+(\S+)', log).group(1))
        verify_finished_checkpoint(resume_checkpoint)
        shutil.copy2(out / 'status.json', out / ('status_before_resume_' + datetime.now().strftime('%H%M%S') + '.json'))
    else:
        out.mkdir(exist_ok=False)
        resume_checkpoint = None
    # Launch fixed copies, never a shell script that is being edited in place.
    snapshot = out / ('runner_snapshot_' + datetime.now().strftime('%H%M%S'))
    snapshot.mkdir()
    for script in ['run_pap_moe_v9_stage_train.sh', 'run_pap_moe_vnext_stage_train.sh']:
        shutil.copy2(ROOT / 'scripts' / script, snapshot / script)
    env = dict(os.environ, PYTHONPATH='/home/ubuntu/lerobot/src', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True',
               PAP_MOE_GATE_ARCHITECTURE='physics_gate_v2',
               PAP_MOE_VERIFY_CORRECTED_CONTRACT='1', PAP_MOE_RUN_FAMILY='pap_corrected',
               PAP_MOE_SAVE_FREQ='3000', PAP_MOE_REBUILD_PROCESSORS='1',
               PAP_MOE_PRESERVE_PRETRAINED_PROCESSOR_STATS='0',
               PAP_MOE_BOUNDED_CONDITIONING='false', PAP_MOE_JOINT_ACTION_EXPERT_LR_SCALE='1.0',
               WORKSPACE50_EVAL_EPISODES='1 11 21 31 41', WORKSPACE50_EVAL_SEEDS='0 1 2',
               PAP_MOE_ROUTING_SOURCE='physicsgate')
    env['PAP_MOE_STAGE_RUNNER'] = str(snapshot / 'run_pap_moe_v9_stage_train.sh')
    state = {'status': 'starting', 'output': str(out), 'baseline': str(BASE), 'dataset': str(DATA), 'completed': []}

    def write():
        state['updated'] = datetime.now().isoformat()
        temporary = out / 'status.json.pending'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(out / 'status.json')

    def run(name, command, hours=48):
        state.update(status='running', task=name, command=list(map(str, command)))
        write()
        print(name, out, flush=True)
        logfile = out / f'{name}.log'
        if logfile.exists() and args.resume_run and name == 'expert_action_joint_flow':
            logfile.rename(out / f'{name}_failed_{datetime.now():%H%M%S}.log')
        with logfile.open('x') as log:
            child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['child_pid'] = child.pid
            write()
            # Process lifetime/timeout monitoring, no automatic retry on failure.
            try:
                code = child.wait(timeout=hours * 3600)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                raise RuntimeError(f'{name}: hard timeout')
        if code:
            raise RuntimeError(f'{name}: exit {code}; see {out / (name + ".log")}')
        state['completed'].append(name)
        write()

    episodes = []
    for ep in (1, 11, 21, 31, 41):
        episodes += ['--episode-npz', ROOT / f'pap_moe_framework/datasets/workspace_50_v10_canonical/pick_up_the_peg_and_insert_it_into_the_hole_episode_{ep:04d}_success/data.npz']

    def flow(name, checkpoint, routing):
        run(name, [PY, ROOT / 'scripts/eval_pap_moe_expert_ablation.py', '--checkpoint', checkpoint,
                   *episodes, '--max-frames-per-expert', '8', '--batch-size', '1', '--seed', '1000',
                   '--num-seeds', '3', '--routing-source', routing, '--mask', 'full', '--mask', 'all_zero',
                   '--continuous-gripper', '--output', out / f'{name}.json'], hours=4)

    try:
        source = BASE
        for stage, steps in [('expert_action_joint', 30000), ('physicsgate_sequence', 15000), ('gate_calibration', 15000)]:
            if stage == 'expert_action_joint' and resume_checkpoint:
                source = resume_checkpoint
                state['completed'].append(stage)
                state['resume_note'] = 'Verified completed 30k weights; previous shell exit 127 occurred after End of training. No retraining.'
            else:
                run(stage, ['bash', snapshot / 'run_pap_moe_vnext_stage_train.sh', stage, source, DATA, str(steps), '1'])
                log = (out / f'{stage}.log').read_text()
                assert 'End of training' in log and 'CORRECTED_PAP_CONTRACT_PASS' in log
                source = Path(re.search(r'final model:\s+(\S+)', log).group(1))
                verify_finished_checkpoint(source)
            state[stage + '_checkpoint'] = str(source)
            write()
            flow(stage + '_flow', source, 'dataset' if stage == 'expert_action_joint' else 'predicted')
            if stage != 'expert_action_joint':
                # Preserve stage-3 assessment before starting stage 4.
                run(stage + '_closed', ['bash', ROOT / 'scripts/run_workspace50_multiseed_eval.sh',
                    out / (stage + '_closed'), 'pap_moe', stage, source], hours=12)
        run('pi05_flow', [PY, ROOT / 'scripts/eval_pi05_matched_flow_mse.py', '--checkpoint', BASE,
                         *episodes, '--max-frames-per-expert', '8', '--batch-size', '1', '--seed', '1000',
                         '--num-seeds', '3', '--output', out / 'pi05_flow.json'], hours=4)
        run('pi05_closed', ['bash', ROOT / 'scripts/run_workspace50_multiseed_eval.sh', out / 'pi05_closed',
                           'pi05', 'pi05', BASE], hours=12)
        state['status'] = 'completed'
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        write()


def verify_finished_checkpoint(checkpoint):
    import numpy as np
    from safetensors import safe_open

    cfg = json.loads((checkpoint / 'config.json').read_text())
    assert cfg['physical_fusion_architecture'] == 'action_input_tokens_v2'
    assert cfg['physics_gate_architecture'] == 'physics_gate_v2'
    with safe_open(checkpoint / 'model.safetensors', framework='np') as tensors:
        keys = list(tensors.keys())
        assert any('action_out_proj.weight' in key for key in keys)
        assert not any('route_forecaster.' in key for key in keys)
        for key in keys:
            tensors.get_slice(key).get_shape()
    stats = json.loads((DATA / 'meta/stats.json').read_text())
    checked = set()
    for file in checkpoint.glob('*processor*.safetensors'):
        with safe_open(file, framework='np') as tensors:
            for feature in ['action', 'observation.state', 'observation.state_history']:
                for metric in ['q01', 'q99']:
                    key = feature + '.' + metric
                    if key in tensors.keys():
                        np.testing.assert_allclose(tensors.get_tensor(key), stats[feature][metric], atol=1e-6, rtol=1e-5)
                        checked.add(key)
    assert len(checked) == 6
    print('CHECKPOINT_VERIFIED', checkpoint, flush=True)


if __name__ == '__main__':
    main()

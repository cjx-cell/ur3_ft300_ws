"""After A/B finishes: new-physics paired 5-position/seed0 diagnostic batch.

Resident model per checkpoint; fresh Gazebo per trial; independent results.
Old 45-trial scores are never imported or overwritten.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

ROOT=Path('/home/ubuntu/ur3_ft300_ws')
AB=ROOT/'artifacts/deployment_validation_20260909_interface_ab'
CODE=AB/'resident_code'
PY='/home/ubuntu/miniconda3/envs/pi0-env/bin/python'


def check_result_status(exit_code, raw):
    assert exit_code in (0,6), exit_code
    assert raw['evaluation_valid'] and not raw['infrastructure_invalid_reasons'],raw
    assert (exit_code==0)==raw['success'],(exit_code,raw['outcome'])


def read_trial(folder, label, checkpoint, episode, contract_id):
    exit_code=json.loads((folder/'completion.json').read_text())['exit_code']
    log=(folder/'run.log').read_text()
    matches=re.findall(r'^  artifacts:\s+(\S+)',log,re.M)
    assert len(matches)==1,matches
    artifact=Path(matches[0]);raw=json.loads((artifact/'result.json').read_text())
    # The existing result writer intentionally returns 6 for valid failures.
    # Do not discard or repeat failed tasks, and never hide an invalid trial.
    check_result_status(exit_code,raw)
    assert raw['checkpoint']==str(checkpoint)
    assert raw['episode']==episode and raw['seed']==0
    assert raw['batch_contract_id']==contract_id
    for path,expected in raw['runtime_source_manifest'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, path
    return dict(raw,eval_label=label,artifact_dir=str(artifact),runner_exit_code=exit_code)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-wait',action='store_true',help='Only a verified interrupted wait with zero started trials')
    parser.add_argument('--output',type=Path,default=AB/'closed_5pos_seed0')
    parser.add_argument('--continue-from',type=Path,help='Import every completed valid task, including failures; never rerun them')
    args=parser.parse_args()
    out=args.output
    if args.resume_wait:
        assert sorted(p.name for p in out.iterdir())==['status.json'], 'Cannot resume any started experiment'
        previous=json.loads((out/'status.json').read_text())
        assert previous['status']=='waiting_for_ab' and previous['completed']==0
        (out/'interrupted_wait_v1.json').write_text(json.dumps(dict(previous=previous,
            observed_exit_code=143,reason='Waiting process terminated before any model load or trial; explicit supervised restart.',
            restarted=datetime.now().isoformat()),indent=2))
    else:
        out.mkdir(exist_ok=False)
    state=dict(status='waiting_for_ab',completed=0,total=15)
    def write(**values):
        state.update(values,updated=datetime.now().isoformat())
        tmp=out/'status.pending';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'status.json')
    write()
    deadline=time.monotonic()+8*3600
    while time.monotonic()<deadline:
        if (AB/'status.json').exists():
            previous=json.loads((AB/'status.json').read_text())
            if previous['status']=='failed':
                write(status='blocked_upstream_failure',upstream=previous);return 1
            if previous['status']=='completed':break
        write(waiting_pid=os.getpid())
        print(f'{datetime.now().isoformat()} waiting for A/B',flush=True)
        time.sleep(30)
    else:write(status='blocked_upstream_timeout');return 1
    checkpoints={
        'pi05_reference':ROOT/'outputs/train/pi05_workspace50_global_stats_expert_only_30k_20260827/checkpoints/030000/pretrained_model',
        'A_legacy':ROOT/'outputs/train/pap_interface_ab_20260909_A_legacy/checkpoints/010000/pretrained_model',
        'B_continuous':ROOT/'outputs/train/pap_interface_ab_20260909_B_continuous/checkpoints/010000/pretrained_model'}
    contract=dict(id='interface_ab_fresh_gripper_read_v1',episodes=[1,11,21,31,41],seeds=[0],
        checkpoints={k:str(v) for k,v in checkpoints.items()},prediction=50,execution=10,
        rtc='arm-only EXP max10 delay0',arm_step_limit_rad=.13,duration_sim_s=120,
        success='original xy<=8mm, peg_z<.89, 5 simulated-time checks; geometry shadow independent',
        gripper_plugin='isolated fresh-read counter candidate; same for all models',
        warning='5 paired positions, one policy seed; diagnostic, not precise population success rates. Pi0.5 reference has different training budget and input modalities; A/B is the single-variable contrast.')
    (out/'contract.json').write_text(json.dumps(contract,indent=2))
    env=dict(os.environ,PYTHONPATH=str(AB/'code'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    rows=[]
    def save_rows():
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        summary={name:dict(trials=len(group),successes=sum(x['success'] for x in group),
            outcomes=dict(Counter(x['outcome'] for x in group)))
            for name in checkpoints for group in [[x for x in rows if x['eval_label']==name]]}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        write(completed=len(rows),summary=summary)
    if args.continue_from:
        previous=json.loads((args.continue_from/'status.json').read_text())
        assert previous['status']=='blocked_no_retry'
        assert json.loads((args.continue_from/'contract.json').read_text())==contract
        imported=[]
        for label,checkpoint in checkpoints.items():
            for episode in contract['episodes']:
                folder=args.continue_from/f'{label}_ep{episode:04d}_seed0'
                if not folder.exists():continue
                row=read_trial(folder,label,checkpoint,episode,contract['id'])
                rows.append(row)
                imported.append(dict(folder=str(folder),artifact=row['artifact_dir'],outcome=row['outcome']))
        assert len(rows)>=previous['completed']
        (out/'continuation.json').write_text(json.dumps(dict(source=str(args.continue_from),
            reason='Supervisor mistook task-failure exit 6 for infrastructure failure; preserve every completed result without retries.',
            imported=imported,changed_runtime_contract=False),indent=2))
        with (out/'results.jsonl').open('x') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        save_rows()
    try:
        for label,checkpoint in checkpoints.items():
            pending=[episode for episode in contract['episodes'] if not any(r['eval_label']==label and r['episode']==episode for r in rows)]
            if not pending:continue
            kind='pi05' if label=='pi05_reference' else 'pap_moe'
            socket_dir=Path(tempfile.mkdtemp(prefix='pap_interface_'))
            socket=socket_dir/'model.sock'
            logfile=(out/f'{label}_daemon.log').open('x')
            daemon=subprocess.Popen([PY,CODE/'daemon.py','--kind',kind,'--checkpoint',checkpoint,
                '--socket',socket,'--output',out/f'{label}_daemon'],env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            write(status='loading',model=label,daemon_pid=daemon.pid)
            try:
                ready_deadline=time.monotonic()+600
                while not socket.exists():
                    if daemon.poll() is not None:raise RuntimeError(f'{label} daemon exited {daemon.returncode}')
                    if time.monotonic()>ready_deadline:raise TimeoutError('model load')
                    time.sleep(2)
                for episode in pending:
                    folder=out/f'{label}_ep{episode:04d}_seed0'
                    write(status='running',model=label,episode=episode)
                    result=subprocess.run([PY,CODE/'run_canary.py','--kind',kind,'--checkpoint',checkpoint,
                        '--episode',str(episode),'--seed','0','--socket',socket,'--output',folder,
                        '--formal-contract',contract['id'],'--record-video','true'],env=env,timeout=900)
                    row=read_trial(folder,label,checkpoint,episode,contract['id']);rows.append(row)
                    with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    save_rows()
            finally:
                if daemon.poll() is None:
                    daemon.terminate()
                    try:daemon.wait(timeout=40)
                    except subprocess.TimeoutExpired:os.killpg(daemon.pid,signal.SIGKILL);daemon.wait()
                logfile.close()
        write(status='completed')
        (out/'completion.json').write_text(json.dumps(dict(completed=True,trials=len(rows))))
    except Exception as exc:
        write(status='blocked_no_retry',error=repr(exc));raise
    return 0


if __name__=='__main__':raise SystemExit(main())

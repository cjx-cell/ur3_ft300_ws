#!/usr/bin/env python3
"""Fixed paired A/B queue: real preflights, four trainings, frozen diagnostics."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/ubuntu/ur3_ft300_ws')
EXP=ROOT/'artifacts/pap_e1_norm_ab_20260906'
SOURCE=ROOT/'outputs/train/pap_lr_low_verified_10k_20260905/checkpoints/010000/pretrained_model'
DATA=ROOT/'pap_moe_framework/datasets/workspace_50_v10_canonical'
PY='/home/ubuntu/miniconda3/envs/pi0-env/bin/python'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def status(**kwargs):
    kwargs['time']=datetime.datetime.now().isoformat()
    (EXP/'status.json').write_text(json.dumps(kwargs,indent=2))
    print(json.dumps(kwargs),flush=True)


def run(name,command,hashes):
    for path,expected in hashes.items():
        assert digest(path)==expected, 'Fingerprint changed: '+path
    logpath=EXP/(name+'.log')
    with logpath.open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        start=time.monotonic()
        while child.poll() is None:
            status(state='RUNNING',stage=name,pid=child.pid,elapsed_s=round(time.monotonic()-start),
                   log=str(logpath),log_age_s=round(time.time()-logpath.stat().st_mtime))
            time.sleep(30)
    if child.returncode:
        status(state='FAILED',stage=name,returncode=child.returncode,log=str(logpath))
        raise RuntimeError(name+' failed')
    status(state='STAGE_COMPLETED',stage=name)


def main():
    EXP.mkdir(exist_ok=False)
    os.environ.update(PYTHONPATH='/home/ubuntu/lerobot/src',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True',LEROBOT_REBUILD_PROCESSORS='0',
        LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS='1',OPENBLAS_NUM_THREADS='1')
    source_cfg=json.loads((SOURCE/'train_config.json').read_text())
    configs={}
    for label,mode in [('A','joint'),('B','branchwise')]:
        view=EXP/('initial_'+label);view.mkdir()
        pc=json.loads((SOURCE/'config.json').read_text())
        pc.update(e1_fusion_normalization=mode,mask_invalid_prefix_tokens=True,mask_invalid_history_cameras=True)
        (view/'config.json').write_text(json.dumps(pc,indent=2))
        for path in SOURCE.iterdir():
            if path.name!='config.json' and path.is_file():(view/path.name).symlink_to(path)
        for seed in [42,43]:
            cfg=copy.deepcopy(source_cfg)
            name=f'{label}_s{seed}'
            cfg.update(resume=False,steps=10000,save_freq=5000,seed=seed,use_policy_training_preset=True,
                output_dir=str(ROOT/f'outputs/train/pap_e1_norm_{name}_20260906'),
                job_name='pap_e1_norm_'+name,eval_freq=20000)
            cfg['policy'].update(pretrained_path=str(SOURCE),e1_fusion_normalization=mode,
                mask_invalid_prefix_tokens=True,mask_invalid_history_cameras=True)
            assert cfg['policy']['joint_action_expert_lr_scale']==.1
            assert cfg['policy']['train_expert_action_joint']
            assert not Path(cfg['output_dir']).exists()
            configs[name]=cfg
            (EXP/(name+'.json')).write_text(json.dumps(cfg,indent=2))
    for seed in [42,43]:
        a,b=copy.deepcopy(configs[f'A_s{seed}']),copy.deepcopy(configs[f'B_s{seed}'])
        for c in [a,b]:
            c.pop('output_dir');c.pop('job_name');c['policy'].pop('e1_fusion_normalization')
        assert a==b,'Unexpected A/B difference'
    watched=list(Path('/home/ubuntu/lerobot/src/lerobot/policies/pap_moe').glob('*.py'))
    watched+=list(Path('/home/ubuntu/lerobot/src/lerobot/policies/pi05').glob('*.py'))
    watched+=list(Path('/home/ubuntu/lerobot/src/lerobot/optim').glob('*.py'))
    watched+=[Path('/home/ubuntu/lerobot/src/lerobot/scripts/lerobot_train.py')]
    watched+=list(EXP.glob('*.json'))+list(EXP.glob('initial_*/config.json'))
    watched+=list(SOURCE.glob('policy_*'))+[SOURCE/'config.json',SOURCE/'model.safetensors']
    watched+=list(DATA.glob('*_success/data.npz'))
    for name in ['train_pap_checked.py','preflight_e1_norm.py','audit_e1_information_fusion.py','summarize_e1_information_fusion.py','run_e1_norm_ab.py']:
        watched.append(ROOT/'scripts'/name)
    hashes={str(path):digest(path) for path in watched}
    (EXP/'manifest.json').write_text(json.dumps(dict(source=str(SOURCE),configs=configs,hashes=hashes,
        comparison='only E1 normalization differs within each seed; masks true in BOTH; no reuse of old A',
        training_seeds=[42,43],expected_steps=10000),indent=2))
    # Both real-batch preflights must pass before either group trains.
    for label in ['A','B']:
        run('preflight_'+label,[PY,'-u',str(ROOT/'scripts/preflight_e1_norm.py'),
            '--checkpoint',str(EXP/('initial_'+label)),'--output',str(EXP/('preflight_'+label)),
            '--episode',str(DATA/'pick_up_the_peg_and_insert_it_into_the_hole_episode_0001_success/data.npz')],hashes)
    for seed in [42,43]:
        for label in ['A','B']:
            name=f'{label}_s{seed}'
            run('train_'+name,[PY,'-u',str(ROOT/'scripts/train_pap_checked.py'),'--config_path='+str(EXP/(name+'.json'))],hashes)
            output=Path(configs[name]['output_dir'])
            assert json.loads((output/'checkpoints/010000/training_state/training_step.json').read_text())['step']==10000
            saved=json.loads((output/'checkpoints/010000/pretrained_model/config.json').read_text())
            assert saved['e1_fusion_normalization']==configs[name]['policy']['e1_fusion_normalization']
            diagnostic=EXP/('eval_'+name)
            run('eval_'+name,[PY,'-u',str(ROOT/'scripts/audit_e1_information_fusion.py'),
                '--checkpoint',str(output/'checkpoints/010000/pretrained_model'),'--dataset',str(DATA),
                '--output',str(diagnostic)],hashes)
            assert len((diagnostic/'actions.jsonl').read_text().splitlines())==720
            run('summarize_'+name,[PY,str(ROOT/'scripts/summarize_e1_information_fusion.py'),
                '--output',str(diagnostic)],hashes)
    rows=['# E1归一化A/B最终结果','','两组共同启用有效mask；oracle开环诊断，不是闭环成功率。','',
        '| 训练seed | 模式 | 臂前10步MSE | 夹爪前10步MSE |','|---|---|---:|---:|']
    for seed in [42,43]:
        for label in ['A','B']:
            result=json.loads((EXP/f'eval_{label}_s{seed}/action_summary.json').read_text())['full']
            rows.append(f"| {seed} | {label} | {result['arm_mse_10']:.8f} | {result['gripper_mse_10']:.8f} |")
    rows+=['','完整probe和分支干预见各eval目录；按事先门槛解释，不只看均值。']
    (EXP/'report.md').write_text('\n'.join(rows)+'\n')
    status(state='COMPLETED',report=str(EXP/'report.md'))


if __name__=='__main__':main()

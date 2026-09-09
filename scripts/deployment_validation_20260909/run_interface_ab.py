"""Sequential equal-budget interface-only training and full-Flow evaluation.

Stops on errors; no hidden retries, no recovery data, no baseline anchoring.
"""
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/ubuntu/ur3_ft300_ws')
OUT=ROOT/'artifacts/deployment_validation_20260909_interface_ab'
PY='/home/ubuntu/miniconda3/envs/pi0-env/bin/python'
SOURCE=ROOT/'outputs/train/pap_corrected_gate_calibration_20260907_104300/checkpoints/015000/pretrained_model'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def main():
    assert not (OUT/'status.json').exists(), 'Refuse to overwrite or silently restart'
    original=ROOT/'outputs/train/pap_corrected_expert_action_joint_20260907_000738/checkpoints/030000/pretrained_model/train_config.json'
    template=json.loads(original.read_text());configs={}
    for arm,prior in [('A_legacy','none'),('B_continuous','log_probability')]:
        cfg=copy.deepcopy(template)
        cfg['output_dir']=str(ROOT/f'outputs/train/pap_interface_ab_20260909_{arm}')
        assert not Path(cfg['output_dir']).exists()
        cfg['job_name']=f'pap_interface_{arm}'
        cfg['policy']['pretrained_path']=str(SOURCE)
        cfg['policy']['action_conditioning_route_prior']=prior
        cfg['steps']=10000;cfg['save_freq']=5000
        cfg['eval_freq']=20000;cfg['wandb']['enable']=False
        cfg.pop('checkpoint_path',None)
        configs[arm]=cfg
        (OUT/f'{arm}.json').write_text(json.dumps(cfg,indent=2))
    a=copy.deepcopy(configs['A_legacy']);b=copy.deepcopy(configs['B_continuous'])
    for d in (a,b):
        d.pop('output_dir');d.pop('job_name');d['policy'].pop('action_conditioning_route_prior')
    assert a==b,'A/B differs beyond one interface option'
    source_files={str(p):digest(p) for p in SOURCE.iterdir() if p.is_file() and (p.suffix in ('.json','.safetensors'))}
    code_files={str(p):digest(p) for p in (OUT/'code/lerobot').rglob('*.py')}
    manifest=dict(source=str(SOURCE),source_hashes=source_files,code_hashes=code_files,
        steps_per_arm=10000,seed=1000,starting_point='same trained S4 weights, not initialization from scratch',
        independent_variable='ActionTokenConditioner attention logits: none vs log(route_probability)',
        training='Original stage2 expert+full Action Expert joint objective under real future route labels; Gate and VLM frozen',
        fixed=['50 existing episodes/14418 frames','saved verified processors','learning rate 2.5e-5 both optimizer groups',
               'seed/sampling/augmentations/losses/1000 warmup/30000 decay schedule','no balance or baseline anchor loss'],
        scope='Controlled adaptation diagnostic, not a claim that 10k is an optimal final training budget',
        acceptance='Compare native predicted-route full-Flow error by phase and boundary stability; do not use training loss alone as closed-loop success')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    state=dict(status='starting',completed=[])
    env=dict(os.environ,PYTHONPATH=str(OUT/'code'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
        PAP_MOE_VERIFY_CORRECTED_CONTRACT='1',LEROBOT_REBUILD_PROCESSORS='0',
        LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS='1')
    def write():
        state['updated']=datetime.now().isoformat()
        temp=OUT/'status.pending';temp.write_text(json.dumps(state,indent=2));temp.replace(OUT/'status.json')
    def run(name,command,hours):
        state.update(status='running',task=name,command=list(map(str,command)));write()
        with (OUT/f'{name}.log').open('x') as log:
            child=subprocess.Popen(list(map(str,command)),env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['pid']=child.pid;write();start=time.monotonic()
            try:
                while child.poll() is None:
                    state['elapsed_s']=time.monotonic()-start;write()
                    if state['elapsed_s']>hours*3600:raise TimeoutError(name)
                    time.sleep(30)
                if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}; inspect log before any retry')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state['completed'].append(name);write()
    try:
        for arm in configs:
            run(arm+'_train',[PY,ROOT/'scripts/deployment_validation_20260909/train_interface_checked.py','--config_path',OUT/f'{arm}.json'],hours=5)
            checkpoint=Path(configs[arm]['output_dir'])/'checkpoints/010000/pretrained_model'
            assert (checkpoint/'model.safetensors').is_file()
            conf=json.loads((checkpoint/'config.json').read_text())
            assert conf['action_conditioning_route_prior']==configs[arm]['policy']['action_conditioning_route_prior']
            assert all(digest(Path(p))==value for p,value in source_files.items()),'Source checkpoint changed'
            run(arm+'_fullflow',[PY,OUT/'eval_native.py','--checkpoint',checkpoint,'--output',OUT/(arm+'_fullflow')],hours=1)
        state['status']='completed'
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write()


if __name__=='__main__':main()

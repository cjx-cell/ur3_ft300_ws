"""Fixed-budget late E1 merge experiment; no silent retries or data changes."""
import copy
import json
import os
from pathlib import Path

import run_e1_norm_ab as queue


def main():
    root = queue.ROOT
    old = root/'artifacts/pap_e1_norm_ab_20260906'
    exp = root/'artifacts/pap_e1_late_sum_20260906'
    exp.mkdir(exist_ok=False)
    queue.EXP = exp
    os.environ.update(PYTHONPATH='/home/ubuntu/lerobot/src', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', PYTORCH_ALLOC_CONF='expandable_segments:True',
        LEROBOT_REBUILD_PROCESSORS='0', LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS='1',
        OPENBLAS_NUM_THREADS='1')
    configs = {}
    for seed in [42,43]:
        original = json.loads((old/f'A_s{seed}.json').read_text())
        cfg = copy.deepcopy(original)
        cfg.update(output_dir=str(root/f'outputs/train/pap_e1_late_sum_s{seed}_20260906'),
                   job_name=f'pap_e1_late_sum_s{seed}')
        cfg['policy']['e1_fusion_mode']='late_sum'
        assert cfg['policy']['e1_fusion_normalization']=='joint'
        assert cfg['steps']==10000 and cfg['policy']['train_expert_action_joint']
        assert not Path(cfg['output_dir']).exists()
        compare = copy.deepcopy(cfg)
        compare['output_dir']=original['output_dir']; compare['job_name']=original['job_name']
        compare['policy'].pop('e1_fusion_mode')
        assert compare==original, 'Unexpected difference from paired A control'
        configs[str(seed)]=cfg
        (exp/f's{seed}.json').write_text(json.dumps(cfg,indent=2))
    view=exp/'initial'; view.mkdir()
    c=json.loads((queue.SOURCE/'config.json').read_text())
    c.update(e1_fusion_mode='late_sum',e1_fusion_normalization='joint',
             mask_invalid_prefix_tokens=True,mask_invalid_history_cameras=True)
    (view/'config.json').write_text(json.dumps(c,indent=2))
    for path in queue.SOURCE.iterdir():
        if path.is_file() and path.name!='config.json': (view/path.name).symlink_to(path)
    old_hashes=json.loads((old/'manifest.json').read_text())['hashes']
    # Existing source/data checksums remain immutable; changed code gets a new manifest.
    immutable={p:h for p,h in old_hashes.items() if p.startswith(str(queue.SOURCE)) or p.startswith(str(queue.DATA))}
    for p,h in immutable.items(): assert queue.digest(p)==h, p
    watched=list(Path('/home/ubuntu/lerobot/src/lerobot/policies/pap_moe').glob('*.py'))
    watched+=list(Path('/home/ubuntu/lerobot/src/lerobot/policies/pi05').glob('*.py'))
    watched+=list(Path('/home/ubuntu/lerobot/src/lerobot/optim').glob('*.py'))
    watched+=[Path('/home/ubuntu/lerobot/src/lerobot/scripts/lerobot_train.py')]
    watched+=list(exp.glob('*.json'))+[view/'config.json']
    watched += [root/'scripts'/n for n in ['run_e1_late_sum.py','run_e1_norm_ab.py',
        'preflight_e1_norm.py','train_pap_checked.py','audit_e1_information_fusion.py','summarize_e1_information_fusion.py']]
    hashes={**immutable,**{str(p):queue.digest(p) for p in watched}}
    (exp/'manifest.json').write_text(json.dumps(dict(configs=configs,hashes=hashes,
        comparison='late_sum vs prior A42/A43 early_sum; parameter-identical; no normalization change',
        deployment_comparison='PENDING: learned-route alignment and matched Pi05/PAP rollout; oracle diagnostics are not deployment evidence'),indent=2))
    queue.run('preflight',[queue.PY,'-u',str(root/'scripts/preflight_e1_norm.py'),
        '--checkpoint',str(view),'--output',str(exp/'preflight'),
        '--episode',str(queue.DATA/'pick_up_the_peg_and_insert_it_into_the_hole_episode_0001_success/data.npz')],hashes)
    for seed in [42,43]:
        queue.run(f'train_s{seed}',[queue.PY,'-u',str(root/'scripts/train_pap_checked.py'),
            '--config_path='+str(exp/f's{seed}.json')],hashes)
        cp=Path(configs[str(seed)]['output_dir'])/'checkpoints/010000'
        assert json.loads((cp/'training_state/training_step.json').read_text())['step']==10000
        assert json.loads((cp/'pretrained_model/config.json').read_text())['e1_fusion_mode']=='late_sum'
        out=exp/f'eval_s{seed}'
        queue.run(f'eval_s{seed}',[queue.PY,'-u',str(root/'scripts/audit_e1_information_fusion.py'),
            '--checkpoint',str(cp/'pretrained_model'),'--dataset',str(queue.DATA),'--output',str(out)],hashes)
        assert len((out/'actions.jsonl').read_text().splitlines())==720
        queue.run(f'summarize_s{seed}',[queue.PY,str(root/'scripts/summarize_e1_information_fusion.py'),
            '--output',str(out)],hashes)
    queue.status(state='ACTION_STAGE_COMPLETED',
        next='Learned-route alignment and matched Pi05 closed-loop comparison remain; no success claim yet')


if __name__=='__main__':main()

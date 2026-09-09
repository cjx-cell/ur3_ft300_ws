#!/usr/bin/env python3
"""Bounded overnight train/eval queue with durable heartbeat and result report."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
EXP = ROOT / 'artifacts/pap_lr_verified_20260905'
PY = '/home/ubuntu/miniconda3/envs/pi0-env/bin/python'
OLD = ROOT / 'artifacts/pap_lr_ab_10k_20260905'
LOW = ROOT / 'outputs/train/pap_lr_low_verified_10k_20260905'
HIGH = ROOT / 'outputs/train/pap_lr_ab_A_10k_20260905/checkpoints/010000/pretrained_model'


def status(**values):
    values['time'] = datetime.datetime.now().isoformat()
    (EXP / 'status.json').write_text(json.dumps(values, indent=2))
    print(json.dumps(values), flush=True)


def run(name, command):
    path = EXP / f'{name}.log'
    start = time.monotonic()
    with path.open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            status(stage=name, pid=proc.pid, elapsed_s=round(time.monotonic()-start),
                   log=str(path), log_age_s=round(time.time()-path.stat().st_mtime))
            time.sleep(30)
    if proc.returncode:
        status(stage=name, state='FAILED', returncode=proc.returncode, log=str(path))
        raise RuntimeError(f'{name} failed: {proc.returncode}')
    status(stage=name, state='COMPLETED', log=str(path))


def evaluate(name, checkpoint):
    command = [PY, '-u', str(ROOT / 'scripts/audit_pap_generated_conditions.py'),
               '--checkpoint', str(checkpoint), '--output', str(EXP/name)]
    for index in ['0001', '0011', '0021', '0031', '0041']:
        command += ['--episode-npz', str(ROOT / f'pap_moe_framework/datasets/workspace_50_v10_canonical/pick_up_the_peg_and_insert_it_into_the_hole_episode_{index}_success/data.npz')]
    run(name, command)
    assert len((EXP/name/'records.jsonl').read_text().splitlines()) == 630


def main():
    EXP.mkdir(exist_ok=True)
    cfg = json.loads((OLD/'A.json').read_text())
    cfg.update(use_policy_training_preset=True, output_dir=str(LOW), job_name='pap_lr_low_verified')
    assert not LOW.exists(), 'Never overwrite a training run'
    (EXP/'low.json').write_text(json.dumps(cfg, indent=2))
    os.environ.update(PYTHONPATH='/home/ubuntu/lerobot/src', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', PYTORCH_ALLOC_CONF='expandable_segments:True',
                      LEROBOT_REBUILD_PROCESSORS='0', LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS='1')
    os.chdir('/home/ubuntu/lerobot')
    evaluate('eval_equal_lr', HIGH)
    run('train_low', [PY, '-u', str(ROOT/'scripts/train_pap_checked.py'), '--config_path='+str(EXP/'low.json')])
    assert json.loads((LOW/'checkpoints/010000/training_state/training_step.json').read_text())['step'] == 10000
    evaluate('eval_low_lr', LOW/'checkpoints/010000/pretrained_model')
    rows = ['# PAP-MoE 学习率对照夜间结果', '',
            '## Material Passport', '', '类型：固定数据离线动作生成诊断；不代表闭环成功率。', '',
            '两组均从同一 30000 步权重继续训练 10000 步。等学习率组为此前误配置但完整保存的运行；低学习率组通过实际优化器校验。',
            '等学习率组使用单参数组，低学习率使用两个参数组；此实现差异需保留在解释中。', '',
            '| 组别 | 模式 | 机械臂前10步MSE | 夹爪前10步MSE | 机械臂50步MSE |',
            '|---|---|---:|---:|---:|']
    for name in ['eval_equal_lr', 'eval_low_lr']:
        result = json.loads((EXP/name/'summary.json').read_text())
        for mode, metrics in result.items():
            rows.append(f"| {name} | {mode} | {metrics['arm_mse_10']:.8f} | {metrics['gripper_mse_10']:.8f} | {metrics['arm_mse_50']:.8f} |")
    rows += ['', '五个位置 × 六个状态 × 三个噪声种子 × 七种条件，共每组630个完整动作块。',
             '真实未来路由仅用于离线诊断，不能作为在线部署策略。allzero 是同一模型去条件，不是独立 Pi0.5 基线。']
    (EXP/'report.md').write_text('\n'.join(rows)+'\n')
    status(state='COMPLETED', report=str(EXP/'report.md'))


if __name__ == '__main__':
    main()

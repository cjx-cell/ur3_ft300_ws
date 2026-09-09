"""Wait for this experiment, then update only its rows and analysis in the report."""
import json
import argparse
import math
import time
from pathlib import Path

ROOT = Path('/home/ubuntu/ur3_ft300_ws')
RUN = ROOT / 'artifacts/pap_corrected_pipeline_20260907_000738'
REPORT = ROOT / 'docs/reports/PAP_MoE推进顺序与实验结论_简版_20260907.md'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def number(value):
    return f'{value:.6f}' if isinstance(value, (float, int)) and math.isfinite(value) else '—'


def update(state):
    rows, results = [], {}
    specs = [('pi05', 'Pi0.5本轮复测', '参考基线'),
             ('expert_action_joint', '新版输入融合＋真实50步路由', '专家＋动作联合30k（oracle）'),
             ('physicsgate_sequence', '新版输入融合＋单Gate预测50步', '阶段3：Gate监督15k'),
             ('gate_calibration', '同上', '阶段4：门控栈校准15k')]
    for key, name, stage in specs:
        data = read(RUN / f'{key}_flow.json')
        diagnostics = data.get('full_condition_action_diagnostics', {})
        full = data.get('flow_mse') if key == 'pi05' else data.get('masks', {}).get('full', {}).get('mse')
        prefix = diagnostics.get('executed_prefix_mse')
        path = RUN / f'{key}_closed/results.jsonl'
        trials = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        attempts = trials
        trials = [row for row in attempts if row.get('evaluation_valid', (row.get('status_count') or 0) > 0)]
        successes = sum(row.get('success') is True for row in trials)
        success = f'{successes}/{len(trials)}' if trials else ('不适用' if key == 'expert_action_joint' else '未完成')
        rate = f'{successes / len(trials):.1%}' if trials else '—'
        if trials and len(trials) != 15:
            success += '（未完成）'
        rows.append(f'| 新{len(rows)+1} | {name} | {stage} | {number(full)} | {number(prefix)} | {success} | {rate} | E |')
        results[key] = dict(full=full, successes=successes, trials=len(trials),
                            infrastructure_failures=len(attempts)-len(trials))
    analysis = ['## 本轮新架构结果分析（自动汇总）', '',
                f'队列状态：`{state["status"]}`。原始结果目录：`{RUN}`。', '',
                'E口径：本轮固定五位置锚点、3噪声seed，归一化flow MSE；闭环50预测／10执行、每位置3seed。旧A–D数值保留作历史记录，不与本轮E口径混作同一次配对评估。', '']
    if state['status'] != 'completed':
        analysis += [f'实验尚未完整结束：{state.get("error", "请检查队列状态")}。已有结果已列出，缺失项不是零误差或零成功率，暂不作完整模型排序。', '']
    base = results['pi05']
    for key, name, _ in specs[2:]:
        result = results[key]
        if base['trials'] == result['trials'] == 15:
            delta = result['successes'] - base['successes']
            analysis.append(f'- {name}（{key}）成功{result["successes"]}/15；本轮Pi0.5为{base["successes"]}/15，相差{delta:+d}次成功。')
            if base['full'] is not None and result['full'] is not None:
                relation = '更低' if result['full'] < base['full'] else '未更低'
                analysis.append(f'  开环flow MSE相对基线{relation}；开环与闭环须分别判断，不以较低平均误差替代任务成功。')
    s3, s4 = results['physicsgate_sequence'], results['gate_calibration']
    if s3['trials'] == s4['trials'] == 15:
        analysis.append(f'- 阶段4相对阶段3变化{s4["successes"]-s3["successes"]:+d}次成功；这是本批点估计，不能仅凭15次试验声称稳健或显著优势。')
    for key, value in results.items():
        if value['infrastructure_failures']:
            analysis.append(f'- {key}有{value["infrastructure_failures"]}次runner_failure，必须人工区分基础设施异常与策略失败，不能据原始分母直接下结论。')
    analysis += ['', '本轮PAP同时包含额外力模态、历史编码、表征监督与额外训练预算；即使超过Pi0.5，也不能单独证明MoE或融合前移的因果收益。oracle行不是可部署策略。', '']
    original = REPORT.read_text()
    lines = [line for line in original.splitlines() if not any(line.startswith(f'| 新{i} |') for i in range(1, 5))]
    anchor = next(i for i, line in enumerate(lines) if line.startswith('| 诊断1 |'))
    lines[anchor:anchor] = rows
    text = '\n'.join(lines) + '\n'
    marker = '<!-- corrected-pap-results -->'
    if marker in text:
        text = text.split(marker)[0].rstrip() + '\n'
    # Preserve the exact pre-update document for recovery.
    backup = RUN / ('report_before_auto_update_' + str(time.time_ns()) + '.md')
    backup.write_text(original)
    temp = REPORT.with_suffix('.md.pending')
    temp.write_text(text + '\n' + marker + '\n' + '\n'.join(analysis))
    temp.replace(REPORT)
    (RUN / 'report_update.json').write_text(json.dumps({'report': str(REPORT), 'status': state['status'], 'results': results}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.run
    # This independent watcher also works for a pipeline process already running.
    while True:
        try:
            state = read(RUN / 'status.json')
        except json.JSONDecodeError:
            time.sleep(30)
            continue
        if state.get('status') in ('completed', 'failed'):
            update(state)
            break
        time.sleep(30)

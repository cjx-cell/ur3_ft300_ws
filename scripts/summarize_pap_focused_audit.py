#!/usr/bin/env python3
"""Summarize completed paired diagnostics without treating anchors as independent trials."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--previous', type=Path)
    args = p.parse_args()
    if not (args.directory / 'summary.json').exists():
        raise RuntimeError('Evaluation has not completed')
    rows = [json.loads(line) for line in (args.directory / 'records.jsonl').read_text().splitlines()]
    lookup = {(r['episode'], r['frame'], r['seed'], r['mode']): r for r in rows}
    assert len(lookup) == len(rows), 'Duplicate paired rows'
    pairs = [('full', 'no_entropy'), ('full', 'E1_mean'), ('full', 'E1_mismatch'),
             ('full', 'drop_E1'), ('full', 'drop_E3'), ('full', 'drop_E4'),
             ('blind_full', 'blind_drop_E2'), ('blind_full', 'blind_no_memory'),
             ('blind_full', 'blind_no_entropy')]
    result = {}
    for reference, variant in pairs:
        selected = [r for r in rows if r['mode'] == variant and
                    (not variant.startswith('blind_') or r['valid_history'])]
        original = [lookup[(r['episode'], r['frame'], r['seed'], reference)] for r in selected]
        stats = {'n_pairs': len(selected), 'reference': reference,
                 'note': 'Blind comparisons exclude t=0 without valid history; descriptive, not independent-trial statistics.'}
        for key in ['arm_mse_10', 'arm_mse_50', 'gripper_mse_10', 'gripper_mse_50']:
            a = sum(r[key] for r in original) / len(original)
            b = sum(r[key] for r in selected) / len(selected)
            stats[key] = {'reference': a, 'variant': b, 'relative_change_percent': 100*(b/a-1) if a else None}
        for group in ['free', 'rigid', 'movable']:
            den = sum(r[f'{group}_weight'] for r in selected)
            stats[group] = {'weight': den}
            for part in ['arm', 'gripper']:
                key = f'{group}_{part}_sse'
                a = sum(r[key] for r in original) / den if den else None
                b = sum(r[key] for r in selected) / den if den else None
                stats[group][part] = {'reference': a, 'variant': b,
                    'relative_change_percent': 100*(b/a-1) if a else None}
        result[variant] = stats
    if args.previous:
        prev = [json.loads(line) for line in (args.previous / 'records.jsonl').read_text().splitlines()]
        matched = []
        for r in prev:
            if r['mode'] != 'full':
                continue
            new = lookup[(r['episode'], r['frame'], r['seed'], 'full')]
            matched.append(abs(r['arm_mse_50']-new['arm_mse_50']))
        result['control_reproduction'] = {'n': len(matched), 'max_full_arm_mse50_abs_diff': max(matched)}
    (args.directory / 'paired_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

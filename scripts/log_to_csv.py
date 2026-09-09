#!/usr/bin/env python3
"""Parse training log and save metrics CSV to output directory.

Usage:
  # SA-MOE log (has stage_acc, expert distribution)
  python scripts/log_to_csv.py /tmp/samoe_train.log --output-dir outputs/train/samoe

  # Pi0 log (loss + grad_norm + lr only)
  python scripts/log_to_csv.py /tmp/pi0_train.log --output-dir outputs/train/pi0
"""
import argparse, csv, os, re, sys
import numpy as np


def parse_number(s: str) -> float:
    s = s.strip()
    if s.upper().endswith('K'):
        return float(s[:-1]) * 1000
    return float(s)


def parse_samoe_log(path: str) -> list[dict]:
    """Parse SA-MOE log: extract step, loss, grad_norm, lr, stage_acc, expert usage."""
    ot_pat = re.compile(
        r"step:([\d\.Kk]+)\s+.*?loss:([\d\.\+e\-]+)\s+grdn:([\d\.\+e\-]+)\s+lr:([\d\.\+e\-]+)"
    )
    sa_pat = re.compile(
        r"SA-MOE v9 step:(\d+) stage_acc\(ema/snap\):([\d.]+)/(\d) "
        r"stage_loss:([\d.]+)\(w=([\d.]+)\) action:([\d.]+) expert:\[([^\]]+)\]"
    )

    rows = []
    with open(path) as f:
        for line in f:
            m = ot_pat.search(line)
            if not m:
                continue
            try:
                step = parse_number(m.group(1))
                loss = float(m.group(2))
                grdn = float(m.group(3))
                lr = float(m.group(4))
                if np.isnan(loss) or np.isinf(loss):
                    continue
            except (ValueError, IndexError):
                continue

            row = {'step': int(step), 'loss': loss, 'grad_norm': grdn, 'lr': lr}
            rows.append(row)

    # Match SA-MOE metrics by step
    sa_metrics = {}
    with open(path) as f:
        for line in f:
            m = sa_pat.search(line)
            if m:
                sa_metrics[int(m.group(1))] = {
                    'ema_acc': float(m.group(2)),
                    'snap_acc': int(m.group(3)),
                    'stage_loss': float(m.group(4)),
                    'stage_weight': float(m.group(5)),
                    'action_loss': float(m.group(6)),
                    'expert': [float(x.strip().strip("'")) for x in m.group(7).split(',')],
                }

    for row in rows:
        sa = sa_metrics.get(row['step'])
        if sa:
            row.update(sa)

    return rows


def parse_pi0_log(path: str) -> list[dict]:
    """Parse Pi0 log: extract step, loss, grad_norm, lr."""
    ot_pat = re.compile(
        r"step:([\d\.Kk]+)\s+.*?loss:([\d\.\+e\-]+)\s+grdn:([\d\.\+e\-]+)\s+lr:([\d\.\+e\-]+)"
    )

    rows = []
    with open(path) as f:
        for line in f:
            m = ot_pat.search(line)
            if not m:
                continue
            try:
                step = parse_number(m.group(1))
                loss = float(m.group(2))
                grdn = float(m.group(3))
                lr = float(m.group(4))
                if np.isnan(loss) or np.isinf(loss):
                    continue
            except (ValueError, IndexError):
                continue
            rows.append({'step': int(step), 'loss': loss, 'grad_norm': grdn, 'lr': lr})
    return rows


def save_csv(rows: list[dict], output_dir: str, is_samoe: bool):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'training_metrics.csv')

    has_sa = is_samoe and 'ema_acc' in rows[0] if rows else False

    if has_sa:
        header = ['step', 'loss', 'grad_norm', 'lr',
                  'ema_acc', 'snap_acc', 'stage_loss', 'stage_weight', 'action_loss',
                  'exp0', 'exp1', 'exp2', 'exp3', 'exp4']
    else:
        header = ['step', 'loss', 'grad_norm', 'lr']

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            vals = [row.get(k, '') for k in header[:4]]
            if has_sa:
                vals += [
                    f"{row.get('ema_acc', 0):.4f}",
                    row.get('snap_acc', 0),
                    f"{row.get('stage_loss', 0):.4f}",
                    f"{row.get('stage_weight', 0):.4f}",
                    f"{row.get('action_loss', 0):.4f}",
                ] + [f"{row.get('expert', [0]*5)[i]:.4f}" for i in range(5)]
            w.writerow(vals)

    print(f"Saved {len(rows)} rows → {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Parse training log to CSV')
    parser.add_argument('logfile', help='Path to training log file')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for CSV')
    parser.add_argument('--type', choices=['samoe', 'pi0', 'auto'], default='auto',
                        help='Log type (default: auto-detect)')
    args = parser.parse_args()

    if not os.path.exists(args.logfile):
        print(f"ERROR: Log file not found: {args.logfile}")
        sys.exit(1)

    # Auto-detect type
    log_type = args.type
    if log_type == 'auto':
        with open(args.logfile) as f:
            sample = f.read(4096)
        log_type = 'samoe' if 'SA-MOE v9' in sample else 'pi0'

    if log_type == 'samoe':
        rows = parse_samoe_log(args.logfile)
    else:
        rows = parse_pi0_log(args.logfile)

    if not rows:
        print(f"ERROR: No training metrics found in {args.logfile}")
        sys.exit(1)

    save_csv(rows, args.output_dir, is_samoe=(log_type == 'samoe'))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Live training curve display for SA-MOE and Pi0.
Shows loss on screen using TkAgg interactive backend.
Usage:
  python scripts/live_plot.py /tmp/samoe_train.log --title "SA-MOE" --output outputs/train/samoe_v2/training_curves.png
  python scripts/live_plot.py /tmp/pi0_train.log --title "Pi0" --output outputs/train/pi0_v2/training_curves.png
"""
import argparse, re, time, os, sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

LOG_PATTERN = re.compile(
    r"step:([\d\.Kk]+)\s+smpl:(\d+)\s+.*?loss:([\d\.\+e\-]+)\s+grdn:([\d\.\+e\-]+)\s+lr:([\d\.\+e\-]+)"
)
SA_PATTERN = re.compile(
    r'SA-MOE v9 step:(\d+) stage_acc\(ema/snap\):([\d.]+)/(\d) '
    r'stage_loss:([\d.]+)\(w=([\d.]+)\) action:([\d.]+) expert:\[([^\]]+)\]'
)

def parse_number(s):
    s = s.strip()
    return float(s[:-1]) * 1000 if s.upper().endswith('K') else float(s)

def parse_log(filepath, is_samoe=False):
    steps, losses, grdns, lrs = [], [], [], []
    sa_steps, sa_action, sa_stage, sa_acc, sa_expert = [], [], [], [], []
    with open(filepath) as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                try:
                    step = int(parse_number(m.group(1)))
                    loss = float(m.group(3))
                    grdn = float(m.group(4))
                    lr = float(m.group(5))
                    if not (np.isnan(loss) or np.isinf(loss)):
                        steps.append(step); losses.append(loss)
                        grdns.append(grdn); lrs.append(lr)
                except (ValueError, IndexError):
                    continue
            if is_samoe:
                m = SA_PATTERN.search(line)
                if m:
                    sa_steps.append(int(m.group(1)))
                    sa_acc.append(float(m.group(2)) * 100)
                    sa_stage.append(float(m.group(4)))
                    sa_action.append(float(m.group(6)))
                    sa_expert.append([float(x.strip().strip("'")) for x in m.group(7).split(',')])
    return steps, losses, grdns, lrs, sa_steps, sa_acc, sa_stage, sa_action, sa_expert

def smooth(y, factor=0.95):
    s = []
    for p in y:
        s.append(p if not s else s[-1] * factor + p * (1 - factor))
    return np.array(s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile')
    parser.add_argument('--title', default='Training')
    parser.add_argument('--output', required=True)
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()

    is_samoe = 'samoe' in args.logfile.lower()

    plt.ion()
    if is_samoe:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    last_n = 0
    while True:
        steps, losses, grdns, lrs, sa_steps, sa_acc, sa_stage, sa_action, sa_expert = \
            parse_log(args.logfile, is_samoe)
        n = len(steps)
        if n > 0 and n != last_n:
            last_n = n
            if is_samoe:
                ax1, ax2, ax3, ax4 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
                for ax in [ax1, ax2, ax3, ax4]: ax.clear()
                # Loss
                ax1.plot(steps, losses, 'b-', alpha=0.3, lw=0.5, label='Raw')
                if len(losses) > 10:
                    ax1.plot(steps, smooth(losses), 'crimson', lw=2, label='EMA')
                ax1.set_ylabel('Loss'); ax1.set_title('Flow Matching Loss')
                ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
                # Grad norm
                ax2.plot(steps, grdns, 'orange', alpha=0.7, lw=0.5)
                ax2.set_ylabel('Grad Norm'); ax2.set_title('Gradient Norm')
                ax2.grid(alpha=0.3)
                # SA-MOE specific
                if sa_steps:
                    ax3.plot(sa_steps, sa_action, 'r-', lw=1)
                    ax3.set_ylabel('Action Loss'); ax3.set_title('Action Loss (SA-MOE)')
                    ax3.grid(alpha=0.3); ax3.set_yscale('log')
                    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
                    for i in range(5):
                        if sa_expert:
                            vals = [e[i] for e in sa_expert if len(e) > i]
                            if len(vals) == len(sa_steps):
                                ax4.plot(sa_steps, vals, color=colors[i], lw=1)
                    ax4.set_ylabel('Prob'); ax4.set_title('Expert Distribution')
                    ax4.grid(alpha=0.3)
                fig.suptitle(f'{args.title}  |  Step {steps[-1]}  Loss {losses[-1]:.2f}')
            else:
                ax1.clear(); ax2.clear()
                ax1.plot(steps, losses, 'b-', alpha=0.3, lw=0.5, label='Raw')
                if len(losses) > 10:
                    ax1.plot(steps, smooth(losses), 'crimson', lw=2, label='EMA')
                ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('Flow Matching Loss')
                ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
                ax2.plot(steps, grdns, 'orange', alpha=0.7, lw=0.5)
                ax2.set_xlabel('Step'); ax2.set_ylabel('Grad Norm'); ax2.set_title('Gradient Norm')
                ax2.grid(alpha=0.3)
                fig.suptitle(f'{args.title}  |  Step {steps[-1]}  Loss {losses[-1]:.2f}')
            plt.draw(); plt.pause(1)
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            fig.savefig(args.output, dpi=100, bbox_inches='tight')

        with open(args.logfile) as f:
            if 'End of training' in f.read():
                print(f'\n*** {args.title} training completed! ***')
                break
        time.sleep(args.interval)

    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Final curve saved: {args.output}')
    plt.ioff(); plt.show(block=True)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Real-time training loss monitor. Reads log file and plots loss curve.
Usage: python scripts/monitor_training.py --log /tmp/samoe_train.log --title "SA-MOE" --output outputs/train/samoe_v2/training_curves.png
"""
import argparse, re, time, os, sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def parse_number(s):
    s = s.strip()
    if s.upper().endswith('K'): return float(s[:-1]) * 1000
    return float(s)

def parse_losses(log_path):
    """Parse training metrics from log file."""
    pattern = re.compile(
        r"step:([\d\.Kk]+)\s+smpl:(\d+)\s+ep:(\d+)\s+.*?loss:([\d\.\+e\-]+)\s+grdn:([\d\.\+e\-]+)\s+lr:([\d\.\+e\-]+)"
    )
    steps, losses, grad_norms, lrs = [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    step = parse_number(m.group(1))
                    loss = float(m.group(4))
                    grdn = float(m.group(5))
                    lr = float(m.group(6))
                    if np.isnan(loss) or np.isinf(loss): continue
                    steps.append(int(step))
                    losses.append(loss)
                    grad_norms.append(grdn)
                    lrs.append(lr)
                except (ValueError, IndexError):
                    continue
    return np.array(steps), np.array(losses), np.array(grad_norms), np.array(lrs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--title', default='Training Loss')
    parser.add_argument('--output', required=True)
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(args.title)

    last_n = 0
    while True:
        steps, losses, grad_norms, lrs = parse_losses(args.log)
        n = len(steps)
        if n > 0 and n != last_n:
            last_n = n
            ax1.clear(); ax2.clear()
            ax1.plot(steps, losses, 'b-', alpha=0.7, linewidth=0.5)
            ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('Flow Matching Loss')
            ax1.grid(True, alpha=0.3)

            ax2.plot(steps, grad_norms, 'r-', alpha=0.7, linewidth=0.5)
            ax2.set_xlabel('Step'); ax2.set_ylabel('Grad Norm'); ax2.set_title('Gradient Norm')
            ax2.grid(True, alpha=0.3)

            fig.suptitle(f"{args.title}  |  Step {steps[-1]}  Loss {losses[-1]:.2f}  GradNorm {grad_norms[-1]:.1f}")
            plt.draw(); plt.pause(1)

            # Save snapshot
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            fig.savefig(args.output, dpi=100, bbox_inches='tight')

        # Check if training completed
        with open(args.log) as f:
            if 'End of training' in f.read():
                print(f"\nTraining completed! Final curve saved to {args.output}")
                break

        time.sleep(args.interval)

    plt.ioff(); plt.show()

if __name__ == '__main__':
    main()

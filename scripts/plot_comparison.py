#!/usr/bin/env python3
"""Pi0 vs SA-MOE loss comparison plot for ablation experiment.
Reads training_metrics.csv from both model output dirs.
Usage:
  python scripts/plot_comparison.py \
    --pi0 outputs/train/pi0_v2/training_metrics.csv \
    --samoe outputs/train/samoe_v2/training_metrics.csv \
    --output outputs/pi0_vs_samoe_comparison.png
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_csv(path):
    """Load training_metrics.csv, return step,loss arrays. Handles both 'loss' and 'action_loss' columns."""
    steps, losses = [], []
    with open(path) as f:
        header = f.readline().strip().split(',')
        step_idx = header.index('step')
        # Try 'action_loss' first (SA-MOE), fall back to 'loss' (Pi0)
        if 'action_loss' in header:
            loss_idx = header.index('action_loss')
        else:
            loss_idx = header.index('loss')
        for line in f:
            parts = line.strip().split(',')
            step = int(float(parts[step_idx]))
            loss = float(parts[loss_idx])
            if not (np.isnan(loss) or np.isinf(loss)):
                steps.append(step)
                losses.append(loss)
    return np.array(steps), np.array(losses)

def smooth(y, factor=0.9):
    s = []
    for p in y:
        s.append(p if not s else s[-1] * factor + p * (1 - factor))
    return np.array(s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pi0', required=True, help='Pi0 training_metrics.csv')
    parser.add_argument('--samoe', required=True, help='SA-MOE training_metrics.csv')
    parser.add_argument('--output', required=True)
    parser.add_argument('--xlim', type=int, default=None, help='Max step for x-axis')
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Pi0 ---
    pi0_steps, pi0_losses = load_csv(args.pi0)
    pi0_color = '#2196F3'
    ax1.plot(pi0_steps, pi0_losses, alpha=0.2, color=pi0_color, lw=0.5, label='Pi0 (raw)')
    if len(pi0_losses) > 10:
        ax1.plot(pi0_steps, smooth(pi0_losses, 0.9), color=pi0_color, lw=2, label='Pi0 (EMA)')
    ax2.plot(pi0_steps, pi0_losses, alpha=0.2, color=pi0_color, lw=0.5)

    # --- SA-MOE ---
    sa_steps, sa_losses = load_csv(args.samoe)
    sa_color = '#FF5722'
    ax1.plot(sa_steps, sa_losses, alpha=0.2, color=sa_color, lw=0.5, label='SA-MOE (raw)')
    if len(sa_losses) > 10:
        ax1.plot(sa_steps, smooth(sa_losses, 0.9), color=sa_color, lw=2, label='SA-MOE (EMA)')
    ax2.plot(sa_steps, sa_losses, alpha=0.2, color=sa_color, lw=0.5)

    # --- Linear-scale panel ---
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Flow Matching Loss', fontsize=12)
    ax1.set_title('Pi0 vs SA-MOE — Ablation Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    if args.xlim: ax1.set_xlim(0, args.xlim)

    # Add final loss annotation
    ax1.annotate(f'Pi0 final: {pi0_losses[-1]:.2f}',
                 xy=(pi0_steps[-1], pi0_losses[-1]), xytext=(10, 10),
                 textcoords='offset points', fontsize=9, color=pi0_color,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.annotate(f'SA-MOE final: {sa_losses[-1]:.2f}',
                 xy=(sa_steps[-1], sa_losses[-1]), xytext=(10, -10),
                 textcoords='offset points', fontsize=9, color=sa_color,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # --- Log-scale panel ---
    ax2.plot(pi0_steps, smooth(pi0_losses, 0.9), color=pi0_color, lw=2, label=f'Pi0 (final={pi0_losses[-1]:.2f})')
    ax2.plot(sa_steps, smooth(sa_losses, 0.9), color=sa_color, lw=2, label=f'SA-MOE (final={sa_losses[-1]:.2f})')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Flow Matching Loss (log scale)', fontsize=12)
    ax2.set_title('Loss Curves — Log Scale', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    if args.xlim: ax2.set_xlim(0, args.xlim)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {args.output}')

if __name__ == '__main__':
    main()

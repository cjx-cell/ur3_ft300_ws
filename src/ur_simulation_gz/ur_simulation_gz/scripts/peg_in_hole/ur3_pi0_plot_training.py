#!/usr/bin/env python3
"""Real-time training curve plotter for Pi0 LoRA training.

Usage (in separate terminal while training runs):
  conda activate pi0-env
  cd ~/ur3_ft300_ws
  python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_pi0_plot_training.py ai-models/pi0/train.log

Or post-hoc:
  python src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole/ur3_pi0_plot_training.py ai-models/pi0/train.log --once

Output image: outputs/<log_basename>.png
"""

import argparse, re, time, os, sys
import numpy as np
import matplotlib

# Check if GUI display is available, fallback to Agg if not
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# Regex to match step, loss, gradient norm, learning rate, update time, and data time
# Example: "step:50 smpl:100 ep:0 epch:0.04 loss:265.559 grdn:20.627 lr:9.9e-07 updt_s:0.612 data_s:0.041"
PATTERN = re.compile(
    r"(\d+)/\d+.*?loss:([\d\.\+e\-]+).*?grdn:([\d\.\+e\-]+).*?lr:([\d\.\+e\-]+).*?updt_s:([\d\.\+e\-]+).*?data_s:([\d\.\+e\-]+)"
)


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)


def parse_log(filepath):
    """Parse all Pi0 log lines from file and return as numpy arrays."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None

    matches = PATTERN.findall(content)
    if not matches:
        return None

    N = len(matches)
    steps = np.zeros(N)
    losses = np.zeros(N)
    grdns = np.zeros(N)
    lrs = np.zeros(N)
    updt_s = np.zeros(N)
    data_s = np.zeros(N)

    for i, m in enumerate(matches):
        steps[i] = int(m[0])
        losses[i] = float(m[1])
        grdns[i] = float(m[2])
        lrs[i] = float(m[3])
        updt_s[i] = float(m[4])
        data_s[i] = float(m[5])

    # Sort to ensure strictly increasing step order
    combined = sorted(zip(steps, losses, grdns, lrs, updt_s, data_s))
    steps, losses, grdns, lrs, updt_s, data_s = zip(*combined)

    return (
        np.array(steps),
        np.array(losses),
        np.array(grdns),
        np.array(lrs),
        np.array(updt_s),
        np.array(data_s),
    )


def plot_curves(steps, losses, grdns, lrs, updt_s, data_s, save_path=None):
    """Generate 4-panel training curves plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Pi0 LoRA Training Curves — {len(steps)} checkpoints, up to step {int(steps[-1])}",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Loss Curve
    ax = axes[0][0]
    ax.plot(steps, losses, alpha=0.3, color="dodgerblue", label="Raw Loss")
    if len(losses) > 10:
        smoothed = smooth_curve(losses, factor=0.9)
        ax.plot(steps, smoothed, color="crimson", linewidth=2, label="Smoothed Loss (EMA)")
    else:
        ax.plot(steps, losses, color="crimson", linewidth=2, label="Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.set_title(f"Flow Matching Loss (final={losses[-1]:.2f})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Panel 2: Learning Rate
    ax = axes[0][1]
    ax.plot(steps, lrs, color="purple", linewidth=2)
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_title(f"Learning Rate (final={lrs[-1]:.2e})")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Gradient Norm
    ax = axes[1][0]
    ax.plot(steps, grdns, color="seagreen", alpha=0.7)
    if len(grdns) > 10:
        smoothed_grdn = smooth_curve(grdns, factor=0.9)
        ax.plot(steps, smoothed_grdn, color="forestgreen", linewidth=2, label="Smoothed Grdn")
    ax.set_ylabel("Gradient Norm")
    ax.set_xlabel("Step")
    ax.set_title(f"Gradient Norm (final={grdns[-1]:.2f})")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Throughput Times
    ax = axes[1][1]
    ax.plot(steps, updt_s, color="orange", alpha=0.6, label="Update Time (s)")
    ax.plot(steps, data_s, color="cyan", alpha=0.6, label="Data Loading Time (s)")
    if len(updt_s) > 10:
        smoothed_updt = smooth_curve(updt_s, factor=0.9)
        ax.plot(steps, smoothed_updt, color="darkorange", linewidth=2, label="Update Time (EMA)")
    ax.set_ylabel("Seconds")
    ax.set_xlabel("Step")
    ax.set_title("Step Execution & Data Loading Throughput")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save first to tmp to avoid locks, then rename
        tmp_path = save_path + ".tmp"
        plt.savefig(tmp_path, format="png", dpi=150)
        plt.close(fig)
        if os.path.exists(tmp_path):
            os.rename(tmp_path, save_path)
            print(f"  Saved: {save_path}")
    return fig


def _redraw_axes(axes, steps, losses, grdns, lrs, updt_s, data_s, n_pts, final_step):
    """Redraw data on existing axes without closing figure."""
    # Panel 1: Loss
    ax = axes[0]
    ax.plot(steps, losses, alpha=0.3, color="dodgerblue", label="Raw Loss")
    if len(losses) > 10:
        smoothed = smooth_curve(losses, factor=0.9)
        ax.plot(steps, smoothed, color="crimson", linewidth=2, label="Smoothed Loss (EMA)")
    else:
        ax.plot(steps, losses, color="crimson", linewidth=2, label="Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.set_title(f"Flow Matching Loss (final={losses[-1]:.2f})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel 2: Learning Rate
    ax = axes[1]
    ax.plot(steps, lrs, color="purple", linewidth=2)
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_title(f"Learning Rate (final={lrs[-1]:.2e})")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Gradient Norm
    ax = axes[2]
    ax.plot(steps, grdns, color="seagreen", alpha=0.7)
    if len(grdns) > 10:
        smoothed_grdn = smooth_curve(grdns, factor=0.9)
        ax.plot(steps, smoothed_grdn, color="forestgreen", linewidth=2, label="Smoothed Grdn")
    ax.set_ylabel("Gradient Norm")
    ax.set_xlabel("Step")
    ax.set_title(f"Gradient Norm (final={grdns[-1]:.2f})")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: execution times
    ax = axes[3]
    ax.plot(steps, updt_s, color="orange", alpha=0.6, label="Update Time (s)")
    ax.plot(steps, data_s, color="cyan", alpha=0.6, label="Data Loading Time (s)")
    if len(updt_s) > 10:
        smoothed_updt = smooth_curve(updt_s, factor=0.9)
        ax.plot(steps, smoothed_updt, color="darkorange", linewidth=2, label="Update (EMA)")
    ax.set_ylabel("Seconds")
    ax.set_xlabel("Step")
    ax.set_title("Step Execution & Data Loading Throughput")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)


def _save_paths(args):
    """Return (live_path, final_path) for output images."""
    basename = os.path.basename(args.logfile).replace(".log", ".png").replace(".txt", ".png")
    
    # Project root
    project_root = "/home/ubuntu/ur3_ft300_ws"
    live_path = args.output or os.path.join(project_root, "outputs", basename)

    final_path = None
    if args.model_dir:
        model_dir = args.model_dir
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(project_root, model_dir)
        os.makedirs(model_dir, exist_ok=True)
        final_path = os.path.join(model_dir, "training_curves.png")

    return live_path, final_path


def _save(fig, *paths):
    for p in paths:
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            # Save first to tmp to avoid locks, then rename
            tmp_path = p + ".tmp"
            fig.savefig(tmp_path, format="png", dpi=150)
            if os.path.exists(tmp_path):
                os.rename(tmp_path, p)
                print(f"  Saved: {p}")


def main():
    parser = argparse.ArgumentParser(description="Pi0 training curve plotter")
    parser.add_argument("logfile", help="Path to training log file")
    parser.add_argument("--once", action="store_true", help="Plot once and exit (no live update)")
    parser.add_argument("--output", "-o", default=None, help="Output image path (default: auto)")
    parser.add_argument("--model-dir", "-m", default=None,
                        help="Training output_dir (also saves training_curves.png there)")
    parser.add_argument("--interval", type=float, default=10, help="Update interval in seconds")
    args = parser.parse_args()

    live_path, final_path = _save_paths(args)

    if args.once:
        data = parse_log(args.logfile)
        if data is None:
            print(f"No valid Pi0 log lines found in {args.logfile}")
            sys.exit(1)
        fig = plot_curves(*data, save_path=live_path)
        if final_path:
            _save(fig, final_path)
        print("Done")
        return

    # Live mode
    print(f"Monitoring {args.logfile} (refresh every {args.interval}s)...")
    print(f"Live save: {live_path}")
    if final_path:
        print(f"Final save: {final_path}")
    print("Live window: close it or Ctrl+C to stop")
    last_count = 0

    plt.ion()  # interactive mode
    fig = None
    try:
        while True:
            if os.path.exists(args.logfile):
                data = parse_log(args.logfile)
                if data is not None and len(data[0]) > last_count:
                    last_count = len(data[0])
                    steps, losses, grdns, lrs, updt_s, data_s = data
                    if fig is None:
                        # Create fig first time
                        fig = plt.figure(figsize=(14, 10))
                        fig.suptitle(
                            f"Pi0 LoRA Training Curves — {len(steps)} checkpoints, up to step {int(steps[-1])}",
                            fontsize=14,
                            fontweight="bold",
                        )
                        # Create 4 subplots
                        axes = fig.subplots(2, 2)
                        axes_flat = axes.flatten()
                        _redraw_axes(axes_flat, steps, losses, grdns, lrs, updt_s, data_s, len(steps), int(steps[-1]))
                    else:
                        for ax in fig.axes:
                            ax.clear()
                        _redraw_axes(fig.axes, steps, losses, grdns, lrs, updt_s, data_s, len(steps), int(steps[-1]))
                        fig.suptitle(
                            f"Pi0 LoRA Training Curves — {len(steps)} checkpoints, up to step {int(steps[-1])}",
                            fontsize=14,
                            fontweight="bold",
                        )
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    _save(fig, live_path)
                    print(f"  Updated: {len(steps)} points")
                    if fig is not None:
                        fig.canvas.flush_events()
            plt.pause(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
        if fig is not None and final_path:
            _save(fig, final_path)
            print(f"Final curve saved to: {final_path}")


if __name__ == "__main__":
    main()

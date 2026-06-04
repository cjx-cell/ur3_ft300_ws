#!/usr/bin/env python3
"""
可视化采集的 UR3 pick-and-place 轨迹数据。

生成:
  - 关节轨迹图 (state/action × 7 joints, 按阶段标注)
  - 关键帧相机图像 (腕部 + 全局)
  - 归一化前后对比

用法:
  /usr/bin/python3.10 visualize_dataset.py
  /usr/bin/python3.10 visualize_dataset.py --episode 0
  /usr/bin/python3.10 visualize_dataset.py --all  # 对比所有 episode
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── 数据路径 ──
RAW_DIR = Path.home() / "ur3_ft300_ws" / "ai-models" / "ur3_pick_place_raw"
OUT_DIR = RAW_DIR / "trajectory_viz"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
]
JOINT_LABELS = ["sh_pan", "sh_lift", "elbow", "wrist1", "wrist2", "wrist3", "grip"]

# 轨迹阶段定义（基于录制脚本的硬编码位姿）
PHASES = {
    "HOME":    [ 0.0,   -1.57,   0.0,   -1.57,   0.0,    0.0],
    "ABOVE":   [-1.834, -1.883, -1.128, -1.646,  1.572,  0.297],
    "GRASP":   [-1.803, -1.942, -1.579, -1.136,  1.570, -0.202],
    "LIFT":    [-1.803, -1.856, -1.293, -1.508,  1.570, -0.202],
    "ABOVE_B": [-0.761, -1.910, -1.230, -1.545,  1.523,  0.839],
    "PLACE":   [-0.765, -1.925, -1.358, -1.402,  1.523,  0.835],
}


def load_episode(ep_dir):
    """加载一个 episode 的 npz 数据"""
    npz = ep_dir / "data.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    return {
        "state": d["state"].astype(np.float32),
        "action": d["action"].astype(np.float32),
        "camera0": d["camera0"].astype(np.float32),
        "camera1": d["camera1"].astype(np.float32),
        "task": str(d.get("task", "")),
        "name": ep_dir.name,
    }


def detect_phases(states):
    """通过最近邻匹配自动标注每帧对应的阶段"""
    n = len(states)
    labels = np.full(n, "", dtype=object)
    phase_frames = {}

    for name, target in PHASES.items():
        target_6d = np.array(target, dtype=np.float32)
        # 找到最接近该目标位姿的帧（仅用前 6 关节）
        dists = np.linalg.norm(states[:, :6] - target_6d, axis=1)
        best = int(np.argmin(dists))
        labels[best] = name
        phase_frames[name] = best

    return labels, phase_frames


def plot_joint_trajectory(ax, states, actions, labels, phase_frames, title):
    """绘制 7 关节 state 和 action 轨迹"""
    n = len(states)
    x = np.arange(n)

    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    for j in range(7):
        ax.plot(x, states[:, j], color=colors[j], alpha=0.5, linewidth=0.6, label=f"{JOINT_LABELS[j]} state")
        ax.plot(x, actions[:, j], color=colors[j], linestyle="--", linewidth=0.8, label=f"{JOINT_LABELS[j]} action")

    # 阶段竖线
    phase_colors = {
        "HOME": "gray", "ABOVE": "blue", "GRASP": "red",
        "LIFT": "green", "ABOVE_B": "orange", "PLACE": "purple",
    }
    for name, fidx in sorted(phase_frames.items(), key=lambda x: x[1]):
        ax.axvline(x=fidx, color=phase_colors.get(name, "black"), linestyle=":", alpha=0.8, linewidth=1)
        ax.text(fidx, ax.get_ylim()[1] * 0.95, name, fontsize=6,
                color=phase_colors.get(name, "black"), rotation=90, va="top", ha="right")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Joint position (rad)")
    ax.legend(fontsize=5, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_single_joint_detail(ax, states, actions, j_idx, labels, phase_frames):
    """单关节详细图"""
    n = len(states)
    x = np.arange(n)
    ax.plot(x, states[:, j_idx], "b-", alpha=0.6, linewidth=0.8, label="state")
    ax.plot(x, actions[:, j_idx], "r--", alpha=0.6, linewidth=0.8, label="action")

    for name, fidx in sorted(phase_frames.items(), key=lambda x: x[1]):
        ax.axvline(x=fidx, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

    ax.set_title(JOINT_NAMES[j_idx], fontsize=9)
    ax.set_xlabel("Frame")
    ax.set_ylabel("rad")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_keyframe_images(axes, cam0, cam1, states, phase_frames, gripper_state):
    """绘制关键帧相机图像"""
    phase_order = ["HOME", "ABOVE", "GRASP", "LIFT", "ABOVE_B", "PLACE"]

    for col, name in enumerate(phase_order):
        if name not in phase_frames:
            continue
        fidx = phase_frames[name]
        grip_val = states[fidx, 6]

        axes[0, col].imshow(np.clip(cam0[fidx], 0, 1))
        axes[0, col].set_title(f"{name}\nframe {fidx}, grip={grip_val:.2f}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(np.clip(cam1[fidx], 0, 1))
        axes[1, col].set_title(f"Global @ {name}", fontsize=8)
        axes[1, col].axis("off")


def plot_normalization_check(ax1, ax2, states, actions, norms):
    """对比归一化前后的数据分布"""
    # 归一化前的 histogram
    for j in range(7):
        ax1.hist(states[:, j], bins=40, alpha=0.4, label=JOINT_LABELS[j])
    ax1.set_title("State raw distribution (before norm)", fontsize=10)
    ax1.set_xlabel("Joint position (rad)")
    ax1.legend(fontsize=6)

    # 归一化后
    state_mean = norms["state_mean"]
    state_std = norms["state_std"]
    states_norm = (states - state_mean) / (state_std + 1e-8)
    for j in range(7):
        ax2.hist(states_norm[:, j], bins=40, alpha=0.4, label=JOINT_LABELS[j])
    ax2.set_title("State after MEAN_STD normalization", fontsize=10)
    ax2.set_xlabel("Normalized value")
    ax2.legend(fontsize=6)


def make_full_figure(ep_data):
    """生成完整的多面板图"""
    states = ep_data["state"]
    actions = ep_data["action"]
    cam0 = ep_data["camera0"]
    cam1 = ep_data["camera1"]
    ep_name = ep_data["name"]

    labels, phase_frames = detect_phases(states)

    # ── 归一化参数（从 checkpoint 提取的真实值） ──
    norms = {
        "state_mean": np.array([-0.98096776, -1.8261101, -1.0006496, -1.4776422, 1.1784788, 0.30134782, 0.28109705], dtype=np.float32),
        "state_std":  np.array([0.6413004, 0.1287974, 0.51718795, 0.13342571, 0.58274025, 0.39418843, 0.36008605], dtype=np.float32),
        "action_mean": np.array([-0.9810236, -1.8261198, -1.0006421, -1.4797164, 1.1785238, 0.30149975, 0.2857143], dtype=np.float32),
        "action_std":  np.array([0.7147818, 0.14523803, 0.5736406, 0.15516822, 0.659141, 0.4443977, 0.36421567], dtype=np.float32),
    }

    fig = plt.figure(figsize=(22, 26))
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1.2, 0.8, 1.5, 1.0, 1.2],
                  hspace=0.35, wspace=0.25)

    # ── Row 0: 关键帧图像 ──
    gs_img = GridSpec(2, 6, figure=fig, left=0.05, right=0.95, top=0.98, bottom=0.90)
    phase_order = ["HOME", "ABOVE", "GRASP", "LIFT", "ABOVE_B", "PLACE"]
    for col, name in enumerate(phase_order):
        if name not in phase_frames:
            continue
        fidx = phase_frames[name]
        grip_val = states[fidx, 6]

        ax_w = fig.add_subplot(gs_img[0, col])
        ax_w.imshow(np.clip(cam0[fidx], 0, 1))
        ax_w.set_title(f"{name} (f{fidx}, grip={grip_val:.2f})", fontsize=8)
        ax_w.axis("off")

        ax_g = fig.add_subplot(gs_img[1, col])
        ax_g.imshow(np.clip(cam1[fidx], 0, 1))
        ax_g.set_title(f"Global", fontsize=7)
        ax_g.axis("off")

    # ── Row 1: 关节轨迹总览 (state vs action) ──
    ax_traj = fig.add_subplot(gs[1, :])
    plot_joint_trajectory(ax_traj, states, actions, labels, phase_frames,
                          f"{ep_name}: Joint Trajectory ({len(states)} frames)")

    # ── Row 2: 前 3 关节详细 ──
    for j in range(3):
        ax = fig.add_subplot(gs[2, j // 2 * 2 + j % 2] if j < 2 else None)
        # Use simpler layout
        pass

    ax_j0 = fig.add_subplot(gs[2, 0])
    plot_single_joint_detail(ax_j0, states, actions, 0, labels, phase_frames)
    ax_j1 = fig.add_subplot(gs[2, 1])
    plot_single_joint_detail(ax_j1, states, actions, 1, labels, phase_frames)

    # ── Row 3: 中间 3 关节详细 ──
    ax_j2 = fig.add_subplot(gs[3, 0])
    plot_single_joint_detail(ax_j2, states, actions, 2, labels, phase_frames)
    ax_j3 = fig.add_subplot(gs[3, 1])
    plot_single_joint_detail(ax_j3, states, actions, 3, labels, phase_frames)

    # ── Row 4: 后 3 关节 + 归一化分布 ──
    ax_j4 = fig.add_subplot(gs[4, 0])
    plot_single_joint_detail(ax_j4, states, actions, 4, labels, phase_frames)
    # 最后两个关节放在一起
    ax_grip = fig.add_subplot(gs[4, 1])
    plot_single_joint_detail(ax_grip, states, actions, 5, labels, phase_frames)

    fig.suptitle(f"UR3 Pick-and-Place Trajectory — {ep_name}  ({len(states)} frames)\n"
                 f"Task: {ep_data['task']}",
                 fontsize=13, fontweight="bold")
    return fig


def make_joint_panel(ep_data):
    """紧凑型：7 关节 × 7 面板（state/action 叠加 + 阶段竖线）"""
    states = ep_data["state"]
    actions = ep_data["action"]
    ep_name = ep_data["name"]
    labels, phase_frames = detect_phases(states)

    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    axes = axes.flatten()

    phase_colors = {"HOME": "gray", "ABOVE": "blue", "GRASP": "red",
                    "LIFT": "green", "ABOVE_B": "orange", "PLACE": "purple"}

    for j in range(7):
        ax = axes[j]
        x = np.arange(len(states))
        ax.plot(x, states[:, j], "b-", alpha=0.7, linewidth=1.0, label="state (recorded)")
        ax.plot(x, actions[:, j], "r--", alpha=0.5, linewidth=0.8, label="action (interpolated)")

        for name, fidx in sorted(phase_frames.items(), key=lambda x: x[1]):
            ax.axvline(x=fidx, color=phase_colors.get(name, "gray"), linestyle=":", alpha=0.7)

        # 标注 phase 名称（只在第一个子图）
        if j == 0:
            for name, fidx in phase_frames.items():
                ax.text(fidx, ax.get_ylim()[1], name, fontsize=7, rotation=90,
                        va="top", ha="right", color=phase_colors.get(name, "black"))

        ax.set_title(f"J{j}: {JOINT_NAMES[j]}  [{states[:, j].min():.2f}, {states[:, j].max():.2f}]", fontsize=10)
        ax.set_ylabel("rad")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # 最后一个 subplot: 统计信息
    ax = axes[7]
    ax.axis("off")
    stats_text = f"""=== {ep_name} Stats ===

Total frames: {len(states)}
State range: [{states.min():.3f}, {states.max():.3f}]

State range per joint:
"""
    for j in range(7):
        stats_text += f"  J{j} {JOINT_NAMES[j]}: [{states[:,j].min():.3f}, {states[:,j].max():.3f}]\n"

    stats_text += f"""
Action range per joint:
"""
    for j in range(7):
        stats_text += f"  J{j} {JOINT_NAMES[j]}: [{actions[:,j].min():.3f}, {actions[:,j].max():.3f}]\n"

    stats_text += f"""
Phase frames:
"""
    for name, fidx in sorted(phase_frames.items(), key=lambda x: x[1]):
        stats_text += f"  {name:10s}: frame {fidx:3d}, state={np.array2string(states[fidx,:6], precision=2)}\n"

    stats_text += f"""
Task: {ep_data['task']}

IMPORTANT: State mean/std and Action mean/std are DIFFERENT!
  State:  policy_preprocessor_step_5_normalizer_processor.safetensors
  Action: policy_postprocessor_step_0_unnormalizer_processor.safetensors
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            fontfamily="monospace", va="top")

    fig.suptitle(f"UR3 Pick-and-Place — {ep_name} — Joint Trajectories", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def make_normalization_figure(ep_data):
    """归一化分析图"""
    states = ep_data["state"]
    actions = ep_data["action"]

    state_mean = np.array([-0.98096776, -1.8261101, -1.0006496, -1.4776422, 1.1784788, 0.30134782, 0.28109705], dtype=np.float32)
    state_std  = np.array([0.6413004, 0.1287974, 0.51718795, 0.13342571, 0.58274025, 0.39418843, 0.36008605], dtype=np.float32)
    action_mean = np.array([-0.9810236, -1.8261198, -1.0006421, -1.4797164, 1.1785238, 0.30149975, 0.2857143], dtype=np.float32)
    action_std  = np.array([0.7147818, 0.14523803, 0.5736406, 0.15516822, 0.659141, 0.4443977, 0.36421567], dtype=np.float32)

    states_norm = (states - state_mean) / (state_std + 1e-8)
    actions_norm = (actions - action_mean) / (action_std + 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    # (0,0): State raw distribution
    for j in range(7):
        axes[0, 0].hist(states[:, j], bins=40, alpha=0.5, color=colors[j], label=JOINT_LABELS[j])
    axes[0, 0].set_title("State — raw joint values (rad)", fontsize=10)
    axes[0, 0].legend(fontsize=7)

    # (0,1): State normalized
    for j in range(7):
        axes[0, 1].hist(states_norm[:, j], bins=40, alpha=0.5, color=colors[j], label=JOINT_LABELS[j])
    axes[0, 1].set_title("State — after (x-mean)/std normalization", fontsize=10)
    axes[0, 1].legend(fontsize=7)

    # (0,2): State normalized trajectories
    x = np.arange(len(states))
    for j in range(7):
        axes[0, 2].plot(x, states_norm[:, j], color=colors[j], alpha=0.7, linewidth=0.8, label=JOINT_LABELS[j])
    axes[0, 2].set_title("State normalized — trajectory", fontsize=10)
    axes[0, 2].set_xlabel("Frame")
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(True, alpha=0.3)

    # (1,0): Action raw distribution
    for j in range(7):
        axes[1, 0].hist(actions[:, j], bins=40, alpha=0.5, color=colors[j], label=JOINT_LABELS[j])
    axes[1, 0].set_title("Action — raw joint values (rad)", fontsize=10)
    axes[1, 0].legend(fontsize=7)

    # (1,1): Action normalized
    for j in range(7):
        axes[1, 1].hist(actions_norm[:, j], bins=40, alpha=0.5, color=colors[j], label=JOINT_LABELS[j])
    axes[1, 1].set_title("Action — after (x-mean)/std normalization", fontsize=10)
    axes[1, 1].legend(fontsize=7)

    # (1,2): Action normalized trajectories
    for j in range(7):
        axes[1, 2].plot(x, actions_norm[:, j], color=colors[j], alpha=0.7, linewidth=0.8, label=JOINT_LABELS[j])
    axes[1, 2].set_title("Action normalized — trajectory", fontsize=10)
    axes[1, 2].set_xlabel("Frame")
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("Normalization Analysis — State vs Action (using checkpoint stats)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def make_episode_comparison(all_eps):
    """跨 episode 对比：同一关节的轨迹叠加"""
    n_eps = len(all_eps)
    if n_eps <= 1:
        print("  只有 1 个 episode，跳过对比")
        return None

    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    axes = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, n_eps))

    for j in range(7):
        ax = axes[j]
        for ei, ep in enumerate(all_eps):
            states = ep["state"]
            x = np.arange(len(states))
            ax.plot(x, states[:, j], color=colors[ei], alpha=0.8, linewidth=0.8,
                    label=f"{ep['name']} ({len(states)}f)")
        ax.set_title(f"J{j}: {JOINT_NAMES[j]} — cross-episode", fontsize=10)
        ax.set_ylabel("rad")
        ax.set_xlabel("Frame")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # 统计对比
    ax = axes[7]
    ax.axis("off")
    text = "Episode Comparison:\n\n"
    for ei, ep in enumerate(all_eps):
        s = ep["state"]
        a = ep["action"]
        text += f"{ep['name']}: frames={len(s)}, "
        text += f"state range=[{s.min():.3f},{s.max():.3f}], "
        text += f"action range=[{a.min():.3f},{a.max():.3f}]\n"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
            fontfamily="monospace", va="top")

    fig.suptitle("Cross-Episode Joint Trajectory Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="可视化 UR3 pick-and-place 采集数据")
    parser.add_argument("--episode", type=int, default=None, help="只看指定 episode")
    parser.add_argument("--all", action="store_true", help="对比所有 episode")
    parser.add_argument("--no-norm", action="store_true", help="不生成归一化图")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载
    episodes = []
    ep_dirs = sorted(RAW_DIR.glob("episode_*"))
    if args.episode is not None:
        target = RAW_DIR / f"episode_{args.episode:04d}"
        ep_dirs = [target] if target.exists() else []

    for ep_dir in ep_dirs:
        data = load_episode(ep_dir)
        if data is not None:
            episodes.append(data)
            print(f"加载: {data['name']} — {len(data['state'])} 帧, "
                  f"cam0 {data['camera0'].shape}, cam1 {data['camera1'].shape}")

    if not episodes:
        print(f"ERROR: 在 {RAW_DIR} 找不到数据")
        sys.exit(1)

    print(f"共 {len(episodes)} 个 episode\n")

    # ── 生成图片 ──
    for ep in episodes:
        print(f"生成 {ep['name']} ...")

        # 1. 关节面板图
        fig1 = make_joint_panel(ep)
        path1 = OUT_DIR / f"{ep['name']}_joints.png"
        fig1.savefig(path1, dpi=100, bbox_inches="tight")
        plt.close(fig1)
        print(f"  → {path1}")

        # 2. 归一化分析
        if not args.no_norm:
            fig2 = make_normalization_figure(ep)
            path2 = OUT_DIR / f"{ep['name']}_normalization.png"
            fig2.savefig(path2, dpi=100, bbox_inches="tight")
            plt.close(fig2)
            print(f"  → {path2}")

    # 3. 跨 episode 对比
    if args.all or (len(episodes) > 1 and args.episode is None):
        fig3 = make_episode_comparison(episodes)
        if fig3 is not None:
            path3 = OUT_DIR / "cross_episode_comparison.png"
            fig3.savefig(path3, dpi=100, bbox_inches="tight")
            plt.close(fig3)
            print(f"  → {path3}")

    print(f"\n完成！图片保存在: {OUT_DIR}")
    print(f"  ls {OUT_DIR}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
将录制的 npz 数据生成可视化视频：
  - 左: 腕部相机 (camera0)
  - 右: 全局相机 (camera1)
  - 中上: 6关节 arm state 实时曲线
  - 中下: 夹爪 gripper state 实时曲线（独立子图，固定Y轴范围）

用法:
  /usr/bin/python3.10 make_video.py
  /usr/bin/python3.10 make_video.py --episode 0 --fps 20
"""

import argparse, os, sys
import numpy as np
import cv2
from pathlib import Path

RAW_DIR = Path.home() / "ur3_ft300_ws" / "ai-models" / "datasets" / "ur3_pick_place_raw"
OUT_DIR = RAW_DIR / "trajectory_viz"
JOINT_NAMES = ["sh_pan", "sh_lift", "elbow", "wrist1", "wrist2", "wrist3"]
GRIP_NAME = "gripper"


def make_video(npz_path, output_path, fps=20):
    d = np.load(npz_path)
    states = d["state"]
    cam0 = d["camera0"]
    cam1 = d["camera1"]

    n_frames = len(states)
    arm_states = states[:, :6]
    grip_states = states[:, 6]
    print(f"Frames: {n_frames}, State shape: {states.shape}")
    print(f"Arm range: [{arm_states.min():.2f}, {arm_states.max():.2f}], "
          f"Grip range: [{grip_states.min():.3f}, {grip_states.max():.3f}]")

    # 画布尺寸
    cam_h, cam_w = 224, 224
    arm_plot_h = 200   # 6关节曲线
    grip_plot_h = 50   # 夹爪曲线（独立子图）
    gap = 10
    total_w = cam_w * 2 + 20
    total_h = cam_h + arm_plot_h + grip_plot_h + gap * 3 + 40

    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, total_h))

    plot_len = min(n_frames, 200)

    for i in range(n_frames):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas.fill(30)

        # ── 左: 腕部相机 ──
        wrist = (np.clip(cam0[i], 0, 1) * 255).astype(np.uint8)
        wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
        canvas[10:10+cam_h, 10:10+cam_w] = wrist_bgr
        cv2.putText(canvas, "Wrist Camera", (14, cam_h+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── 右: 全局相机 ──
        global_cam = (np.clip(cam1[i], 0, 1) * 255).astype(np.uint8)
        global_bgr = cv2.cvtColor(global_cam, cv2.COLOR_RGB2BGR)
        canvas[10:10+cam_h, cam_w+20:cam_w+20+cam_w] = global_bgr
        cv2.putText(canvas, "Global Camera", (cam_w+24, cam_h+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── 中上: 6DOF arm 关节曲线 ──
        arm_y0 = cam_h + 40
        start = max(0, i - plot_len)
        x_vals = np.arange(start, i + 1)
        x_vals = np.arange(start, i + 1)
        for j in range(6):
            y = arm_states[start:i+1, j]
            j_min, j_max = arm_states[:, j].min(), arm_states[:, j].max()
            if j_max - j_min < 0.01:
                j_min -= 0.1; j_max += 0.1
            y_mapped = arm_y0 + arm_plot_h - 10 - \
                       (y - j_min) / (j_max - j_min) * (arm_plot_h - 20)
            pts = np.column_stack([
                (x_vals - start) / plot_len * total_w,
                y_mapped
            ]).astype(np.int32)
            cv2.polylines(canvas, [pts], False, colors[j], 1)

        # arm 图例
        for j in range(6):
            cv2.putText(canvas, JOINT_NAMES[j],
                        (10 + j * 78, arm_y0 + arm_plot_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[j], 1)

        # ── 中下: 夹爪曲线（固定 Y 轴范围 0→0.55） ──
        grip_y0 = arm_y0 + arm_plot_h + gap
        grip_min, grip_max = -0.05, 0.55  # 固定范围，看清开合
        grip_y = grip_states[start:i+1]
        grip_mapped = grip_y0 + grip_plot_h - 5 - \
                      (grip_y - grip_min) / (grip_max - grip_min) * (grip_plot_h - 10)
        grip_pts = np.column_stack([
            (np.arange(len(grip_y))) / plot_len * total_w,
            grip_mapped
        ]).astype(np.int32)
        cv2.polylines(canvas, [grip_pts], False, (0, 255, 255), 2)

        # 夹爪数值 + 状态文字
        grip_val = grip_states[i]
        if grip_val < 0.05:
            state_text = "OPEN"
        elif grip_val > 0.30:
            state_text = "CLOSE"
        else:
            state_text = "..."
        cv2.putText(canvas, f"Gripper: {grip_val:.3f} [{state_text}]",
                    (10, grip_y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 帧计数
        cv2.putText(canvas, f"Frame {i}/{n_frames}",
                    (total_w - 160, total_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        video.write(canvas)

        if i % 200 == 0:
            print(f"  rendering... {i}/{n_frames}")

    video.release()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.episode is not None:
        episodes = [(ep_dir.name, RAW_DIR / ep_dir / "data.npz")
                    for ep_dir in sorted(RAW_DIR.glob("*_episode_*"))
                    if (RAW_DIR / ep_dir / "data.npz").exists()
                    and str(args.episode) in ep_dir.name]
    else:
        episodes = []
        for ep_dir in sorted(RAW_DIR.glob("*_episode_*")):
            npz = ep_dir / "data.npz"
            if npz.exists():
                episodes.append((ep_dir.name, npz))

    if not episodes:
        print(f"ERROR: no episodes found in {RAW_DIR}")
        sys.exit(1)

    print(f"找到 {len(episodes)} 个 episode, 开始生成视频...\n")

    for ep_name, npz_path in episodes:
        out = args.output if args.output and len(episodes) == 1 \
              else str(OUT_DIR / f"{ep_name}.mp4")
        print(f"{ep_name}: {npz_path}")
        make_video(npz_path, out, args.fps)
        print(f"  → {out}\n")

    print(f"完成！视频保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()

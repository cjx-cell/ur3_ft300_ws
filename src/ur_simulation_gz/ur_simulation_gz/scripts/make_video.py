#!/usr/bin/env python3
"""
将录制的 npz 数据生成可视化视频：
  - 左: 腕部相机 (camera0)
  - 右: 全局相机 (camera1)
  - 底部: 7关节 state/action 实时曲线

用法:
  python3 make_video.py
  python3 make_video.py --episode 0 --fps 20
  python3 make_video.py --output /tmp/ur3_ep0.mp4
"""

import argparse, os, sys
import numpy as np
import cv2
from pathlib import Path

RAW_DIR = Path.home() / "ur3_ft300_ws" / "ai-models" / "ur3_pick_place_raw"
OUT_DIR = RAW_DIR / "trajectory_viz"
JOINT_NAMES = ["sh_pan", "sh_lift", "elbow", "wrist1", "wrist2", "wrist3", "grip"]


def make_video(npz_path, output_path, fps=20):
    d = np.load(npz_path)
    states = d["state"]
    actions = d["action"]
    cam0 = d["camera0"]    # (N, 224, 224, 3) float32 [0,1]
    cam1 = d["camera1"]

    n_frames = len(states)
    print(f"Frames: {n_frames}, State shape: {states.shape}")

    # 视频画布尺寸
    cam_h, cam_w = 224, 224
    plot_h = 280  # 关节曲线高度
    total_w = cam_w * 2 + 20   # 两个相机 + 间隔
    total_h = cam_h + plot_h + 40

    # 颜色
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255), (200, 200, 200),
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, total_h))

    plot_len = min(n_frames, 200)  # 曲线显示的窗口大小

    for i in range(n_frames):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas.fill(30)  # 深灰背景

        # 左: 腕部相机
        wrist = (np.clip(cam0[i], 0, 1) * 255).astype(np.uint8)
        wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
        canvas[10:10+cam_h, 10:10+cam_w] = wrist_bgr
        cv2.putText(canvas, "Wrist Camera", (14, cam_h+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 右: 全局相机
        global_cam = (np.clip(cam1[i], 0, 1) * 255).astype(np.uint8)
        global_bgr = cv2.cvtColor(global_cam, cv2.COLOR_RGB2BGR)
        canvas[10:10+cam_h, cam_w+20:cam_w+20+cam_w] = global_bgr
        cv2.putText(canvas, "Global Camera", (cam_w+24, cam_h+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 底部: 关节曲线
        plot_y0 = cam_h + 40
        start = max(0, i - plot_len)
        x = np.arange(start, i + 1)
        # 将关节值缩放到 plot 区域
        for j in range(7):
            y = states[start:i+1, j]
            # 映射: [min, max] → [plot_y0+plot_h-10, plot_y0+10]
            j_min, j_max = states[:, j].min(), states[:, j].max()
            if j_max - j_min < 0.01:
                j_min -= 0.1
                j_max += 0.1
            y_mapped = plot_y0 + plot_h - 10 - (y - j_min) / (j_max - j_min) * (plot_h - 20)
            # 转换为整数坐标
            pts = np.column_stack([(x - start) / plot_len * total_w, y_mapped]).astype(np.int32)
            cv2.polylines(canvas, [pts], False, colors[j], 1)

        # 图例
        for j in range(7):
            cv2.putText(canvas, f"{JOINT_NAMES[j]}", (10 + j * 85, plot_y0 + plot_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[j], 1)

        # 帧计数
        cv2.putText(canvas, f"Frame {i}/{n_frames}", (total_w - 160, total_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        video.write(canvas)

        if i % 200 == 0:
            print(f"  rendering... {i}/{n_frames}")

    video.release()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=None, help="只生成指定 episode (默认: 全部)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.episode is not None:
        episodes = [(args.episode, RAW_DIR / f"episode_{args.episode:04d}" / "data.npz")]
    else:
        episodes = []
        for ep_dir in sorted(RAW_DIR.glob("episode_*")):
            npz = ep_dir / "data.npz"
            if npz.exists():
                ep_idx = int(ep_dir.name.split("_")[-1])
                episodes.append((ep_idx, npz))

    if not episodes:
        print(f"ERROR: no episodes found in {RAW_DIR}")
        sys.exit(1)

    print(f"找到 {len(episodes)} 个 episode, 开始生成视频...\n")

    for ep_idx, npz_path in episodes:
        out = args.output if args.output and len(episodes) == 1 else str(OUT_DIR / f"episode_{ep_idx:04d}.mp4")
        print(f"Episode {ep_idx}: {npz_path}")
        make_video(npz_path, out, args.fps)
        print(f"  → {out}\n")

    print(f"完成！视频保存在: {OUT_DIR}")
    print(f"  ls {OUT_DIR}/")


if __name__ == "__main__":
    main()

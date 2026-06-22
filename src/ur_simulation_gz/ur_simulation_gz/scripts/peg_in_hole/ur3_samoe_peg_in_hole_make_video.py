#!/usr/bin/env python3
"""
Peg-in-hole 数据可视化视频生成器。

布局:
  ┌──────────────┬──────────────┐
  │  Wrist Cam   │ Global Cam   │  224x224  cameras
  ├──────────────┴──────────────┤
  │  6-DOF Arm Joints (sh_pan, sh_lift, elbow, wrist1/2/3)  │  180px
  ├─────────────────────────────┤
  │  |F| force magnitude        │  50px
  ├─────────────────────────────┤
  │  Gripper (open/close)       │  50px
  ├─────────────────────────────┤
  │  Stage bar + Frame counter  │  30px
  └─────────────────────────────┘

用法:
  /usr/bin/python3 make_video_peg_in_hole.py
  /usr/bin/python3 make_video_peg_in_hole.py --episode 0 --fps 20
"""

import argparse, os, sys
import numpy as np
import cv2
from pathlib import Path

RAW_DIR = Path.home() / "ur3_ft300_ws" / "ai-models" / "ur3_peg_in_hole_raw"
OUT_DIR = RAW_DIR / "trajectory_viz"
JOINT_NAMES = ["sh_pan", "sh_lift", "elbow", "wrist1", "wrist2", "wrist3"]
STAGE_NAMES = {0:"approach", 1:"align", 2:"grasp", 3:"insert", 4:"confirm"}
STAGE_COLORS = {
    0: (100, 200, 100),   # approach: green
    1: (200, 200, 100),   # align: yellow-green
    2: (255, 180, 50),    # grasp: orange
    3: (100, 150, 255),   # insert: blue
    4: (255, 100, 120),   # confirm: red-pink
}


def make_video(npz_path, output_path, fps=20):
    d = np.load(npz_path, allow_pickle=True)
    states = d["state"]
    cam0 = d["camera0"]
    cam1 = d["camera1"]
    force = d["force"]       # (N, 6): Fx,Fy,Fz,Tx,Ty,Tz
    has_stage = "stage" in d.files
    stages = d["stage"] if has_stage else np.zeros(len(states), dtype=int)

    n_frames = len(states)
    arm_states = states[:, :6]
    grip_states = states[:, 6]

    # Force magnitude and torque magnitude
    f_mag = np.linalg.norm(force[:, :3], axis=1)  # ||F||
    t_mag = np.linalg.norm(force[:, 3:], axis=1)  # ||T||
    fz = force[:, 2]  # axial force (most informative for peg-in-hole)

    print(f"Frames: {n_frames}, State: {states.shape}, Force: {force.shape}")
    print(f"Arm range: [{arm_states.min():.2f}, {arm_states.max():.2f}]")
    print(f"Grip range: [{grip_states.min():.3f}, {grip_states.max():.3f}]")
    print(f"|F| range: [{f_mag.min():.1f}, {f_mag.max():.1f}] N")
    print(f"|T| range: [{t_mag.min():.2f}, {t_mag.max():.2f}] Nm")

    if has_stage:
        unique, counts = np.unique(stages, return_counts=True)
        stage_info = ", ".join(f"{STAGE_NAMES.get(s,'?')}:{c}" for s, c in zip(unique, counts))
        print(f"Stages: {stage_info}")

    # ── Layout geometry ──
    cam_h, cam_w = 224, 224
    arm_plot_h = 180
    force_plot_h = 50
    grip_plot_h = 50
    stage_bar_h = 30
    gap = 8
    pad = 10

    total_w = cam_w * 2 + 20
    total_h = pad + cam_h + gap + arm_plot_h + gap + force_plot_h + gap + grip_plot_h + gap + stage_bar_h + pad

    # Arm joint colors
    joint_colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, total_h))

    plot_len = min(n_frames, 200)

    for i in range(n_frames):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas.fill(30)

        # ── Row 1: Cameras ──
        cam_y0 = pad

        # Left: wrist camera
        wrist = (np.clip(cam0[i], 0, 1) * 255).astype(np.uint8)
        wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
        canvas[cam_y0:cam_y0+cam_h, pad:pad+cam_w] = wrist_bgr
        cv2.putText(canvas, "Wrist Camera", (pad+4, cam_y0+cam_h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Right: global camera
        global_cam = (np.clip(cam1[i], 0, 1) * 255).astype(np.uint8)
        global_bgr = cv2.cvtColor(global_cam, cv2.COLOR_RGB2BGR)
        gx = cam_w + 20
        canvas[cam_y0:cam_y0+cam_h, gx:gx+cam_w] = global_bgr
        cv2.putText(canvas, "Global Camera", (gx+4, cam_y0+cam_h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── Row 2: 6-DOF Arm Joints ──
        arm_y0 = cam_y0 + cam_h + gap
        start = max(0, i - plot_len)
        x_vals = np.arange(start, i + 1)

        for j in range(6):
            y = arm_states[start:i+1, j]
            j_min, j_max = arm_states[:, j].min(), arm_states[:, j].max()
            if j_max - j_min < 0.01:
                j_min -= 0.1
                j_max += 0.1
            y_mapped = arm_y0 + arm_plot_h - 10 - \
                       (y - j_min) / (j_max - j_min) * (arm_plot_h - 20)
            pts = np.column_stack([
                (x_vals - start) / plot_len * total_w,
                y_mapped
            ]).astype(np.int32)
            cv2.polylines(canvas, [pts], False, joint_colors[j], 1)

        # Arm legend
        for j in range(6):
            cv2.putText(canvas, JOINT_NAMES[j],
                        (pad + j * 80, arm_y0 + arm_plot_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, joint_colors[j], 1)

        # ── Row 3: Force |F| + |T| ──
        force_y0 = arm_y0 + arm_plot_h + gap
        # |F| trace (blue-green)
        f_vals = f_mag[start:i+1]
        f_min, f_max = 0, max(f_mag.max(), 10)
        f_mapped = force_y0 + force_plot_h - 5 - \
                   (f_vals - f_min) / max(f_max - f_min, 0.1) * (force_plot_h - 10)
        f_pts = np.column_stack([
            (np.arange(len(f_vals))) / plot_len * total_w,
            f_mapped
        ]).astype(np.int32)
        cv2.polylines(canvas, [f_pts], False, (0, 255, 200), 2)

        # |T| trace (orange)
        t_vals = t_mag[start:i+1]
        t_min, t_max = 0, max(t_mag.max(), 2)
        t_mapped = force_y0 + force_plot_h - 5 - \
                   (t_vals - t_min) / max(t_max - t_min, 0.1) * (force_plot_h - 10)
        t_pts = np.column_stack([
            (np.arange(len(t_vals))) / plot_len * total_w,
            t_mapped
        ]).astype(np.int32)
        cv2.polylines(canvas, [t_pts], False, (0, 180, 255), 2)

        # Force legend + current values
        f_cur = f_mag[i]
        t_cur = t_mag[i]
        cv2.putText(canvas, f"|F|={f_cur:6.1f}N", (pad, force_y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)
        cv2.putText(canvas, f"|T|={t_cur:5.2f}Nm", (pad + 120, force_y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)
        # Fz (axial force)
        fz_cur = fz[i]
        cv2.putText(canvas, f"Fz={fz_cur:+6.1f}N",
                    (pad + 250, force_y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 255, 100) if abs(fz_cur) < 30 else (100, 100, 255), 1)

        # ── Row 4: Gripper ──
        grip_y0 = force_y0 + force_plot_h + gap
        grip_min, grip_max = -0.05, 0.60
        grip_y = grip_states[start:i+1]
        grip_mapped = grip_y0 + grip_plot_h - 5 - \
                      (grip_y - grip_min) / (grip_max - grip_min) * (grip_plot_h - 10)
        grip_pts = np.column_stack([
            (np.arange(len(grip_y))) / plot_len * total_w,
            grip_mapped
        ]).astype(np.int32)
        cv2.polylines(canvas, [grip_pts], False, (0, 255, 255), 2)

        grip_val = grip_states[i]
        if grip_val < 0.05:
            g_state = "OPEN"
        elif grip_val < 0.20:
            g_state = "partial"
        elif grip_val < 0.40:
            g_state = "GRIP"
        else:
            g_state = "CLOSE"
        cv2.putText(canvas, f"Gripper: {grip_val:.3f} [{g_state}]",
                    (pad, grip_y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # ── Row 5: Stage bar + frame counter ──
        stage_y0 = grip_y0 + grip_plot_h + gap
        stage_id = stages[i]
        color = STAGE_COLORS.get(stage_id, (128, 128, 128))
        bar_h = stage_bar_h - 8
        cv2.rectangle(canvas, (pad, stage_y0), (total_w - pad, stage_y0 + bar_h), color, -1)

        stage_name = STAGE_NAMES.get(stage_id, "?")
        cv2.putText(canvas, f"Stage {stage_id}: {stage_name}",
                    (pad + 4, stage_y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Frame counter (right-aligned)
        frame_text = f"Frame {i}/{n_frames-1}"
        (tw, th), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(canvas, frame_text,
                    (total_w - tw - pad, stage_y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        video.write(canvas)

        if i % 200 == 0:
            print(f"  rendering... {i}/{n_frames}")

    video.release()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Peg-in-hole trajectory video generator")
    parser.add_argument("--episode", type=str, default=None,
                        help="Episode number or substring to match (e.g. '0', '0000', 'success')")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path (single episode only)")
    parser.add_argument("--raw_dir", type=str, default=None,
                        help=f"Override raw data directory (default: {RAW_DIR})")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR
    out_dir = raw_dir / "trajectory_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find episodes
    if args.episode is not None:
        episodes = [(ep_dir.name, raw_dir / ep_dir / "data.npz")
                    for ep_dir in sorted(raw_dir.glob("*_episode_*"))
                    if (raw_dir / ep_dir / "data.npz").exists()
                    and args.episode in ep_dir.name]
        if not episodes and args.episode.isdigit():
            # Try matching by number
            ep_num = int(args.episode)
            episodes = [(ep_dir.name, raw_dir / ep_dir / "data.npz")
                        for ep_dir in sorted(raw_dir.glob(f"*_episode_{ep_num:04d}_*"))
                        if (raw_dir / ep_dir / "data.npz").exists()]
    else:
        episodes = []
        for ep_dir in sorted(raw_dir.glob("*_episode_*")):
            npz = ep_dir / "data.npz"
            if npz.exists():
                episodes.append((ep_dir.name, npz))

    if not episodes:
        print(f"ERROR: no episodes found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(episodes)} episode(s), generating videos...\n")

    for ep_name, npz_path in episodes:
        out = args.output if args.output and len(episodes) == 1 \
              else str(out_dir / f"{ep_name}.mp4")
        print(f"{ep_name}: {npz_path}")
        make_video(npz_path, out, args.fps)
        print(f"  → {out}\n")

    print(f"Done! Videos saved to: {out_dir}")


if __name__ == "__main__":
    main()

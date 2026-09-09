#!/usr/bin/env python3
"""
PAP-MOE Peg-in-hole trajectory video generator.
Visualizes multi-modal visual observations, forces, joint positions, and stages side-by-side.
"""

import argparse, os, sys
import numpy as np
import cv2
from pathlib import Path

RAW_DIR = Path.home() / "ur3_ft300_ws" / "pap_moe_framework" / "datasets" / "raw"
OUT_DIR = RAW_DIR / "trajectory_viz"
JOINT_NAMES = ["sh_pan", "sh_lift", "elbow", "wrist1", "wrist2", "wrist3"]
STAGE_NAMES = {
    0: "E1: Normal-Vision Free Motion",
    1: "E2: Visual-Degradation Backup",
    2: "E3: Rigid Constraint Contact",
    3: "E4: Movable/Compliant Contact",
}
STAGE_COLORS = {
    0: (100, 200, 100),   # green (E1)
    1: (200, 180, 60),    # amber (E2)
    2: (255, 140, 40),    # orange (E3)
    3: (80, 140, 255),    # blue (E4)
}
STAGE_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in STAGE_COLORS.items()}


def make_video(npz_path, output_path, fps=20):
    d = np.load(npz_path, allow_pickle=True)
    states = d["state"]
    cam0 = d["camera0"]
    cam1 = d["camera1"]
    force = d["force"]       # (N, 6): Fx,Fy,Fz,Tx,Ty,Tz
    
    n_frames = len(states)
    # v8 keeps the language prompt global for baseline/PAP fairness and stores
    # per-frame review labels separately.
    task_data = d.get(
        "semantic_subtask",
        d.get("task", "pick up the peg and insert it into the hole"),
    )
    if isinstance(task_data, (np.ndarray, list)) and len(task_data) == n_frames:
        tasks = [str(t) for t in task_data]
    else:
        tasks = [str(task_data)] * n_frames

    # V6 episodes already contain the final factorized soft-routing prior.
    # Recomputing the legacy v2 labels here makes the review video disagree
    # with the exact targets that will be used for training.
    stage_v2_path = str(npz_path).replace(".npz", "_stage_v2.npy")
    stages = None
    if "stage" in d.files:
        stored_stages = d["stage"]
        if stored_stages.ndim == 2 and stored_stages.shape[1] == 4:
            stages = stored_stages.astype(np.float32)
            print("  ✓ Using stored factorized soft-routing targets")

    # Retain the legacy relabel path only for older datasets without v6
    # four-expert targets.
    if stages is None and os.path.exists(stage_v2_path):
        loaded_stages = np.load(stage_v2_path)
        if len(loaded_stages) == len(states):
            stages = loaded_stages
        else:
            print(f"  ⚠ Warning: {stage_v2_path} has stale size {len(loaded_stages)} (expected {len(states)}). Re-computing...")

    if stages is None:
        # Try to compute clean v2 labels on-the-fly using tasks & joint states context
        try:
            from pap_moe_relabel import compute_labels, calibrate_force_with_payload
            tasks_arr = np.array(tasks)
            force_calibrated = calibrate_force_with_payload(force, tasks_arr)
            joint_states = states[:, :6]
            img_means = np.mean(cam0, axis=(1, 2, 3))
            stages = compute_labels(
                force_calibrated, d.get("tool0_z", np.zeros(len(states))), 
                img_means, states[:, 6], joint_states=joint_states, 
                tasks=tasks_arr
            )
            print(f"  ✓ Computed clean stage v2 on-the-fly for visualization")
        except Exception as e:
            print(f"  ⚠ Failed to compute stage v2 on-the-fly ({e}). Falling back to old labels...")

    if stages is None and "stage" in d.files:
        stages_raw = d["stage"]
        if stages_raw.ndim == 2 and stages_raw.shape[1] == 4:
            stages = stages_raw.astype(np.float32)
        elif stages_raw.ndim == 1 or stages_raw.shape[1] == 1:
            s = np.squeeze(stages_raw).astype(int)
            stages = np.zeros((len(states), 4), dtype=np.float32)
            for idx, val in enumerate(s):
                m = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}.get(val, 0)
                stages[idx, m] = 1.0
        else:
            stages = np.zeros((len(states), 4), dtype=np.float32)
            stages[:, 0] = 1.0
    elif stages is None:
        stages = np.zeros((len(states), 4), dtype=np.float32)
        stages[:, 0] = 1.0  # default to free motion

    n_frames = len(states)
    arm_states = states[:, :6]
    grip_states = states[:, 6]

    # Force magnitude and torque magnitude
    f_mag = np.linalg.norm(force[:, :3], axis=1)  # ||F||
    t_mag = np.linalg.norm(force[:, 3:], axis=1)  # ||T||
    fz = force[:, 2]  # axial force

    print(f"Frames: {n_frames}, State: {states.shape}, Force: {force.shape}")
    print(f"Arm range: [{arm_states.min():.2f}, {arm_states.max():.2f}]")
    print(f"Grip range: [{grip_states.min():.3f}, {grip_states.max():.3f}]")
    print(f"|F| range: [{f_mag.min():.1f}, {f_mag.max():.1f}] N")
    print(f"|T| range: [{t_mag.min():.2f}, {t_mag.max():.2f}] Nm")

    if stages is not None:
        avg_weights = stages.mean(axis=0)
        print(f"Avg Gate Weights: E1_free_load:{avg_weights[0]:.2f}, E2_optical_blind:{avg_weights[1]:.2f}, E3_rigid_micro:{avg_weights[2]:.2f}, E4_flexible_brittle:{avg_weights[3]:.2f}")

    # Layout geometry
    cam_h, cam_w = 224, 224
    arm_plot_h = 160
    force_plot_h = 45
    ft_plot_h = 35       # Fz + |T| sub-row
    grip_plot_h = 45
    stage_bar_h = 60     # increased for stacked bar and subtask labels
    gap = 6
    pad = 10

    total_w = cam_w * 2 + 20
    total_h = pad + cam_h + gap + arm_plot_h + gap + force_plot_h + gap + ft_plot_h + gap + grip_plot_h + gap + stage_bar_h + pad

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

        # Row 1: Cameras
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

        # Row 2: 6-DOF Arm Joints
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

        # Row 3: Force components Fx, Fy, Fz
        force_y0 = arm_y0 + arm_plot_h + gap
        f_min = min(force[:, :3].min(), -10.0)
        f_max = max(force[:, :3].max(), 10.0)
        f_range = max(f_max - f_min, 1.0)
        
        # Zero line for forces
        f_zy = int(force_y0 + force_plot_h - 3 - (0.0 - f_min) / f_range * (force_plot_h - 8))
        cv2.line(canvas, (0, f_zy), (total_w, f_zy), (60, 60, 60), 1)
        
        f_colors = [(50, 50, 255), (50, 255, 50), (255, 150, 50)]  # BGR: Red (Fx), Green (Fy), Blue (Fz)
        for j in range(3):
            f_vals = force[start:i+1, j]
            f_mapped = force_y0 + force_plot_h - 3 - (f_vals - f_min) / f_range * (force_plot_h - 8)
            f_pts = np.column_stack([
                (np.arange(len(f_vals))) / plot_len * total_w,
                f_mapped
            ]).astype(np.int32)
            cv2.polylines(canvas, [f_pts], False, f_colors[j], 1)
            
        fx, fy, fz_val = force[i, 0], force[i, 1], force[i, 2]
        cv2.putText(canvas, f"Fx={fx:+5.1f}N", (pad, force_y0 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, f_colors[0], 1)
        cv2.putText(canvas, f"Fy={fy:+5.1f}N", (pad + 100, force_y0 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, f_colors[1], 1)
        cv2.putText(canvas, f"Fz={fz_val:+5.1f}N", (pad + 200, force_y0 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, f_colors[2], 1)

        # Row 4: Torque components Tx, Ty, Tz
        ft_y0 = force_y0 + force_plot_h + gap
        t_min = min(force[:, 3:].min(), -1.0)
        t_max = max(force[:, 3:].max(), 1.0)
        t_range = max(t_max - t_min, 0.2)
        
        # Zero line for torques
        t_zy = int(ft_y0 + ft_plot_h - 3 - (0.0 - t_min) / t_range * (ft_plot_h - 6))
        cv2.line(canvas, (0, t_zy), (total_w, t_zy), (60, 60, 60), 1)
        
        t_colors = [(80, 80, 255), (80, 255, 80), (255, 180, 80)]  # BGR: Red (Tx), Green (Ty), Blue (Tz)
        for j in range(3):
            t_vals = force[start:i+1, 3 + j]
            t_mapped = ft_y0 + ft_plot_h - 3 - (t_vals - t_min) / t_range * (ft_plot_h - 6)
            t_pts = np.column_stack([
                (np.arange(len(t_vals))) / plot_len * total_w,
                t_mapped
            ]).astype(np.int32)
            cv2.polylines(canvas, [t_pts], False, t_colors[j], 1)
            
        tx, ty, tz = force[i, 3], force[i, 4], force[i, 5]
        cv2.putText(canvas, f"Tx={tx:+5.2f}Nm", (pad, ft_y0 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, t_colors[0], 1)
        cv2.putText(canvas, f"Ty={ty:+5.2f}Nm", (pad + 100, ft_y0 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, t_colors[1], 1)
        cv2.putText(canvas, f"Tz={tz:+5.2f}Nm", (pad + 200, ft_y0 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, t_colors[2], 1)

        # Row 5: Gripper
        grip_y0 = ft_y0 + ft_plot_h + gap
        grip_min, grip_max = -0.05, 0.85
        grip_y = grip_states[start:i+1]
        grip_mapped = grip_y0 + grip_plot_h - 3 - \
                      (grip_y - grip_min) / (grip_max - grip_min) * (grip_plot_h - 8)
        grip_pts = np.column_stack([
            (np.arange(len(grip_y))) / plot_len * total_w,
            grip_mapped
        ]).astype(np.int32)
        cv2.polylines(canvas, [grip_pts], False, (0, 255, 255), 2)
        # Universal full-close command endpoint.  The measured joint normally
        # stalls around 0.627 rad on object contact; that is state, not a
        # task-specific command target.
        grip_target_y = int(grip_y0 + grip_plot_h - 3 - (0.8 - grip_min) / (grip_max - grip_min) * (grip_plot_h - 8))
        cv2.line(canvas, (0, grip_target_y), (total_w, grip_target_y), (255, 255, 0), 1)

        grip_val = grip_states[i]
        bin_val = 0 if grip_val <= 0.12 else 1
        g_state = "CLOSED (1)" if bin_val == 1 else "OPEN (0)"
        cv2.putText(canvas, f"Gripper: {grip_val:.3f} [{g_state}]",
                    (pad, grip_y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

        # Row 6: Stacked 4-expert weight bar + frame counter
        stage_y0 = grip_y0 + grip_plot_h + gap
        stage_vec = np.clip(stages[i], 0, 1)
        bar_h = 16
        bar_w = total_w - pad * 2
        # Draw stacked segments proportional to weight
        x_cursor = pad
        seg_order = [0, 1, 2, 3]  # E1, E2, E3, E4
        for eid in seg_order:
            seg_w = int(bar_w * stage_vec[eid])
            if seg_w > 0:
                color_bgr = STAGE_COLORS_BGR[eid]
                cv2.rectangle(canvas, (x_cursor, stage_y0),
                              (x_cursor + seg_w, stage_y0 + bar_h),
                              color_bgr, -1)
                if seg_w > 40:
                    cv2.putText(canvas, f"E{eid+1}",
                                (x_cursor + 3, stage_y0 + bar_h - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
                x_cursor += seg_w

        # Weight labels below bar
        desc_text = f"E1:{stage_vec[0]:.2f}  E2:{stage_vec[1]:.2f}  E3:{stage_vec[2]:.2f}  E4:{stage_vec[3]:.2f}"
        dominant_id = int(np.argmax(stage_vec))
        stage_name = STAGE_NAMES.get(dominant_id, "?")
        cv2.putText(canvas, desc_text,
                    (pad + 4, stage_y0 + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
        cv2.putText(canvas, stage_name,
                    (pad + 4, stage_y0 + bar_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, STAGE_COLORS_BGR.get(dominant_id, (160, 160, 160)), 1)
        # Dynamic subtask text
        cv2.putText(canvas, f"Subtask: {tasks[i]}",
                    (pad + 4, stage_y0 + bar_h + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Frame counter (right-aligned)
        frame_text = f"Frame {i}/{n_frames-1}"
        (tw, th), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, frame_text,
                    (total_w - tw - pad, stage_y0 + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        video.write(canvas)

        if i % 200 == 0:
            print(f"  rendering... {i}/{n_frames}")

    video.release()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PAP-MOE Peg-in-hole trajectory video generator")
    parser.add_argument("--episode", type=str, default=None,
                        help="Episode number or substring to match (e.g. '0', '0000', 'success', 'failed')")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path (single episode only)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for one or more episode videos")
    parser.add_argument("--raw_dir", type=str, default=None,
                        help=f"Override raw data directory (default: {RAW_DIR})")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR
    if args.output and args.output_dir:
        parser.error("--output and --output-dir are mutually exclusive")
    out_dir = Path(args.output_dir) if args.output_dir else raw_dir / "trajectory_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old videos in the output directory
    # print(f"Cleaning up old videos in {out_dir}...")
    # for f in out_dir.glob("*.mp4"):
    #     try:
    #         f.unlink()
    #     except Exception as e:
    #         print(f"  Failed to delete {f.name}: {e}")

    # Find episodes
    if args.episode is not None:
        episodes = [(ep_dir.name, ep_dir / "data.npz")
                    for ep_dir in sorted(raw_dir.glob("*_episode_*"))
                    if (ep_dir / "data.npz").exists()
                    and args.episode in ep_dir.name]
        if not episodes and args.episode.isdigit():
            ep_num = int(args.episode)
            episodes = [(ep_dir.name, ep_dir / "data.npz")
                        for ep_dir in sorted(raw_dir.glob(f"*_episode_{ep_num:04d}_*"))
                        if (ep_dir / "data.npz").exists()]
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

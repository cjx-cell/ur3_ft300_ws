#!/usr/bin/env python3
"""
Build a trimmed LeRobot dataset from raw episodes 301+.
Steps:
  1. Scan raw/ for episodes >= 301, skip _failed
  2. For each success episode, trim after gripper release (remove return-to-home)
  3. Binarize gripper (same logic as original converter)
  4. Save as LeRobot v2 dataset
"""

import argparse, os, sys, time, shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

RAW_DIR = Path("/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/raw")
OUTPUT_DIR = Path("/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_trimmed_301plus")
REPO_ID = "pap_moe/ur3_peg_trimmed_301plus"
FPS = 10
MIN_EP = 301


def find_trim_frame(actions, tasks):
    """Find the frame where gripper releases after insertion (task goes to 'go back to home').
    Returns the cut index: keep frames [0, cut_idx).
    """
    n = len(actions)

    # Strategy 1: Find last frame before 'go back to home' task starts
    if tasks is not None:
        for i in range(n):
            if str(tasks[i]) == "go back to home":
                # Keep 5 extra frames after insertion completes for margin
                return min(i + 5, n)

    # Strategy 2: Find gripper open transition (continuous value drops back toward 0)
    gripper = actions[:, 6]
    # Find where gripper was closed (> 0.12) then opens (< 0.12)
    was_closed = False
    for i in range(n):
        if gripper[i] > 0.12:
            was_closed = True
        elif was_closed and gripper[i] <= 0.12:
            return min(i + 10, n)

    return n  # No trim needed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_ep", type=int, default=MIN_EP)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    # 1. Scan raw episodes
    all_dirs = sorted(RAW_DIR.iterdir())
    episodes = []
    failed = []
    for d in all_dirs:
        if not d.is_dir() or "_episode_" not in d.name:
            continue
        # Extract episode number
        parts = d.name.split("_episode_")
        if len(parts) < 2:
            continue
        ep_num_str = parts[1].split("_")[0]
        try:
            ep_num = int(ep_num_str)
        except ValueError:
            continue
        if ep_num < args.min_ep:
            continue
        npz_path = d / "data.npz"
        if not npz_path.exists():
            continue
        if d.name.endswith("_failed"):
            failed.append((ep_num, d.name))
        else:
            episodes.append((ep_num, str(npz_path), d.name))

    print(f"=" * 70)
    print(f"📊 Raw Data Scan (Episode >= {args.min_ep})")
    print(f"   Success: {len(episodes)} episodes")
    print(f"   Failed (skipped): {len(failed)} episodes")
    for ep_num, name in failed:
        print(f"     ❌ {name}")
    print(f"=" * 70)

    if not episodes:
        print("No episodes found!")
        return

    # 2. Inspect first episode for feature shapes
    first = np.load(episodes[0][1], allow_pickle=True)
    state_shape = first["state"].shape[1]
    action_shape = first["action"].shape[1]
    force_shape = first["force"].shape[1]
    cam0_shape = first["camera0"].shape[1:]
    cam1_shape = first["camera1"].shape[1:]

    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        "robotiq_85_left_knuckle_joint",
    ]

    features = {
        "action": {"dtype": "float32", "shape": (action_shape,), "names": joint_names},
        "observation.state": {"dtype": "float32", "shape": (state_shape,), "names": joint_names},
        "observation.force": {"dtype": "float32", "shape": (force_shape,), "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
        "observation.images.camera0": {"dtype": "video", "shape": tuple(cam0_shape), "names": ["height", "width", "channels"]},
        "observation.images.camera1": {"dtype": "video", "shape": tuple(cam1_shape), "names": ["height", "width", "channels"]},
        "observation.stage": {"dtype": "float32", "shape": (4,), "names": ["E1_free_load", "E2_optical_blind", "E3_rigid_micro", "E4_flexible_brittle"]},
    }

    # 3. Create LeRobot dataset
    print(f"\n🔨 Creating LeRobot dataset: {REPO_ID}")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        robot_type="ur3",
        use_videos=True,
        vcodec="h264",
    )

    total_frames = 0
    total_trimmed = 0
    t0 = time.time()

    for idx, (ep_num, npz_path, dir_name) in enumerate(episodes):
        data = np.load(npz_path, allow_pickle=True)
        states = data["state"]
        actions = data["action"]
        forces = data["force"]
        cam0 = data["camera0"]
        cam1 = data["camera1"]

        # Load stage labels
        stage_v2_path = npz_path.replace(".npz", "_stage_v2.npy")
        stage_v2_path2 = str(Path(npz_path).parent / "data_stage_v2.npy")
        stages = None
        for sp in [stage_v2_path2, stage_v2_path]:
            if os.path.exists(sp):
                loaded = np.load(sp)
                if len(loaded) == len(states):
                    stages = loaded
                    break
        if stages is None:
            stages = data.get("stage", np.zeros((len(states), 4), dtype=np.float32))

        # Task labels
        n_frames_raw = len(states)
        task_data = data.get("task", None)
        if isinstance(task_data, np.ndarray) and len(task_data) == n_frames_raw:
            tasks = list(task_data)
        else:
            tasks = None

        # 4. Find trim point (remove return-to-home)
        trim_idx = find_trim_frame(actions, tasks)
        trimmed_count = n_frames_raw - trim_idx

        peg_x = float(data.get("peg_x", 0))
        peg_y = float(data.get("peg_y", 0))
        hole_x = float(data.get("hole_x", 0))
        hole_y = float(data.get("hole_y", 0))

        # 5. Write frames (trimmed)
        for i in range(trim_idx):
            obs_state = states[i].copy().astype(np.float32)
            task_str = str(tasks[i]) if tasks is not None else "pick up the peg and insert it into the hole"
            if task_str == "go back to home" or states[i, 6] <= 0.12:
                obs_state[6] = 0.0
            else:
                obs_state[6] = 1.0

            act = actions[i].copy().astype(np.float32)
            if task_str == "go back to home" or actions[i, 6] <= 0.12:
                act[6] = 0.0
            else:
                act[6] = 1.0

            stage_vec = stages[i].astype(np.float32)

            frame = {
                "observation.state": obs_state,
                "action": act,
                "observation.force": forces[i].astype(np.float32),
                "observation.stage": stage_vec,
                "observation.images.camera0": (cam0[i] * 255).astype(np.uint8),
                "observation.images.camera1": (cam1[i] * 255).astype(np.uint8),
                "task": task_str,
            }
            dataset.add_frame(frame)
            total_frames += 1

        dataset.save_episode()
        total_trimmed += trimmed_count
        print(f"  ✓ Ep {ep_num:4d} ({idx+1:2d}/{len(episodes)}) | {trim_idx} frames kept, {trimmed_count} trimmed | peg=({peg_x:.3f},{peg_y:.3f}) hole=({hole_x:.3f},{hole_y:.3f})")

    dataset.finalize()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"✅ Dataset build complete!")
    print(f"   Total episodes:       {len(episodes)}")
    print(f"   Total frames kept:    {total_frames}")
    print(f"   Total frames trimmed: {total_trimmed}")
    print(f"   Time elapsed:         {elapsed:.1f}s")

    # Copy to output directory
    output_path = Path(args.output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(dataset.root, output_path)
    print(f"   Saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

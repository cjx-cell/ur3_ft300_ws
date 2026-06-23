#!/usr/bin/env python3
"""
Convert peg-in-hole npz files to LeRobot v3.0 format with force modality.

Usage (pi0-env):
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env
  python3 ur3_convert_peg_in_hole_to_lerobot.py --input ~/ur3_ft300_ws/ai-models/ur3_peg_in_hole_raw --repo_id cjx-cell/ur3_peg_in_hole
"""

import argparse, os, sys, time
import numpy as np
from pathlib import Path

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print(f"ERROR: LeRobot not installed ({e})")
    print("Please run in pi0-env: source ~/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env")
    sys.exit(1)


def find_episodes(input_dir, skip_failed=True):
    """Scan all *_episode_*/data.npz files, optionally skipping failed episodes.

    Failed episodes (dir name ends with _failed) contain trajectories where the
    peg was not successfully inserted. In Behavior Cloning these are harmful:
    the model learns to reproduce the failed motion, not to avoid it.
    """
    episodes = []
    skipped = []
    for d in sorted(os.listdir(input_dir)):
        ep_dir = os.path.join(input_dir, d)
        if "_episode_" in d and os.path.isdir(ep_dir):
            npz_path = os.path.join(ep_dir, "data.npz")
            if os.path.exists(npz_path):
                if skip_failed and d.endswith("_failed"):
                    skipped.append(d)
                else:
                    episodes.append(npz_path)
    return episodes, skipped


def main():
    parser = argparse.ArgumentParser(description="npz → LeRobot format (peg-in-hole, force modality)")
    parser.add_argument("--input", type=str, required=True, help="npz dataset directory")
    parser.add_argument("--repo_id", type=str, default="cjx-cell/ur3_peg_in_hole",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--fps", type=int, default=10, help="Target frame rate (default 10Hz)")
    parser.add_argument("--source_fps", type=int, default=10,
                        help="Source recording frame rate (default 10Hz, for downsampling)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Local output directory")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--keep_failed", action="store_true",
                        help="Include failed episodes (default: skip, they harm BC training)")
    args = parser.parse_args()

    step = max(1, args.source_fps // args.fps)
    print(f"Downsampling: {args.source_fps}Hz → {args.fps}Hz (every {step}th frame)")
    print("Tip: use --policy.use_relative_actions=true during training to avoid Identity Shortcut")

    input_dir = Path(args.input)
    episodes, skipped = find_episodes(input_dir, skip_failed=not args.keep_failed)
    print(f"Found {len(episodes)} successful episodes")
    if skipped:
        print(f"Skipped {len(skipped)} failed episodes (use --keep_failed to include):")
        for s in skipped:
            print(f"  ✗ {s}")

    if not episodes:
        print("No data found! Run ur3_record_peg_in_hole.py first.")
        return

    # Read first episode to get feature shapes
    first = np.load(episodes[0], allow_pickle=True)
    state_shape = first["state"].shape[1]   # (N, state_dim)
    action_shape = first["action"].shape[1]  # (N, action_dim)
    force_shape = first["force"].shape[1]    # (N, 6)
    cam0_shape = first["camera0"].shape[1:]  # (224, 224, 3)
    cam1_shape = first["camera1"].shape[1:]

    print(f"State: {state_shape}D | Action: {action_shape}D | Force: {force_shape}D")
    print(f"Camera0: {cam0_shape} | Camera1: {cam1_shape}")

    # Check if stage labels are present
    has_stage = "stage" in first.files
    if has_stage:
        print("Stage labels found in npz")

    # Joint names
    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        "robotiq_85_left_knuckle_joint",
    ]

    # ── Build features dict ──
    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_shape,),
            "names": joint_names if action_shape == 7 else [f"joint_{i}" for i in range(action_shape)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_shape,),
            "names": joint_names if state_shape == 7 else [f"joint_{i}" for i in range(state_shape)],
        },
        "observation.force": {
            "dtype": "float32",
            "shape": (force_shape,),
            "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
        },
        "observation.images.camera0": {
            "dtype": "video",
            "shape": tuple(cam0_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera1": {
            "dtype": "video",
            "shape": tuple(cam1_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.stage": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["stage"],
        },
    }

    print(f"\nCreating LeRobot dataset: {args.repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type="ur3",
        use_videos=True,
        vcodec="h264",
    )

    # ── Add frames per episode ──
    total_frames = 0
    t0 = time.time()
    for ep_idx, npz_path in enumerate(episodes):
        data = np.load(npz_path, allow_pickle=True)
        states = data["state"]
        actions = data["action"]
        forces = data["force"]
        cam0 = data["camera0"]
        cam1 = data["camera1"]
        stages = data.get("stage", np.zeros(len(states), dtype=np.int64))

        n_frames = len(states)
        task_text = str(data.get("task", "pick up the peg and insert it into the hole"))

        for i in range(0, n_frames, step):
            frame = {
                "observation.state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "observation.force": forces[i].astype(np.float32),
                "observation.stage": np.array([stages[i]], dtype=np.float32),
                "observation.images.camera0": (cam0[i] * 255).astype(np.uint8),
                "observation.images.camera1": (cam1[i] * 255).astype(np.uint8),
                "task": task_text,
            }
            dataset.add_frame(frame)
            total_frames += 1

        dataset.save_episode()
        elapsed = time.time() - t0
        print(f"  Episode {ep_idx+1}/{len(episodes)}: {n_frames} frames "
              f"({elapsed:.1f}s, {total_frames} total)")

    # ── Finalize ──
    dataset.finalize()
    print(f"\nDataset created: {total_frames} frames, {len(episodes)} episodes, "
          f"{time.time()-t0:.1f}s elapsed")

    # Print statistics
    stats = getattr(dataset, "stats", None)
    if not stats:
        try:
            stats = dataset.meta.stats
        except Exception:
            stats = None
    if stats:
        print(f"\nNormalization stats:")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}: mean={v.get('mean', 'N/A')}, std={v.get('std', 'N/A')}")

    output_path = dataset.root
    print(f"\nDataset path: {output_path}")

    # Copy to local directory
    import shutil
    local_dir = args.output_dir or os.path.expanduser(
        "~/ur3_ft300_ws/ai-models/datasets/ur3_peg_in_hole_lerobot")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    shutil.copytree(output_path, local_dir)
    print(f"Copied to: {local_dir}")

    if args.push_to_hub:
        print("Pushing to HuggingFace Hub...")
        dataset.push_to_hub()
        print(f"Pushed: https://huggingface.co/datasets/{args.repo_id}")

    print("\nNext: train SA-MOE on this dataset")
    print(f"  python -m lerobot.scripts.lerobot_train \\")
    print(f"    --policy.path=lerobot/sa_moe_pi0 \\")
    print(f"    --dataset.repo_id={args.repo_id}")


if __name__ == "__main__":
    main()

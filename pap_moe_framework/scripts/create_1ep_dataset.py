#!/usr/bin/env python3
"""
Create single episode LeRobot dataset (lerobot_overfit_1ep) from raw Episode 301.
Computes true absolute statistics and saves info.json, stats.json, parquet, and videos.
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def create_1ep_dataset():
    raw_ep_dir = Path("/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/raw_5ep/pick_up_the_peg_and_insert_it_into_the_hole_episode_0301_success")
    data_file = raw_ep_dir / "data.npz"

    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        sys.exit(1)

    print(f"📦 Converting Episode 301 from {data_file} to LeRobot dataset format...")
    npz_data = np.load(data_file)

    # 1. Target dataset directory
    out_dir = Path("/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_overfit_1ep")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    chunk_data_dir = out_dir / "data/chunk-000"
    chunk_data_dir.mkdir(parents=True, exist_ok=True)
    chunk_vid_dir = out_dir / "videos"
    chunk_vid_dir.mkdir(parents=True, exist_ok=True)

    state = npz_data["state"].astype(np.float32)       # (N, 7)
    action = npz_data["action"].astype(np.float32)     # (N, 7)
    force = npz_data["force"].astype(np.float32)       # (N, 6)
    timestamp = npz_data["timestamp"].astype(np.float64) # (N,)
    N = len(state)

    print(f"   Total frames: {N}")

    # 2. Build parquet file
    df_data = {
        "episode_index": np.zeros(N, dtype=np.int64),
        "frame_index": np.arange(N, dtype=np.int64),
        "timestamp": (np.arange(N, dtype=np.float32) / 10.0),
        "index": np.arange(N, dtype=np.int64),
        "task_index": np.zeros(N, dtype=np.int64),
        "action": list(action),
        "observation.state": list(state),
        "observation.force": list(force),
    }

    df = pd.DataFrame(df_data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, chunk_data_dir / "file-000.parquet")

    # 3. Create MP4 videos for camera0 & camera1
    import cv2
    for cam_name in ["camera0", "camera1"]:
        cam_vid_dir = chunk_vid_dir / f"observation.images.{cam_name}/chunk-000"
        cam_vid_dir.mkdir(parents=True, exist_ok=True)
        vid_path = cam_vid_dir / "file-000.mp4"

        frames = npz_data[cam_name]  # (N, 224, 224, 3)
        if frames.max() <= 1.0:
            frames = (frames * 255.0).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (224, 224))
        for img in frames:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out_writer.write(img_bgr)
        out_writer.release()
        print(f"   Saved video {cam_name} ({len(frames)} frames) to {vid_path}")

    # 4. Create info.json
    info_json = {
        "codebase_version": "v3.0",
        "robot_type": "ur3",
        "total_episodes": 1,
        "total_frames": N,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "robotiq_85_left_knuckle_joint"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [7],
                "names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "robotiq_85_left_knuckle_joint"]
            },
            "observation.force": {
                "dtype": "float32",
                "shape": [6],
                "names": ["fx", "fy", "fz", "tx", "ty", "tz"]
            },
            "observation.images.camera0": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": 224, "video.width": 224, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.fps": 10, "video.channels": 3, "has_audio": False}
            },
            "observation.images.camera1": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": 224, "video.width": 224, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.fps": 10, "video.channels": 3, "has_audio": False}
            }
        }
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info_json, f, indent=4)

    # 5. Compute true stats.json
    def compute_stats(arr):
        return {
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "count": [N],
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q10": np.percentile(arr, 10, axis=0).tolist(),
            "q50": np.percentile(arr, 50, axis=0).tolist(),
            "q90": np.percentile(arr, 90, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    stats_json = {
        "episode_index": compute_stats(np.zeros((N, 1))),
        "index": compute_stats(np.arange(N, dtype=np.float32)[:, None]),
        "frame_index": compute_stats(np.arange(N, dtype=np.float32)[:, None]),
        "timestamp": compute_stats(df_data["timestamp"][:, None]),
        "action": compute_stats(action),
        "observation.state": compute_stats(state),
        "observation.force": compute_stats(force),
    }

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats_json, f, indent=4)

    print(f"✓ Single episode dataset successfully created at {out_dir}!")

if __name__ == "__main__":
    create_1ep_dataset()

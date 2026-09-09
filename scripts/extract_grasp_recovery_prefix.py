#!/usr/bin/env python3
"""Extract a validated grasp-only recovery prefix from a recorded episode."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


FRAME_KEYS = {
    "state",
    "action",
    "force",
    "force_fast",
    "force_slow",
    "state_history",
    "visual_quality",
    "force_fast_valid",
    "force_slow_valid",
    "state_history_valid",
    "force_reference_payload",
    "force_fast_reference_payload",
    "force_slow_reference_payload",
    "stage",
    "tool0_z",
    "camera0",
    "camera1",
    "timestamp",
    "timestamp_ros",
    "task",
    "search_mode_frame",
    "recovery_active",
    "recovery_attempt",
    "recovery_direction",
    "recovery_result",
    "recovery_event_id",
    "recovery_safety",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    with np.load(args.input, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    tasks = np.asarray([str(value) for value in data["task"]], dtype=object)
    non_grasp = np.flatnonzero(tasks != "grasp the peg")
    end = int(non_grasp[0]) if len(non_grasp) else len(tasks)
    if end < 20:
        raise ValueError(f"Grasp prefix is implausibly short: {end} frames")
    if not np.any(np.asarray(data["state"][:end, 6]) > 0.60):
        raise ValueError("Grasp prefix never reaches a closed gripper state")
    offset = np.asarray(data.get("grasp_recovery_offset", np.zeros(2)), dtype=np.float32)
    if offset.shape != (2,) or float(np.linalg.norm(offset)) < 0.005:
        raise ValueError(f"Episode has no meaningful recovery offset: {offset}")

    output: dict[str, np.ndarray] = {}
    for key, value in data.items():
        output[key] = value[:end] if key in FRAME_KEYS else value
    last_ros_timestamp = float(np.asarray(output["timestamp_ros"])[-1])
    raw_mask = np.asarray(data["raw_force_timestamp"]) <= last_ros_timestamp
    for key in ("raw_force_timestamp", "raw_force_wall_timestamp", "raw_force"):
        output[key] = np.asarray(data[key])[raw_mask]
    output["controller_result"] = np.str_("grasp_recovery_success")
    output["trajectory_scope"] = np.str_("grasp_recovery_prefix_v1")
    output["source_episode"] = np.str_(str(args.input.resolve()))

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        np.savez_compressed(temp_dir / "data.npz", **output)
        os.replace(temp_dir, args.output_dir)
    except Exception:
        for child in temp_dir.iterdir():
            child.unlink()
        temp_dir.rmdir()
        raise
    print(
        f"Saved {end}-frame grasp recovery prefix to {args.output_dir} "
        f"(offset={offset.tolist()})"
    )


if __name__ == "__main__":
    main()

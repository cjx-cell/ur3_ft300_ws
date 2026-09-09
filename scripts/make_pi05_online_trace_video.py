#!/usr/bin/env python3
"""Render a dual-camera MP4 from a Pi0.5 online trace.

The trace contains the exact camera observations consumed at each policy replan.
Frames are held for one execution horizon by default, so the output follows
simulated task time instead of the slower Gazebo wall-clock rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _to_bgr(image: np.ndarray, size: int) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seconds-per-chunk", type=float, default=1.0)
    parser.add_argument("--view-size", type=int, default=448)
    parser.add_argument("--title", default="Baseline pure-model peg-in-hole inference")
    args = parser.parse_args()

    paths = sorted(args.trace_dir.glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No chunk_*.npz under {args.trace_dir}")
    if args.fps <= 0 or args.seconds_per_chunk <= 0:
        raise ValueError("fps and seconds-per-chunk must be positive")

    result = {}
    if args.result_json is not None:
        result = json.loads(args.result_json.read_text())

    width = 2 * args.view_size
    header_height = 70
    height = args.view_size + header_height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video {args.output}")

    repeats = max(1, round(args.fps * args.seconds_per_chunk))
    feedback_first_chunk = None
    try:
        for path in paths:
            with np.load(path, allow_pickle=False) as trace:
                chunk_id = int(trace["chunk_id"])
                feedback = bool(trace.get("insertion_feedback_active", False))
                state = np.asarray(trace["state"])
                wrist = _to_bgr(trace["camera0"], args.view_size)
                global_camera = _to_bgr(trace["camera1"], args.view_size)
            if feedback and feedback_first_chunk is None:
                feedback_first_chunk = chunk_id

            frame = cv2.hconcat([wrist, global_camera])
            frame = cv2.copyMakeBorder(
                frame, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(22, 22, 22)
            )
            cv2.putText(
                frame, args.title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.63, (255, 255, 255), 1, cv2.LINE_AA,
            )
            phase = "learned joint+FT300 feedback" if feedback else "vision policy + stage adapter"
            cv2.putText(
                frame,
                f"chunk={chunk_id:03d}  task_time~{chunk_id * args.seconds_per_chunk:5.1f}s  "
                f"gripper={state[6]:.3f}  phase={phase}",
                (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (120, 230, 160) if feedback else (180, 220, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(frame, "Wrist camera", (16, 66), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.putText(frame, "Global camera", (args.view_size + 16, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
            for _ in range(repeats):
                writer.write(frame)

        if result:
            final = np.full((height, width, 3), 22, dtype=np.uint8)
            success = bool(result.get("success", False))
            color = (80, 230, 120) if success else (80, 80, 240)
            cv2.putText(final, "GAZEBO RESULT: SUCCESS" if success else "GAZEBO RESULT: FAILURE",
                        (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 3, cv2.LINE_AA)
            cv2.putText(
                final,
                f"peg-hole XY={1000 * float(result.get('success_xy_m', float('nan'))):.2f} mm   "
                f"peg z={float(result.get('success_peg_z_m', float('nan'))):.4f} m",
                (90, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2, cv2.LINE_AA,
            )
            cv2.putText(
                final, f"ever_attached={result.get('ever_attached')}   pure physical grasp",
                (90, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA,
            )
            for _ in range(round(3 * args.fps)):
                writer.write(final)
    finally:
        writer.release()

    print(f"Saved {len(paths)} trace observations to {args.output}")
    print(f"feedback_first_chunk={feedback_first_chunk}, frames={len(paths) * repeats}")


if __name__ == "__main__":
    main()

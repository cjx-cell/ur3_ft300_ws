#!/usr/bin/env python3
"""Build deployable-observation DAgger corrections from a failed Pi0.5 rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--executed-prefix",
        type=int,
        default=10,
        help="Number of receding-horizon actions that deployment actually executes.",
    )
    parser.add_argument("--blend-steps", type=int, default=20)
    parser.add_argument(
        "--target-gripper-threshold",
        type=float,
        default=0.12,
        help=(
            "Physical demonstration gripper threshold used by the v9 dataset "
            "conversion (0.100 open, 0.629 closed)."
        ),
    )
    parser.add_argument(
        "--target-frame-offset",
        type=int,
        default=0,
        help=(
            "Advance correction targets this many expert frames beyond the aligned "
            "state so progress appears inside a short receding-horizon prefix."
        ),
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("monotonic", "independent"),
        default="monotonic",
        help="Prevent a sequential rollout from being relabeled with backward demo progress.",
    )
    parser.add_argument(
        "--alignment-domain",
        choices=("preclose-grasp", "postgrasp-task", "align-to-insert"),
        default="preclose-grasp",
        help="Choose the demonstration phase used to relabel rollout states.",
    )
    args = parser.parse_args()
    if args.target_frame_offset < 0:
        raise ValueError("--target-frame-offset must be non-negative")
    if not 1 <= args.executed_prefix <= args.horizon:
        raise ValueError("--executed-prefix must be within [1, --horizon]")
    if not np.isfinite(args.target_gripper_threshold):
        raise ValueError("--target-gripper-threshold must be finite")

    all_trace_paths = sorted(args.trace_dir.glob("chunk_*.npz"))
    trace_paths = all_trace_paths[args.start_chunk : args.start_chunk + args.max_chunks]
    if not trace_paths:
        raise FileNotFoundError(f"No rollout traces under {args.trace_dir}")
    with np.load(args.episode_npz, allow_pickle=True) as archive:
        demo = {key: archive[key] for key in ("state", "action", "task")}

    task = np.asarray([str(value) for value in demo["task"]])
    if args.alignment_domain == "preclose-grasp":
        valid = np.flatnonzero((task == "grasp the peg") & (demo["state"][:, 6] < 0.2))
    elif args.alignment_domain == "postgrasp-task":
        postgrasp_tasks = np.isin(task, (
            "grasp the peg",
            "transport to the hole",
            "approach and align with the hole",
            "insert the peg into the hole",
            "verify insertion success",
        ))
        valid = np.flatnonzero(postgrasp_tasks & (demo["state"][:, 6] >= 0.5))
    else:
        # Contact-preparation labels must not fall back to transport or grasp.
        # Align only against the successful approach phase, then use
        # --target-frame-offset to place the executed prefix at the end of
        # alignment / beginning of insertion.
        valid = np.flatnonzero(
            (task == "approach and align with the hole")
            & (demo["state"][:, 6] >= 0.5)
        )
    if len(valid) == 0:
        raise ValueError(f"Demonstration has no states for {args.alignment_domain}")

    states = []
    camera0 = []
    camera1 = []
    targets = []
    source_predictions = []
    demo_frames = []
    alignment_l2 = []
    first_target_jumps = []
    prefix_net_l2 = []
    prefix_max_joint_steps = []
    prefix_close = []
    previous_demo_frame = int(valid[0])
    for trace_path in trace_paths:
        with np.load(trace_path) as trace:
            state = trace["state"].astype(np.float32)
            candidates = valid
            if args.alignment_mode == "monotonic":
                candidates = valid[valid >= previous_demo_frame]
                if len(candidates) == 0:
                    candidates = valid[-1:]
            distances = np.linalg.norm(demo["state"][candidates, :6] - state[None, :6], axis=1)
            demo_frame = int(candidates[int(np.argmin(distances))])
            previous_demo_frame = demo_frame
            target_start_frame = min(
                demo_frame + args.target_frame_offset, len(demo["action"]) - 1
            )
            indices = np.minimum(
                np.arange(target_start_frame, target_start_frame + args.horizon),
                len(demo["action"]) - 1,
            )
            target = demo["action"][indices].astype(np.float32).copy()
            # Match the v9 dataset converter.  The raw demonstration stores the
            # physical 0.100-open / 0.629-closed joint and begins a real close
            # transition well below 0.5; thresholding at 0.5 delays the label
            # by roughly two seconds and makes short-horizon replanning discard
            # the grasp transition indefinitely.
            target[:, 6] = (
                target[:, 6] > args.target_gripper_threshold
            ).astype(np.float32)

            # Make the correction executable from the actual off-policy state:
            # begin at the measured joints, then smoothly merge into the expert
            # trajectory.  This changes labels offline only; deployment never
            # reads the demonstration or simulator object truth.
            blend_steps = min(args.blend_steps, args.horizon)
            offset = state[:6] - demo["state"][target_start_frame, :6]
            weights = 1.0 - (np.arange(1, blend_steps + 1, dtype=np.float32) / blend_steps)
            target[:blend_steps, :6] += weights[:, None] * offset[None, :]
            target[:blend_steps, 6] = (
                0.0 if args.alignment_domain == "preclose-grasp" else 1.0
            )

            states.append(state)
            camera0.append(trace["camera0"].astype(np.float32))
            camera1.append(trace["camera1"].astype(np.float32))
            targets.append(target)
            source_predictions.append(trace["predicted_action_chunk"].astype(np.float32))
            demo_frames.append(demo_frame)
            alignment_l2.append(float(distances.min()))
            first_target_jumps.append(float(np.max(np.abs(target[0, :6] - state[:6]))))
            prefix = target[: args.executed_prefix]
            prefix_net_l2.append(float(np.linalg.norm(prefix[-1, :6] - state[:6])))
            prefix_steps = np.diff(
                np.concatenate((state[None, :6], prefix[:, :6]), axis=0), axis=0
            )
            prefix_max_joint_steps.append(float(np.max(np.abs(prefix_steps))))
            prefix_close.extend((prefix[:, 6] >= 0.5).astype(np.float32).tolist())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        state=np.stack(states),
        camera0=np.stack(camera0),
        camera1=np.stack(camera1),
        target_action=np.stack(targets),
        source_prediction=np.stack(source_predictions),
        demo_frame=np.asarray(demo_frames, dtype=np.int64),
        alignment_l2_rad=np.asarray(alignment_l2, dtype=np.float32),
    )
    report = {
        "trace_dir": str(args.trace_dir.resolve()),
        "episode": str(args.episode_npz.resolve()),
        "samples": len(states),
        "start_chunk": args.start_chunk,
        "horizon": args.horizon,
        "executed_prefix": args.executed_prefix,
        "blend_steps": args.blend_steps,
        "target_frame_offset": args.target_frame_offset,
        "target_gripper_threshold": args.target_gripper_threshold,
        "alignment_domain": args.alignment_domain,
        "alignment_mode": args.alignment_mode,
        "demo_frames": demo_frames,
        "alignment_l2_rad": {
            "mean": float(np.mean(alignment_l2)),
            "max": float(np.max(alignment_l2)),
        },
        "max_first_target_jump_rad": float(np.max(first_target_jumps)),
        "executed_prefix_progress_l2_rad": {
            "mean": float(np.mean(prefix_net_l2)),
            "min": float(np.min(prefix_net_l2)),
            "max": float(np.max(prefix_net_l2)),
        },
        "executed_prefix_max_joint_step_rad": float(np.max(prefix_max_joint_steps)),
        "executed_prefix_close_rate": float(np.mean(prefix_close)),
        "uses_online_object_truth": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

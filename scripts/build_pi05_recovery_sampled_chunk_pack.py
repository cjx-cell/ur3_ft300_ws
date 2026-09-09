#!/usr/bin/env python3
"""Extract auditable Pi0.5 critical chunks from one validated recovery episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ALIGN_TASK = "approach and align with the hole"
INSERT_TASKS = {"insert the peg into the hole", "verify insertion success"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--anchor-episode", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--descent-start-frame", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.recovery, allow_pickle=False) as raw:
        recovery = {key: raw[key] for key in raw.files}
    with np.load(args.anchor_episode, allow_pickle=True) as raw:
        anchor = {key: raw[key] for key in ("state", "action", "task")}
        anchor["semantic_subtask"] = raw["semantic_subtask"]

    if str(np.asarray(recovery["recovery_outcome"]).item()) != "success":
        raise ValueError("recovery episode is not successful")
    intervention = np.asarray(recovery["intervention_mask"], dtype=bool)
    action = np.asarray(recovery["executed_action"], dtype=np.float32)
    state = np.asarray(recovery["state"], dtype=np.float32)
    task = np.asarray([str(value) for value in anchor["semantic_subtask"]])
    anchor_state = np.asarray(anchor["state"], dtype=np.float32)
    if len(action) != len(state) or len(intervention) != len(state):
        raise ValueError("recovery arrays have inconsistent lengths")

    samples: dict[str, list[np.ndarray | int]] = {
        "state": [],
        "camera0": [],
        "camera1": [],
        "target_action": [],
        "demo_frame": [],
    }
    audit = []
    for frame in args.frames:
        if frame < 0 or frame + args.chunk_size > len(state):
            raise ValueError(f"frame {frame} has no complete action chunk")
        if not intervention[frame]:
            raise ValueError(f"frame {frame} is not expert-controlled")
        allowed = (
            np.isin(task, list(INSERT_TASKS))
            if frame >= args.descent_start_frame
            else task == ALIGN_TASK
        )
        candidates = np.flatnonzero(allowed)
        candidates = candidates[candidates + args.chunk_size <= len(anchor_state)]
        if len(candidates) == 0:
            raise ValueError(f"no anchor candidates for frame {frame}")
        distances = np.linalg.norm(
            anchor_state[candidates, :6] - state[frame, :6], axis=1
        ) + 0.25 * np.abs(anchor_state[candidates, 6] - state[frame, 6])
        nearest = int(candidates[int(np.argmin(distances))])
        samples["state"].append(state[frame])
        samples["camera0"].append(np.asarray(recovery["camera0"][frame]))
        samples["camera1"].append(np.asarray(recovery["camera1"][frame]))
        samples["target_action"].append(action[frame : frame + args.chunk_size])
        samples["demo_frame"].append(nearest)
        audit.append(
            {
                "recovery_frame": frame,
                "kind": "insert" if frame >= args.descent_start_frame else "align",
                "anchor_frame": nearest,
                "anchor_task": task[nearest],
                "anchor_distance": float(distances.min()),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        state=np.stack(samples["state"]).astype(np.float32),
        camera0=np.stack(samples["camera0"]),
        camera1=np.stack(samples["camera1"]),
        target_action=np.stack(samples["target_action"]).astype(np.float32),
        demo_frame=np.asarray(samples["demo_frame"], dtype=np.int64),
    )
    report = {
        "recovery": str(args.recovery.resolve()),
        "anchor_episode": str(args.anchor_episode.resolve()),
        "chunk_size": args.chunk_size,
        "descent_start_frame": args.descent_start_frame,
        "samples": audit,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

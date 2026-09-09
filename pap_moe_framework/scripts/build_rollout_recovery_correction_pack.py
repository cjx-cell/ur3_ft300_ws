#!/usr/bin/env python3
"""Convert one validated raw recovery episode into sampled-chunk corrections."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pap_moe_framework.rollout_recovery.schema import load_and_validate


def _binarize_state(state: np.ndarray) -> np.ndarray:
    result = np.asarray(state, dtype=np.float32).copy()
    result[..., 6] = (result[..., 6] > 0.12).astype(np.float32)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--anchor-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--anchor-task", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.chunk_size < 1 or args.stride < 1 or args.max_samples < 1:
        parser.error("chunk size, stride and max samples must be positive")

    summary = load_and_validate(args.episode)
    with np.load(args.episode, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    with np.load(args.anchor_episode, allow_pickle=True) as archive:
        anchor = {key: archive[key] for key in ("state", "task")}

    allowed_tasks = args.anchor_task or ["transport to the hole", "approach and align with the hole"]
    anchor_mask = np.isin(anchor["task"].astype(str), allowed_tasks)
    anchor_mask &= anchor["state"][:, 6] > 0.12
    anchor_indices = np.flatnonzero(anchor_mask)
    if len(anchor_indices) == 0:
        parser.error(f"no closed-gripper anchor frames match {allowed_tasks}")

    expert_indices = np.flatnonzero(raw["control_mode"] == 1)
    # Reduce state samples to one observation per distinct scheduled expert
    # action point. Recorder may retain multiple camera/state snapshots close
    # to the same action timestamp.
    action_times = raw["expert_action_timestamp"][expert_indices]
    _, first_positions = np.unique(np.round(action_times, decimals=6), return_index=True)
    unique_indices = expert_indices[np.sort(first_positions)]
    if len(unique_indices) < args.chunk_size:
        parser.error(
            f"recovery has only {len(unique_indices)} unique expert actions; "
            f"need {args.chunk_size}"
        )

    candidates = unique_indices[:: args.stride]
    candidates = candidates[candidates <= unique_indices[-args.chunk_size]]
    if len(candidates) > args.max_samples:
        choice = np.linspace(0, len(candidates) - 1, args.max_samples).round().astype(int)
        candidates = candidates[choice]

    states, camera0, camera1, targets, demo_frames = [], [], [], [], []
    expert_position = {int(index): position for position, index in enumerate(unique_indices)}
    anchor_states = _binarize_state(anchor["state"])
    for raw_index in candidates:
        position = expert_position[int(raw_index)]
        future_indices = unique_indices[position : position + args.chunk_size]
        target = raw["expert_action"][future_indices].astype(np.float32)
        if target.shape != (args.chunk_size, 7) or not np.isfinite(target).all():
            raise ValueError(f"invalid expert target at raw frame {raw_index}")
        state = _binarize_state(raw["state"][raw_index])
        distances = np.linalg.norm(
            anchor_states[anchor_indices, :6] - state[None, :6], axis=1
        )
        demo_frame = int(anchor_indices[int(np.argmin(distances))])
        states.append(state)
        camera0.append(raw["camera0"][raw_index].astype(np.float32) / 255.0)
        camera1.append(raw["camera1"][raw_index].astype(np.float32) / 255.0)
        targets.append(target)
        demo_frames.append(demo_frame)

    values = {
        "state": np.stack(states).astype(np.float32),
        "camera0": np.stack(camera0).astype(np.float32),
        "camera1": np.stack(camera1).astype(np.float32),
        "target_action": np.stack(targets).astype(np.float32),
        "demo_frame": np.asarray(demo_frames, dtype=np.int64),
        "raw_frame": np.asarray(candidates, dtype=np.int64),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, args.output)
    print(
        f"Saved {len(candidates)} corrections from validated {summary.recovery_phase} "
        f"episode ({summary.policy_frames} policy, {summary.expert_frames} expert frames): "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

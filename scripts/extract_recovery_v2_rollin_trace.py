#!/usr/bin/env python3
"""Extract policy roll-in observations from a recovery-v2 episode as replay chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--last-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.last_frames <= 0 or args.stride <= 0:
        raise ValueError("--last-frames and --stride must be positive")

    keys = (
        "state",
        "camera0",
        "camera1",
        "force",
        "force_fast",
        "force_slow",
        "state_history",
    )
    args.output_dir.mkdir(parents=True)
    with np.load(args.episode, allow_pickle=True) as archive:
        intervention = np.asarray(archive["intervention_mask"], dtype=bool)
        takeover_indices = np.flatnonzero(intervention)
        if not len(takeover_indices):
            raise ValueError("Episode has no takeover frames")
        takeover = int(takeover_indices[0])
        start = max(0, takeover - args.last_frames)
        indices = list(range(start, takeover, args.stride))
        if not indices or indices[-1] != takeover - 1:
            indices.append(takeover - 1)
        for chunk_index, frame in enumerate(indices):
            values = {key: archive[key][frame] for key in keys}
            values["source_frame_index"] = np.asarray(frame, dtype=np.int64)
            values["takeover_index"] = np.asarray(takeover, dtype=np.int64)
            np.savez_compressed(args.output_dir / f"chunk_{chunk_index:04d}.npz", **values)

    report = {
        "episode": str(args.episode.resolve()),
        "takeover_index": takeover,
        "start_index": start,
        "stride": args.stride,
        "frames": indices,
    }
    (args.output_dir / "trace_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report expert-frame semantic balance for explicit recovery episodes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    args = parser.parse_args()

    episodes: dict[str, object] = {}
    total: Counter[str] = Counter()
    for path in args.episodes:
        with np.load(path, allow_pickle=False) as data:
            intervention = np.asarray(data["intervention_mask"], dtype=bool)
            labels = [str(value) for value in data["semantic_subtask"][intervention]]
            counts = Counter(labels)
            total.update(counts)
            gripper = np.asarray(data["executed_action"], dtype=np.float32)[intervention, 6]
            starts = np.flatnonzero(intervention & ~np.r_[False, intervention[:-1]])
            ends = np.flatnonzero(intervention & ~np.r_[intervention[1:], False])
            episodes[path.parent.name] = {
                "frames": len(intervention),
                "policy_frames": int((~intervention).sum()),
                "expert_frames": int(intervention.sum()),
                "intervention_ranges": [
                    [int(start), int(end)] for start, end in zip(starts, ends, strict=True)
                ],
                "expert_semantic_frames": dict(sorted(counts.items())),
                "expert_gripper_action_min": float(gripper.min()),
                "expert_gripper_action_max": float(gripper.max()),
            }
    print(json.dumps({
        "episodes": episodes,
        "total_expert_semantic_frames": dict(sorted(total.items())),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit episode-safe camera history sampling before enabling PAP-MoE E2 memory."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


CAMERA_KEYS = ("observation.images.camera0", "observation.images.camera1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="pap_moe/pap_moe_v9_stage_training")
    parser.add_argument("--history-indices", default="-9,-4,-1,0")
    parser.add_argument("--video-backend", default="torchcodec")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indices = [int(value) for value in args.history_indices.split(",")]
    if not indices or indices[-1] != 0 or any(value > 0 for value in indices):
        raise ValueError("history indices must be non-positive and end at current frame 0")

    delta_timestamps = {
        key: [index / 10.0 for index in indices] for key in CAMERA_KEYS
    }
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    if dataset.meta.fps != 10:
        raise ValueError(
            f"This audit requested 10 Hz deltas but dataset fps={dataset.meta.fps}"
        )

    checked = 0
    for episode_index in range(len(dataset.meta.episodes)):
        episode = dataset.meta.episodes[episode_index]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        for absolute_index in (start, min(start + abs(min(indices)), end - 1)):
            relative_index = absolute_index
            item = dataset[relative_index]
            if int(item["episode_index"]) != int(episode_index):
                raise AssertionError("history query crossed an episode boundary")
            for key in CAMERA_KEYS:
                expected = (len(indices), 3, 224, 224)
                if tuple(item[key].shape) != expected:
                    raise AssertionError(
                        f"{key} shape={tuple(item[key].shape)}, expected={expected}"
                    )
                padding = item[f"{key}_is_pad"]
                if absolute_index == start and not bool(padding[:-1].all()):
                    raise AssertionError(
                        f"episode {episode_index} start did not pad all past frames"
                    )
            checked += 1

    print(
        f"PASS: {checked} boundary/interior samples across "
        f"{len(dataset.meta.episodes)} episodes; camera history is episode-safe."
    )


if __name__ == "__main__":
    main()

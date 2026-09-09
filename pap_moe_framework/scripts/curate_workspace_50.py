#!/usr/bin/env python3
"""Build the canonical five-position, ten-style workspace dataset.

The raw NPZ files remain byte-identical.  The canonical manifest records the
formal episode number and supersedes raw collection bookkeeping fields such as
``position_group_id`` that reflect interrupted/retried collection batches.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


TASK_PREFIX = "pick_up_the_peg_and_insert_it_into_the_hole"
SOURCE_GROUPS = (
    (7, 10, 11, 13, 14, 15, 16, 17, 18, 19),
    tuple(range(20, 30)),
    tuple(range(30, 40)),
    tuple(range(40, 50)),
    tuple(range(50, 60)),
)
COORD_KEYS = ("peg_x", "peg_y", "hole_x", "hole_y")


def scalar(data: np.lib.npyio.NpzFile, key: str):
    if key not in data:
        raise ValueError(f"missing required field: {key}")
    return data[key].item()


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_episodes: list[dict] = []
    link_modes: set[str] = set()
    formal_episode = 1
    for group_index, source_episodes in enumerate(SOURCE_GROUPS):
        group_coords: tuple[float, ...] | None = None
        styles: list[int] = []
        for repeat_index, source_episode in enumerate(source_episodes):
            source_dir = source_root / (
                f"{TASK_PREFIX}_episode_{source_episode:04d}_success"
            )
            source_npz = source_dir / "data.npz"
            if not source_npz.is_file():
                raise FileNotFoundError(source_npz)

            with np.load(source_npz, allow_pickle=True) as data:
                controller_result = str(scalar(data, "controller_result"))
                if controller_result != "success":
                    raise ValueError(
                        f"source episode {source_episode:04d} is not successful: "
                        f"{controller_result}"
                    )
                style = int(scalar(data, "trajectory_style_id"))
                coords = tuple(float(scalar(data, key)) for key in COORD_KEYS)
                frames = int(data["action"].shape[0])
                force_hz = float(scalar(data, "policy_hz"))
                release_delta_z = float(scalar(data, "terminal_release_delta_z"))

            if style != repeat_index:
                raise ValueError(
                    f"source episode {source_episode:04d}: expected style "
                    f"{repeat_index}, got {style}"
                )
            styles.append(style)
            if group_coords is None:
                group_coords = coords
            elif not np.allclose(coords, group_coords, atol=1e-7, rtol=0.0):
                raise ValueError(
                    f"position group {group_index + 1} has mixed coordinates: "
                    f"{group_coords} vs {coords}"
                )

            formal_dir = output_root / (
                f"{TASK_PREFIX}_episode_{formal_episode:04d}_success"
            )
            formal_dir.mkdir()
            link_modes.add(link_or_copy(source_npz, formal_dir / "data.npz"))

            metadata = {
                "formal_episode": formal_episode,
                "source_episode": source_episode,
                "position_group_id": group_index,
                "position_repeat_index": repeat_index,
                "position_repeat_count": 10,
                "trajectory_style_id": style,
                "peg_x": coords[0],
                "peg_y": coords[1],
                "hole_x": coords[2],
                "hole_y": coords[3],
                "frames": frames,
                "policy_hz": force_hz,
                "terminal_release_delta_z": release_delta_z,
                "source_npz": str(source_npz),
            }
            (formal_dir / "formal_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest_episodes.append(metadata)
            formal_episode += 1

        if styles != list(range(10)):
            raise ValueError(
                f"position group {group_index + 1} has invalid styles: {styles}"
            )

    unique_positions = {
        tuple(round(float(ep[key]), 7) for key in COORD_KEYS)
        for ep in manifest_episodes[::10]
    }
    if len(manifest_episodes) != 50 or len(unique_positions) != 5:
        raise ValueError(
            f"expected 50 episodes and 5 positions, got "
            f"{len(manifest_episodes)} and {len(unique_positions)}"
        )

    manifest = {
        "schema": "pap_moe_workspace_5_positions_x_10_styles_v1",
        "task": "pick up the peg and insert it into the hole",
        "episode_count": 50,
        "position_count": 5,
        "styles_per_position": 10,
        "npz_storage": sorted(link_modes),
        "raw_npz_bookkeeping_note": (
            "formal_metadata.json and this manifest are authoritative for "
            "formal episode/group/repeat indices; raw NPZ fields preserve "
            "collection provenance"
        ),
        "episodes": manifest_episodes,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Curated {len(manifest_episodes)} episodes across "
        f"{len(unique_positions)} positions into {output_root}"
    )


if __name__ == "__main__":
    main()

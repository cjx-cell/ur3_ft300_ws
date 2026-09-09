#!/usr/bin/env python3
"""Compute action statistics for current-state-relative action chunks.

LeRobot stores one absolute next-state action per row. Pi0.5 expands that
column into a K-step chunk using offsets ``0..K-1`` and repeats the episode's
last action when the requested horizon crosses the episode boundary.

For relative-action training, every arm target in the chunk is expressed
relative to the *same current observation state*:

    relative_action[t, k] = absolute_action[t + k] - state[t]

Excluded dimensions (the gripper by default) remain absolute.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq


STAT_KEYS = ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def _load_rows(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parquet_files = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files found below {dataset_root}")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for parquet_path in parquet_files:
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "episode_index", "frame_index"],
        )
        states.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32))
        actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float32))
        episodes.append(np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64))
        frames.append(np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64))

    return (
        np.concatenate(states),
        np.concatenate(actions),
        np.concatenate(episodes),
        np.concatenate(frames),
    )


def _make_relative_chunks(
    states: np.ndarray,
    actions: np.ndarray,
    episodes: np.ndarray,
    frames: np.ndarray,
    *,
    chunk_size: int,
    exclude_indices: set[int],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for episode_index in np.unique(episodes):
        row_indices = np.flatnonzero(episodes == episode_index)
        row_indices = row_indices[np.argsort(frames[row_indices])]
        episode_states = states[row_indices]
        episode_actions = actions[row_indices]
        episode_length = len(row_indices)

        offsets = np.arange(chunk_size, dtype=np.int64)
        future_rows = np.minimum(
            np.arange(episode_length, dtype=np.int64)[:, None] + offsets[None, :],
            episode_length - 1,
        )
        episode_chunks = episode_actions[future_rows].copy()
        relative_mask = np.ones(actions.shape[1], dtype=bool)
        for index in exclude_indices:
            if index < 0 or index >= actions.shape[1]:
                raise ValueError(
                    f"Excluded action index {index} is outside action_dim={actions.shape[1]}"
                )
            relative_mask[index] = False
        episode_chunks[:, :, relative_mask] -= episode_states[:, None, relative_mask]
        chunks.append(episode_chunks.reshape(-1, actions.shape[1]))

    return np.concatenate(chunks, axis=0)


def _compute_stats(values: np.ndarray) -> dict[str, list[float]]:
    result = {
        "mean": values.mean(axis=0),
        "std": np.maximum(values.std(axis=0), 1e-8),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
    }
    for key, quantile in QUANTILES.items():
        result[key] = np.quantile(values, quantile, axis=0)
    return {key: np.asarray(value).tolist() for key, value in result.items()}


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--exclude-indices",
        type=int,
        nargs="*",
        default=[6],
        help="Action dimensions kept absolute (default: gripper index 6)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output stats JSON; default: meta/stats_relative_h<chunk_size>.json",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Install the generated stats as meta/stats.json after creating a backup",
    )
    parser.add_argument(
        "--view-root",
        type=Path,
        help=(
            "Create a lightweight LeRobot dataset view that symlinks data/videos, "
            "copies metadata, and installs the generated stats as its meta/stats.json"
        ),
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    dataset_root = args.dataset_root.resolve()
    source_stats_path = dataset_root / "meta" / "stats.json"
    if not source_stats_path.is_file():
        raise FileNotFoundError(f"Dataset stats not found: {source_stats_path}")

    with source_stats_path.open(encoding="utf-8") as file:
        source_stats = json.load(file)

    states, actions, episodes, frames = _load_rows(dataset_root)
    if states.shape != actions.shape:
        raise ValueError(f"State/action shape mismatch: {states.shape} vs {actions.shape}")

    relative_chunks = _make_relative_chunks(
        states,
        actions,
        episodes,
        frames,
        chunk_size=args.chunk_size,
        exclude_indices=set(args.exclude_indices),
    )
    action_stats = _compute_stats(relative_chunks)

    output_stats = deepcopy(source_stats)
    output_stats["action"].update(action_stats)
    output_stats["action"]["count"] = [float(len(relative_chunks))]

    output_path = args.output
    if output_path is None:
        output_path = dataset_root / "meta" / f"stats_relative_h{args.chunk_size}.json"
    output_path = output_path.resolve()
    _atomic_write_json(output_path, output_stats)

    q01 = np.asarray(action_stats["q01"])
    q99 = np.asarray(action_stats["q99"])
    normalized = 2.0 * (relative_chunks - q01) / np.maximum(q99 - q01, 1e-8) - 1.0
    outside_fraction = np.mean(np.abs(normalized) > 1.0, axis=0)
    audit = {
        "dataset_root": str(dataset_root),
        "source_rows": int(len(states)),
        "episodes": int(len(np.unique(episodes))),
        "chunk_size": args.chunk_size,
        "statistics_samples": int(len(relative_chunks)),
        "exclude_indices": args.exclude_indices,
        "mean_abs_relative_arm_action": np.mean(
            np.abs(relative_chunks[:, :6]), axis=0
        ).tolist(),
        "fraction_outside_q01_q99": outside_fraction.tolist(),
        "output_stats": str(output_path),
    }
    audit_path = output_path.with_suffix(".audit.json")
    _atomic_write_json(audit_path, audit)

    if args.apply:
        backup_path = source_stats_path.with_name("stats.json.absolute_backup")
        if not backup_path.exists():
            shutil.copy2(source_stats_path, backup_path)
        shutil.copy2(output_path, source_stats_path)
        print(f"Installed relative chunk stats: {source_stats_path}")
        print(f"Absolute stats backup: {backup_path}")

    if args.view_root is not None:
        view_root = args.view_root.resolve()
        if view_root == dataset_root:
            raise ValueError("--view-root must differ from --dataset-root")
        view_root.mkdir(parents=True, exist_ok=True)

        for directory_name in ("data", "videos"):
            source_directory = dataset_root / directory_name
            if not source_directory.exists():
                continue
            view_directory = view_root / directory_name
            if view_directory.is_symlink():
                if view_directory.resolve() != source_directory.resolve():
                    raise ValueError(
                        f"Existing symlink {view_directory} targets {view_directory.resolve()}, "
                        f"expected {source_directory.resolve()}"
                    )
            elif view_directory.exists():
                raise FileExistsError(
                    f"Refusing to replace non-symlink dataset view path: {view_directory}"
                )
            else:
                view_directory.symlink_to(source_directory, target_is_directory=True)

        view_meta = view_root / "meta"
        shutil.copytree(dataset_root / "meta", view_meta, dirs_exist_ok=True)
        shutil.copy2(output_path, view_meta / "stats.json")
        provenance = {
            "source_dataset": str(dataset_root),
            "action_semantics": "current_state_relative_chunk",
            "chunk_size": args.chunk_size,
            "exclude_indices": args.exclude_indices,
            "generated_stats": str(output_path),
        }
        _atomic_write_json(view_root / "RELATIVE_ACTION_VIEW.json", provenance)
        print(f"Relative-action dataset view: {view_root}")

    print(f"Rows: {len(states)}, episodes: {len(np.unique(episodes))}")
    print(f"Chunk statistics samples: {len(relative_chunks)}")
    print(f"Action stats: {output_path}")
    print(f"Audit: {audit_path}")
    print(
        "Mean |relative arm action|: "
        + np.array2string(
            np.mean(np.abs(relative_chunks[:, :6]), axis=0),
            precision=5,
        )
    )
    print(
        "Fraction outside [q01, q99]: "
        + np.array2string(outside_fraction, precision=4)
    )


if __name__ == "__main__":
    main()

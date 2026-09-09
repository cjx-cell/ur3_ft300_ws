#!/usr/bin/env python3
"""Create a lightweight LeRobot view whose language task is globally constant.

The source dataset keeps its absolute actions and per-frame semantic subtask
labels.  This view rewrites only parquet metadata/task indices and symlinks the
large video directory, so a standard Pi0.5 baseline is trained with the same
global instruction that it receives at evaluation time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


DEFAULT_GLOBAL_TASK = "pick up the peg and insert it into the hole"
MARKER_NAME = "GLOBAL_TASK_VIEW.json"


def _replace_with_zero(value: object) -> object:
    if isinstance(value, np.ndarray):
        return np.zeros_like(value)
    if isinstance(value, list):
        return [0 for _ in value]
    return type(value)(0) if value is not None else 0


def _write_global_tasks(meta_dir: Path, global_task: str) -> None:
    tasks = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([global_task], name="task"),
    )
    tasks.to_parquet(meta_dir / "tasks.parquet")

    for episode_file in sorted((meta_dir / "episodes").glob("chunk-*/file-*.parquet")):
        episodes = pd.read_parquet(episode_file)
        episodes["tasks"] = [[global_task] for _ in range(len(episodes))]
        for column in episodes.columns:
            if column.startswith("stats/task_index/") and not column.endswith("/count"):
                episodes[column] = episodes[column].map(_replace_with_zero)
        episodes.to_parquet(episode_file, index=False)

    info_path = meta_dir / "info.json"
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    info["total_tasks"] = 1
    with info_path.open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=4, ensure_ascii=False)
        file.write("\n")


def create_view(source_root: Path, view_root: Path, global_task: str) -> None:
    source_root = source_root.resolve()
    view_root = view_root.resolve()
    if source_root == view_root:
        raise ValueError("Source and view roots must differ")
    if not (source_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Invalid LeRobot source dataset: {source_root}")

    if view_root.exists():
        marker = view_root / MARKER_NAME
        if not marker.is_file():
            raise FileExistsError(
                f"Refusing to replace unmarked directory: {view_root}"
            )
        shutil.rmtree(view_root)

    view_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{view_root.name}.", dir=view_root.parent)
    )
    try:
        shutil.copytree(source_root / "meta", temp_root / "meta")

        source_data = source_root / "data"
        for source_file in sorted(source_data.glob("chunk-*/file-*.parquet")):
            relative_path = source_file.relative_to(source_root)
            destination = temp_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            frames = pd.read_parquet(source_file)
            frames["task_index"] = np.zeros(len(frames), dtype=np.int64)
            frames.to_parquet(destination, index=False)

        for directory_name in ("videos", "images"):
            source_directory = source_root / directory_name
            if source_directory.exists():
                os.symlink(source_directory, temp_root / directory_name)

        _write_global_tasks(temp_root / "meta", global_task)
        marker = {
            "source_dataset": str(source_root),
            "global_task": global_task,
            "purpose": "standard Pi0.5 baseline without subtask-label leakage",
        }
        with (temp_root / MARKER_NAME).open("w", encoding="utf-8") as file:
            json.dump(marker, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(temp_root, view_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--global-task", default=DEFAULT_GLOBAL_TASK)
    args = parser.parse_args()

    create_view(args.source_root, args.view_root, args.global_task)
    print(f"Global-task dataset view ready: {args.view_root.resolve()}")


if __name__ == "__main__":
    main()

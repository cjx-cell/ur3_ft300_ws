#!/usr/bin/env python3
"""Recompute exact dataset-wide numeric statistics for a LeRobot v3 dataset.

LeRobot currently aggregates per-episode quantiles by averaging them.  An
average of quantiles is not the quantile of the union and becomes badly biased
when episodes cover different positions or trajectory styles.  This utility
reads the finalized parquet data and replaces floating-point feature statistics
with exact dataset-wide values.  It also computes mean/std in float64 so nearly
constant robot joints do not suffer float32 cancellation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def _load_float_feature(parquet_files: list[Path], feature: str) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    frame_count = 0
    for path in parquet_files:
        table = pq.read_table(path, columns=[feature])
        values = np.asarray(table[feature].to_pylist(), dtype=np.float64)
        frame_count += len(values)
        if values.ndim == 1:
            values = values[:, None]
        # LeRobot vector statistics reduce every leading dimension and retain
        # the final feature dimension (for example [N, 64, 6] -> [N*64, 6]).
        chunks.append(values.reshape(-1, values.shape[-1]))
    if not chunks:
        raise ValueError(f"No values found for feature {feature!r}")
    return np.concatenate(chunks, axis=0), frame_count


def _exact_stats(values: np.ndarray, frame_count: int) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "count": [float(frame_count)],
    }
    for quantile in QUANTILES:
        result[f"q{int(quantile * 100):02d}"] = np.quantile(
            values, quantile, axis=0
        ).tolist()
    return result


def recompute_numeric_stats(dataset_root: str | Path) -> dict[str, object]:
    root = Path(dataset_root).resolve()
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    parquet_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not info_path.is_file() or not stats_path.is_file() or not parquet_files:
        raise FileNotFoundError(f"Incomplete LeRobot dataset: {root}")

    info = json.loads(info_path.read_text())
    stats = json.loads(stats_path.read_text())
    parquet_columns = set(pq.read_schema(parquet_files[0]).names)
    repaired: dict[str, object] = {}

    for feature, spec in info["features"].items():
        dtype = str(spec.get("dtype", ""))
        if feature not in parquet_columns or not dtype.startswith("float"):
            continue
        values, frame_count = _load_float_feature(parquet_files, feature)
        exact = _exact_stats(values, frame_count)
        stats[feature] = exact
        repaired[feature] = {
            "frames": frame_count,
            "samples": int(len(values)),
            "shape": list(values.shape[1:]),
        }

    backup_path = stats_path.with_name("stats.json.episode_aggregate_backup")
    if not backup_path.exists():
        shutil.copy2(stats_path, backup_path)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    receipt = {
        "schema": "lerobot_exact_global_numeric_stats_v1",
        "dataset_root": str(root),
        "parquet_files": len(parquet_files),
        "quantiles": list(QUANTILES),
        "numeric_features": repaired,
        "backup": str(backup_path),
    }
    receipt_path = root / "meta" / "exact_global_stats_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _hardlink_view(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    shutil.copytree(source, output, copy_function=os.link)
    # Metadata must not share inodes with the immutable source view.
    for path in (output / "meta").rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            path.unlink()
            path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Create a hard-linked dataset view and repair that view only.",
    )
    args = parser.parse_args()

    target = args.dataset_root.resolve()
    if args.output is not None:
        target = args.output.resolve()
        _hardlink_view(args.dataset_root.resolve(), target)
    receipt = recompute_numeric_stats(target)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

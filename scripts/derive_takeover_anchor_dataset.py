#!/usr/bin/env python3
"""Derive takeover-entry sampling weights without re-encoding dataset videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


WEIGHT_KEY = "d1.sample_weight"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/home/ubuntu/ur3_ft300_ws"))
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.weight <= 0 or args.window <= 0:
        raise ValueError("weight and window must be positive")

    receipt_path = args.source / "meta/d1_materialization.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    recovery_assets = {
        entry["episode_id"]: args.workspace / entry["asset"]["path"]
        for entry in manifest["recovery_episodes"]
    }
    recovery_episodes = {
        int(entry["dataset_episode_index"]): entry["source_id"]
        for entry in receipt["episodes"]
        if entry["source_id"] in recovery_assets
    }

    # Hard-link immutable payloads (especially videos), then atomically replace
    # every file that this derivation changes.  The source dataset is untouched.
    shutil.copytree(args.source, args.output, copy_function=os.link)
    parquet_paths = sorted((args.output / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError("No parquet files found in source dataset")

    total_anchor_frames = 0
    newly_elevated_frames = 0
    weight_sum_before = 0.0
    weight_sum_after = 0.0
    takeover_count = 0
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        episode_indices = table["episode_index"].to_numpy()
        frame_indices = table["frame_index"].to_numpy()
        weights = table[WEIGHT_KEY].to_numpy().astype(np.float32, copy=True)
        weight_sum_before += float(weights.sum())

        for episode_index, source_id in recovery_episodes.items():
            row_mask = episode_indices == episode_index
            row_count = int(row_mask.sum())
            if row_count == 0:
                continue
            with np.load(recovery_assets[source_id], allow_pickle=False) as raw:
                intervention = np.asarray(raw["intervention_mask"], dtype=bool)
            if len(intervention) != row_count:
                raise ValueError(
                    f"Episode {episode_index} frame mismatch: raw={len(intervention)}, parquet={row_count}"
                )
            previous = np.concatenate([np.asarray([False]), intervention[:-1]])
            starts = np.flatnonzero(intervention & ~previous)
            takeover_count += len(starts)
            anchor = np.zeros_like(intervention)
            for start in starts:
                anchor[start : min(len(anchor), int(start) + args.window)] = True
            anchor &= intervention

            episode_rows = np.flatnonzero(row_mask)
            ordered = episode_rows[np.argsort(frame_indices[row_mask])]
            target_rows = ordered[anchor]
            total_anchor_frames += len(target_rows)
            newly_elevated_frames += int((weights[target_rows] < args.weight).sum())
            weights[target_rows] = np.maximum(weights[target_rows], args.weight)

        weight_sum_after += float(weights.sum())
        replacement = pa.array(weights, type=pa.float32())
        column_index = table.schema.get_field_index(WEIGHT_KEY)
        updated = table.set_column(column_index, WEIGHT_KEY, replacement)
        temporary = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(updated, temporary, compression="zstd")
        os.replace(temporary, parquet_path)

    receipt["recovery_takeover_anchor_frames"] = total_anchor_frames
    receipt["recovery_takeover_anchor_contract"] = {
        "weight": args.weight,
        "expert_frames_after_each_takeover": args.window,
        "source": "logged intervention-mask transition only",
        "scope": "offline sampling metadata only; never a policy input",
    }
    receipt["takeover_anchor_derivation"] = {
        "source_dataset": str(args.source.resolve()),
        "source_receipt_sha256": _sha256(receipt_path),
        "manifest": str(args.manifest.resolve()),
        "takeovers": takeover_count,
        "anchor_frames": total_anchor_frames,
        "newly_elevated_frames": newly_elevated_frames,
        "weight_sum_before": weight_sum_before,
        "weight_sum_after": weight_sum_after,
        "videos_reused_without_reencoding": True,
    }
    output_receipt = args.output / "meta/d1_materialization.json"
    temporary_receipt = output_receipt.with_suffix(".json.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_receipt, output_receipt)
    print(json.dumps(receipt["takeover_anchor_derivation"], indent=2), flush=True)


if __name__ == "__main__":
    main()

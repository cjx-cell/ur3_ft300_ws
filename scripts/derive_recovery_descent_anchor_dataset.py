#!/usr/bin/env python3
"""Boost short physical-descent windows in every validated recovery episode.

The source LeRobot dataset is copied with hard links. Only the parquet sample
weights and the materialization receipt are replaced; videos and actions are
left byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


WEIGHT_KEY = "d1.sample_weight"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/home/ubuntu/ur3_ft300_ws"))
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--xy-m", type=float, default=0.0005)
    parser.add_argument("--min-drop-m", type=float, default=0.0005)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.weight <= 0 or args.window <= 0 or args.xy_m <= 0 or args.min_drop_m <= 0:
        raise ValueError("weight, window, xy-m and min-drop-m must be positive")

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

    shutil.copytree(args.source, args.output, copy_function=os.link)
    summaries: dict[int, dict[str, object]] = {}
    total_anchor_frames = 0
    for parquet_path in sorted((args.output / "data").glob("*/*.parquet")):
        table = pq.read_table(parquet_path)
        episode_indices = table["episode_index"].to_numpy()
        frame_indices = table["frame_index"].to_numpy()
        weights = table[WEIGHT_KEY].to_numpy().astype(np.float32, copy=True)

        for episode_index, source_id in recovery_episodes.items():
            row_mask = episode_indices == episode_index
            if not np.any(row_mask):
                continue
            episode_rows = np.flatnonzero(row_mask)
            ordered = episode_rows[np.argsort(frame_indices[row_mask])]
            with np.load(recovery_assets[source_id], allow_pickle=False) as raw:
                intervention = np.asarray(raw["intervention_mask"], dtype=bool)
                peg = np.asarray(raw["peg_position"], dtype=np.float64)
                hole = np.asarray(raw["hole_position"], dtype=np.float64)
            if len(intervention) != len(ordered):
                raise ValueError(
                    f"Episode {episode_index} frame mismatch: raw={len(intervention)}, "
                    f"parquet={len(ordered)}"
                )
            previous = np.concatenate([np.asarray([False]), intervention[:-1]])
            starts = np.flatnonzero(intervention & ~previous)
            xy = np.linalg.norm(peg[:, :2] - hole[:, :2], axis=1)
            anchor = np.zeros_like(intervention)
            descent_starts: list[int] = []
            indices = np.arange(len(intervention))
            for start in starts:
                # Stay inside this contiguous expert intervention. A grasp or
                # transport intervention with no concentric descent contributes
                # no anchor and therefore receives no accidental stage boost.
                end_candidates = np.flatnonzero(~intervention[int(start) :])
                end = len(intervention) if len(end_candidates) == 0 else int(start) + int(end_candidates[0])
                candidates = np.flatnonzero(
                    intervention
                    & (indices >= int(start))
                    & (indices < end)
                    & (xy <= args.xy_m)
                    & (peg[:, 2] <= peg[int(start), 2] - args.min_drop_m)
                )
                if len(candidates) == 0:
                    continue
                descent_start = int(candidates[0])
                descent_starts.append(descent_start)
                anchor[descent_start : min(end, descent_start + args.window)] = True
            anchor &= intervention
            # Policy context remains zero weight. Only already-positive expert
            # labels can be promoted by this derivation.
            positive_anchor = anchor & (weights[ordered] > 0)
            weights[ordered[positive_anchor]] = np.maximum(
                weights[ordered[positive_anchor]], args.weight
            )
            count = int(positive_anchor.sum())
            total_anchor_frames += count
            summaries[episode_index] = {
                "source_id": source_id,
                "interventions": int(len(starts)),
                "descent_starts": descent_starts,
                "anchor_frames": count,
            }

        updated = table.set_column(
            table.schema.get_field_index(WEIGHT_KEY),
            WEIGHT_KEY,
            pa.array(weights, type=pa.float32()),
        )
        temporary = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(updated, temporary, compression="zstd")
        os.replace(temporary, parquet_path)

    receipt["recovery_descent_anchor_derivation"] = {
        "source_dataset": str(args.source.resolve()),
        "weight": args.weight,
        "window": args.window,
        "xy_m": args.xy_m,
        "min_drop_m": args.min_drop_m,
        "total_anchor_frames": total_anchor_frames,
        "episodes": summaries,
        "policy_failure_weights_unchanged": True,
        "videos_reused_without_reencoding": True,
    }
    output_receipt = args.output / "meta/d1_materialization.json"
    temporary_receipt = output_receipt.with_suffix(".json.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_receipt, output_receipt)
    print(json.dumps(receipt["recovery_descent_anchor_derivation"], indent=2))


if __name__ == "__main__":
    main()

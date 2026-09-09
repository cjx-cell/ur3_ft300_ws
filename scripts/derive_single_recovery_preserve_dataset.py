#!/usr/bin/env python3
"""Keep one recovery episode and preserve its correct pre-takeover rollout.

The source dataset is copied with hard links.  Only parquet sampling weights
and the materialization receipt are replaced; videos are never re-encoded.
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
    parser.add_argument("--keep-source-id", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/home/ubuntu/ur3_ft300_ws"))
    parser.add_argument("--preserve-prefix-weight", type=float, default=1.0)
    parser.add_argument("--takeover-weight", type=float, default=10.0)
    parser.add_argument("--takeover-window", type=int, default=20)
    parser.add_argument("--descent-anchor-weight", type=float, default=None)
    parser.add_argument("--descent-anchor-window", type=int, default=20)
    parser.add_argument("--descent-anchor-xy-m", type=float, default=0.0005)
    parser.add_argument("--descent-anchor-min-drop-m", type=float, default=0.0005)
    parser.add_argument(
        "--source-policy-adapter",
        type=Path,
        default=None,
        help="Optional rollout LoRA adapter used together with the base checkpoint.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.preserve_prefix_weight <= 0 or args.takeover_weight <= 0:
        raise ValueError("weights must be positive")
    if args.takeover_window <= 0:
        raise ValueError("takeover-window must be positive")
    if args.descent_anchor_weight is not None and args.descent_anchor_weight <= 0:
        raise ValueError("descent-anchor-weight must be positive")
    if args.descent_anchor_window <= 0:
        raise ValueError("descent-anchor-window must be positive")

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
    selected = [index for index, source_id in recovery_episodes.items() if source_id == args.keep_source_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected recovery episode, got {selected}")
    selected_episode = selected[0]

    shutil.copytree(args.source, args.output, copy_function=os.link)
    summaries: dict[int, dict[str, float | int | str]] = {}
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
            if episode_index != selected_episode:
                weights[ordered] = 0.0
                summaries[episode_index] = {
                    "source_id": source_id,
                    "frames": len(ordered),
                    "weight_sum": 0.0,
                }
                continue

            with np.load(recovery_assets[source_id], allow_pickle=False) as raw:
                intervention = np.asarray(raw["intervention_mask"], dtype=bool)
                peg_position = np.asarray(raw["peg_position"], dtype=np.float64)
                hole_position = np.asarray(raw["hole_position"], dtype=np.float64)
            if len(intervention) != len(ordered):
                raise ValueError(
                    f"Episode {episode_index} frame mismatch: raw={len(intervention)}, parquet={len(ordered)}"
                )
            previous = np.concatenate([np.asarray([False]), intervention[:-1]])
            starts = np.flatnonzero(intervention & ~previous)
            if len(starts) == 0:
                raise ValueError(f"Selected recovery episode {episode_index} has no takeover")

            # The selected rollout is known to complete grasp and transport
            # before its first true alignment failure.  Replaying its actual
            # model actions prevents recovery tuning from overwriting that
            # already-correct behavior.
            prefix_end = int(starts[0])
            weights[ordered[:prefix_end]] = np.maximum(
                weights[ordered[:prefix_end]], args.preserve_prefix_weight
            )
            anchor = np.zeros_like(intervention)
            for start in starts:
                anchor[start : min(len(anchor), int(start) + args.takeover_window)] = True
            anchor &= intervention
            weights[ordered[anchor]] = np.maximum(weights[ordered[anchor]], args.takeover_weight)
            descent_anchor = np.zeros_like(intervention)
            descent_starts: list[int] = []
            if args.descent_anchor_weight is not None:
                peg_hole_xy = np.linalg.norm(
                    peg_position[:, :2] - hole_position[:, :2], axis=1
                )
                for start in starts:
                    candidates = np.flatnonzero(
                        intervention
                        & (np.arange(len(intervention)) >= int(start))
                        & (peg_hole_xy <= args.descent_anchor_xy_m)
                        & (
                            peg_position[:, 2]
                            <= peg_position[int(start), 2]
                            - args.descent_anchor_min_drop_m
                        )
                    )
                    if len(candidates) == 0:
                        raise ValueError(
                            f"Selected recovery episode {episode_index} has no physical descent anchor"
                        )
                    descent_start = int(candidates[0])
                    descent_starts.append(descent_start)
                    descent_anchor[
                        descent_start : min(
                            len(descent_anchor), descent_start + args.descent_anchor_window
                        )
                    ] = True
                descent_anchor &= intervention
                weights[ordered[descent_anchor]] = np.maximum(
                    weights[ordered[descent_anchor]], args.descent_anchor_weight
                )
            summaries[episode_index] = {
                "source_id": source_id,
                "frames": len(ordered),
                "correct_policy_prefix_frames": prefix_end,
                "takeovers": len(starts),
                "takeover_anchor_frames": int(anchor.sum()),
                "descent_anchor_starts": descent_starts,
                "descent_anchor_frames": int(descent_anchor.sum()),
                "weight_sum": float(weights[ordered].sum()),
            }

        replacement = pa.array(weights, type=pa.float32())
        updated = table.set_column(table.schema.get_field_index(WEIGHT_KEY), WEIGHT_KEY, replacement)
        temporary = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(updated, temporary, compression="zstd")
        os.replace(temporary, parquet_path)

    receipt["single_recovery_preservation_derivation"] = {
        "source_dataset": str(args.source.resolve()),
        "kept_source_id": args.keep_source_id,
        "kept_dataset_episode_index": selected_episode,
        "preserve_prefix_weight": args.preserve_prefix_weight,
        "takeover_weight": args.takeover_weight,
        "takeover_window": args.takeover_window,
        "descent_anchor_weight": args.descent_anchor_weight,
        "descent_anchor_window": args.descent_anchor_window,
        "descent_anchor_xy_m": args.descent_anchor_xy_m,
        "descent_anchor_min_drop_m": args.descent_anchor_min_drop_m,
        "excluded_recovery_episode_indices": sorted(set(recovery_episodes) - {selected_episode}),
        "episodes": summaries,
        "videos_reused_without_reencoding": True,
    }
    if args.source_policy_adapter is not None:
        receipt["single_recovery_preservation_derivation"]["source_policy_adapter"] = str(
            args.source_policy_adapter.resolve()
        )
    output_receipt = args.output / "meta/d1_materialization.json"
    temporary_receipt = output_receipt.with_suffix(".json.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_receipt, output_receipt)
    print(json.dumps(receipt["single_recovery_preservation_derivation"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pair takeover observations with expert chunks after planning-delay frames.

Recovery recording includes real wall-clock expert setup.  When that setup is
longer than the deployment execution horizon, ordinary chunk training teaches
the policy to repeat the static prefix forever.  This derivation keeps the
actual takeover observation but shifts the first genuinely moving expert chunk
to that frame.  It is offline supervision only and uses no policy-time oracle.
"""

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
    parser.add_argument("--weight", type=float, default=64.0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--moving-threshold-rad", type=float, default=0.01)
    parser.add_argument(
        "--only-source-id",
        action="append",
        default=[],
        help="Restrict correction derivation to these recovery source IDs.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.weight <= 0 or args.chunk_size <= 0 or args.execution_horizon <= 0:
        raise ValueError("weight, chunk-size and execution-horizon must be positive")

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
        and (
            not args.only_source_id
            or entry["source_id"] in set(args.only_source_id)
        )
    }
    if args.only_source_id and not recovery_episodes:
        raise ValueError("None of --only-source-id matched a materialized recovery")

    shutil.copytree(args.source, args.output, copy_function=os.link)
    corrections: list[dict[str, object]] = []
    for parquet_path in sorted((args.output / "data").glob("*/*.parquet")):
        table = pq.read_table(parquet_path)
        episode_indices = table["episode_index"].to_numpy()
        frame_indices = table["frame_index"].to_numpy()
        weights = table[WEIGHT_KEY].to_numpy().astype(np.float32, copy=True)
        actions = np.stack(table["action"].to_pylist()).astype(np.float32)

        for episode_index, source_id in recovery_episodes.items():
            episode_rows = np.flatnonzero(episode_indices == episode_index)
            if not len(episode_rows):
                continue
            ordered = episode_rows[np.argsort(frame_indices[episode_rows])]
            with np.load(recovery_assets[source_id], allow_pickle=False) as raw:
                intervention = np.asarray(raw["intervention_mask"], dtype=bool)
                raw_state = np.asarray(raw["state"], dtype=np.float32)
                raw_action = np.asarray(raw["executed_action"], dtype=np.float32)
            if len(intervention) != len(ordered):
                raise ValueError(f"Episode {episode_index} raw/parquet frame mismatch")
            previous = np.concatenate([np.asarray([False]), intervention[:-1]])
            for start in np.flatnonzero(intervention & ~previous):
                segment_end = start
                while segment_end < len(intervention) and intervention[segment_end]:
                    segment_end += 1
                arm_distance = np.max(
                    np.abs(raw_action[start:segment_end, :6] - raw_state[start, :6]), axis=1
                )
                moving_offsets = np.flatnonzero(arm_distance > args.moving_threshold_rad)
                if not len(moving_offsets):
                    continue
                delay = int(moving_offsets[0])
                # A short prefix is naturally consumed by deployment.  Only
                # correct prefixes that would be replanned before motion starts.
                if delay < args.execution_horizon:
                    continue
                moving_start = int(start) + delay
                if moving_start + args.chunk_size > segment_end:
                    continue

                source_rows = ordered[moving_start : moving_start + args.chunk_size]
                target_rows = ordered[start : start + args.chunk_size]
                actions[target_rows] = actions[source_rows]
                weights[target_rows] = 0.0
                weights[target_rows[0]] = args.weight
                corrections.append(
                    {
                        "dataset_episode_index": episode_index,
                        "source_id": source_id,
                        "takeover_frame": int(start),
                        "moving_frame": moving_start,
                        "static_delay_frames": delay,
                        "target_chunk_frames": args.chunk_size,
                        "sample_weight": args.weight,
                    }
                )

        weight_index = table.schema.get_field_index(WEIGHT_KEY)
        action_index = table.schema.get_field_index("action")
        updated = table.set_column(weight_index, WEIGHT_KEY, pa.array(weights, type=pa.float32()))
        action_type = table.schema.field(action_index).type
        action_values = pa.array(actions.reshape(-1), type=pa.float32())
        action_column = pa.FixedSizeListArray.from_arrays(action_values, action_type.list_size)
        updated = updated.set_column(action_index, "action", action_column)
        temporary = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(updated, temporary, compression="zstd")
        os.replace(temporary, parquet_path)

    if not corrections:
        raise RuntimeError("No takeover had a static prefix longer than the execution horizon")
    receipt["shifted_takeover_correction_contract"] = {
        "source_dataset": str(args.source.resolve()),
        "source_receipt_sha256": _sha256(receipt_path),
        "manifest": str(args.manifest.resolve()),
        "execution_horizon": args.execution_horizon,
        "moving_threshold_rad": args.moving_threshold_rad,
        "chunk_size": args.chunk_size,
        "policy_input": "recorded takeover observation only",
        "target": "first expert chunk after the recorded static planning prefix",
        "runtime_oracle": False,
        "only_source_ids": list(args.only_source_id),
        "corrections": corrections,
    }
    output_receipt = args.output / "meta/d1_materialization.json"
    temporary_receipt = output_receipt.with_suffix(".json.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_receipt, output_receipt)
    print(json.dumps(receipt["shifted_takeover_correction_contract"], indent=2), flush=True)


if __name__ == "__main__":
    main()

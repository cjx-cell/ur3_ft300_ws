#!/usr/bin/env python3
"""Create a new immutable D1 manifest by appending validated v4 recoveries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from pap_moe_framework.rollout_recovery.schema import load_and_validate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_digest(manifest: dict) -> str:
    content = dict(manifest)
    content.pop("frozen_at_utc", None)
    content.pop("manifest_sha256", None)
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--sample-weight", type=float, default=1.0)
    parser.add_argument(
        "--drop-existing-recoveries",
        action="store_true",
        help="Keep the base success episodes but replace its recovery curriculum.",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    manifest = json.loads(args.base.read_text(encoding="utf-8"))
    manifest["dataset_id"] = args.dataset_id
    manifest["supersedes"] = str(args.base.resolve())
    manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()

    if args.drop_existing_recoveries:
        old_recovery_ids = {
            entry["episode_id"] for entry in manifest["recovery_episodes"]
        }
        manifest["recovery_episodes"] = []
        manifest["split_episode_ids"]["train"] = [
            episode_id
            for episode_id in manifest["split_episode_ids"]["train"]
            if episode_id not in old_recovery_ids
        ]
        manifest["split_episode_ids"]["validation"] = [
            episode_id
            for episode_id in manifest["split_episode_ids"]["validation"]
            if episode_id not in old_recovery_ids
        ]
        manifest["recovery_checkpoint_lineage"] = []

    existing_ids = {entry["episode_id"] for entry in manifest["recovery_episodes"]}
    for recovery_path in args.recovery:
        path = recovery_path.resolve()
        summary = load_and_validate(path)
        with np.load(path, allow_pickle=False) as raw:
            schema_version = str(np.asarray(raw["schema_version"]).item())
            if schema_version != "pap_moe_rollout_recovery_v4_multi_handoff_full_episode":
                raise ValueError(f"{path}: expected v4 full episode, got {schema_version}")
            episode_name = path.parent.name
            episode_id = f"recovery:{episode_name}"
            if episode_id in existing_ids:
                raise ValueError(f"duplicate recovery episode: {episode_id}")
            source_checkpoint = str(np.asarray(raw["source_policy_checkpoint"]).item())
            recovery_phase = str(np.asarray(raw["recovery_phase"]).item())
            intervention = np.asarray(raw["intervention_mask"], dtype=bool)
            frames = len(intervention)
            expert_frames = int(intervention.sum())
            policy_frames = frames - expert_frames

        try:
            relative_path = path.relative_to(workspace).as_posix()
        except ValueError as error:
            raise ValueError(f"recovery must be inside workspace: {path}") from error
        entry = {
            "episode_id": episode_id,
            "kind": "full_modal_rollout_recovery_episode",
            "recovery_phase": recovery_phase,
            "split": "train",
            "frames": frames,
            "expert_frames": expert_frames,
            "policy_frames": policy_frames,
            "materialize_from_frame": 0,
            "action_source": "executed_action",
            "positive_weight_scope": "expert_frames",
            "schema_version": schema_version,
            "sample_weight": args.sample_weight,
            "independent_recovery_episodes": 1,
            "asset": {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            },
            "modality_validity": "all_native_full_modal_hard_gate",
            "source_policy_checkpoint": source_checkpoint,
        }
        if not summary.recovery_success:
            raise ValueError(f"{path}: recovery did not end in success")
        manifest["recovery_episodes"].append(entry)
        manifest["split_episode_ids"]["train"].append(episode_id)
        existing_ids.add(episode_id)
        if source_checkpoint not in manifest["recovery_checkpoint_lineage"]:
            manifest["recovery_checkpoint_lineage"].append(source_checkpoint)

    counts = manifest["counts"]
    counts["formal_recovery_episodes"] = len(manifest["recovery_episodes"])
    counts["independent_recovery_episodes"] = sum(
        int(entry.get("independent_recovery_episodes", 1))
        for entry in manifest["recovery_episodes"]
    )
    counts["train_episode_groups"] = len(manifest["split_episode_ids"]["train"])
    counts["validation_episode_groups"] = len(
        manifest["split_episode_ids"]["validation"]
    )
    manifest["manifest_sha256"] = _manifest_digest(manifest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "manifest_sha256": manifest["manifest_sha256"],
        "recovery_episodes": counts["formal_recovery_episodes"],
        "train_episode_groups": counts["train_episode_groups"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

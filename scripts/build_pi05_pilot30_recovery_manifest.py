#!/usr/bin/env python3
"""Freeze the current pilot-30 successes plus validated Pi0.5 recoveries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    return str(np.asarray(data[key]).item())


def asset(path: Path, workspace: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve().relative_to(workspace)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/home/ubuntu/ur3_ft300_ws"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/ubuntu/ur3_ft300_ws/outputs/train/"
            "pi05_v9_absolute_1000step_20260818_233017/checkpoints/010000/pretrained_model"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/manifests/"
            "pi05_pilot30_plus_recovery_v1.json"
        ),
    )
    parser.add_argument(
        "--include-checkpoint",
        action="append",
        type=Path,
        default=[],
        help=(
            "Also include validated recoveries collected by an ancestor checkpoint. "
            "May be repeated to build a cumulative DAgger dataset."
        ),
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        help=(
            "Extend an existing frozen cumulative manifest. Its success and recovery "
            "episodes are retained before explicitly selected new recoveries are added."
        ),
    )
    parser.add_argument(
        "--recovery-episode",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit validated recovery data.npz allowlist. When provided, no other "
            "source-checkpoint-matched recovery is included. May be repeated."
        ),
    )
    parser.add_argument(
        "--recovery-sample-weight",
        type=float,
        default=1.0,
        help="Sampling weight assigned to every explicitly selected recovery episode.",
    )
    parser.add_argument(
        "--recovery-only",
        action="store_true",
        help=(
            "Build a compact manifest containing only the explicitly selected "
            "recoveries. This is used to materialize and then merge a new DAgger "
            "batch without re-encoding the frozen success videos."
        ),
    )
    args = parser.parse_args()
    if args.recovery_sample_weight <= 0.0:
        parser.error("--recovery-sample-weight must be positive")
    workspace = args.workspace.resolve()
    success_root = workspace / "pap_moe_framework/datasets/raw_v9_exactfit_scripted"
    legacy_recovery_root = (
        workspace
        / "pap_moe_framework/datasets/raw_rollout_recovery_v3_multitask_full_modalities"
    )
    full_episode_recovery_root = (
        workspace
        / "pap_moe_framework/datasets/raw_rollout_recovery_v4_multi_handoff_full_episode"
    )
    base_manifest = None
    if args.recovery_only:
        if not args.recovery_episode:
            parser.error("--recovery-only requires at least one --recovery-episode")
        success_entries = []
    elif args.base_manifest is not None:
        base_manifest = json.loads(args.base_manifest.resolve().read_text(encoding="utf-8"))
        if Path(base_manifest["workspace"]).resolve() != workspace:
            raise ValueError("base manifest belongs to a different workspace")
        success_entries = list(base_manifest["success_episodes"])
    else:
        successes = sorted(success_root.glob("*_episode_*_success/data.npz"))
        if len(successes) != 30:
            raise ValueError(f"expected exactly 30 pilot successes, found {len(successes)}")

        success_entries: list[dict[str, object]] = []
        for path in successes:
            with np.load(path, allow_pickle=False) as data:
                if scalar(data, "schema_version") not in {"pap_moe_v6", "pap_moe_v8"}:
                    raise ValueError(f"wrong success schema: {path}")
                frames = len(data["state"])
                match = re.search(r"episode_(\d+)_success", path.parent.name)
                if match is None:
                    raise ValueError(f"cannot parse success episode id: {path}")
                episode_id = int(match.group(1))
            success_entries.append(
                {
                    "episode_id": f"success:{episode_id:04d}",
                    "kind": "complete_success_episode",
                    "split": "train",
                    "frames": frames,
                    "sample_weight": 1.0,
                    "independent_recovery_episodes": 0,
                    "asset": asset(path, workspace),
                    "modality_validity": "all_native",
                }
            )

    expected_checkpoints = {
        str(args.checkpoint.resolve()),
        *(str(path.resolve()) for path in args.include_checkpoint),
    }
    recovery_paths = (
        sorted(path.resolve() for path in args.recovery_episode)
        if args.recovery_episode
        else sorted(legacy_recovery_root.glob("pi05_ep*_grasp_lift_*/data.npz"))
    )
    recovery_entries: list[dict[str, object]] = (
        []
        if args.recovery_only
        else list(base_manifest["recovery_episodes"])
        if base_manifest is not None
        else []
    )
    existing_recovery_ids = {str(entry["episode_id"]) for entry in recovery_entries}
    for path in recovery_paths:
        if not path.is_file():
            raise ValueError(f"explicit recovery episode is missing: {path}")
        if not any(
            root.resolve() in path.parents
            for root in (legacy_recovery_root, full_episode_recovery_root)
        ):
            raise ValueError(f"recovery episode is outside the canonical root: {path}")
        with np.load(path, allow_pickle=False) as data:
            schema_version = scalar(data, "schema_version")
            if schema_version not in {
                "pap_moe_rollout_recovery_v3_multitask_full_modalities",
                "pap_moe_rollout_recovery_v4_multi_handoff_full_episode",
            }:
                continue
            if scalar(data, "recovery_outcome") != "success":
                continue
            source_checkpoint = str(
                Path(scalar(data, "source_policy_checkpoint")).resolve()
            )
            if source_checkpoint not in expected_checkpoints:
                continue
            frames = len(data["state"])
            expert_frames = int(np.asarray(data["intervention_mask"], dtype=bool).sum())
            policy_frames = frames - expert_frames
            phase = scalar(data, "recovery_phase")
            full_episode = schema_version == "pap_moe_rollout_recovery_v4_multi_handoff_full_episode"
        new_entry = {
                "episode_id": f"recovery:{path.parent.name}",
                "kind": "full_modal_rollout_recovery_episode",
                "recovery_phase": phase,
                "split": "train",
                "frames": frames,
                "expert_frames": expert_frames,
                "policy_frames": policy_frames,
                "materialize_from_frame": 0 if full_episode else "first_takeover",
                "action_source": "executed_action" if full_episode else "expert_action",
                "positive_weight_scope": "expert_frames" if full_episode else "expert_suffix",
                "schema_version": schema_version,
                "sample_weight": args.recovery_sample_weight,
                "independent_recovery_episodes": 1,
                "asset": asset(path, workspace),
                "modality_validity": "all_native_full_modal_hard_gate",
                "source_policy_checkpoint": source_checkpoint,
            }
        if str(new_entry["episode_id"]) not in existing_recovery_ids:
            recovery_entries.append(new_entry)
            existing_recovery_ids.add(str(new_entry["episode_id"]))
    if not recovery_entries:
        raise ValueError("no validated source-matched Pi0.5 recovery was found")

    manifest: dict[str, object] = {
        "schema_version": "pap_moe_baseline_comparison_manifest_v4_full_modal_d2",
        "dataset_id": "pi05_cumulative_plus_v4_full_episode_recovery",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "purpose": "fixed-workspace pure Pi0.5 recovery adaptation with PAP-MoE-compatible modalities",
        "supersedes": None,
        "field_contract": {
            "action": "50x7 absolute joint targets; gripper physical radians 0.0 open to 0.8 full-close command",
            "state": "measured 7D joint positions in physical radians",
            "baseline_inputs": "two RGB images plus measured 7D state only",
            "pap_moe_auxiliary": "FT300 current/fast/slow, state history, visual quality, routing and progress supervision",
        },
        "derivation_contract": {
            "action_chunk": "current plus next 49 absolute targets within one episode; pad only at episode end",
            "recovery_actions": (
                "v4 full episodes preserve executed_action for every policy/expert frame; "
                "legacy v3 episodes retain expert_action after their monotonic takeover"
            ),
            "recovery_sampling": (
                "v4 stores every frame but only expert intervention frames have positive "
                "training weight; source-policy frames remain available for audit/context"
            ),
            "gripper": "preserve continuous physical command; contact determines measured aperture",
            "physics_gate_target": "computed only from observable modality statistics during materialization",
        },
        "sampling_contract": {
            "weights": {
                "complete_success_episode": 1.0,
                "full_modal_rollout_recovery_episode": args.recovery_sample_weight,
            },
            "episode_atomic_split": True,
            "random_frame_split_forbidden": True,
        },
        "recovery_checkpoint_lineage": sorted(
            expected_checkpoints
            | set(base_manifest.get("recovery_checkpoint_lineage", []) if base_manifest else [])
        ),
        "split_episode_ids": {
            "train": [entry["episode_id"] for entry in success_entries + recovery_entries],
            "validation": [],
        },
        "coverage_warnings": [
            "This is an L0 fixed-workspace adaptation set, not the final held-out benchmark.",
            "Legacy v3 recoveries contain only the expert suffix; v4 recoveries retain the full multi-handoff episode.",
        ],
        "counts": {
            "success_episodes": len(success_entries),
            "formal_recovery_episodes": len(recovery_entries),
            "qualified_correction_packs": 0,
            "qualified_correction_samples": 0,
            "matched_failure_state_samples": 0,
            "independent_recovery_episodes": len(recovery_entries),
            "train_episode_groups": len(success_entries) + len(recovery_entries),
            "validation_episode_groups": 0,
        },
        "success_episodes": success_entries,
        "recovery_episodes": recovery_entries,
        "qualified_corrections": [],
        "auxiliary_supervision": [],
        "qualification_reports": [],
        "excluded": [],
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["counts"], indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

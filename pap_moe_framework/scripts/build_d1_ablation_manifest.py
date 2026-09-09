#!/usr/bin/env python3
"""Freeze the episode-safe D1 manifest for Pi0.5/PAP-MoE comparison.

This script intentionally creates a small manifest rather than copying image
arrays.  It records immutable source fingerprints, episode lineage, split and
sampling contracts, and the exact rules used to derive the common multimodal
sample fields.  A future dataset reader must reject sources that no longer
match these fingerprints.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np


SCHEMA_VERSION = "pap_moe_baseline_comparison_manifest_v3"
DATASET_ID = "D1_24success_16004_16007_qualified_corrections_v3"
ACTION_CHUNK_SIZE = 50

# This is the exact source set used to build lerobot_v9_ground_train_24ep_absolute.
D0_EPISODES = (
    13001, 13002, 13003, 13004, 13005, 13006, 13008, 13009, 13010,
    14001, 14002, 14005, 14006, 14007, 14008, 14009, 14010, 14011,
    14014, 14016, 14017, 14018, 14019, 14020,
)

# Frozen, independent-episode validation split.  Recovery episodes remain in
# train because there is currently only one episode in each recovery phase;
# the manifest reports that recovery validation coverage is absent.
VALIDATION_EPISODES = frozenset({14018, 14019, 14020})

DEPLOYED_ADAPTER_ROOTS = (
    "pi05_v9_ep13001_left_lora_r16_round3_robust_v2_20260812",
    "pi05_v9_ep13001_descent_prefix10_stage_lora_r16_v2b_20260812",
    "pi05_v9_ep13001_lower_descent_final_stage_lora_r16_v1b_20260813",
    "pi05_v9_ep13001_stable_preclose_close_lora_local_action_out_r16_v2b_20260813",
    "pi05_v9_ep13001_postgrasp_raw_recovery_lora_local_action_out_r16_v4_20260813",
    "pi05_v9_ep13001_postgrasp_align_lora_local_action_out_r16_v6b_20260813",
    "pi05_v9_ep13001_contact_preinsert_lora_local_action_out_r16_v1_20260813",
)

# Data qualification and adapter deployment are deliberately different gates.
# The 16007 expert correction pack is an immutable extraction from the formal,
# schema-qualified recovery episode. Its v3b consumer passed the offline data
# gate even though that adapter family was later rejected for online rollout.
# Keep the safe expert data without treating the rejected model as deployed.
ADDITIONAL_QUALIFIED_DATA_REPORTS = (
    "pi05_v9_ep13001_contact_insertion_raw_recovery_lora_local_action_out_r16_v3b_20260813_report.json",
)

RAW_RECOVERIES = (
    ("16004", "align_precontact", "episode_16004_align"),
    ("16007", "insertion", "episode_16007_insertion"),
)

SUBTASK_NAMES = (
    "grasp the peg",
    "transport to the hole",
    "approach and align with the hole",
    "recover contact and relocate the hole",
    "insert the peg into the hole",
    "verify insertion success",
    "release the peg after verification",
    "retract and go back to home",
)

GRASP_PROGRESS_NAMES = (
    "approach the peg",
    "descend onto the peg",
    "stabilize over the peg",
    "close the gripper on the peg",
    "lift the grasped peg",
    "not grasping",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path, workspace: Path, *, content_hash: bool) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        portable_path = str(resolved.relative_to(workspace.resolve()))
    except ValueError:
        portable_path = str(resolved)
    result: dict[str, object] = {
        "path": portable_path,
        "bytes": resolved.stat().st_size,
    }
    if content_hash:
        result["sha256"] = _sha256(resolved)
    return result


def _npz_headers(path: Path) -> dict[str, dict[str, object]]:
    """Read NPY headers from an NPZ without decompressing array payloads."""
    headers: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in {(2, 0), (3, 0)}:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported NPY version {version} in {path}")
            headers[Path(member).stem] = {
                "shape": list(shape),
                "dtype": str(dtype),
            }
    return headers


def _require_shape(
    headers: dict[str, dict[str, object]], key: str, trailing: tuple[int, ...]
) -> int:
    if key not in headers:
        raise ValueError(f"missing {key!r}")
    shape = tuple(int(value) for value in headers[key]["shape"])
    if len(shape) != len(trailing) + 1 or shape[1:] != trailing:
        raise ValueError(f"{key} has shape {shape}, expected [T,{','.join(map(str, trailing))}]")
    return shape[0]


def _read_scalar(path: Path, key: str) -> str:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            raise ValueError(f"{path}: missing scalar {key}")
        value = np.asarray(archive[key])
        if value.ndim != 0:
            raise ValueError(f"{path}: {key} is not scalar")
        return str(value.item())


def _episode_number(path: Path) -> int:
    match = re.search(r"episode_(\d+)_", path.parent.name)
    if not match:
        raise ValueError(f"cannot parse episode number from {path}")
    return int(match.group(1))


def _collect_qualified_corrections(artifacts: Path) -> tuple[list[Path], list[Path]]:
    """Collect packs that passed either deployment or explicit data gates."""
    seen_adapters: set[str] = set()
    reports: list[Path] = []
    corrections: dict[str, Path] = {}

    def visit(adapter: str | Path) -> None:
        name = Path(adapter).name
        if name in seen_adapters:
            return
        seen_adapters.add(name)
        report = artifacts / f"{name}_report.json"
        if not report.is_file():
            raise FileNotFoundError(f"missing deployed-adapter report: {report}")
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("qualified") is not True or data.get("selected") is None:
            raise ValueError(f"deployed adapter is not qualified: {report}")
        reports.append(report)
        values = data.get("corrections") or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            path = artifacts / Path(value).name
            if not path.is_file():
                raise FileNotFoundError(path)
            corrections[path.name] = path
        initial = data.get("initial_adapter")
        if initial:
            visit(initial)

    for root in DEPLOYED_ADAPTER_ROOTS:
        visit(root)
    for report_name in ADDITIONAL_QUALIFIED_DATA_REPORTS:
        report = artifacts / report_name
        if not report.is_file():
            raise FileNotFoundError(report)
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("qualified") is not True or data.get("selected") is None:
            raise ValueError(f"explicit data qualification did not pass: {report}")
        reports.append(report)
        values = data.get("corrections") or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            path = artifacts / Path(value).name
            if not path.is_file():
                raise FileNotFoundError(path)
            corrections[path.name] = path
    return sorted(corrections.values()), sorted(reports)


def _sidecar_for(pack: Path) -> Path | None:
    sidecar = pack.with_suffix(".json")
    return sidecar if sidecar.is_file() else None


def _raw_parent_id(pack: Path) -> str | None:
    match = re.search(r"raw_recovery_(16004|16007)", pack.name)
    return None if match is None else f"recovery:{match.group(1)}"


def _correction_entry(
    pack: Path, workspace: Path, *, content_hash: bool
) -> dict[str, object]:
    headers = _npz_headers(pack)
    samples = _require_shape(headers, "state", (7,))
    for key, trailing in (
        ("camera0", (224, 224, 3)),
        ("camera1", (224, 224, 3)),
        ("target_action", (ACTION_CHUNK_SIZE, 7)),
    ):
        if _require_shape(headers, key, trailing) != samples:
            raise ValueError(f"{pack}: {key} length differs from state")

    parent = _raw_parent_id(pack)
    sidecar = _sidecar_for(pack)
    trace_dir = None
    anchor_episode = None
    uses_object_truth = False
    if sidecar is not None:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if int(metadata.get("samples", -1)) != samples:
            raise ValueError(f"{sidecar}: sample count disagrees with NPZ")
        trace_dir = Path(metadata["trace_dir"]).resolve()
        parent = f"rollout:{trace_dir.parent.name}"
        anchor_episode = _episode_number(Path(metadata["episode"]))
        uses_object_truth = bool(metadata.get("uses_online_object_truth", True))
        if uses_object_truth:
            raise ValueError(f"correction used online object truth: {sidecar}")
    if parent is None:
        raise ValueError(f"cannot establish parent episode lineage for {pack}")

    native_force = parent.startswith("recovery:")
    entry: dict[str, object] = {
        "source_id": f"correction:{pack.stem}",
        "kind": "qualified_correction_pack",
        "split": "train",
        "parent_episode_id": parent,
        "independent_recovery_episodes": 0,
        "samples": samples,
        "sample_weight": 4.0,
        "asset": _fingerprint(pack, workspace, content_hash=content_hash),
        "action_contract": "absolute_joint_target_chunk_50x7",
        "online_object_truth": False,
        "modality_validity": {
            "camera0": True,
            "camera1": True,
            "state": True,
            "action_chunk": True,
            "force_current": native_force,
            "force_fast": native_force,
            "force_slow": native_force,
            "state_history": native_force,
        },
        "missing_modality_policy": (
            "derive_from_parent_raw_frame"
            if native_force
            else "zero_fill_and_set_validity_false"
        ),
    }
    if sidecar is not None:
        entry["sidecar"] = _fingerprint(sidecar, workspace, content_hash=content_hash)
        entry["trace_dir"] = str(trace_dir.relative_to(workspace.resolve()))
        entry["anchor_episode"] = anchor_episode
    return entry


def build_manifest(
    workspace: Path,
    *,
    content_hash: bool = True,
    include_full_modal_recovery_v2: bool = False,
    include_qualified_corrections: bool = True,
    include_legacy_recoveries: bool = True,
    full_modal_recovery_prefix: str | None = None,
    full_modal_source_checkpoint: str | None = None,
) -> dict[str, object]:
    workspace = workspace.resolve()
    raw_root = workspace / "pap_moe_framework/datasets/raw_v6_admittance"
    recovery_root = workspace / "pap_moe_framework/datasets/raw_rollout_recovery_v1"
    artifacts = workspace / "artifacts"

    success_entries: list[dict[str, object]] = []
    for episode in D0_EPISODES:
        path = raw_root / (
            "pick_up_the_peg_and_insert_it_into_the_hole_"
            f"episode_{episode}_success/data.npz"
        )
        headers = _npz_headers(path)
        frames = _require_shape(headers, "state", (7,))
        for key, trailing in (
            ("action", (7,)),
            ("force", (6,)),
            ("force_fast", (64, 6)),
            ("force_slow", (50, 6)),
            ("state_history", (10, 7)),
            ("visual_quality", (4,)),
            ("camera0", (224, 224, 3)),
            ("camera1", (224, 224, 3)),
            ("stage", (4,)),
        ):
            if _require_shape(headers, key, trailing) != frames:
                raise ValueError(f"{path}: {key} length differs from state")
        success_entries.append(
            {
                "episode_id": f"success:{episode}",
                "kind": "complete_success_episode",
                "split": "validation" if episode in VALIDATION_EPISODES else "train",
                "frames": frames,
                "sample_weight": 1.0,
                "independent_recovery_episodes": 0,
                "asset": _fingerprint(path, workspace, content_hash=content_hash),
                "modality_validity": "all_native",
            }
        )

    recovery_entries: list[dict[str, object]] = []
    for episode_id, phase, directory in (
        RAW_RECOVERIES if include_legacy_recoveries else ()
    ):
        path = recovery_root / directory / "data.npz"
        if _read_scalar(path, "schema_version") != "pap_moe_rollout_recovery_v1":
            raise ValueError(f"{path}: wrong recovery schema")
        if _read_scalar(path, "recovery_outcome") != "success":
            raise ValueError(f"{path}: recovery is not successful")
        if _read_scalar(path, "recovery_phase") != phase:
            raise ValueError(f"{path}: recovery phase mismatch")
        headers = _npz_headers(path)
        frames = _require_shape(headers, "state", (7,))
        for key, trailing in (
            ("expert_action", (7,)),
            ("executed_action", (7,)),
            ("force", (6,)),
            ("camera0", (224, 224, 3)),
            ("camera1", (224, 224, 3)),
        ):
            if _require_shape(headers, key, trailing) != frames:
                raise ValueError(f"{path}: {key} length differs from state")
        recovery_entries.append(
            {
                "episode_id": f"recovery:{episode_id}",
                "kind": "formal_rollout_recovery_episode",
                "recovery_phase": phase,
                "split": "train",
                "frames": frames,
                "sample_weight": 4.0,
                "independent_recovery_episodes": 1,
                "asset": _fingerprint(path, workspace, content_hash=content_hash),
                "modality_validity": {
                    "camera0": True,
                    "camera1": True,
                    "state": True,
                    "force_current": True,
                    "force_fast": "derived_from_timestamped_episode_force",
                    "force_slow": "derived_from_timestamped_episode_force",
                    "state_history": "derived_from_timestamped_episode_state",
                    "action_chunk": "expert_only_after_takeover",
                },
            }
        )

    full_modal_paths: list[Path] = []
    if include_full_modal_recovery_v2:
        full_modal_roots = (
            workspace / "pap_moe_framework/datasets/raw_rollout_recovery_v2_full_modalities",
            workspace / "pap_moe_framework/datasets/raw_rollout_recovery_v3_multitask_full_modalities",
        )
        v2_paths = sorted(full_modal_roots[0].glob("*/data.npz")) if full_modal_roots[0].exists() else []
        v3_paths = sorted(full_modal_roots[1].glob("*/data.npz")) if full_modal_roots[1].exists() else []
        v3_episode_names = {path.parent.name for path in v3_paths}
        # A migrated v3 episode and its v2 source are one physical rollout,
        # never two independent training episodes. Prefer v3 by episode name.
        full_modal_paths = sorted(
            [path for path in v2_paths if path.parent.name not in v3_episode_names] + v3_paths
        )
        if full_modal_recovery_prefix is not None:
            full_modal_paths = [
                path
                for path in full_modal_paths
                if path.parent.name.startswith(full_modal_recovery_prefix)
            ]
        if full_modal_source_checkpoint is not None:
            expected_checkpoint = str(Path(full_modal_source_checkpoint).resolve())
            full_modal_paths = [
                path
                for path in full_modal_paths
                if str(Path(_read_scalar(path, "source_policy_checkpoint")).resolve())
                == expected_checkpoint
            ]
        for path in full_modal_paths:
            schema_version = _read_scalar(path, "schema_version")
            if schema_version not in {
                "pap_moe_rollout_recovery_v2_full_modalities",
                "pap_moe_rollout_recovery_v3_multitask_full_modalities",
            }:
                raise ValueError(f"{path}: wrong full-modality recovery schema")
            if _read_scalar(path, "recovery_outcome") != "success":
                raise ValueError(f"{path}: recovery is not successful")
            phase = _read_scalar(path, "recovery_phase")
            headers = _npz_headers(path)
            frames = _require_shape(headers, "state", (7,))
            for key, trailing in (
                ("expert_action", (7,)),
                ("executed_action", (7,)),
                ("force", (6,)),
                ("force_fast", (64, 6)),
                ("force_slow", (50, 6)),
                ("state_history", (10, 7)),
                ("visual_quality", (4,)),
                ("stage", (4,)),
                ("modality_validity", (7,)),
                ("camera0", (224, 224, 3)),
                ("camera1", (224, 224, 3)),
            ):
                if _require_shape(headers, key, trailing) != frames:
                    raise ValueError(f"{path}: {key} length differs from state")
            if schema_version.endswith("v3_multitask_full_modalities"):
                for key in (
                    "skill_progress_phase", "skill_progress", "transition_readiness",
                    "skill_progress_valid", "skill_progress_confidence",
                ):
                    if _require_shape(headers, key, ()) != frames:
                        raise ValueError(f"{path}: {key} length differs from state")
            recovery_entries.append(
                {
                    "episode_id": f"recovery:{path.parent.name}",
                    "kind": "full_modal_rollout_recovery_episode",
                    "recovery_phase": phase,
                    "split": "train",
                    "frames": frames,
                    "sample_weight": 4.0,
                    "independent_recovery_episodes": 1,
                    "asset": _fingerprint(path, workspace, content_hash=content_hash),
                    "modality_validity": "all_native_full_modal_hard_gate",
                }
            )

    correction_paths, report_paths = _collect_qualified_corrections(artifacts)
    if not include_qualified_corrections:
        correction_paths = []
        report_paths = []
    corrections = [
        _correction_entry(path, workspace, content_hash=content_hash)
        for path in correction_paths
    ]

    # 074701 has neither FT300 history nor an executed expert recovery. Keep it
    # outside authoritative D1; it may only be studied in a separately named
    # pseudo-label ablation.
    auxiliary: list[dict[str, object]] = []

    split_episode_ids = {
        split: sorted(
            entry["episode_id"]
            for entry in [*success_entries, *recovery_entries]
            if entry["split"] == split
        )
        for split in ("train", "validation")
    }
    if set(split_episode_ids["train"]) & set(split_episode_ids["validation"]):
        raise AssertionError("episode leakage across train and validation")
    independent_recoveries = sum(
        int(entry["independent_recovery_episodes"])
        for entry in [*recovery_entries, *corrections, *auxiliary]
    )
    expected_recoveries = (
        len(RAW_RECOVERIES) if include_legacy_recoveries else 0
    ) + (len(full_modal_paths) if include_full_modal_recovery_v2 else 0)
    if independent_recoveries != expected_recoveries:
        raise AssertionError(
            f"manifest must contain {expected_recoveries} independent recoveries, "
            f"got {independent_recoveries}"
        )

    field_contract = {
        "observation.images.camera0": {"dtype": "uint8", "shape": [224, 224, 3]},
        "observation.images.camera1": {"dtype": "uint8", "shape": [224, 224, 3]},
        "observation.state": {"dtype": "float32", "shape": [7]},
        "observation.state_history": {"dtype": "float32", "shape": [10, 7]},
        "observation.force": {"dtype": "float32", "shape": [6]},
        "observation.force_fast": {"dtype": "float32", "shape": [64, 6]},
        "observation.force_slow": {"dtype": "float32", "shape": [50, 6]},
        "observation.visual_quality": {"dtype": "float32", "shape": [4]},
        "observation.physics_gate_target": {"dtype": "float32", "shape": [4]},
        "subtask_label": {"dtype": "int64", "classes": list(SUBTASK_NAMES)},
        "skill_progress_phase": {
            "dtype": "int64",
            "classes": ["enter", "approach", "align", "interact", "stabilize", "verify", "exit", "recover"],
        },
        "skill_progress": {"dtype": "float32", "range": [0.0, 1.0]},
        "transition_readiness": {"dtype": "float32", "range": [0.0, 1.0]},
        "skill_progress_valid": {"dtype": "bool", "semantics": "loss mask"},
        "skill_progress_confidence": {"dtype": "float32", "range": [0.0, 1.0]},
        "action": {
            "dtype": "float32",
            "shape": [ACTION_CHUNK_SIZE, 7],
            "semantics": "absolute joint targets; gripper open=0 closed=1 after >0.12 binarization",
        },
        "sample_weight": {"dtype": "float32", "shape": []},
        "modality_validity": {
            "dtype": "bool",
            "names": ["camera0", "camera1", "state", "state_history", "force_current", "force_fast", "force_slow"],
        },
        "lineage": {
            "names": ["source_id", "parent_episode_id", "source_frame", "independent_recovery_episodes"]
        },
    }

    return {
        "schema_version": (
            "pap_moe_baseline_comparison_manifest_v4_full_modal_d2"
            if include_full_modal_recovery_v2
            else SCHEMA_VERSION
        ),
        "dataset_id": (
            "D2_pi05_matching_recovery_v1"
            if full_modal_recovery_prefix is not None or full_modal_source_checkpoint is not None
            else "D2_D1plus_full_modal_rollout_recovery_v1"
            if include_full_modal_recovery_v2
            else DATASET_ID
        ),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "purpose": (
            "D2 adds independently collected full-modality rollout recovery to shared D1"
            if include_full_modal_recovery_v2
            else "shared D1 data contract for true Pi0.5 baseline and PAP-MoE"
        ),
        "supersedes": (
            "pap_moe_d1_v3.json"
            if include_full_modal_recovery_v2
            else "pap_moe_d1_v2.json"
        ),
        "field_contract": field_contract,
        "derivation_contract": {
            "action_chunk": "take current plus next 49 absolute targets within the same parent episode; repeat final target at episode end",
            "recovery_actions": "use only explicit expert_action after monotonic takeover; never infer expert_action from next state",
            "state_history": "causal 1.0 s history resampled to 10 positions, left-edge padded",
            "force_fast": "causal 0.64 s history resampled to 64 positions, left-edge padded",
            "force_slow": "causal 5.0 s history resampled to 50 positions, left-edge padded",
            "visual_quality": "same four observable image statistics used by D0",
            "recovery_subtask": {
                "transport": "transport to the hole",
                "align_precontact": "approach and align with the hole",
                "contact": "recover contact and relocate the hole",
                "insertion": "insert the peg into the hole",
                "grasp_lift": "grasp the peg",
            },
            "recovery_skill_progress": "shared local phase plus continuous progress/readiness; train only where skill_progress_valid is true",
            "physics_gate_target": "observable factorized PAP-MoE routing prior; it is not an FSM-stage label",
            "legacy_correction_missing_force": "zero-fill force/history and set modality_validity false; do not borrow force from the matched demonstration",
        },
        "sampling_contract": {
            "weights": {"complete_success_episode": 1.0, "formal_recovery_episode": 4.0, "qualified_correction": 4.0},
            "weight_interpretation": "single authoritative per-observation weight; launchers must not apply a second recovery boost",
            "episode_atomic_split": True,
            "correction_split_inherits_parent_rollout": True,
            "random_frame_split_forbidden": True,
        },
        "split_episode_ids": split_episode_ids,
        "coverage_warnings": [
            "validation contains independent success episodes but no held-out formal recovery episode",
            "legacy rollout correction traces do not contain FT300/history and are explicitly masked",
            "16004 and 16007 are one independent recovery episode each regardless of extracted frame or chunk count",
            "074701 is excluded because it has no FT300 history or executed expert recovery",
        ],
        "counts": {
            "success_episodes": len(success_entries),
            "formal_recovery_episodes": independent_recoveries,
            "qualified_correction_packs": len(corrections),
            "qualified_correction_samples": sum(int(item["samples"]) for item in corrections),
            "matched_failure_state_samples": 0,
            "independent_recovery_episodes": independent_recoveries,
            "train_episode_groups": len(split_episode_ids["train"]),
            "validation_episode_groups": len(split_episode_ids["validation"]),
        },
        "success_episodes": success_entries,
        "recovery_episodes": recovery_entries,
        "qualified_corrections": corrections,
        "auxiliary_supervision": auxiliary,
        "qualification_reports": [
            _fingerprint(path, workspace, content_hash=content_hash) for path in report_paths
        ],
        "excluded": [
            {"episode_id": "recovery:16006", "reason": "schema-rejected sustained force overload"},
            {"episode_id": "rollout:074701", "reason": "not an expert recovery; excluded from authoritative D1 and reserved for a separate pseudo-label ablation"},
            {"pattern": "gazebo_pi05_v9_absolute_20260813_080146/080753/081455", "reason": "L0 online success evidence; validation-only evidence and never training data"},
        ],
    }


def _canonical_digest(manifest: dict[str, object]) -> str:
    content = dict(manifest)
    content.pop("frozen_at_utc", None)
    content.pop("manifest_sha256", None)
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pap_moe_framework/datasets/manifests/pap_moe_d1_v3.json"),
    )
    parser.add_argument(
        "--exclude-qualified-corrections",
        action="store_true",
        help="Omit legacy synthetic correction carriers (useful for native-modal recovery adaptation)",
    )
    parser.add_argument("--skip-content-hash", action="store_true", help="For tests only; a frozen manifest should include hashes")
    parser.add_argument(
        "--include-full-modal-recovery-v2",
        action="store_true",
        help="Build D2 from hard-gated v2/v3 recoveries, preferring v3 by episode id",
    )
    parser.add_argument(
        "--exclude-legacy-recoveries",
        action="store_true",
        help="Omit the legacy v1 recovery episodes from a source-matched ablation",
    )
    parser.add_argument(
        "--full-modal-recovery-prefix",
        help="Include only v2/v3 recovery directory names with this prefix",
    )
    parser.add_argument(
        "--full-modal-source-checkpoint",
        help="Include only v2/v3 recoveries rolled out by this exact source checkpoint",
    )
    parser.add_argument("--check", action="store_true", help="Validate an existing manifest instead of replacing it")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    manifest = build_manifest(
        workspace,
        content_hash=not args.skip_content_hash,
        include_full_modal_recovery_v2=args.include_full_modal_recovery_v2,
        include_qualified_corrections=not args.exclude_qualified_corrections,
        include_legacy_recoveries=not args.exclude_legacy_recoveries,
        full_modal_recovery_prefix=args.full_modal_recovery_prefix,
        full_modal_source_checkpoint=args.full_modal_source_checkpoint,
    )
    manifest["manifest_sha256"] = _canonical_digest(manifest)
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not output.is_file():
            parser.error(f"manifest does not exist: {output}")
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != _canonical_digest(existing):
            raise ValueError(f"stored manifest digest is invalid: {output}")
        # frozen_at_utc is intentionally ignored; all other content must match.
        existing.pop("frozen_at_utc", None)
        manifest.pop("frozen_at_utc", None)
        if existing != manifest:
            raise ValueError(f"manifest inputs or contract changed: {output}")
        print(f"D1 manifest check passed: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"refusing to overwrite frozen manifest: {output}; use --check")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Frozen D1 manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

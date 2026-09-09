from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_d1_ablation_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_d1_ablation_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _correction(path: Path, samples: int = 2) -> None:
    np.savez_compressed(
        path,
        state=np.zeros((samples, 7), dtype=np.float32),
        camera0=np.zeros((samples, 224, 224, 3), dtype=np.float32),
        camera1=np.zeros((samples, 224, 224, 3), dtype=np.float32),
        target_action=np.zeros((samples, 50, 7), dtype=np.float32),
    )


def test_npz_headers_do_not_materialize_payload(tmp_path: Path) -> None:
    path = tmp_path / "pack.npz"
    _correction(path, samples=3)
    headers = MODULE._npz_headers(path)
    assert headers["state"]["shape"] == [3, 7]
    assert headers["target_action"]["shape"] == [3, 50, 7]


def test_correction_lineage_and_missing_force_are_explicit(tmp_path: Path) -> None:
    workspace = tmp_path
    artifacts = workspace / "artifacts"
    trace = artifacts / "rollout_a" / "online_trace"
    trace.mkdir(parents=True)
    pack = artifacts / "qualified_corrections.npz"
    _correction(pack)
    sidecar = pack.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "trace_dir": str(trace),
                "episode": str(
                    workspace
                    / "raw/pick_up_the_peg_and_insert_it_into_the_hole_episode_13001_success/data.npz"
                ),
                "samples": 2,
                "uses_online_object_truth": False,
            }
        )
    )
    entry = MODULE._correction_entry(pack, workspace, content_hash=False)
    assert entry["parent_episode_id"] == "rollout:rollout_a"
    assert entry["independent_recovery_episodes"] == 0
    assert entry["modality_validity"]["force_current"] is False
    assert entry["missing_modality_policy"] == "zero_fill_and_set_validity_false"


def test_raw_recovery_correction_inherits_parent_modalities(tmp_path: Path) -> None:
    workspace = tmp_path
    artifacts = workspace / "artifacts"
    artifacts.mkdir()
    pack = artifacts / "pi05_raw_recovery_16007_corrections.npz"
    _correction(pack)
    entry = MODULE._correction_entry(pack, workspace, content_hash=False)
    assert entry["parent_episode_id"] == "recovery:16007"
    assert entry["independent_recovery_episodes"] == 0
    assert entry["modality_validity"]["force_fast"] is True
    assert entry["missing_modality_policy"] == "derive_from_parent_raw_frame"


def test_object_truth_correction_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path
    artifacts = workspace / "artifacts"
    trace = artifacts / "rollout_b" / "online_trace"
    trace.mkdir(parents=True)
    pack = artifacts / "bad_corrections.npz"
    _correction(pack)
    pack.with_suffix(".json").write_text(
        json.dumps(
            {
                "trace_dir": str(trace),
                "episode": str(
                    workspace
                    / "raw/pick_up_the_peg_and_insert_it_into_the_hole_episode_13001_success/data.npz"
                ),
                "samples": 2,
                "uses_online_object_truth": True,
            }
        )
    )
    with pytest.raises(ValueError, match="online object truth"):
        MODULE._correction_entry(pack, workspace, content_hash=False)


def test_digest_ignores_only_freeze_time() -> None:
    first = {"schema_version": "v1", "frozen_at_utc": "one", "counts": {"x": 2}}
    second = {"schema_version": "v1", "frozen_at_utc": "two", "counts": {"x": 2}}
    assert MODULE._canonical_digest(first) == MODULE._canonical_digest(second)
    second["counts"]["x"] = 3
    assert MODULE._canonical_digest(first) != MODULE._canonical_digest(second)

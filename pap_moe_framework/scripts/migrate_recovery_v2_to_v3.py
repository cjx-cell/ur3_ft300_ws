#!/usr/bin/env python3
"""Create a non-destructive v3 multi-task view of full-modal recovery v2 data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pap_moe_framework.rollout_recovery.schema import (
    SCHEMA_VERSION,
    TRAJECTORY_SCOPE,
    V2_SCHEMA_VERSION,
    load_and_validate,
    validate_episode,
)


def _labels(data: dict[str, np.ndarray], takeover: int) -> dict[str, np.ndarray]:
    frames = len(data["state"])
    phase_name = str(np.asarray(data["recovery_phase"]).item())
    phase = np.full(frames, 7, dtype=np.int64)
    progress = np.zeros(frames, dtype=np.float32)
    readiness = np.zeros(frames, dtype=np.float32)
    valid = np.zeros(frames, dtype=bool)
    confidence = np.zeros(frames, dtype=np.float32)

    peg = np.asarray(data["peg_position"], dtype=np.float32)
    hole = np.asarray(data["hole_position"], dtype=np.float32)
    gripper = np.asarray(data["gripper_position"], dtype=np.float32)
    attached = np.asarray(data["peg_attached"], dtype=bool)
    expert = np.asarray(data["expert_action"], dtype=np.float32)
    xy_hole = np.linalg.norm(peg[:, :2] - hole[:, :2], axis=1)

    for index in range(takeover, frames):
        if phase_name == "grasp_lift":
            xy = float(np.linalg.norm(gripper[index, :2] - peg[index, :2]))
            z_error = abs(float(gripper[index, 2] - peg[index, 2]) - 0.105)
            if attached[index]:
                phase[index] = 6  # exit/lift
                lift = max(0.0, float(peg[index, 2] - peg[takeover, 2]))
                readiness[index] = np.clip(lift / 0.05, 0.0, 1.0)
                progress[index] = 0.80 + 0.20 * readiness[index]
            elif expert[index, 6] >= 0.5:
                phase[index], progress[index] = 3, 0.72  # interact/close
                readiness[index] = np.clip(float(data["state"][index, 6]) / 0.6, 0.0, 1.0)
            elif xy > 0.008:
                phase[index] = 1  # approach
                readiness[index] = np.clip(1.0 - xy / 0.20, 0.0, 1.0)
                progress[index] = 0.20 + 0.25 * readiness[index]
            else:
                phase[index] = 2  # align/descend
                readiness[index] = np.clip(1.0 - z_error / 0.05, 0.0, 1.0)
                progress[index] = 0.50 + 0.15 * readiness[index]
        else:
            if xy_hole[index] > 0.003:
                phase[index] = 2  # align
                readiness[index] = np.clip(1.0 - xy_hole[index] / 0.05, 0.0, 1.0)
                progress[index] = 0.25 + 0.35 * readiness[index]
            else:
                phase[index] = 3  # interact/contact/insertion
                readiness[index] = np.clip(1.0 - xy_hole[index] / 0.003, 0.0, 1.0)
                progress[index] = 0.65 + 0.25 * readiness[index]
        valid[index] = True
        confidence[index] = 0.70

    names = np.asarray(
        [("enter", "approach", "align", "interact", "stabilize", "verify", "exit", "recover")[i]
         for i in phase],
        dtype=np.str_,
    )
    source = np.where(valid, "offline_v2_geometry_heuristic_v1", "unlabelled_policy_rollin")
    return {
        "skill_progress_phase": phase,
        "skill_progress_phase_name": names,
        "skill_progress": progress,
        "transition_readiness": readiness,
        "skill_progress_valid": valid,
        "skill_progress_label_source": source.astype(np.str_),
        "skill_progress_confidence": confidence,
    }


def migrate(source: Path, destination: Path) -> None:
    summary = load_and_validate(source, require_full_modalities=True)
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(np.asarray(data["schema_version"]).item()) != V2_SCHEMA_VERSION:
        raise ValueError(f"not a v2 full-modal recovery: {source}")
    data.pop("grasp_progress_label", None)
    data["schema_version"] = np.asarray(SCHEMA_VERSION)
    data["trajectory_scope"] = np.asarray(TRAJECTORY_SCOPE)
    force = np.asarray(data["force"], dtype=np.float32)
    baseline = np.median(force[: min(10, len(force)), :3], axis=0)
    contact = np.linalg.norm(force[:, :3] - baseline[None, :], axis=1) >= 0.5
    stage = np.zeros((len(force), 4), dtype=np.float32)
    stage[:, 0] = 1.0
    phase_name = str(np.asarray(data["recovery_phase"]).item())
    stage[contact, 0] = 0.0
    stage[contact, 3 if phase_name == "insertion" else 2] = 1.0
    data["stage"] = stage
    data.update(_labels(data, summary.takeover_index))
    validate_episode(data, require_full_modalities=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    sources = sorted(args.source.rglob("data.npz")) if args.source.is_dir() else [args.source]
    if not sources:
        parser.error("no data.npz files found")
    for source in sources:
        relative = source.relative_to(args.source) if args.source.is_dir() else Path(source.name)
        destination = args.destination / relative if args.source.is_dir() else args.destination
        migrate(source, destination)
        print(f"migrated: {source} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare PAP-MoE soft routes with Gazebo contact teacher metadata.

This is a read-only audit.  Gazebo collision truth is not a policy input and
is deliberately not treated as a complete routing target: a stable gripper
hold is geometrically in contact but need not always dominate the action
condition.  It is used as a hard negative for free-space E3/E4 activation and
as an independent check that hole contact is represented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _confusion(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.count_nonzero(prediction & target)),
        "fp": int(np.count_nonzero(prediction & ~target)),
        "fn": int(np.count_nonzero(~prediction & target)),
        "tn": int(np.count_nonzero(~prediction & ~target)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    args = parser.parse_args()

    files = sorted(args.raw_root.glob("**/data.npz"))
    if not files:
        raise FileNotFoundError(f"No data.npz below {args.raw_root}")

    required = (
        "stage",
        "contact_truth_valid",
        "contact_truth_task",
        "contact_truth_gripper",
        "contact_truth_hole",
        "contact_truth_table",
    )
    episodes = []
    skipped = []
    skipped_invalid_truth = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in required if key not in data]
            if missing:
                skipped.append({"path": str(path), "missing": missing})
                continue
            episode = {
                "path": str(path),
                **{key: np.asarray(data[key]) for key in required},
            }
            if not np.any(episode["contact_truth_valid"]):
                skipped_invalid_truth.append(str(path))
                continue
            episodes.append(episode)
    if not episodes:
        raise RuntimeError("No episode contains the Gazebo contact-truth contract")

    routes = np.concatenate([episode["stage"] for episode in episodes])
    routes = np.maximum(routes, 0.0)
    routes /= np.maximum(routes.sum(axis=-1, keepdims=True), 1e-12)
    truth_valid = np.concatenate(
        [episode["contact_truth_valid"] for episode in episodes]
    ).astype(bool)
    task = np.concatenate(
        [episode["contact_truth_task"] for episode in episodes]
    ).astype(bool)
    gripper = np.concatenate(
        [episode["contact_truth_gripper"] for episode in episodes]
    ).astype(bool)
    hole = np.concatenate(
        [episode["contact_truth_hole"] for episode in episodes]
    ).astype(bool)
    table = np.concatenate(
        [episode["contact_truth_table"] for episode in episodes]
    ).astype(bool)

    predicted_contact = (routes[:, 2] + routes[:, 3]) >= args.contact_threshold
    # The peg resting on the table before pickup is not robot-task contact;
    # otherwise every initial free-space approach would be mislabeled.
    policy_free_space = ~task
    report = {
        "raw_root": str(args.raw_root.resolve()),
        "episodes_audited": len(episodes),
        "frames": int(len(routes)),
        "contact_threshold": args.contact_threshold,
        "truth_valid_frames": int(np.count_nonzero(truth_valid)),
        "truth_valid_fraction": float(truth_valid.mean()),
        "truth_frames": {
            "task_gripper_or_hole": int(np.count_nonzero(task)),
            "gripper": int(np.count_nonzero(gripper)),
            "hole": int(np.count_nonzero(hole)),
            "table": int(np.count_nonzero(table)),
            "policy_free_space_ignoring_peg_table_support": int(
                np.count_nonzero(policy_free_space)
            ),
        },
        "soft_contact_vs_task_truth": _confusion(predicted_contact, task),
        "soft_contact_vs_hole_truth": _confusion(predicted_contact, hole),
        "hard_negative_checks": {
            "soft_contact_in_strict_free_space_frames": int(
                np.count_nonzero(predicted_contact & policy_free_space)
            ),
            "soft_contact_in_strict_free_space_fraction": float(
                np.mean(predicted_contact & policy_free_space)
            ),
            "hole_contact_without_soft_contact_frames": int(
                np.count_nonzero(hole & ~predicted_contact)
            ),
        },
        "skipped_legacy_episodes": skipped,
        "skipped_invalid_truth_episodes": skipped_invalid_truth,
        "notes": [
            "Gazebo contact truth is teacher/audit metadata, never a model input.",
            "A gripper hold is contact but is not automatically an E4 target; force and motion still determine routing salience.",
            "Free-space E3/E4 activation and missed hole contact are routing-label defects.",
        ],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()

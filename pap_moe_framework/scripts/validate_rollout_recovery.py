#!/usr/bin/env python3
"""Validate independent rollout-failure-recovery raw episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pap_moe_framework.rollout_recovery.schema import load_and_validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--allow-failed-recovery", action="store_true")
    parser.add_argument(
        "--allow-legacy-missing-modalities",
        action="store_true",
        help="audit historical v1 only; never use this for new-data admission",
    )
    parser.add_argument("--max-camera-skew-s", type=float, default=0.15)
    parser.add_argument("--max-pose-skew-s", type=float, default=1.0)
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.rglob("data.npz")) if path.is_dir() else [path])
    if not files:
        parser.error("no data.npz files found")

    failures = 0
    for path in files:
        try:
            # A full-task DAgger episode intentionally retains the policy's
            # failed contact context.  Those frames have zero imitation
            # weight; only sustained overload while the expert owns control
            # remains a hard admission failure.  Keep the standalone validator
            # consistent with save_validated_episode(), which applies the same
            # contract at write time.
            with np.load(path, allow_pickle=False) as episode:
                recovery_phase = str(np.asarray(episode["recovery_phase"]).item())
            summary = load_and_validate(
                path,
                max_camera_skew_s=args.max_camera_skew_s,
                max_pose_skew_s=args.max_pose_skew_s,
                require_success=not args.allow_failed_recovery,
                require_full_modalities=not args.allow_legacy_missing_modalities,
                allow_policy_failure_force_context=(recovery_phase == "full_task"),
            )
            print(json.dumps({"path": str(path), "valid": True, **summary.__dict__}))
        except Exception as error:
            failures += 1
            print(json.dumps({"path": str(path), "valid": False, "error": str(error)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Parse a peg-in-hole ROS log into a machine-readable closed-loop result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


STATUS_PATTERN = re.compile(
    r"EVAL status: t=(?P<time>[0-9.]+)s, .*?"
    r"gripper=(?P<gripper>-?[0-9.]+), .*?"
    r"attached=(?P<attached>True|False), "
    r"gripper_peg_xy=(?P<gripper_xy>[0-9.]+)m, "
    r"gripper_peg_z=(?P<gripper_z>-?[0-9.]+)m"
)


def geometry_summary(path):
    if not path.exists():
        return {"available": False}
    rows, invalid, malformed = [], 0, 0
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if row.get("valid") and row.get("geometry", {}).get("schema") == "cad_seating_shadow_v2":
            rows.append(row)
        else:
            invalid += 1
    return dict(available=bool(rows), schema="cad_seating_shadow_v2",
                valid_samples=len(rows), invalid_or_other_schema_samples=invalid,
                malformed_samples=malformed,
                observed_inserted=any(r["geometry"]["candidate_inserted"] for r in rows),
                observed_fully_seated=any(r["geometry"]["candidate_fully_seated"] for r in rows),
                last_valid_geometry=rows[-1]["geometry"] if rows else None,
                release_stability_verified=False,
                note="Sampled diagnostic observations, not sustained-success or terminal-state verification")


def _parse_status_metrics(text: str) -> dict[str, object]:
    samples = []
    for match in STATUS_PATTERN.finditer(text):
        samples.append(
            {
                "time_s": float(match.group("time")),
                "gripper": float(match.group("gripper")),
                "attached": match.group("attached") == "True",
                "gripper_peg_xy_m": float(match.group("gripper_xy")),
                "gripper_peg_z_m": float(match.group("gripper_z")),
            }
        )

    if not samples:
        return {
            "min_gripper_peg_xy_m": None,
            "min_gripper_peg_xy_time_s": None,
            "gripper_peg_z_at_min_xy_m": None,
            "min_gripper_peg_z_m": None,
            "first_close": None,
            "ever_attached": False,
            "first_attached_time_s": None,
        }

    min_xy_sample = min(samples, key=lambda sample: sample["gripper_peg_xy_m"])
    first_close = next((sample for sample in samples if sample["gripper"] >= 0.5), None)
    first_attached = next((sample for sample in samples if sample["attached"]), None)
    return {
        "min_gripper_peg_xy_m": min_xy_sample["gripper_peg_xy_m"],
        "min_gripper_peg_xy_time_s": min_xy_sample["time_s"],
        "gripper_peg_z_at_min_xy_m": min_xy_sample["gripper_peg_z_m"],
        "min_gripper_peg_z_m": min(sample["gripper_peg_z_m"] for sample in samples),
        "first_close": first_close,
        "ever_attached": first_attached is not None,
        "first_attached_time_s": None if first_attached is None else first_attached["time_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--execution-prefix", type=int, default=None)
    parser.add_argument("--physical-arm-residual-scale", type=float, default=None)
    parser.add_argument("--max-arm-step-rad", type=float, default=None)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    if "SUCCESS: peg is inserted" in text:
        outcome = "success"
    elif "OUT-OF-DISTRIBUTION:" in text:
        outcome = "out_of_distribution"
    elif "ABORT:" in text:
        outcome = "abort"
    elif "TIMEOUT:" in text:
        outcome = "timeout"
    elif "DEMONSTRATION COMPLETE:" in text:
        outcome = "demonstration_exhausted"
    else:
        outcome = "unknown_failure"

    statuses = [match.group(1).strip() for match in re.finditer(r"EVAL status: (.*)", text)]
    invalid_reasons = re.findall(r"INFRASTRUCTURE INVALID: ([^\n]+)", text)
    geometry = geometry_summary(args.log.parent / "geometry_shadow.jsonl")
    if os.environ.get("WORKSPACE50_GEOMETRY_SHADOW") == "true" and not geometry.get("available"):
        invalid_reasons.append("Required geometry shadow observations unavailable")
    source_manifest = json.loads(os.environ.get("WORKSPACE50_SOURCE_MANIFEST_JSON", "{}"))
    for path, expected_hash in source_manifest.items():
        if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            invalid_reasons.append(f"Runtime source changed during evaluation: {path}")
    evaluation_valid = bool(statuses) and not invalid_reasons
    if not evaluation_valid:
        outcome = "infrastructure_failure"
    success_match = re.search(
        r"SUCCESS: peg is inserted \(xy=([0-9.]+)m, z=([0-9.]+)m\)",
        text,
    )
    result = {
        "geometry_shadow": geometry,
        "batch_contract_id": os.environ.get("WORKSPACE50_BATCH_CONTRACT_ID"),
        "evaluation_kind": os.environ.get(
            "WORKSPACE50_EVALUATION_KIND",
            "engineering_smoke" if float(os.environ.get("WORKSPACE50_MAX_EPISODE_DURATION_S", "120")) != 120 else "formal",
        ),
        "max_episode_duration_s": float(os.environ.get("WORKSPACE50_MAX_EPISODE_DURATION_S", "120")),
        "goal_time_tolerance_override_s": float(os.environ.get("POLICY_GOAL_TIME_TOLERANCE_S", "0")),
        "controller_max_arm_step_rad": (
            float(os.environ["POLICY_ACTION_CHUNK_MAX_STEP_RAD"])
            if "POLICY_ACTION_CHUNK_MAX_STEP_RAD" in os.environ else None
        ),
        "success_contract": {
            "max_xy_m": float(os.environ.get("WORKSPACE50_SUCCESS_MAX_XY_M", "0.008")),
            "max_peg_z_m": float(os.environ.get("WORKSPACE50_SUCCESS_MAX_PEG_Z_M", "0.890")),
            "required_checks": int(os.environ.get("WORKSPACE50_SUCCESS_REQUIRED_CHECKS", "5")),
            "check_timebase": "simulation",
            "release_stability_verified": False,
        },
        "policy": args.policy,
        "checkpoint": str(args.checkpoint.resolve()),
        "episode": args.episode,
        "seed": args.seed,
        "execution_prefix": args.execution_prefix,
        "physical_arm_residual_scale": args.physical_arm_residual_scale,
        "max_arm_step_rad": args.max_arm_step_rad,
        "outcome": outcome,
        "evaluation_valid": evaluation_valid,
        "infrastructure_invalid_reasons": invalid_reasons,
        "runtime_source_manifest": source_manifest,
        "action_exchange_contract": "paired-v1" if os.environ.get("POLICY_PAIRED_ACTION_FILE") else "legacy",
        "success": outcome == "success",
        "success_xy_m": (float(success_match.group(1)) if success_match is not None else None),
        "success_peg_z_m": (float(success_match.group(2)) if success_match is not None else None),
        "status_count": len(statuses),
        "last_status": statuses[-1] if statuses else None,
        "ros_log": str(args.log.resolve()),
        **_parse_status_metrics(text),
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["success"]:
        raise SystemExit(6)


if __name__ == "__main__":
    main()

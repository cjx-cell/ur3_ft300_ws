#!/usr/bin/env python3
"""Apply explicit offline gates before allowing a Pi0.5 checkpoint into Gazebo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-episode-improvement", type=float, default=0.05)
    parser.add_argument("--min-startup-improvement", type=float, default=0.10)
    parser.add_argument("--max-phase-regression", type=float, default=0.10)
    parser.add_argument("--max-executed-step-rad", type=float, default=0.12)
    parser.add_argument("--min-gripper-accuracy", type=float, default=0.95)
    parser.add_argument("--max-std-ratio", type=float, default=1.05)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = summary.get("ranking", [])
    source = next((row for row in rows if row["tag"] == "source_030000"), None)
    if source is None:
        raise ValueError("summary does not contain source_030000")

    audits = []
    for row in rows:
        if row["tag"] == "source_030000":
            continue
        improvements = {
            "train_episode": _relative_improvement(
                source["train_mean_executed_arm_mae_rad"],
                row["train_mean_executed_arm_mae_rad"],
            ),
            "holdout_episode": _relative_improvement(
                source["holdout_mean_executed_arm_mae_rad"],
                row["holdout_mean_executed_arm_mae_rad"],
            ),
            "train_startup": _relative_improvement(
                source["train_startup_mean_executed_arm_mae_rad"],
                row["train_startup_mean_executed_arm_mae_rad"],
            ),
            "holdout_startup": _relative_improvement(
                source["holdout_startup_mean_executed_arm_mae_rad"],
                row["holdout_startup_mean_executed_arm_mae_rad"],
            ),
        }
        phase_improvements = {}
        for split in ("train", "holdout"):
            field = f"{split}_phase_mean_executed_arm_mae_rad"
            source_phases = source[field]
            candidate_phases = row[field]
            if set(source_phases) != set(candidate_phases):
                raise ValueError(
                    f"{row['tag']} {split} phase mismatch: "
                    f"source={sorted(source_phases)}, "
                    f"candidate={sorted(candidate_phases)}"
                )
            phase_improvements[split] = {
                phase: _relative_improvement(
                    source_phases[phase], candidate_phases[phase]
                )
                for phase in source_phases
            }
        gates = {
            "train_episode_improved": improvements["train_episode"]
            >= args.min_episode_improvement,
            "holdout_episode_improved": improvements["holdout_episode"]
            >= args.min_episode_improvement,
            "train_startup_improved": improvements["train_startup"]
            >= args.min_startup_improvement,
            "holdout_startup_improved": improvements["holdout_startup"]
            >= args.min_startup_improvement,
            "train_phases_not_regressed": all(
                improvement >= -args.max_phase_regression
                for improvement in phase_improvements["train"].values()
            ),
            "holdout_phases_not_regressed": all(
                improvement >= -args.max_phase_regression
                for improvement in phase_improvements["holdout"].values()
            ),
            "executed_step_safe": row["max_executed_predicted_step_rad"]
            <= args.max_executed_step_rad,
            "gripper_accurate": row["mean_gripper_accuracy"]
            >= args.min_gripper_accuracy,
            "stochasticity_not_worse": row["mean_prediction_std"]
            <= source["mean_prediction_std"] * args.max_std_ratio,
        }
        audits.append(
            {
                **row,
                "relative_improvements": improvements,
                "phase_relative_improvements": phase_improvements,
                "gates": gates,
                "qualified": all(gates.values()),
            }
        )

    qualified = sorted(
        (item for item in audits if item["qualified"]),
        key=lambda item: (
            item["mean_executed_arm_mae_rad"],
            item["worst_executed_arm_mae_rad"],
        ),
    )
    result = {
        "source": source,
        "thresholds": {
            "min_episode_improvement": args.min_episode_improvement,
            "min_startup_improvement": args.min_startup_improvement,
            "max_phase_regression": args.max_phase_regression,
            "max_executed_step_rad": args.max_executed_step_rad,
            "min_gripper_accuracy": args.min_gripper_accuracy,
            "max_std_ratio": args.max_std_ratio,
        },
        "recommended_checkpoint": qualified[0]["checkpoint"] if qualified else None,
        "qualified_tags": [item["tag"] for item in qualified],
        "audits": audits,
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

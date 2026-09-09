#!/usr/bin/env python3
"""Gate a gripper-repaired Pi0.5 checkpoint using deployed inference semantics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

OPEN_PHASES = {
    "release the peg after verification",
    "retract and go back to home",
}


def _relative_improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(abs(reference), 1e-12)


def _mean(frames: list[dict[str, object]], field: str) -> float:
    return float(np.mean([float(frame[field]) for frame in frames]))


def _summarize(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = []
    for episode in data["episodes"]:
        frames = episode["frames"]
        phases = sorted({str(frame["phase"]) for frame in frames})
        phase_arm = {
            phase: _mean(
                [frame for frame in frames if frame["phase"] == phase],
                "ensemble_rate_limited_executed_arm_mae_rad",
            )
            for phase in phases
        }
        phase_gripper = {
            phase: _mean(
                [frame for frame in frames if frame["phase"] == phase],
                "ensemble_gripper_accuracy",
            )
            for phase in phases
        }
        startup = [frame for frame in frames if frame["phase"] == "grasp the peg"]
        episodes.append(
            {
                "episode": episode["episode"],
                "deployed_arm_mae_rad": _mean(
                    frames, "ensemble_rate_limited_executed_arm_mae_rad"
                ),
                "startup_deployed_arm_mae_rad": _mean(
                    startup, "ensemble_rate_limited_executed_arm_mae_rad"
                ),
                "ensemble_gripper_accuracy": _mean(
                    frames, "ensemble_gripper_accuracy"
                ),
                "open_phase_gripper_accuracy": float(
                    min(phase_gripper[phase] for phase in OPEN_PHASES)
                ),
                "phase_deployed_arm_mae_rad": phase_arm,
                "phase_ensemble_gripper_accuracy": phase_gripper,
            }
        )

    all_frames = [frame for episode in data["episodes"] for frame in episode["frames"]]
    return {
        "path": str(path.resolve()),
        "checkpoint": data["checkpoint"],
        "episodes": episodes,
        "deployed_arm_mae_rad": _mean(
            all_frames, "ensemble_rate_limited_executed_arm_mae_rad"
        ),
        "ensemble_gripper_accuracy": _mean(
            all_frames, "ensemble_gripper_accuracy"
        ),
        "max_deployed_arm_step_rad": float(
            data["aggregate_max_ensemble_rate_limited_executed_step_rad"]
        ),
        "raw_prediction_std": float(data["aggregate_mean_prediction_std"]),
        "inference_contract": {
            "num_seeds": data["num_seeds"],
            "ensemble": "physical_action_mean",
            "causal_arm_step_limit_rad": 0.12,
            "gripper_threshold": 0.5,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-source", type=Path, required=True)
    parser.add_argument("--repair-parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parameter-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parent-regression", type=float, default=0.05)
    parser.add_argument("--max-parent-startup-regression", type=float, default=0.10)
    parser.add_argument("--max-parent-phase-regression", type=float, default=0.10)
    parser.add_argument("--max-deployed-episode-arm-mae", type=float, default=0.08)
    parser.add_argument("--min-gripper-accuracy", type=float, default=0.95)
    parser.add_argument("--max-arm-step-rad", type=float, default=0.12)
    parser.add_argument("--max-std-ratio", type=float, default=1.05)
    args = parser.parse_args()

    source = _summarize(args.original_source)
    parent = _summarize(args.repair_parent)
    candidate = _summarize(args.candidate)
    parameter_audit = json.loads(args.parameter_audit.read_text(encoding="utf-8"))

    if not (
        len(source["episodes"])
        == len(parent["episodes"])
        == len(candidate["episodes"])
        == 2
    ):
        raise ValueError("Expected exactly train and holdout episodes in every evaluation")

    source_improvements: dict[str, float] = {}
    parent_changes: dict[str, float] = {}
    source_phase_improvements: dict[str, dict[str, float]] = {}
    parent_phase_changes: dict[str, dict[str, float]] = {}
    for split_index, split in enumerate(("train", "holdout")):
        source_episode = source["episodes"][split_index]
        parent_episode = parent["episodes"][split_index]
        candidate_episode = candidate["episodes"][split_index]
        if not (
            source_episode["episode"]
            == parent_episode["episode"]
            == candidate_episode["episode"]
        ):
            raise ValueError(f"{split} episode sources do not match")
        source_improvements[f"{split}_episode"] = _relative_improvement(
            source_episode["deployed_arm_mae_rad"],
            candidate_episode["deployed_arm_mae_rad"],
        )
        source_improvements[f"{split}_startup"] = _relative_improvement(
            source_episode["startup_deployed_arm_mae_rad"],
            candidate_episode["startup_deployed_arm_mae_rad"],
        )
        parent_changes[f"{split}_episode"] = _relative_improvement(
            parent_episode["deployed_arm_mae_rad"],
            candidate_episode["deployed_arm_mae_rad"],
        )
        parent_changes[f"{split}_startup"] = _relative_improvement(
            parent_episode["startup_deployed_arm_mae_rad"],
            candidate_episode["startup_deployed_arm_mae_rad"],
        )
        source_phases = source_episode["phase_deployed_arm_mae_rad"]
        parent_phases = parent_episode["phase_deployed_arm_mae_rad"]
        candidate_phases = candidate_episode["phase_deployed_arm_mae_rad"]
        if not set(source_phases) == set(parent_phases) == set(candidate_phases):
            raise ValueError(f"{split} phase sets do not match")
        source_phase_improvements[split] = {
            phase: _relative_improvement(source_phases[phase], candidate_phases[phase])
            for phase in source_phases
        }
        parent_phase_changes[split] = {
            phase: _relative_improvement(parent_phases[phase], candidate_phases[phase])
            for phase in parent_phases
        }

    gates = {
        "parameter_integrity": parameter_audit.get("passed") is True,
        "train_parent_preserved": parent_changes["train_episode"]
        >= -args.max_parent_regression,
        "holdout_parent_preserved": parent_changes["holdout_episode"]
        >= -args.max_parent_regression,
        "train_startup_parent_preserved": parent_changes["train_startup"]
        >= -args.max_parent_startup_regression,
        "holdout_startup_parent_preserved": parent_changes["holdout_startup"]
        >= -args.max_parent_startup_regression,
        "train_parent_phases_preserved": all(
            value >= -args.max_parent_phase_regression
            for value in parent_phase_changes["train"].values()
        ),
        "holdout_parent_phases_preserved": all(
            value >= -args.max_parent_phase_regression
            for value in parent_phase_changes["holdout"].values()
        ),
        "train_deployed_arm_accurate": candidate["episodes"][0][
            "deployed_arm_mae_rad"
        ]
        <= args.max_deployed_episode_arm_mae,
        "holdout_deployed_arm_accurate": candidate["episodes"][1][
            "deployed_arm_mae_rad"
        ]
        <= args.max_deployed_episode_arm_mae,
        "train_gripper_accurate": candidate["episodes"][0][
            "ensemble_gripper_accuracy"
        ]
        >= args.min_gripper_accuracy,
        "holdout_gripper_accurate": candidate["episodes"][1][
            "ensemble_gripper_accuracy"
        ]
        >= args.min_gripper_accuracy,
        "train_open_phases_accurate": candidate["episodes"][0][
            "open_phase_gripper_accuracy"
        ]
        >= args.min_gripper_accuracy,
        "holdout_open_phases_accurate": candidate["episodes"][1][
            "open_phase_gripper_accuracy"
        ]
        >= args.min_gripper_accuracy,
        "deployed_step_safe": candidate["max_deployed_arm_step_rad"]
        <= args.max_arm_step_rad + 1e-6,
        "raw_stochasticity_not_worse": candidate["raw_prediction_std"]
        <= parent["raw_prediction_std"] * args.max_std_ratio,
    }
    qualified = all(gates.values())
    result = {
        "recommended_checkpoint": candidate["checkpoint"] if qualified else None,
        "qualified": qualified,
        "gates": gates,
        "thresholds": {
            "max_parent_regression": args.max_parent_regression,
            "max_parent_startup_regression": args.max_parent_startup_regression,
            "max_parent_phase_regression": args.max_parent_phase_regression,
            "max_deployed_episode_arm_mae": args.max_deployed_episode_arm_mae,
            "min_gripper_accuracy": args.min_gripper_accuracy,
            "max_arm_step_rad": args.max_arm_step_rad,
            "max_std_ratio": args.max_std_ratio,
        },
        "source_improvements": source_improvements,
        "parent_changes": parent_changes,
        "source_phase_improvements": source_phase_improvements,
        "parent_phase_changes": parent_phase_changes,
        "original_source": source,
        "repair_parent": parent,
        "candidate": candidate,
        "parameter_audit": parameter_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

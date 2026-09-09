#!/usr/bin/env python3
"""Probe stage prompts on critical Pi0.5 closed-loop rollout observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS = (
    "pick up the peg and insert it into the hole",
    "grasp the peg",
    "approach the peg",
    "move down toward the peg",
    "descend and align the gripper with the peg",
    "stabilize above the peg and close the gripper",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--execution-horizon", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(0, str(SCRIPT_DIR))
    from eval_pi05_offline_action_chunk import _postprocess_chunk, _raw_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401

    with np.load(args.corrections) as archive:
        corrections = {key: archive[key] for key in archive.files}
    policy = PI05Policy.from_pretrained(str(args.checkpoint), strict=False)
    policy.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    dynamic_prompt_steps = 0
    for step in preprocessor.steps:
        if hasattr(step, "global_task"):
            step.global_task = None
            dynamic_prompt_steps += 1
    if dynamic_prompt_steps != 1:
        raise RuntimeError(
            f"Expected one global-task override step, found {dynamic_prompt_steps}"
        )

    results = []
    horizon = args.execution_horizon
    for prompt in PROMPTS:
        per_observation = []
        for index in args.indices:
            batch = preprocessor(
                _raw_observation(
                    corrections["state"][index],
                    corrections["camera0"][index],
                    corrections["camera1"][index],
                    prompt,
                )
            )
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                prediction = _postprocess_chunk(
                    policy.predict_action_chunk(batch), postprocessor
                )[0].float().cpu().numpy()
            target = corrections["target_action"][index]
            deltas = np.diff(
                np.concatenate([corrections["state"][index, None, :6], prediction[:horizon, :6]], axis=0),
                axis=0,
            )
            per_observation.append(
                {
                    "index": index,
                    "arm_mae_rad": float(np.abs(prediction[:horizon, :6] - target[:horizon, :6]).mean()),
                    "endpoint_arm_mae_rad": float(
                        np.abs(prediction[horizon - 1, :6] - target[horizon - 1, :6]).mean()
                    ),
                    "max_step_rad": float(np.max(np.abs(deltas))),
                    "closed_fraction": float(np.mean(prediction[:horizon, 6] >= 0.5)),
                    "endpoint_arm": prediction[horizon - 1, :6].tolist(),
                }
            )
        results.append(
            {
                "prompt": prompt,
                "mean_arm_mae_rad": float(np.mean([value["arm_mae_rad"] for value in per_observation])),
                "worst_arm_mae_rad": float(np.max([value["arm_mae_rad"] for value in per_observation])),
                "mean_endpoint_arm_mae_rad": float(
                    np.mean([value["endpoint_arm_mae_rad"] for value in per_observation])
                ),
                "max_step_rad": float(np.max([value["max_step_rad"] for value in per_observation])),
                "observations": per_observation,
            }
        )
    results.sort(key=lambda value: (value["mean_arm_mae_rad"], value["worst_arm_mae_rad"]))
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "corrections": str(args.corrections.resolve()),
        "indices": args.indices,
        "seed": args.seed,
        "preprocessor_global_task_override_disabled": True,
        "ranking": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

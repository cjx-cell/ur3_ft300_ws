#!/usr/bin/env python3
"""Rank Pi0.5 sampling seeds on saved closed-loop observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPT_DIR = Path(__file__).resolve().parent
TASK = "pick up the peg and insert it into the hole"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--execution-horizon", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(0, str(SCRIPT_DIR))
    from eval_pi05_offline_action_chunk import (
        _binarize_observed_gripper,
        _binarize_target_gripper,
        _postprocess_chunk,
        _raw_observation,
    )
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401

    trace_paths = sorted(args.trace_dir.glob("chunk_*.npz"))[: args.chunks]
    if len(trace_paths) != args.chunks:
        raise ValueError(f"Expected {args.chunks} traces, found {len(trace_paths)}")
    with np.load(args.episode_npz, allow_pickle=True) as archive:
        demo = {key: archive[key] for key in ("state", "action", "task")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PI05Policy.from_pretrained(str(args.checkpoint), strict=False)
    policy.to(device=device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )

    examples = []
    for trace_path in trace_paths:
        with np.load(trace_path) as archive:
            state = _binarize_observed_gripper(archive["state"][None])[0]
            camera0 = archive["camera0"].astype(np.float32)
            camera1 = archive["camera1"].astype(np.float32)
        distances = np.linalg.norm(demo["state"][:, :6] - state[None, :6], axis=1)
        nearest_distance = float(distances.min())
        nearest = int(np.flatnonzero(distances <= nearest_distance + 0.01)[0])
        indices = np.minimum(
            np.arange(nearest, nearest + policy.config.chunk_size), len(demo["action"]) - 1
        )
        target = _binarize_target_gripper(demo["action"][indices], demo["task"][indices])
        batch = preprocessor(_raw_observation(state, camera0, camera1, TASK))
        examples.append((trace_path, state, nearest, nearest_distance, target, batch))

    seed_results = []
    horizon = args.execution_horizon
    for seed in range(args.num_seeds):
        per_chunk = []
        for trace_path, _state, nearest, nearest_distance, target, batch in examples:
            torch.manual_seed(seed)
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(batch)
                prediction = _postprocess_chunk(normalized, postprocessor)[0].float().cpu().numpy()
            arm_mae = float(np.abs(prediction[:horizon, :6] - target[:horizon, :6]).mean())
            endpoint_mae = float(
                np.abs(prediction[horizon - 1, :6] - target[horizon - 1, :6]).mean()
            )
            gripper_accuracy = float(
                np.mean(
                    (prediction[:horizon, 6] >= 0.5)
                    == (target[:horizon, 6] >= 0.5)
                )
            )
            per_chunk.append(
                {
                    "trace": str(trace_path.resolve()),
                    "nearest_demo_frame": nearest,
                    "nearest_state_l2_rad": nearest_distance,
                    "arm_mae_rad": arm_mae,
                    "endpoint_arm_mae_rad": endpoint_mae,
                    "gripper_accuracy": gripper_accuracy,
                }
            )
        seed_results.append(
            {
                "seed": seed,
                "mean_arm_mae_rad": float(np.mean([x["arm_mae_rad"] for x in per_chunk])),
                "max_arm_mae_rad": float(np.max([x["arm_mae_rad"] for x in per_chunk])),
                "mean_endpoint_arm_mae_rad": float(
                    np.mean([x["endpoint_arm_mae_rad"] for x in per_chunk])
                ),
                "mean_gripper_accuracy": float(
                    np.mean([x["gripper_accuracy"] for x in per_chunk])
                ),
                "chunks": per_chunk,
            }
        )

    ranking = sorted(
        seed_results,
        key=lambda item: (
            -item["mean_gripper_accuracy"],
            item["max_arm_mae_rad"],
            item["mean_arm_mae_rad"],
        ),
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "trace_dir": str(args.trace_dir.resolve()),
        "episode": str(args.episode_npz.resolve()),
        "execution_horizon": horizon,
        "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"top_seeds": ranking[:10]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Counterfactually decode one observation under deployable grasp-stage prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
DEFAULT_PROMPTS = (
    "pick up the peg and insert it into the hole",
    "approach the peg",
    "descend to the peg",
    "stabilize above the peg",
    "close the gripper on the peg",
    "lift the peg",
)


def _raw_observation(episode: dict[str, np.ndarray], frame: int, prompt: str) -> dict[str, object]:
    state = episode["state"][frame].copy().astype(np.float32)
    state[6] = float(state[6] > 0.12)
    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.camera0": torch.from_numpy(
            np.ascontiguousarray(episode["camera0"][frame].transpose(2, 0, 1))
        ),
        "observation.images.camera1": torch.from_numpy(
            np.ascontiguousarray(episode["camera1"][frame].transpose(2, 0, 1))
        ),
        "task": prompt,
    }


def _postprocess_chunk(chunk: torch.Tensor, postprocessor: object) -> torch.Tensor:
    return torch.stack(
        [postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])],
        dim=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, required=True)
    parser.add_argument("--frame", type=int, action="append", required=True)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    required = ("state", "action", "camera0", "camera1", "tool0_z", "task")
    with np.load(args.episode_npz, allow_pickle=True) as archive:
        episode = {key: archive[key] for key in required}

    sys.path.insert(0, str(LEROBOT_SRC))
    import lerobot.policies.pi05.processor_pi05  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

    checkpoint = args.checkpoint.resolve()
    policy = PI05Policy.from_pretrained(str(checkpoint), strict=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
    )
    prompt_steps = [
        step
        for step in preprocessor.steps
        if isinstance(step, Pi05PrepareStateTokenizerProcessorStep)
    ]
    if len(prompt_steps) != 1:
        raise RuntimeError(f"Expected one Pi0.5 prompt step, found {len(prompt_steps)}")
    prompt_steps[0].global_task = None

    prompts = tuple(args.prompt or DEFAULT_PROMPTS)
    rows = []
    for frame in args.frame:
        if not 0 <= frame < len(episode["state"]):
            raise ValueError(f"Frame {frame} is outside episode length {len(episode['state'])}")
        for prompt in prompts:
            batch = preprocessor(_raw_observation(episode, frame, prompt))
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(batch)
                physical = _postprocess_chunk(normalized, postprocessor)[0].float().cpu().numpy()
            prefix = physical[: args.execute_steps]
            rows.append(
                {
                    "frame": frame,
                    "dataset_phase": str(episode["task"][frame]),
                    "prompt": prompt,
                    "observed_tool0_z_m": float(episode["tool0_z"][frame]),
                    "first_action": physical[0].tolist(),
                    "execute_endpoint": prefix[-1].tolist(),
                    "execute_arm_delta_l2": float(
                        np.linalg.norm(prefix[-1, :6] - episode["state"][frame, :6])
                    ),
                    "execute_closed_fraction": float((prefix[:, 6] >= 0.5).mean()),
                    "full_closed_fraction": float((physical[:, 6] >= 0.5).mean()),
                    "key_actions": {
                        str(index): physical[index].tolist()
                        for index in (0, 9, 19, 29, 39, 49)
                        if index < len(physical)
                    },
                }
            )
            print(
                f"frame={frame} prompt={prompt!r} "
                f"endpoint={np.round(prefix[-1], 4)}",
                flush=True,
            )

    result = {
        "checkpoint": str(checkpoint),
        "episode": str(args.episode_npz.resolve()),
        "seed": args.seed,
        "execute_steps": args.execute_steps,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

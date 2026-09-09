#!/usr/bin/env python3
"""Probe a pure Pi0.5 checkpoint on exact expert-recovery observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
ROS_SCRIPT_DIR = Path(
    "/home/ubuntu/ur3_ft300_ws/src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole"
)
for path in (LEROBOT_SRC, ROS_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lerobot.policies.pi05.processor_pi05  # noqa: F401, E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from ur3_peg_in_hole_inference_common import postprocess_action_chunk  # noqa: E402


TASK = "pick up the peg and insert it into the hole"
GRIPPER_CLOSED_RAD = 0.629


def _observation(
    state: np.ndarray,
    camera0: np.ndarray,
    camera1: np.ndarray,
    gripper_state_mode: str,
) -> dict[str, object]:
    semantic_state = np.asarray(state, dtype=np.float32).copy()
    if gripper_state_mode == "semantic_binary":
        semantic_state[6] = float(semantic_state[6] > 0.12)
    elif gripper_state_mode == "continuous_0_1":
        semantic_state[6] = np.clip(semantic_state[6] / GRIPPER_CLOSED_RAD, 0.0, 1.0)
    else:  # Keep this helper safe when called outside argparse.
        raise ValueError(f"Unsupported gripper state mode: {gripper_state_mode}")
    return {
        "observation.state": torch.from_numpy(semantic_state),
        "observation.images.camera0": torch.from_numpy(
            np.ascontiguousarray(camera0.transpose(2, 0, 1))
        ),
        "observation.images.camera1": torch.from_numpy(
            np.ascontiguousarray(camera1.transpose(2, 0, 1))
        ),
        "task": TASK,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 80, 160])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--gripper-state-mode",
        choices=("semantic_binary", "continuous_0_1"),
        default="semantic_binary",
        help=(
            "State contract used by the checkpoint. continuous_0_1 maps the "
            "measured 0..0.629 rad joint to the official-style 0..1 feature."
        ),
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    with np.load(args.recovery, allow_pickle=False) as raw:
        intervention = np.asarray(raw["intervention_mask"], dtype=bool)
        takeover = int(np.flatnonzero(intervention)[0])
        selected = {}
        for offset in args.offsets:
            index = takeover + offset
            if index >= len(intervention) or not intervention[index]:
                raise ValueError(f"offset {offset} is outside the expert recovery")
            stop = min(index + 50, len(intervention))
            selected[offset] = {
                "state": np.asarray(raw["state"][index], dtype=np.float32),
                "camera0": np.asarray(raw["camera0"][index]),
                "camera1": np.asarray(raw["camera1"][index]),
                "expert": np.asarray(raw["expert_action"][index:stop], dtype=np.float32),
            }

    policy = PI05Policy.from_pretrained(str(checkpoint), strict=False)
    policy.config.rtc_config = None
    policy.init_rtc_processor()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()
    torch.set_grad_enabled(False)

    results = []
    for offset, sample in selected.items():
        batch = preprocessor(
            _observation(
                sample["state"],
                sample["camera0"],
                sample["camera1"],
                args.gripper_state_mode,
            )
        )
        chunks = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(batch)
                physical = postprocess_action_chunk(normalized, postprocessor)
            chunks.append(physical[0].detach().float().cpu().numpy())
        predicted = np.stack(chunks)
        predicted_mean = predicted.mean(axis=0)
        expert = sample["expert"]
        horizon = min(10, len(expert))
        results.append(
            {
                "offset": offset,
                "state_arm": sample["state"][:6].round(5).tolist(),
                "expert_first10_endpoint": expert[horizon - 1, :6].round(5).tolist(),
                "predicted_first10_endpoint_mean": predicted_mean[horizon - 1, :6].round(5).tolist(),
                "expert_first10_arm_delta_l2": float(
                    np.linalg.norm(expert[horizon - 1, :6] - sample["state"][:6])
                ),
                "predicted_first10_arm_delta_l2_mean": float(
                    np.linalg.norm(predicted_mean[horizon - 1, :6] - sample["state"][:6])
                ),
                "predicted_vs_expert_first10_endpoint_l2": float(
                    np.linalg.norm(predicted_mean[horizon - 1, :6] - expert[horizon - 1, :6])
                ),
                "expert_close_in_first10": bool(np.any(expert[:horizon, 6] >= 0.5)),
                "predicted_close_fraction_first10": float(
                    np.mean(predicted[:, :horizon, 6] >= 0.5)
                ),
                "predicted_close_fraction_full50": float(np.mean(predicted[:, :, 6] >= 0.5)),
                "predicted_gripper_first10_mean": predicted_mean[:horizon, 6].round(4).tolist(),
            }
        )

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "takeover": takeover,
                "gripper_state_mode": args.gripper_state_mode,
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

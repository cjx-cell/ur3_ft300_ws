#!/usr/bin/env python3
# ruff: noqa: I001
"""Isolate which online observation group changes a Pi0.5 action chunk."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPT_DIR = Path(__file__).resolve().parent


def _decode_case(
    *,
    policy: object,
    preprocessor: object,
    postprocessor: object,
    raw_observation: object,
    postprocess_chunk: object,
    state: np.ndarray,
    camera0: np.ndarray,
    camera1: np.ndarray,
    force: np.ndarray,
    force_fast: np.ndarray,
    force_slow: np.ndarray,
    state_history: np.ndarray,
) -> np.ndarray:
    batch = preprocessor(
        raw_observation(
            state,
            camera0,
            camera1,
            force,
            force_fast,
            force_slow,
            state_history,
        )
    )
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(batch)
        decoded = postprocess_chunk(normalized, postprocessor)
    return decoded[0].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, required=True)
    parser.add_argument("--online-trace-npz", type=Path, required=True)
    parser.add_argument("--episode-frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(0, str(SCRIPT_DIR))
    import lerobot.policies.pi05.processor_pi05  # noqa: F401
    from eval_pi05_checkpoint_stages import (
        _binarize_observed_gripper,
        _binarize_target_gripper,
        _postprocess_chunk,
        _raw_observation,
    )
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    required = (
        "state",
        "action",
        "camera0",
        "camera1",
        "force",
        "force_fast",
        "force_slow",
        "state_history",
        "task",
    )
    with np.load(args.episode_npz, allow_pickle=True) as archive:
        episode = {key: archive[key] for key in required}
    with np.load(args.online_trace_npz) as archive:
        online = {key: archive[key] for key in archive.files}
    frame = args.episode_frame
    if frame < 0 or frame + 1 > len(episode["state"]):
        raise ValueError(f"Invalid --episode-frame {frame}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PI05Policy.from_pretrained(str(args.checkpoint.resolve()), strict=False)
    policy.to(device=device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint.resolve()),
    )

    data_values = {
        "state": _binarize_observed_gripper(episode["state"][frame]),
        "camera0": episode["camera0"][frame].astype(np.float32),
        "camera1": episode["camera1"][frame].astype(np.float32),
        "force": episode["force"][frame].astype(np.float32),
        "force_fast": episode["force_fast"][frame].astype(np.float32),
        "force_slow": episode["force_slow"][frame].astype(np.float32),
        "state_history": episode["state_history"][frame].astype(np.float32),
    }
    online_values = {key: online[key].astype(np.float32) for key in data_values}
    cases = {
        "data_all": data_values,
        "online_all": online_values,
        "online_images_data_state_sensors": {
            **data_values,
            "camera0": online_values["camera0"],
            "camera1": online_values["camera1"],
        },
        "data_images_online_state_sensors": {
            **online_values,
            "camera0": data_values["camera0"],
            "camera1": data_values["camera1"],
        },
        "online_current_state_only": {
            **data_values,
            "state": online_values["state"],
        },
        "online_state_history_only": {
            **data_values,
            "state_history": online_values["state_history"],
        },
        "online_all_force_only": {
            **data_values,
            "force": online_values["force"],
            "force_fast": online_values["force_fast"],
            "force_slow": online_values["force_slow"],
        },
        "online_current_force_only": {
            **data_values,
            "force": online_values["force"],
        },
        "online_fast_force_only": {
            **data_values,
            "force_fast": online_values["force_fast"],
        },
        "online_slow_force_only": {
            **data_values,
            "force_slow": online_values["force_slow"],
        },
    }
    target = _binarize_target_gripper(
        episode["action"][frame : frame + policy.config.chunk_size],
        episode["task"][frame : frame + policy.config.chunk_size],
    )
    if len(target) != policy.config.chunk_size:
        raise ValueError(f"Frame {frame} has no complete {policy.config.chunk_size}-step target")
    horizon = int(policy.config.n_action_steps)
    results = {}
    for name, values in cases.items():
        prediction = _decode_case(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            raw_observation=_raw_observation,
            postprocess_chunk=_postprocess_chunk,
            **values,
        )
        results[name] = {
            "executed_arm_mae_rad": float(np.abs(prediction[:horizon, :6] - target[:horizon, :6]).mean()),
            "first_arm": prediction[0, :6].tolist(),
            "last_executed_arm": prediction[horizon - 1, :6].tolist(),
            "executed_arm": prediction[:horizon, :6].tolist(),
        }

    data_prediction = np.asarray(results["data_all"]["executed_arm"])
    for result in results.values():
        result["mean_shift_from_data_prediction_rad"] = float(
            np.abs(np.asarray(result["executed_arm"]) - data_prediction).mean()
        )
    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "episode": str(args.episode_npz.resolve()),
        "online_trace": str(args.online_trace_npz.resolve()),
        "episode_frame": frame,
        "execution_horizon": horizon,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

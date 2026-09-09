#!/usr/bin/env python3
"""Replay a recorded successful episode through the production ROS action path.

This is a causal plumbing test, not a deployable policy: each new observation
advances a fixed 10-step window through the demonstration actions.  It accepts
the baseline inference CLI so the normal Gazebo evaluation launcher can be
used unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
ACTION_FILE = "/tmp/ur3_action.txt"
ACTION_CHUNK_FILE = "/tmp/ur3_action_chunk.npy"
ACTION_CHUNK_TMP_FILE = "/tmp/ur3_action_chunk_tmp.npy"
READY_FILE = "/tmp/ur3_inference_ready.txt"
PREDICTED_ACTION_STEPS = 50
EXECUTED_ACTION_STEPS = 10


def _atomic_write_text(path: str, value: str) -> None:
    temporary = f"{path}.tmp"
    Path(temporary).write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--physical-arm-residual-scale", type=float, default=1.0)
    parser.add_argument("--max-arm-step-rad", type=float, default=0.0)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-first-chunks", type=int, default=0)
    parser.add_argument("--fixed-noise-per-replan", action="store_true")
    args = parser.parse_args()

    episode_path_text = os.environ.get("PI05_ORACLE_EPISODE_NPZ")
    if not episode_path_text:
        raise ValueError("PI05_ORACLE_EPISODE_NPZ is required")
    episode_path = Path(episode_path_text).resolve()
    with np.load(episode_path, allow_pickle=True) as archive:
        actions = archive["action"].astype(np.float32)
        tasks = np.asarray([str(value) for value in archive["task"]])
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected episode actions [T, 7], got {actions.shape}")
    open_phases = {
        "release the peg after verification",
        "retract and go back to home",
    }
    actions[:, 6] = (
        (actions[:, 6] > 0.12) & ~np.isin(tasks, list(open_phases))
    ).astype(np.float32)

    checkpoint_config = json.loads(
        (args.checkpoint.resolve() / "config.json").read_text(encoding="utf-8")
    )
    release_override = bool(checkpoint_config.get("use_release_gripper_override", False))
    ready = {
        "mode": "oracle_episode_temporal_replay",
        "episode": str(episode_path),
        "predicted_action_steps": PREDICTED_ACTION_STEPS,
        "executed_action_steps": EXECUTED_ACTION_STEPS,
        "action_dt_s": 0.1,
        "seed": args.seed,
        "fixed_noise_per_replan": args.fixed_noise_per_replan,
        "ensemble_size": args.ensemble_size,
        "physical_arm_residual_scale": args.physical_arm_residual_scale,
        "max_arm_step_rad": args.max_arm_step_rad,
        "gripper_action_mode": "binary_threshold_0.5",
        "state_gripper_mode": "binary_threshold_0.12",
        "release_gripper_override": release_override,
        "release_head_probability_threshold": checkpoint_config.get(
            "release_head_probability_threshold"
        ),
    }
    _atomic_write_text(READY_FILE, json.dumps(ready) + "\n")
    print(
        f"Oracle replay ready: {episode_path}, frames={len(actions)}, "
        f"publish={EXECUTED_ACTION_STEPS}",
        flush=True,
    )

    while not os.path.exists(JOINT_STATE_FILE):
        time.sleep(0.01)
    last_observation_mtime: float | None = None
    frame = 0
    chunk_id = 0
    while True:
        try:
            observation_mtime = os.path.getmtime(JOINT_STATE_FILE)
            if observation_mtime == last_observation_mtime:
                time.sleep(0.001)
                continue
            last_observation_mtime = observation_mtime

            end = min(frame + EXECUTED_ACTION_STEPS, len(actions))
            chunk = actions[frame:end].copy()
            if len(chunk) < EXECUTED_ACTION_STEPS:
                chunk = np.concatenate(
                    [
                        chunk,
                        np.repeat(actions[-1][None], EXECUTED_ACTION_STEPS - len(chunk), axis=0),
                    ],
                    axis=0,
                )
            np.save(ACTION_CHUNK_TMP_FILE, chunk)
            os.replace(ACTION_CHUNK_TMP_FILE, ACTION_CHUNK_FILE)
            _atomic_write_text(
                ACTION_FILE,
                " ".join(f"{value:.6f}" for value in chunk[0]) + "\n",
            )
            print(
                f"[oracle chunk {chunk_id:05d}] frames={frame}:{end}, "
                f"first_action={np.round(chunk[0], 4)}",
                flush=True,
            )
            frame = end
            chunk_id += 1
        except KeyboardInterrupt:
            break
        except Exception as error:
            print(f"Oracle replay error: {error}", flush=True)
            time.sleep(0.05)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate one raw corpus before deriving PAP-MoE and baseline datasets."""

import argparse
from pathlib import Path

import numpy as np


BASELINE_INPUTS = (
    "observation.images.camera0",
    "observation.images.camera1",
    "observation.state",
)
PAP_MOE_EXTRA_INPUTS = (
    "observation.force",
    "observation.force_fast",
    "observation.force_slow",
    "observation.state_history",
    "observation.visual_quality",
)


def canonical_gripper(values: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(values, dtype=np.float32), 0.0, 0.8)
    result[np.abs(result) <= 1.0e-3] = 0.0
    result[np.abs(result - 0.8) <= 1.0e-3] = 0.8
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_episode(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        required = {
            "state", "action", "camera0", "camera1", "force",
            "force_fast", "force_slow", "state_history", "stage",
            "timestamp_ros", "task", "semantic_subtask", "controller_result",
            "trajectory_scope", "policy_episode_endpoint", "peg_x", "peg_y",
            "hole_x", "hole_y", "trajectory_style_id", "position_group_id",
            "position_repeat_index", "position_repeat_count",
        }
        missing = sorted(required.difference(data.files))
        require(not missing, f"{path}: missing fields {missing}")

        state = np.asarray(data["state"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        n = len(state)
        require(state.shape == (n, 7), f"{path}: state shape {state.shape}")
        require(action.shape == (n, 7), f"{path}: action shape {action.shape}")
        require(n >= 100, f"{path}: only {n} frames; ACT horizon cannot be sampled")
        require(np.isfinite(state).all(), f"{path}: non-finite state")
        require(np.isfinite(action).all(), f"{path}: non-finite action")

        for camera in ("camera0", "camera1"):
            image = np.asarray(data[camera])
            require(image.shape == (n, 224, 224, 3), f"{path}: {camera} shape {image.shape}")
            require(np.isfinite(image).all(), f"{path}: non-finite {camera}")
            require(float(image.min()) >= 0.0 and float(image.max()) <= 1.0,
                    f"{path}: {camera} outside [0,1]")

        require(np.asarray(data["force"]).shape == (n, 6), f"{path}: force shape")
        require(np.asarray(data["force_fast"]).shape == (n, 64, 6), f"{path}: force_fast shape")
        require(np.asarray(data["force_slow"]).shape == (n, 50, 6), f"{path}: force_slow shape")
        require(np.asarray(data["state_history"]).shape == (n, 10, 7), f"{path}: state_history shape")

        timestamp = np.asarray(data["timestamp_ros"], dtype=np.float64)
        dt = np.diff(timestamp)
        require(np.all(dt > 0), f"{path}: timestamps are not strictly increasing")
        require(0.09 <= float(np.median(dt)) <= 0.11,
                f"{path}: policy dt median={np.median(dt):.6f}")

        task = np.asarray(data["task"], dtype=object)
        require(len(np.unique(task)) == 1, f"{path}: per-frame task oracle detected")
        require(str(task[0]) == "pick up the peg and insert it into the hole",
                f"{path}: unexpected global task {task[0]!r}")
        subtasks = np.asarray(data["semantic_subtask"], dtype=object)
        require(not np.any(subtasks == "retract and go back to home"),
                f"{path}: post-success reset leaked into policy episode")
        require(str(data["controller_result"]) == "success", f"{path}: not successful")
        require(str(data["trajectory_scope"]) == "full_task", f"{path}: not full task")

        routing = np.asarray(data["stage"], dtype=np.float32)
        require(routing.shape == (n, 4), f"{path}: routing shape {routing.shape}")
        require(np.all(routing >= 0.0) and np.all(routing <= 1.0),
                f"{path}: routing outside [0,1]")
        require(np.allclose(routing.sum(axis=1), 1.0, atol=2e-4),
                f"{path}: routing weights do not sum to one")

        gripper = canonical_gripper(action[:, 6])
        require(float(gripper.min()) == 0.0, f"{path}: no universal open endpoint")
        require(np.isclose(float(gripper.max()), 0.8, atol=1e-7),
                f"{path}: no universal closed endpoint")
        require(np.any((gripper > 0.01) & (gripper < 0.79)),
                f"{path}: continuous gripper transition is missing")
        # The physical state must remain measured and may stop below 0.8 on contact.
        require(float(state[:, 6].max()) < 0.75,
                f"{path}: measured gripper state looks overwritten by command")

        return {
            "path": str(path),
            "frames": n,
            "state": state,
            "action": np.column_stack((action[:, :6], gripper)),
            "position": tuple(float(data[key]) for key in ("peg_x", "peg_y", "hole_x", "hole_y")),
            "style": int(data["trajectory_style_id"]),
            "group": int(data["position_group_id"]),
            "repeat": int(data["position_repeat_index"]),
            "repeat_count": int(data["position_repeat_count"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("*_success/data.npz"))
    require(bool(paths), f"No successful episodes under {args.root}")
    episodes = [validate_episode(path) for path in paths]

    states = np.concatenate([episode["state"] for episode in episodes])
    actions = np.concatenate([episode["action"] for episode in episodes])
    q01, q99 = np.quantile(actions, (0.01, 0.99), axis=0)
    require(np.all(q99 - q01 > 1e-6), "Pi0.5 quantile action range collapsed")
    require(q01[6] == 0.0 and np.isclose(q99[6], 0.8, atol=1e-7),
            "Pi0.5 gripper quantiles do not span [0,0.8]")
    require(np.all(np.std(states, axis=0) > 1e-7), "ACT state std collapsed")
    require(np.all(np.ptp(actions, axis=0) > 1e-6), "DP action min/max collapsed")

    groups: dict[tuple[int, tuple[float, ...]], list[dict]] = {}
    for episode in episodes:
        key = (episode["group"], episode["position"])
        groups.setdefault(key, []).append(episode)
    repeated = [values for values in groups.values() if len(values) > 1]
    if repeated:
        for values in repeated:
            styles = [value["style"] for value in values]
            require(len(styles) == len(set(styles)),
                    f"Repeated scene has duplicate trajectory styles {styles}")

    print(f"PASS: {len(episodes)} episodes, {sum(e['frames'] for e in episodes)} frames")
    print(f"Baseline inputs (Pi0.5/ACT/DP): {BASELINE_INPUTS} -> action[7]")
    print(f"PAP-MoE extras: {PAP_MOE_EXTRA_INPUTS} + auxiliary routing/subtask targets")
    print("Pi0.5 normalization: q01/q99; ACT: mean/std; DP: min/max")
    print("action q01:", np.array2string(q01, precision=6))
    print("action q99:", np.array2string(q99, precision=6))
    for episode in episodes:
        print(
            f"  {Path(episode['path']).parent.name}: frames={episode['frames']} "
            f"position={episode['position']} style={episode['style']} "
            f"repeat={episode['repeat']}/{episode['repeat_count']}"
        )


if __name__ == "__main__":
    main()

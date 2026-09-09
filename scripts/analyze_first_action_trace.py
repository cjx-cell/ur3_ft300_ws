"""Summarize opt-in policy/ROS trace without inferring a causal failure label."""
import argparse
import json
from pathlib import Path

import numpy as np


def analyze(root):
    events = [json.loads(line) for line in (root / "controller.jsonl").read_text().splitlines()]
    rows = []
    for path in sorted(root.glob("chunk_*.npz")):
        chunk_id = int(path.stem.split("_")[-1])
        with np.load(path, allow_pickle=False) as data:
            state = data["observation/state"]
            action = data["published_action"]
            route = data["route_sequence"]
            dispatch = next((e for e in events if e["kind"] == "dispatch" and e["chunk_id"] == chunk_id), None)
            feedback = [e for e in events if e["kind"] == "feedback" and e["chunk_id"] == chunk_id]
            row = {
                "chunk": chunk_id, "observed_state": state.tolist(),
                "first_target": action[0].tolist(), "last_target": action[-1].tolist(),
                "first_arm_step_linf": float(np.abs(action[0, :6] - state[:6]).max()),
                "internal_arm_step_linf": float(np.abs(np.diff(action[:, :6], axis=0)).max()),
                "gripper_targets": action[:, 6].tolist(),
                "force": data["observation/force"].tolist(),
                "fast_force_max_abs": float(np.abs(data["observation/force_fast"]).max()),
                "expert_token_norms": data["expert_token_norms"].tolist(),
                "route_first": route[0].tolist() if route.ndim == 2 else route.tolist(),
                "route_executed_mean": route[:len(action)].mean(axis=0).tolist() if route.ndim == 2 else route.tolist(),
                "condition_norm": float(data["metadata/condition_residual_norm"]),
                "feedback_count": len(feedback),
            }
            if dispatch:
                row["dispatch_state"] = dispatch["state"]
                row["gripper_peg_distance_at_dispatch"] = dispatch["gripper_distance"]
                row["published_vs_requested_linf"] = float(np.abs(action - dispatch["requested"]).max())
                row["requested_vs_controller_linf"] = float(np.abs(np.array(dispatch["requested"]) - dispatch["controller"]).max())
            if feedback:
                row["last_feedback"] = feedback[-1]
                row["max_arm_feedback_error"] = max(float(np.abs(e["error"][:6]).max()) for e in feedback)
            rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace), indent=2))

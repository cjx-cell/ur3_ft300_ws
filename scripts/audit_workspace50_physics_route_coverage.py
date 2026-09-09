#!/usr/bin/env python3
"""Audit four-expert soft-route coverage in canonical Workspace50 episodes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPERT_NAMES = ("E1_free_motion", "E2_visual_blind", "E3_rigid_contact", "E4_movable")


def _summary(routes: np.ndarray) -> dict[str, object]:
    dominant = routes.argmax(axis=-1)
    experts = {}
    for index, name in enumerate(EXPERT_NAMES):
        selected = routes[dominant == index]
        experts[name] = {
            "mean_weight": float(routes[:, index].mean()),
            "max_weight": float(routes[:, index].max()),
            "frames_argmax": int(len(selected)),
            "fraction_gt_0_1": float(np.mean(routes[:, index] > 0.1)),
            "fraction_gt_0_5": float(np.mean(routes[:, index] > 0.5)),
            "fraction_gt_0_9": float(np.mean(routes[:, index] > 0.9)),
            "dominant_centroid": None if not len(selected) else selected.mean(axis=0).tolist(),
            "dominant_median": None if not len(selected) else np.median(selected, axis=0).tolist(),
        }
    return {"frames": int(len(routes)), "experts": experts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    episode_paths = sorted(args.dataset.glob("*_success/data.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No successful episode NPZ files found in {args.dataset}")

    all_routes = []
    grouped: dict[str, dict[str, list[np.ndarray]]] = {
        "semantic_subtask": defaultdict(list),
        "skill_progress_phase_name": defaultdict(list),
    }
    for episode_path in episode_paths:
        with np.load(episode_path, allow_pickle=True) as episode:
            routes = np.asarray(episode["stage"], dtype=np.float32)
            if routes.ndim != 2 or routes.shape[1] != len(EXPERT_NAMES):
                raise ValueError(f"Invalid stage shape in {episode_path}: {routes.shape}")
            if not np.isfinite(routes).all() or np.any(routes < 0):
                raise ValueError(f"Invalid route values in {episode_path}")
            routes = routes / np.clip(routes.sum(axis=-1, keepdims=True), 1e-8, None)
            all_routes.append(routes)
            for key, values in grouped.items():
                if key not in episode:
                    continue
                for label, route in zip(episode[key], routes, strict=True):
                    values[str(label)].append(route)

    result = {
        "dataset": str(args.dataset.resolve()),
        "episodes": len(episode_paths),
        "overall": _summary(np.concatenate(all_routes, axis=0)),
        "groups": {
            key: {
                label: _summary(np.asarray(routes, dtype=np.float32))
                for label, routes in sorted(values.items())
            }
            for key, values in grouped.items()
        },
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()

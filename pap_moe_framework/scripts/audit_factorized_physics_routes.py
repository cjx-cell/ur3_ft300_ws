#!/usr/bin/env python3
"""Audit four-expert targets and their task-agnostic b/c/m factorization.

This script is read-only. It never rewrites a LeRobot dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROUTE_KEY = "observation.physics_gate_target"
FORCE_KEY = "observation.force"
QUALITY_KEY = "observation.visual_quality"


def _stack(table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float64)


def expert_probs_to_factors(routes: np.ndarray) -> np.ndarray:
    routes = np.maximum(routes, 0.0)
    routes /= np.maximum(routes.sum(axis=-1, keepdims=True), 1e-12)
    x = routes[:, 1]
    y = routes[:, 2] + routes[:, 3]
    product = x * y
    discriminant = np.maximum(1.0 - 4.0 * product, 0.0)
    z = np.ones_like(product)
    active = product > 1e-12
    z[active] = (1.0 - np.sqrt(discriminant[active])) / (2.0 * product[active])
    blindness = np.clip(x * z, 0.0, 1.0)
    contact = np.clip(y * z, 0.0, 1.0)
    mobility = np.divide(
        routes[:, 3],
        y,
        out=np.zeros_like(y),
        where=y > 1e-12,
    )
    return np.stack([blindness, contact, np.clip(mobility, 0.0, 1.0)], axis=-1)


def factors_to_expert_probs(factors: np.ndarray) -> np.ndarray:
    blindness, contact, mobility = factors.T
    raw = np.stack(
        [
            (1.0 - blindness) * (1.0 - contact),
            blindness,
            contact * (1.0 - mobility),
            contact * mobility,
        ],
        axis=-1,
    )
    return raw / np.maximum(raw.sum(axis=-1, keepdims=True), 1e-12)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    parquet_files = sorted((args.dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files below {args.dataset_root / 'data'}")
    tables = [
        pq.read_table(
            path,
            columns=[ROUTE_KEY, FORCE_KEY, QUALITY_KEY, "episode_index"],
        )
        for path in parquet_files
    ]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    routes = _stack(table, ROUTE_KEY)
    force = _stack(table, FORCE_KEY)
    quality = _stack(table, QUALITY_KEY)
    episode = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    routes = np.maximum(routes, 0.0)
    routes /= np.maximum(routes.sum(axis=-1, keepdims=True), 1e-12)
    factors = expert_probs_to_factors(routes)
    reconstructed = factors_to_expert_probs(factors)
    dominant = routes.argmax(axis=-1)
    force_norm = np.linalg.norm(force[:, :3], axis=-1)

    transitions = 0
    for episode_id in np.unique(episode):
        sequence = dominant[episode == episode_id]
        transitions += int(np.count_nonzero(sequence[1:] != sequence[:-1]))

    normal_visual = quality[:, 3] > 0.5
    e1_force_reference = force_norm[dominant == 0]
    e1_q99 = float(np.quantile(e1_force_reference, 0.99))
    suspicious_contact = (routes[:, 2] + routes[:, 3] > 0.5) & (force_norm <= e1_q99)
    suspicious_blind = normal_visual & (routes[:, 1] > 0.5)

    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "frames": int(len(routes)),
        "episodes": int(len(np.unique(episode))),
        "dominant_expert_counts": {
            f"E{index + 1}": int(np.count_nonzero(dominant == index)) for index in range(4)
        },
        "mean_expert_route": routes.mean(axis=0).tolist(),
        "mean_bcm": factors.mean(axis=0).tolist(),
        "bcm_names": ["visual_blindness_b", "contact_c", "mobility_m"],
        "factorization_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - routes))
        ),
        "force_norm_by_dominant_expert": {
            f"E{index + 1}": _quantiles(force_norm[dominant == index])
            for index in range(4)
            if np.any(dominant == index)
        },
        "route_transitions": transitions,
        "route_transitions_per_episode": float(transitions / len(np.unique(episode))),
        "audit_flags": {
            "e1_force_q99_reference_n": e1_q99,
            "dominant_contact_at_or_below_e1_force_q99_frames": int(
                suspicious_contact.sum()
            ),
            "dominant_contact_at_or_below_e1_force_q99_fraction": float(
                suspicious_contact.mean()
            ),
            "e2_dominant_while_visual_valid_frames": int(suspicious_blind.sum()),
            "e2_dominant_while_visual_valid_fraction": float(suspicious_blind.mean()),
        },
        "notes": [
            "Audit flags identify rows for review; they are not automatic relabel decisions.",
            "Gazebo collision truth should replace force-only contact heuristics in the next collector.",
        ],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()

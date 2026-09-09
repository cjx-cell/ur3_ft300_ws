#!/usr/bin/env python3
"""Verify that a Pi0.5 action-only repair changed only the effective action path."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open


VLM_PREFIX = "model.paligemma_with_expert.paligemma."
ACTION_EXPERT_TOKEN = "model.paligemma_with_expert.gemma_expert."
UNUSED_ACTION_LM_HEAD = f"{ACTION_EXPERT_TOKEN}lm_head."


def _model_path(checkpoint: Path) -> Path:
    model = checkpoint.resolve() / "model.safetensors"
    if not model.is_file():
        raise FileNotFoundError(model)
    return model


def _group(key: str) -> str:
    if key.startswith(VLM_PREFIX):
        return "frozen_vlm"
    if key.startswith(UNUSED_ACTION_LM_HEAD):
        return "unused_lm_head"
    if key.startswith("model."):
        return "effective_action"
    return "unclassified"


def _max_abs_diff(
    source_tensor: torch.Tensor,
    repaired_tensor: torch.Tensor,
    chunk_elements: int = 1 << 20,
) -> float:
    """Compute a float32 max difference without allocating full-size copies."""
    source_flat = source_tensor.reshape(-1)
    repaired_flat = repaired_tensor.reshape(-1)
    maximum = 0.0
    for start in range(0, source_flat.numel(), chunk_elements):
        end = min(start + chunk_elements, source_flat.numel())
        difference = (
            source_flat[start:end].float()
            - repaired_flat[start:end].float()
        ).abs().max().item()
        maximum = max(maximum, difference)
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_checkpoint = args.source_checkpoint.resolve()
    checkpoint = args.checkpoint.resolve()
    source_model = _model_path(source_checkpoint)
    repaired_model = _model_path(checkpoint)
    if source_model == repaired_model:
        raise ValueError("The source and repaired checkpoints must be different")

    group_names = (
        "effective_action",
        "frozen_vlm",
        "unused_lm_head",
        "unclassified",
    )
    groups = {
        name: {
            "tensor_count": 0,
            "elements": 0,
            "equal_tensors": 0,
            "changed_tensors": 0,
            "max_abs_diff": 0.0,
        }
        for name in group_names
    }
    started = time.monotonic()
    with (
        safe_open(source_model, framework="pt", device="cpu") as source_file,
        safe_open(repaired_model, framework="pt", device="cpu") as repaired_file,
    ):
        source_keys = set(source_file.keys())
        repaired_keys = set(repaired_file.keys())
        if source_keys != repaired_keys:
            raise ValueError(
                "Checkpoint key mismatch: "
                f"source_only={sorted(source_keys - repaired_keys)[:10]}, "
                f"repaired_only={sorted(repaired_keys - source_keys)[:10]}"
            )
        for index, key in enumerate(sorted(source_keys), start=1):
            source_tensor = source_file.get_tensor(key)
            repaired_tensor = repaired_file.get_tensor(key)
            if (
                source_tensor.shape != repaired_tensor.shape
                or source_tensor.dtype != repaired_tensor.dtype
            ):
                raise ValueError(
                    f"Tensor metadata changed for {key}: "
                    f"{source_tensor.shape}/{source_tensor.dtype} != "
                    f"{repaired_tensor.shape}/{repaired_tensor.dtype}"
                )
            group = groups[_group(key)]
            equal = torch.equal(source_tensor, repaired_tensor)
            group["tensor_count"] += 1
            group["elements"] += source_tensor.numel()
            group["equal_tensors" if equal else "changed_tensors"] += 1
            if not equal:
                max_abs_diff = _max_abs_diff(
                    source_tensor, repaired_tensor
                )
                group["max_abs_diff"] = max(
                    group["max_abs_diff"], max_abs_diff
                )
            del source_tensor, repaired_tensor
            if index % 25 == 0:
                gc.collect()

    action = groups["effective_action"]
    vlm = groups["frozen_vlm"]
    lm_head = groups["unused_lm_head"]
    gates = {
        "effective_action_present": action["tensor_count"] > 0,
        "every_effective_action_tensor_changed": (
            action["changed_tensors"] == action["tensor_count"]
        ),
        "frozen_vlm_present": vlm["tensor_count"] > 0,
        "every_frozen_vlm_tensor_exact": (
            vlm["equal_tensors"] == vlm["tensor_count"]
        ),
        "unused_lm_head_present": lm_head["tensor_count"] > 0,
        "unused_lm_head_exact": (
            lm_head["equal_tensors"] == lm_head["tensor_count"]
        ),
        "no_unclassified_tensors": groups["unclassified"]["tensor_count"] == 0,
    }
    result = {
        "source_checkpoint": str(source_checkpoint),
        "checkpoint": str(checkpoint),
        "groups": groups,
        "gates": gates,
        "passed": all(gates.values()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()

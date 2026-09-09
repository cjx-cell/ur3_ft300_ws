#!/usr/bin/env python3
"""Interpolate a fine-tuned Pi0.5 checkpoint back toward its source.

The produced directory is a normal standalone checkpoint.  No runtime adapter,
stage signal, or privileged observation is introduced:

    merged = source + alpha * (fine_tuned - source)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MODEL_FILE = "model.safetensors"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fine-tuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    source_model = args.source / MODEL_FILE
    tuned_model = args.fine_tuned / MODEL_FILE
    if not source_model.is_file() or not tuned_model.is_file():
        raise FileNotFoundError("Both checkpoints must contain model.safetensors")

    args.output.mkdir(parents=True)
    for item in args.source.iterdir():
        if item.name == MODEL_FILE:
            continue
        destination = args.output / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    merged: dict[str, torch.Tensor] = {}
    changed_tensors = 0
    exact_source_tensors = 0
    with safe_open(source_model, framework="pt", device="cpu") as source_reader:
        with safe_open(tuned_model, framework="pt", device="cpu") as tuned_reader:
            source_keys = set(source_reader.keys())
            tuned_keys = set(tuned_reader.keys())
            if source_keys != tuned_keys:
                raise ValueError(
                    "Checkpoint tensor keys differ: "
                    f"source_only={sorted(source_keys - tuned_keys)[:10]}, "
                    f"fine_tuned_only={sorted(tuned_keys - source_keys)[:10]}"
                )
            for key in sorted(source_keys):
                source = source_reader.get_tensor(key)
                tuned = tuned_reader.get_tensor(key)
                if source.shape != tuned.shape or source.dtype != tuned.dtype:
                    raise ValueError(f"Tensor contract differs for {key}")
                if not source.is_floating_point() or torch.equal(source, tuned):
                    merged[key] = source.contiguous()
                    exact_source_tensors += 1
                    continue
                # Compute in fp32 so small recovery deltas are not rounded before
                # scaling, then restore the checkpoint's original dtype.
                source_fp32 = source.float()
                tuned_fp32 = tuned.float()
                value = source_fp32.lerp(tuned_fp32, args.alpha)
                merged[key] = value.to(dtype=source.dtype).contiguous()
                changed_tensors += 1

    save_file(merged, args.output / MODEL_FILE)
    provenance = {
        "contract": "standalone_pi05_checkpoint_interpolation_v1",
        "formula": "source + alpha * (fine_tuned - source)",
        "source": str(args.source.resolve()),
        "fine_tuned": str(args.fine_tuned.resolve()),
        "alpha": args.alpha,
        "changed_floating_tensors": changed_tensors,
        "exact_source_tensors": exact_source_tensors,
        "runtime_adapter": False,
        "privileged_policy_input": False,
    }
    (args.output / "checkpoint_interpolation.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compose independently gated PAP-MoE arm and gripper heads.

The arm checkpoint is authoritative for every parameter and configuration
field except ``model.gripper_head.*`` and the explicit ``gripper_head_*``
configuration contract copied from the gripper donor.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-checkpoint", type=Path, required=True)
    parser.add_argument("--gripper-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    if args.output_checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_checkpoint}")
    for checkpoint in (args.arm_checkpoint, args.gripper_checkpoint):
        for required in ("model.safetensors", "config.json"):
            if not (checkpoint / required).is_file():
                raise FileNotFoundError(checkpoint / required)

    args.output_checkpoint.mkdir(parents=True)
    for source in args.arm_checkpoint.iterdir():
        if source.name not in {"model.safetensors", "config.json"} and source.is_file():
            shutil.copy2(source, args.output_checkpoint / source.name)

    state = load_file(args.arm_checkpoint / "model.safetensors", device="cpu")
    donor_keys: list[str] = []
    with safe_open(
        args.gripper_checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as donor:
        for key in donor.keys():
            if key.startswith("model.gripper_head."):
                state[key] = donor.get_tensor(key)
                donor_keys.append(key)
    if not donor_keys:
        raise RuntimeError("Gripper donor contains no model.gripper_head parameters")
    save_file(state, args.output_checkpoint / "model.safetensors")

    arm_config = json.loads((args.arm_checkpoint / "config.json").read_text())
    gripper_config = json.loads((args.gripper_checkpoint / "config.json").read_text())
    copied_config: list[str] = []
    for key, value in gripper_config.items():
        if key.startswith("gripper_head_") or key in {
            "use_deterministic_gripper_head",
            "gripper_action_index",
        }:
            arm_config[key] = value
            copied_config.append(key)
    (args.output_checkpoint / "config.json").write_text(
        json.dumps(arm_config, indent=4) + "\n", encoding="utf-8"
    )
    receipt = {
        "arm_checkpoint": str(args.arm_checkpoint.resolve()),
        "gripper_checkpoint": str(args.gripper_checkpoint.resolve()),
        "gripper_parameter_keys": donor_keys,
        "gripper_config_keys": sorted(copied_config),
    }
    (args.output_checkpoint / "checkpoint_composition.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

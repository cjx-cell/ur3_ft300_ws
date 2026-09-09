#!/usr/bin/env python3
"""Repair/check persisted Pi0.5 prompt-step configs for a checkpoint tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


STEP_NAME = "pi05_prepare_state_tokenizer_processor_step"


def _checkpoint_dirs(path: Path) -> list[Path]:
    if (path / "config.json").is_file():
        return [path]
    # ``checkpoints/last`` is a symlink to the newest numbered checkpoint.
    # Resolve and deduplicate it so reports contain one row per physical model.
    return sorted(
        {
            candidate.parent.resolve()
            for candidate in path.glob("*/pretrained_model/config.json")
        }
    )


def _repair(checkpoint: Path) -> dict[str, object]:
    config_path = checkpoint / "config.json"
    processor_path = checkpoint / "policy_preprocessor.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    global_task = config.get("global_task")
    if not global_task:
        return {
            "checkpoint": str(checkpoint.resolve()),
            "changed": False,
            "skipped": "config has no global_task",
        }
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    matches = [
        step for step in processor["steps"] if step["registry_name"] == STEP_NAME
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{checkpoint}: expected exactly one {STEP_NAME}, got {len(matches)}"
        )
    expected = {
        "max_state_dim": int(config["max_state_dim"]),
        "task_key": "task",
        "global_task": str(global_task),
    }
    changed = matches[0].get("config") != expected
    matches[0]["config"] = expected
    if changed:
        temporary = processor_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(processor, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, processor_path)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "changed": changed,
        "global_task": global_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    checkpoints = _checkpoint_dirs(args.path.resolve())
    if not checkpoints:
        raise ValueError(f"No pretrained checkpoints found under {args.path}")
    results = [_repair(checkpoint) for checkpoint in checkpoints]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

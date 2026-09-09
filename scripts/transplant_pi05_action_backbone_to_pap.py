#!/usr/bin/env python3
"""Transplant a repaired Pi0.5 action backbone into a trained PAP-MoE checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file


VLM_PREFIX = "model.paligemma_with_expert.paligemma."
ACTION_EXPERT_TOKEN = "model.paligemma_with_expert.gemma_expert."
UNUSED_ACTION_LM_HEAD = f"{ACTION_EXPERT_TOKEN}lm_head."
LORA_PROJECTIONS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def _is_action_backbone_key(key: str) -> bool:
    """Select tensors on the effective Pi0.5 action-generation path."""
    return (
        key.startswith("model.")
        and not key.startswith(VLM_PREFIX)
        and not key.startswith(UNUSED_ACTION_LM_HEAD)
    )


def _pap_target_key(source_key: str, pap_keys: set[str]) -> str | None:
    if source_key in pap_keys:
        return source_key
    if ACTION_EXPERT_TOKEN not in source_key:
        return None
    parts = source_key.split(".")
    if (
        len(parts) >= 2
        and parts[-2] in LORA_PROJECTIONS
        and parts[-1] in {"weight", "bias"}
    ):
        parts.insert(-1, "base")
        mapped = ".".join(parts)
        if mapped in pap_keys:
            return mapped
    return None


def _load_config(checkpoint: Path) -> dict:
    return json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))


def _validate_contract(pi05_checkpoint: Path, pap_checkpoint: Path) -> None:
    pi05 = _load_config(pi05_checkpoint)
    pap = _load_config(pap_checkpoint)
    required_equal = (
        "chunk_size",
        "n_action_steps",
        "max_state_dim",
        "max_action_dim",
        "paligemma_variant",
        "action_expert_variant",
        "use_relative_actions",
        "action_feature_names",
    )
    mismatches = {
        key: (pi05.get(key), pap.get(key))
        for key in required_equal
        if pi05.get(key) != pap.get(key)
    }
    if mismatches:
        raise ValueError(f"Pi0.5/PAP action contract mismatch: {mismatches}")


def _build_mapping(
    pi05_model: Path, pap_model: Path
) -> tuple[dict[str, str], dict[str, list[int]], dict[str, list[int]]]:
    with safe_open(pi05_model, framework="pt", device="cpu") as pi05_file:
        pi05_shapes = {
            key: list(pi05_file.get_slice(key).get_shape())
            for key in pi05_file.keys()
        }
    with safe_open(pap_model, framework="pt", device="cpu") as pap_file:
        pap_shapes = {
            key: list(pap_file.get_slice(key).get_shape())
            for key in pap_file.keys()
        }

    pap_keys = set(pap_shapes)
    selected = [key for key in pi05_shapes if _is_action_backbone_key(key)]
    mapping = {}
    unmapped = []
    shape_mismatches = {}
    for source_key in selected:
        target_key = _pap_target_key(source_key, pap_keys)
        if target_key is None:
            unmapped.append(source_key)
            continue
        if pi05_shapes[source_key] != pap_shapes[target_key]:
            shape_mismatches[source_key] = {
                "target": target_key,
                "pi05_shape": pi05_shapes[source_key],
                "pap_shape": pap_shapes[target_key],
            }
            continue
        mapping[source_key] = target_key

    if unmapped or shape_mismatches:
        raise ValueError(
            "Action-backbone mapping is incomplete: "
            f"unmapped={unmapped[:10]}, shape_mismatches={shape_mismatches}"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Multiple Pi0.5 tensors map to the same PAP tensor")
    return mapping, pi05_shapes, pap_shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05-checkpoint", type=Path, required=True)
    parser.add_argument("--pap-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pi05_checkpoint = args.pi05_checkpoint.resolve()
    pap_checkpoint = args.pap_checkpoint.resolve()
    pi05_model = pi05_checkpoint / "model.safetensors"
    pap_model = pap_checkpoint / "model.safetensors"
    for required in (
        pi05_model,
        pap_model,
        pi05_checkpoint / "config.json",
        pap_checkpoint / "config.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    _validate_contract(pi05_checkpoint, pap_checkpoint)
    mapping, pi05_shapes, pap_shapes = _build_mapping(pi05_model, pap_model)
    mapped_direct = sum(source == target for source, target in mapping.items())
    mapped_lora_base = len(mapping) - mapped_direct
    preserved_pap = len(pap_shapes) - len(mapping)
    report = {
        "pi05_checkpoint": str(pi05_checkpoint),
        "pap_checkpoint": str(pap_checkpoint),
        "action_tensors_transplanted": len(mapping),
        "direct_tensors": mapped_direct,
        "lora_base_tensors": mapped_lora_base,
        "pap_tensors_preserved": preserved_pap,
        "pi05_action_parameter_elements": int(
            sum(np.prod(pi05_shapes[key]) for key in mapping)
        ),
        "mapping": mapping,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.output is None:
        raise ValueError("--output is required unless --dry-run is used")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    pap_state = load_file(pap_model, device="cpu")
    pi05_state = load_file(pi05_model, device="cpu")
    for source_key, target_key in mapping.items():
        pap_state[target_key] = pi05_state[source_key].to(
            dtype=pap_state[target_key].dtype
        )

    with safe_open(pap_model, framework="pt", device="cpu") as pap_file:
        metadata = pap_file.metadata()
    save_file(pap_state, output / "model.safetensors", metadata=metadata)

    for source in pap_checkpoint.iterdir():
        if source.name == "model.safetensors" or not source.is_file():
            continue
        shutil.copy2(source, output / source.name)
    output_config = _load_config(output)
    processor_path = output / "policy_preprocessor.json"
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    prompt_steps = [
        step
        for step in processor["steps"]
        if step["registry_name"]
        == "pap_moe_prepare_state_tokenizer_processor_step"
    ]
    if len(prompt_steps) != 1:
        raise ValueError(
            "Expected exactly one PAP-MoE prompt processor in transplanted checkpoint"
        )
    prompt_steps[0]["config"] = {
        "max_state_dim": int(output_config["max_state_dim"]),
        "task_key": "task",
        "global_task": str(output_config["global_task"]),
    }
    processor_path.write_text(
        json.dumps(processor, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "action_backbone_transplant.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

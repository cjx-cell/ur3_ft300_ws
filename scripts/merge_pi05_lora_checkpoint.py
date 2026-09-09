#!/usr/bin/env python3
"""Merge a LeRobot Pi0.5 LoRA checkpoint into its full base checkpoint.

The command fails closed: every LoRA A/B pair must map to one base tensor and
have a compatible shape.  Processor metadata and normalization tensors are
copied from the adapter checkpoint, because those files define the inference
contract that produced the successful rollout.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def base_key_for_lora_target(target: str) -> str:
    key = target.removeprefix("base_model.model.") + ".weight"
    # PEFT addresses SigLIP's encoder through the wrapper, whereas the saved
    # Pi0.5 state dict includes the intermediate vision_model component.
    return key.replace(
        ".vision_tower.encoder.", ".vision_tower.vision_model.encoder."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_model = require(args.base / "model.safetensors")
    adapter_model = require(args.adapter / "adapter_model.safetensors")
    adapter_config_path = require(args.adapter / "adapter_config.json")
    require(args.base / "config.json")

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")

    adapter_config = json.loads(adapter_config_path.read_text())
    rank = int(adapter_config["r"])
    alpha = float(adapter_config.get("lora_alpha", rank))
    scaling = alpha / rank

    base_state = load_file(str(base_model), device="cpu")
    adapter_state = load_file(str(adapter_model), device="cpu")
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    unexpected_adapter_keys: list[str] = []
    for key, tensor in adapter_state.items():
        if ".lora_A." in key:
            pairs[key.split(".lora_A.", 1)[0]]["A"] = tensor
        elif ".lora_B." in key:
            pairs[key.split(".lora_B.", 1)[0]]["B"] = tensor
        else:
            unexpected_adapter_keys.append(key)

    errors: list[str] = []
    merged: list[dict[str, object]] = []
    for target, tensors in sorted(pairs.items()):
        if set(tensors) != {"A", "B"}:
            errors.append(f"{target}: incomplete pair {sorted(tensors)}")
            continue
        base_key = base_key_for_lora_target(target)
        if base_key not in base_state:
            errors.append(f"{target}: missing base tensor {base_key}")
            continue
        a, b = tensors["A"], tensors["B"]
        if a.ndim != 2 or b.ndim != 2:
            errors.append(f"{target}: expected matrices, got A{tuple(a.shape)} B{tuple(b.shape)}")
            continue
        delta = (b.float() @ a.float()) * scaling
        original = base_state[base_key]
        if delta.shape != original.shape:
            errors.append(
                f"{target}: delta {tuple(delta.shape)} != base {tuple(original.shape)}"
            )
            continue
        base_state[base_key] = (original.float() + delta).to(original.dtype)
        merged.append(
            {
                "lora_target": target,
                "base_key": base_key,
                "shape": list(original.shape),
                "dtype": str(original.dtype),
            }
        )

    if unexpected_adapter_keys:
        errors.append(f"unsupported non-LoRA adapter keys: {unexpected_adapter_keys}")
    if errors or not merged:
        raise RuntimeError("LoRA merge validation failed:\n" + "\n".join(errors))

    args.output.mkdir(parents=True, exist_ok=True)
    output_model = args.output / "model.safetensors"
    save_file(base_state, str(output_model))
    del base_state, adapter_state
    gc.collect()

    # A merged checkpoint is a normal full Pi0.5 model, not a PEFT model.
    merged_config = json.loads((args.adapter / "config.json").read_text())
    merged_config["use_peft"] = False
    merged_config["pretrained_path"] = str(args.output.resolve())
    (args.output / "config.json").write_text(
        json.dumps(merged_config, ensure_ascii=False, indent=4) + "\n"
    )

    processor_names = ["policy_preprocessor.json", "policy_postprocessor.json"]
    processor_names.extend(path.name for path in args.adapter.glob("policy_*_processor.safetensors"))
    for name in sorted(set(processor_names)):
        shutil.copy2(require(args.adapter / name), args.output / name)

    receipt = {
        "format": "pi05_full_checkpoint_from_lora_v1",
        "base": str(args.base.resolve()),
        "base_model_sha256": sha256(base_model),
        "adapter": str(args.adapter.resolve()),
        "adapter_model_sha256": sha256(adapter_model),
        "rank": rank,
        "alpha": alpha,
        "scaling": scaling,
        "merged_target_count": len(merged),
        "merged_targets": merged,
        "output_model_sha256": sha256(output_model),
    }
    (args.output / "merge_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"merged {len(merged)} LoRA targets")
    print(f"output: {output_model} ({output_model.stat().st_size / 1e9:.2f} GB)")
    print(f"sha256: {receipt['output_model_sha256']}")


if __name__ == "__main__":
    main()

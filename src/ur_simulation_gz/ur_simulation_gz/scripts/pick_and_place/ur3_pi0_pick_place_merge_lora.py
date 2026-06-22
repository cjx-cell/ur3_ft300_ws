#!/usr/bin/env python3
"""离线 merge LoRA adapter 到 base 权重，输出单个 model.safetensors

用法:
  python3 merge_lora.py --base <预训练权> --lora <LoRA checkpoint> --output <输出目录>

示例:
  python3 merge_lora.py \
    --base /home/ubuntu/ur3_ft300_ws/ai-models/lerobot/pi0 \
    --lora /home/ubuntu/ur3_ft300_ws/ai-models/ur3_pi0_lora_r16/checkpoints/040000/pretrained_model \
    --output /home/ubuntu/ur3_ft300_ws/ai-models/ur3_pi0_lora_r16_merged
"""

import argparse, torch, json, gc, shutil
from pathlib import Path
from collections import defaultdict
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base weights")
    parser.add_argument("--base", type=Path, required=True, help="预训练权重目录 (含 model_bf16.safetensors + config.json)")
    parser.add_argument("--lora", type=Path, required=True, help="LoRA checkpoint 目录 (含 adapter_model.safetensors)")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    args = parser.parse_args()

    base_dir = args.base
    lora_dir = args.lora
    output_dir = args.output

    # Load LoRA adapter config
    with open(lora_dir / "adapter_config.json") as f:
        lora_cfg = json.load(f)
    print(f"LoRA config: r={lora_cfg.get('r')}, alpha={lora_cfg.get('lora_alpha')}, "
          f"targets={lora_cfg.get('target_modules')}")
    r = lora_cfg["r"]
    alpha = lora_cfg.get("lora_alpha", r)
    scaling = alpha / r

    # Load base weights
    base_weights = base_dir / "model_bf16.safetensors"
    if not base_weights.exists():
        base_weights = base_dir / "model.safetensors"
    print(f"\nLoading base weights: {base_weights}")
    sd = load_file(str(base_weights), device="cpu")
    print(f"  Base: {len(sd)} keys")

    # Load LoRA weights
    print(f"Loading LoRA adapter: {lora_dir / 'adapter_model.safetensors'}")
    lora_sd = load_file(str(lora_dir / "adapter_model.safetensors"), device="cpu")
    print(f"  LoRA: {len(lora_sd)} keys")

    # Parse LoRA keys to find A/B pairs
    lora_pairs = defaultdict(dict)
    for key, tensor in lora_sd.items():
        parts = key.split(".lora_")
        if len(parts) != 2:
            continue
        target = parts[0]
        suffix = parts[1]
        if suffix.startswith("A."):
            lora_pairs[target]["A"] = tensor
        elif suffix.startswith("B."):
            lora_pairs[target]["B"] = tensor

    print(f"\nMerging {len(lora_pairs)} LoRA target modules...")
    merged_count = 0
    for target, tensors in lora_pairs.items():
        base_key = target.replace("base_model.model.", "") + ".weight"

        if "A" in tensors and "B" in tensors:
            lora_A = tensors["A"]
            lora_B = tensors["B"]

            if lora_B.ndim == 2 and lora_A.ndim == 2:
                delta = (lora_B @ lora_A) * scaling
            else:
                print(f"  Skip {target}: unexpected shapes A={lora_A.shape} B={lora_B.shape}")
                continue

            if base_key in sd:
                sd[base_key] = (sd[base_key].float() + delta.float()).to(torch.bfloat16)
                merged_count += 1
            else:
                print(f"  Skip {target}: base_key={base_key} not found")
        else:
            print(f"  Skip {target}: missing A or B")
    print(f"  Merged {merged_count} weights")

    # modules_to_save (full finetune layers)
    for key, tensor in lora_sd.items():
        if "lora_" not in key and "base_model.model." in key:
            base_key = key.replace("base_model.model.", "")
            if base_key in sd:
                sd[base_key] = tensor.to(torch.bfloat16)

    # Save
    print(f"\nSaving merged model to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(output_dir / "model.safetensors"))
    del sd; gc.collect()

    # Copy config files
    for f in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        src = base_dir / f
        if src.exists():
            shutil.copy(src, output_dir / f)
    # Copy normalizer stats from LoRA checkpoint
    for f in lora_dir.glob("policy_*_step_*"):
        shutil.copy(f, output_dir / f.name)

    size_gb = (output_dir / "model.safetensors").stat().st_size / 1e9
    print(f"Done: {output_dir / 'model.safetensors'} ({size_gb:.1f}GB)")


if __name__ == "__main__":
    main()

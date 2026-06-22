#!/usr/bin/env python3
"""修复 Pi0 checkpoint 中 vision encoder 的 key 命名——去掉多余的 vision_model 层级。"""
from safetensors.torch import load_file, save_file
from pathlib import Path

model_dir = Path("/home/ubuntu/ur3_ft300_ws/ai-models/lerobot/pi0")

# 加载原始权重
sd = load_file(str(model_dir / "model.safetensors"), device="cpu")

new_sd = {}
changed = 0
for k, v in sd.items():
    new_k = k.replace(".vision_tower.vision_model.", ".vision_tower.")
    new_sd[new_k] = v
    if new_k != k:
        changed += 1

print(f"Remapped {changed} / {len(sd)} keys")

# 保存修复后的权重
save_file(new_sd, str(model_dir / "model_fixed.safetensors"))
print("Saved model_fixed.safetensors")

# 验证
test_sd = load_file(str(model_dir / "model_fixed.safetensors"), device="cpu")
bad = [k for k in test_sd if "vision_tower.vision_model" in k]
print(f"Remaining vision_model keys: {len(bad)}")
print("Done")

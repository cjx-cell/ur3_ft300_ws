#!/usr/bin/env python3
"""
Pi0 单次推理测试 — 验证模型加载、观测输入、动作输出是否正常。

用法 (conda pi0-env):
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env
  python3 ur3_pi0_test.py              # 使用 /tmp/ 中的真实数据
  python3 ur3_pi0_test.py --dummy      # 使用随机数据测试
  python3 ur3_pi0_test.py --mode bf16  # GPU bfloat16 模式
"""

import os, sys, json, time, argparse
import numpy as np
import torch
from pathlib import Path

MODEL_DIR = Path("/home/ubuntu/ur3_ft300_ws/ai-models/lerobot/pi0")
JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"

# UR3 关节近似归一化参数（无原始数据集统计，估算值）
STATE_MEAN = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
STATE_STD = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float32)
ACTION_MEAN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ACTION_STD = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

TASK_PROMPT = "pick up the red cube and place it into the bowl\n"
IMG_SIZE = (224, 224)


def gpu_memory_str():
    if not torch.cuda.is_available():
        return "N/A"
    a = torch.cuda.memory_allocated(0) / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{a:.2f}GB / {t:.1f}GB"


def load_model(mode="cpu"):
    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.pi0.configuration_pi0 import PI0Config

    device = "cpu" if mode == "cpu" else "cuda"
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32

    with open(MODEL_DIR / "config.json") as f:
        raw = json.load(f)
    input_features = {}
    for k, v in raw.get("input_features", {}).items():
        input_features[k] = PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
    output_features = {}
    for k, v in raw.get("output_features", {}).items():
        output_features[k] = PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))

    print(f"加载模型 (device={device}, dtype={dtype})...")
    t0 = time.time()

    config = PI0Config(
        device="cpu", dtype=str(dtype).split(".")[-1],
        empty_cameras=1,
        input_features=input_features,
        output_features=output_features,
    )
    policy = PI0Policy(config)
    print(f"  模型创建: {time.time() - t0:.1f}s")

    weights_path = MODEL_DIR / ("model_bf16.safetensors" if mode == "bf16" else "model.safetensors")
    print(f"  加载权重: {weights_path.name}")
    t1 = time.time()
    sd = load_file(str(weights_path), device="cpu")
    if mode == "bf16":
        for k in list(sd.keys()):
            if sd[k].dtype in (torch.float16, torch.float32, torch.bfloat16):
                sd[k] = sd[k].to(dtype=torch.bfloat16)
    print(f"  权重加载: {time.time() - t1:.1f}s ({len(sd)} keys)")

    t2 = time.time()
    policy.load_state_dict(sd, strict=False)
    del sd; gc.collect()
    print(f"  权重应用: {time.time() - t2:.1f}s")

    if device == "cuda":
        print("  移动到 CUDA...")
        t3 = time.time()
        policy = policy.to(device="cuda", dtype=dtype)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  移动完成: {time.time() - t3:.1f}s | GPU: {gpu_memory_str()}")

    policy.eval()
    torch.set_grad_enabled(False)
    print(f"  总加载时间: {time.time() - t0:.1f}s | GPU: {gpu_memory_str()}")
    return policy, device, dtype


def load_tokenizer():
    """加载本地 PaliGemma tokenizer。"""
    from transformers import AutoTokenizer
    tokenizer_path = "/home/ubuntu/ur3_ft300_ws/ai-models/paligemma_tokenizer"
    print(f"加载 PaliGemma tokenizer ({tokenizer_path})...")
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        print("  Tokenizer 加载成功")
        return tok
    except Exception as e:
        print(f"  WARNING: Tokenizer 加载失败 ({e})")
        print("  将使用随机 token (模型不受任务指令制约，仅用于验证)")
        return None


def tokenize_prompt(tokenizer, prompt, device, max_length=48):
    """Tokenize 任务指令。若 tokenizer 不可用，返回随机 token。"""
    if tokenizer is not None:
        tokens = tokenizer(prompt, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=max_length)
        return (tokens["input_ids"].to(device),
                tokens["attention_mask"].to(device).bool())
    else:
        # 离线模式：随机 token，PaliGemma 词表大小 256000
        ids = torch.randint(0, 256000, (1, max_length), device=device)
        mask = torch.ones(1, max_length, device=device, dtype=torch.bool)
        return ids, mask


def read_joint_state():
    try:
        with open(JOINT_STATE_FILE, "r") as f:
            line = f.readline().strip()
            if line:
                return np.array([float(x) for x in line.split()], dtype=np.float32)
    except Exception:
        pass
    return np.zeros(6, dtype=np.float32)


def read_camera_image(path):
    if path and os.path.exists(path):
        try:
            img = np.load(path, allow_pickle=False)
            if img.ndim == 3:
                return img.astype(np.float32)
        except Exception:
            pass
    return np.zeros((*IMG_SIZE, 3), dtype=np.float32)


def normalize_state(joint_pos):
    return (joint_pos - STATE_MEAN) / (STATE_STD + 1e-8)


def build_observation(tokenizer, joint_pos, wrist_img, global_img, device, dtype, task_prompt=TASK_PROMPT):
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    # 归一化状态
    state_norm = normalize_state(joint_pos)
    state_t = torch.from_numpy(state_norm).unsqueeze(0).to(device=device, dtype=dtype)

    # 图像: HWC [0,1] → CHW, BCHW
    def img_to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return t.to(device=device, dtype=dtype)

    # Tokenize 语言指令
    lang_ids, lang_mask = tokenize_prompt(tokenizer, task_prompt, device)

    batch = {
        "observation.state": state_t,
        "observation.images.camera0": img_to_tensor(wrist_img),
        "observation.images.camera1": img_to_tensor(global_img),
        "observation.images.camera2": torch.zeros(1, 3, *IMG_SIZE, device=device, dtype=dtype),
        OBS_LANGUAGE_TOKENS: lang_ids,
        OBS_LANGUAGE_ATTENTION_MASK: lang_mask,
    }
    return batch


def run_inference(policy, batch, mode):
    print("推理中...")
    t0 = time.time()

    policy.reset()  # 初始化 action queue

    with torch.no_grad():
        if mode == "bf16":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                action = policy.select_action(batch)
        else:
            action = policy.select_action(batch)

    elapsed = (time.time() - t0) * 1000
    action_np = action.cpu().float().numpy().flatten()

    # 逆归一化: 从归一化空间回到实际关节增量
    action_unnorm = action_np * ACTION_STD + ACTION_MEAN

    return action_np, action_unnorm, elapsed


def main():
    parser = argparse.ArgumentParser(description="Pi0 单次推理测试")
    parser.add_argument("--dummy", action="store_true", help="使用随机数据（不需要 ROS 仿真运行）")
    parser.add_argument("--mode", type=str, default="cpu", choices=["cpu", "bf16", "fp32"],
                        help="推理模式 (默认 cpu)")
    parser.add_argument("--prompt", type=str, default=TASK_PROMPT,
                        help="任务指令 (默认: pick up the red cube...)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Pi0 推理测试 | 模式={args.mode} | 数据={'随机' if args.dummy else '/tmp/ 文件'}")
    print(f"任务指令: {args.prompt.strip()}")
    print("=" * 60)

    # 1. 加载模型
    policy, device, dtype = load_model(args.mode)

    # 2. 加载 tokenizer
    tokenizer = load_tokenizer()

    # 3. 读取观测
    if args.dummy:
        joint_pos = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        wrist_img = np.random.rand(*IMG_SIZE, 3).astype(np.float32)
        global_img = np.random.rand(*IMG_SIZE, 3).astype(np.float32)
        print("使用随机数据")
    else:
        joint_pos = read_joint_state()
        wrist_img = read_camera_image(CAMERA0_FILE)
        global_img = read_camera_image(CAMERA1_FILE)
        print(f"关节状态: {np.array2string(joint_pos, precision=3)}")
        print(f"腕部相机: {'有数据' if wrist_img.any() else '无数据(全零)'} shape={wrist_img.shape}")
        print(f"全局相机: {'有数据' if global_img.any() else '无数据(全零)'} shape={global_img.shape}")

    # 4. 构造 batch
    print("\n构造观测 batch...")
    batch = build_observation(tokenizer, joint_pos, wrist_img, global_img, device, dtype, args.prompt)
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}, device={v.device}")

    # 5. 推理
    print()
    action_raw, action_unnorm, elapsed = run_inference(policy, batch, args.mode)

    # 6. 输出结果
    print(f"\n推理耗时: {elapsed:.0f}ms")
    print(f"动作输出 (归一化): {np.array2string(action_raw, precision=4)}")
    print(f"动作输出 (去归一化, rad): {np.array2string(action_unnorm, precision=4)}")
    print(f"当前关节 (rad): {np.array2string(joint_pos, precision=4)}")
    print(f"目标关节 (rad): {np.array2string(joint_pos + action_unnorm, precision=4)}")

    # 7. 检查输出合理性
    print("\n合理性检查:")
    max_delta = np.max(np.abs(action_unnorm))
    if max_delta > 0.5:
        print(f"  WARNING: 动作增量过大 ({max_delta:.3f} rad)，可能异常")
    elif max_delta < 0.001:
        print(f"  WARNING: 动作增量接近零，模型可能未正常工作")
    else:
        print(f"  OK: 最大增量 {max_delta:.3f} rad")
    print("=" * 60)


if __name__ == "__main__":
    main()

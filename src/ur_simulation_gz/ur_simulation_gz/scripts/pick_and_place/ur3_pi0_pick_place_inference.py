#!/usr/bin/env python3
"""
UR3 Pi0 LoRA 推理端 — Conda pi0-env 环境运行

循环读取 /tmp/ 中的观测数据，运行 Pi0 推理，输出动作到 /tmp/ur3_action.txt。

用法:
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env
  python3 ur3_pi0_inference.py --mode bf16 --hz 10
"""

import os, sys, json, time, argparse, gc, tempfile
import numpy as np
import torch
from pathlib import Path

from accelerate import init_empty_weights
from safetensors.torch import load_file
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from transformers import AutoTokenizer

# ── 路径配置 ──
MODEL_DIR = Path("/home/ubuntu/ur3_ft300_ws/ai-models/pi0/pi0_full_v1/checkpoints/084000/pretrained_model")
CKPT_DIR  = Path("/home/ubuntu/ur3_ft300_ws/ai-models/pi0/pi0_full_v1/checkpoints/084000/pretrained_model")
JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
ACTION_FILE = "/tmp/ur3_action.txt"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"
IMG_SIZE = (224, 224)

TASK_PROMPT = "pick up the red cube and place it into the bowl\n"


def gpu_memory_str():
    if not torch.cuda.is_available():
        return "N/A"
    a = torch.cuda.memory_allocated(0) / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{a:.2f}GB / {t:.1f}GB"


def load_normalizer_stats(ckpt_dir):
    """从 checkpoint 加载 state 归一化和 action 反归一化参数"""
    pre_file = None
    post_file = None
    for f in ckpt_dir.glob("policy_*step*.safetensors"):
        name = f.name
        if "preprocessor" in name and "normalizer" in name:
            pre_file = f
        elif "postprocessor" in name and "unnormalizer" in name:
            post_file = f

    if pre_file:
        pre = load_file(str(pre_file))
        state_mean = pre["observation.state.mean"].numpy().astype(np.float32)
        state_std  = pre["observation.state.std"].numpy().astype(np.float32)
        print(f"  状态归一化: mean={np.array2string(state_mean, precision=3)}")
        print(f"               std={np.array2string(state_std, precision=3)}")
    else:
        print("  WARNING: 未找到 state normalizer，使用 identity")
        state_mean = np.zeros(7, dtype=np.float32)
        state_std  = np.ones(7, dtype=np.float32)

    if post_file:
        post = load_file(str(post_file))
        action_mean = post["action.mean"].numpy().astype(np.float32)
        action_std  = post["action.std"].numpy().astype(np.float32)
        print(f"  动作反归一化: mean={np.array2string(action_mean, precision=3)}")
        print(f"                 std={np.array2string(action_std, precision=3)}")
    else:
        print("  WARNING: 未找到 action unnormalizer，使用 identity")
        action_mean = np.zeros(7, dtype=np.float32)
        action_std  = np.ones(7, dtype=np.float32)

    return state_mean, state_std, action_mean, action_std


class UR3Pi0Inference:
    def __init__(self, mode="cpu", hz=5, model_dir=None):
        self.mode = mode
        self.hz = hz
        self.model_dir = model_dir or MODEL_DIR
        self.device = "cuda" if mode in ("bf16", "fp32") and torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if mode == "bf16" else torch.float32

        print(f"UR3 Pi0 LoRA 推理 | 模式={self.mode.upper()} | 频率={hz}Hz | 设备={self.device}")
        self.policy = None
        self.tokenizer = None
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self._init_files()
        self._load()

    def _init_files(self):
        for path, default in [(JOINT_STATE_FILE, "0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"),
                               (ACTION_FILE, "0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")]:
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(default)

    def _load(self):
        md = self.model_dir
        print(f"加载模型: {md}")
        t0 = time.time()

        # ── 加载 normalizer stats ──
        self.state_mean, self.state_std, self.action_mean, self.action_std = load_normalizer_stats(CKPT_DIR)

        # ── Config ──
        with open(md / "config.json") as f:
            raw = json.load(f)
        input_features = {}
        for k, v in raw.get("input_features", {}).items():
            input_features[k] = PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        output_features = {}
        for k, v in raw.get("output_features", {}).items():
            output_features[k] = PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))

        # 检测相对动作模式 (use_relative_actions=true 时模型预测 delta = action - state)
        self.use_relative = raw.get("use_relative_actions", False)
        # 也可以通过 normalizer 自动检测: relative 模式下 action_mean ≈ 0
        if not self.use_relative and np.abs(self.action_mean).max() < 0.05:
            self.use_relative = True
            print("  自动检测到相对动作模式 (action_mean≈0)")
        if self.use_relative:
            print("  ✓ 相对动作模式: 推理输出 delta + state → 绝对关节角")
        else:
            print("  ⚠ 绝对动作模式: 推理直接输出绝对关节角 (Identity Shortcut 风险)")

        # FIX 1: empty_cameras=0 匹配训练配置
        config = PI0Config(
            device="cpu",
            dtype="bfloat16" if self.mode == "bf16" else "float32",
            empty_cameras=0,
            num_inference_steps=50,  # 与训练配置一致 (config.json num_inference_steps=50)
            input_features=input_features,
            output_features=output_features,
        )
        config.device = "meta"

        # ── 加载权重 ──
        weights_path = md / "model.safetensors"
        print(f"  加载权重: {weights_path.name}")
        t1 = time.time()
        sd = load_file(str(weights_path), device="cpu")
        if self.mode == "bf16":
            for k in list(sd.keys()):
                if sd[k].dtype in (torch.float16, torch.float32, torch.bfloat16):
                    sd[k] = sd[k].to(dtype=torch.bfloat16)
        print(f"  权重加载: {time.time() - t1:.1f}s ({len(sd)} keys)")

        # ── 创建模型 (meta device) ──
        print("  创建模型 (meta device, 零 CPU 内存)...")
        with init_empty_weights():
            self.policy = PI0Policy(config)
        print(f"  模型创建: {time.time() - t0:.1f}s")

        # ── 注入权重 ──
        t2 = time.time()
        print("  注入权重...")
        self.policy.load_state_dict(sd, strict=False, assign=True)
        del sd; gc.collect()
        print(f"  权重注入: {time.time() - t2:.1f}s")

        # ── 物化到 GPU ──
        if self.device == "cuda":
            print("  物化到 CUDA (to_empty)...")
            t3 = time.time()
            self.policy = self.policy.to_empty(device="cuda")
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  物化完成: {time.time() - t3:.1f}s | GPU: {gpu_memory_str()}")

        self.policy.eval()
        torch.set_grad_enabled(False)
        self.policy.reset()

        # ── Tokenizer ──
        tokenizer_path = "/home/ubuntu/ur3_ft300_ws/ai-models/paligemma_tokenizer"
        print(f"加载 PaliGemma tokenizer ({tokenizer_path})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            print("  Tokenizer 加载成功")
        except Exception as e:
            print(f"  WARNING: {e}，回退随机 token")
            self.tokenizer = None

        print(f"模型加载完成 ({time.time() - t0:.1f}s) | GPU: {gpu_memory_str()}")

    # ── 数据读取 ──
    def _read_joint_state(self):
        try:
            with open(JOINT_STATE_FILE, "r") as f:
                line = f.readline().strip()
                if line:
                    vals = [float(x) for x in line.split()]
                    if len(vals) == 6: vals.append(0.0)
                    return np.array(vals, dtype=np.float32)
        except Exception: pass
        return np.zeros(7, dtype=np.float32)

    def _read_image(self, path):
        if path and os.path.exists(path):
            try:
                img = np.load(path, allow_pickle=False)
                if img.ndim == 3: return img.astype(np.float32)
            except Exception: pass
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)

    def _build_batch(self, joint_pos, wrist_img, global_img):
        # 归一化 state
        state_norm = (joint_pos - self.state_mean) / (self.state_std + 1e-8)
        state_t = torch.from_numpy(state_norm).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        def img_tensor(img):
            return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        if self.tokenizer is not None:
            tokens = self.tokenizer(TASK_PROMPT, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=48)
            lang_ids = tokens["input_ids"].to(self.device)
            lang_mask = tokens["attention_mask"].to(self.device).bool()
        else:
            lang_ids = torch.randint(0, 256000, (1, 48), device=self.device)
            lang_mask = torch.ones(1, 48, device=self.device, dtype=torch.bool)

        # FIX 2: 只传 2 个相机 (empty_cameras=0, 匹配训练)
        return {
            "observation.state": state_t,
            "observation.images.camera0": img_tensor(wrist_img),
            "observation.images.camera1": img_tensor(global_img),
            OBS_LANGUAGE_TOKENS: lang_ids,
            OBS_LANGUAGE_ATTENTION_MASK: lang_mask,
        }

    def _infer(self, batch):
        with torch.no_grad():
            if self.mode == "bf16":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    action = self.policy.select_action(batch)
            else:
                action = self.policy.select_action(batch)
        # 反归一化 → 得到模型原始输出 (绝对动作 or 相对增量取决于训练配置)
        action_np = action.cpu().float().numpy().flatten()
        action_unnorm = action_np * self.action_std + self.action_mean
        return action_unnorm

    def _to_absolute(self, model_output, current_state):
        """将模型输出转为绝对关节角。

        相对模式 (use_relative_actions=true): model_output = delta → action = state + delta
        绝对模式 (use_relative_actions=false): model_output = absolute → 直接使用
        """
        if self.use_relative:
            return current_state + model_output
        return model_output

    # 原子写入，避免 ROS 端读到不完整的文件
    def _write_action(self, action):
        fd, tmp = tempfile.mkstemp(dir="/tmp", prefix="ur3_action_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(" ".join(f"{a:.6f}" for a in action[:7]) + "\n")
            os.replace(tmp, ACTION_FILE)  # atomic rename
        except Exception:
            os.unlink(tmp)

    def run(self):
        period = 1.0 / self.hz
        print(f"推理循环启动 ({self.hz} Hz)...")
        step = 0

        # 动作 EMA 平滑，减少抖动
        ema_action = None
        ema_alpha = 0.3  # 平滑系数 (0=不平滑, 1=完全用历史)

        while True:
            try:
                t0 = time.time()
                joint_pos = self._read_joint_state()
                wrist_img = self._read_image(CAMERA0_FILE)
                global_img = self._read_image(CAMERA1_FILE)

                batch = self._build_batch(joint_pos, wrist_img, global_img)
                raw_action = self._infer(batch)
                # 转绝对关节角 (相对模式下 raw_action 是 delta)
                absolute_action = self._to_absolute(raw_action, joint_pos)

                # ── EMA 平滑 ──
                if ema_action is None:
                    ema_action = absolute_action.copy()
                else:
                    ema_action = ema_alpha * ema_action + (1 - ema_alpha) * absolute_action
                action = ema_action.copy()

                self._write_action(action)

                step += 1
                if step % 10 == 0:
                    elapsed = (time.time() - t0) * 1000
                    has_w = "Y" if wrist_img.any() else "N"
                    has_g = "Y" if global_img.any() else "N"
                    print(f"  [{step}] {elapsed:.0f}ms | 关节={np.array2string(joint_pos, precision=2)} | "
                          f"动作={np.array2string(action, precision=3)} | 相机(w/g)={has_w}/{has_g}"
                          f" | {gpu_memory_str()}")

                sleep_time = period - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except KeyboardInterrupt:
                print("停止推理...")
                break
            except Exception as e:
                print(f"循环异常: {e}")
                import traceback; traceback.print_exc()
                time.sleep(period)


def main():
    parser = argparse.ArgumentParser(description="UR3 Pi0 LoRA 推理端")
    parser.add_argument("--mode", type=str, default="cpu", choices=["cpu", "bf16", "fp32"])
    parser.add_argument("--hz", type=int, default=5)
    parser.add_argument("--model", type=str, default=None, help="模型路径覆盖")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    model_dir = Path(args.model) if args.model else MODEL_DIR
    engine = UR3Pi0Inference(mode=args.mode, hz=args.hz, model_dir=model_dir)

    if args.warmup > 0 and engine.device == "cuda":
        print(f"\nGPU 预热 ({args.warmup} 次)...")
        joint_pos = np.zeros(7, dtype=np.float32)
        wrist_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        global_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        for i in range(args.warmup):
            t0 = time.time()
            batch = engine._build_batch(joint_pos, wrist_img, global_img)
            _ = engine._infer(batch)
            print(f"  预热 {i+1}/{args.warmup}: {(time.time()-t0)*1000:.0f}ms")
        print(f"预热完成 | GPU: {gpu_memory_str()}\n")

    engine.run()


if __name__ == "__main__":
    main()

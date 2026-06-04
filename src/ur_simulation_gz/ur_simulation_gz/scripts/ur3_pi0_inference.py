#!/usr/bin/env python3
"""
UR3 Pi0 推理端 — Conda pi0-env 环境运行

循环读取 /tmp/ 中的观测数据，运行 Pi0 推理，输出动作到 /tmp/ur3_action.txt。

用法:
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env
  python3 ur3_pi0_inference.py                # CPU 模式
  python3 ur3_pi0_inference.py --mode bf16    # GPU bfloat16 (推荐, ~7GB)
  python3 ur3_pi0_inference.py --mode bf16 --hz 5  # 5 Hz 推理
"""

import os, sys, json, time, argparse, gc
import numpy as np
import torch
from pathlib import Path

from safetensors.torch import load_file
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from transformers import AutoTokenizer

# ── 配置 ──
MODEL_DIR = Path("/home/ubuntu/ur3_ft300_ws/outputs/train/pi0_ur3_v3/checkpoints/last/pretrained_model")
JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
ACTION_FILE = "/tmp/ur3_action.txt"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"
IMG_SIZE = (224, 224)

# 7D 关节归一化参数（从 checkpoint preprocessor/postprocessor 提取）
# State 归一化 → 来自 policy_preprocessor_step_5_normalizer_processor.safetensors
STATE_MEAN  = np.array([-0.98096776, -1.8261101,  -1.0006496,  -1.4776422,   1.1784788,   0.30134782,  0.28109705], dtype=np.float32)
STATE_STD   = np.array([ 0.6413004,   0.1287974,   0.51718795,  0.13342571,  0.58274025,  0.39418843,  0.36008605], dtype=np.float32)
# Action 反归一化 → 来自 policy_postprocessor_step_0_unnormalizer_processor.safetensors
ACTION_MEAN = np.array([-0.9810236,  -1.8261198,  -1.0006421,  -1.4797164,   1.1785238,   0.30149975,  0.2857143], dtype=np.float32)
ACTION_STD  = np.array([ 0.7147818,   0.14523803,  0.5736406,   0.15516822,  0.659141,    0.4443977,   0.36421567], dtype=np.float32)

TASK_PROMPT = "pick up the red cube and place it into the bowl\n"


def gpu_memory_str():
    if not torch.cuda.is_available():
        return "N/A"
    a = torch.cuda.memory_allocated(0) / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{a:.2f}GB / {t:.1f}GB"


class UR3Pi0Inference:
    def __init__(self, mode="cpu", hz=5):
        self.mode = mode
        self.hz = hz
        self.device = "cuda" if mode in ("bf16", "fp32") and torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if mode == "bf16" else torch.float32

        print(f"UR3 Pi0 推理端 | 模式={self.mode.upper()} | 频率={hz}Hz | 设备={self.device}")
        self.policy = None
        self.tokenizer = None
        self._init_files()
        self._load()

    def _init_files(self):
        for path, default in [(JOINT_STATE_FILE, "0.0 0.0 0.0 0.0 0.0 0.0\n"),
                               (ACTION_FILE, "0.0 0.0 0.0 0.0 0.0 0.0\n")]:
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(default)

    def _load(self):
        print(f"加载模型: {MODEL_DIR}")
        t0 = time.time()

        with open(MODEL_DIR / "config.json") as f:
            raw = json.load(f)
        input_features = {}
        for k, v in raw.get("input_features", {}).items():
            input_features[k] = PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        output_features = {}
        for k, v in raw.get("output_features", {}).items():
            output_features[k] = PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))

        config = PI0Config(
            device="cpu",
            dtype="bfloat16" if self.mode == "bf16" else "float32",
            empty_cameras=1,
            input_features=input_features,
            output_features=output_features,
        )
        print("  创建模型 (CPU, bf16 → ~7GB)...")
        self.policy = PI0Policy(config)
        print(f"  模型创建: {time.time() - t0:.1f}s")

        if self.mode == "bf16":
            weights_path = MODEL_DIR / "model_bf16.safetensors"
            if not weights_path.exists():
                weights_path = MODEL_DIR / "model.safetensors"
        else:
            weights_path = MODEL_DIR / "model.safetensors"
        print(f"  加载权重: {weights_path.name}")
        t1 = time.time()
        sd = load_file(str(weights_path), device="cpu")
        if self.mode == "bf16":
            for k in list(sd.keys()):
                if sd[k].dtype in (torch.float16, torch.float32, torch.bfloat16):
                    sd[k] = sd[k].to(dtype=torch.bfloat16)
        print(f"  权重加载: {time.time() - t1:.1f}s ({len(sd)} keys)")

        t2 = time.time()
        self.policy.load_state_dict(sd, strict=False)
        del sd; gc.collect()
        print(f"  权重应用: {time.time() - t2:.1f}s")

        if self.device == "cuda":
            print("  移动到 CUDA...")
            t3 = time.time()
            self.policy = self.policy.to(device="cuda", dtype=self.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  移动完成: {time.time() - t3:.1f}s | GPU: {gpu_memory_str()}")

        self.policy.eval()
        torch.set_grad_enabled(False)
        self.policy.reset()

        # ── Tokenizer ──
        tokenizer_path = "/home/ubuntu/ur3_ft300_ws/ai-models/paligemma_tokenizer"
        print(f"加载 PaliGemma tokenizer ({tokenizer_path})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            self._use_random_tokens = False
            print("  Tokenizer 加载成功")
        except Exception as e:
            print(f"  WARNING: {e}，回退随机 token")
            self.tokenizer = None
            self._use_random_tokens = True

        print(f"模型加载完成 ({time.time() - t0:.1f}s) | GPU: {gpu_memory_str()}")

    # ── 数据读取 ──

    def _read_joint_state(self):
        try:
            with open(JOINT_STATE_FILE, "r") as f:
                line = f.readline().strip()
                if line:
                    vals = [float(x) for x in line.split()]
                    if len(vals) == 6:
                        vals.append(0.0)  # pad 6→7
                    return np.array(vals, dtype=np.float32)
        except Exception:
            pass
        return np.zeros(7, dtype=np.float32)

    def _read_image(self, path):
        if path and os.path.exists(path):
            try:
                img = np.load(path, allow_pickle=False)
                if img.ndim == 3:
                    return img.astype(np.float32)
            except Exception:
                pass
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)

    # ── 批构造 ──

    def _build_batch(self, joint_pos, wrist_img, global_img):
        state_norm = (joint_pos - STATE_MEAN) / (STATE_STD + 1e-8)
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

        return {
            "observation.state": state_t,
            "observation.images.camera0": img_tensor(wrist_img),
            "observation.images.camera1": img_tensor(global_img),
            "observation.images.camera2": torch.zeros(1, 3, *IMG_SIZE, device=self.device, dtype=self.dtype),
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
        action_np = action.cpu().float().numpy().flatten()
        action_unnorm = action_np * ACTION_STD + ACTION_MEAN
        return action_unnorm[:7]  # 绝对关节位置，不裁剪

    def _write_action(self, action):
        with open(ACTION_FILE, "w") as f:
            f.write(" ".join(f"{a:.6f}" for a in action[:7]) + "\n")

    # ── 夹爪-机械臂同步状态机（时间控制） ──
    # 训练数据中夹爪是瞬时跳变（move_gripper 阻塞完成后才录制），但物理夹爪闭合需要时间。
    # 模型在夹爪闭合过程中就输出下一阶段的关节位置，导致还没夹稳就抬起。
    # 解决：用时间冻结 —— 夹爪动作开始后 N 秒内机械臂不能动。
    #
    # 状态转换:
    #   FREE ──(model grip>0.5)──> CLOSING  冻结机械臂，夹爪→0.75，计时 2 秒
    #   CLOSING ──(2秒到)──> HOLDING        解除冻结，夹爪锁 0.75
    #   HOLDING ──(model grip<0.1)──> OPENING 冻结机械臂，夹爪→0.0，计时 2 秒
    #   OPENING ──(2秒到)──> FREE

    GRIP_CLOSE_THRESHOLD = 0.5      # model action > 此值 → 开始闭合
    GRIP_OPEN_THRESHOLD = 0.1       # model action < 此值 → 开始打开
    GRIP_FREEZE_TIME_CLOSE = 5.0    # 闭合时冻结机械臂的秒数
    GRIP_FREEZE_TIME_OPEN = 5.0     # 打开时冻结机械臂的秒数

    def run(self):
        period = 1.0 / self.hz
        print(f"推理循环启动 ({self.hz} Hz)...")
        step = 0

        # 夹爪状态机
        grip_state = "FREE"         # FREE | CLOSING | HOLDING | OPENING
        frozen_arm = None           # 冻结时的 6 关节位置
        freeze_until = 0.0          # 冻结截止时间 (time.monotonic)

        while True:
            try:
                t0 = time.time()
                now = time.monotonic()
                joint_pos = self._read_joint_state()
                wrist_img = self._read_image(CAMERA0_FILE)
                global_img = self._read_image(CAMERA1_FILE)

                batch = self._build_batch(joint_pos, wrist_img, global_img)
                action = self._infer(batch)

                raw_grip_action = action[6]
                actual_grip = joint_pos[6]

                # ── 状态机 ──
                if grip_state == "FREE":
                    if raw_grip_action > self.GRIP_CLOSE_THRESHOLD:
                        grip_state = "CLOSING"
                        frozen_arm = joint_pos[:6].copy()
                        freeze_until = now + self.GRIP_FREEZE_TIME_CLOSE
                        print(f"  [step {step}] FREE → CLOSING (arm frozen {self.GRIP_FREEZE_TIME_CLOSE}s, grip→0.75)")

                elif grip_state == "CLOSING":
                    action[:6] = frozen_arm   # 机械臂保持不动
                    action[6] = 0.75
                    if now >= freeze_until:
                        grip_state = "HOLDING"
                        frozen_arm = None
                        print(f"  [step {step}] CLOSING → HOLDING (timer done, arm released)")

                elif grip_state == "HOLDING":
                    action[6] = 0.75          # 强制保持闭合
                    if raw_grip_action < self.GRIP_OPEN_THRESHOLD:
                        grip_state = "OPENING"
                        frozen_arm = joint_pos[:6].copy()
                        freeze_until = now + self.GRIP_FREEZE_TIME_OPEN
                        print(f"  [step {step}] HOLDING → OPENING (arm frozen {self.GRIP_FREEZE_TIME_OPEN}s, grip→0.0)")

                elif grip_state == "OPENING":
                    action[:6] = frozen_arm   # 机械臂保持不动
                    action[6] = 0.0
                    if now >= freeze_until:
                        grip_state = "FREE"
                        frozen_arm = None
                        print(f"  [step {step}] OPENING → FREE (timer done, arm released)")

                self._write_action(action)

                elapsed = (time.time() - t0) * 1000
                step += 1
                if step % 10 == 0:
                    has_w = "Y" if wrist_img.any() else "N"
                    has_g = "Y" if global_img.any() else "N"
                    grip_state_str = grip_state if grip_state != "FREE" else ""
                    grip_marker = f" [{grip_state_str}]" if grip_state_str else ""
                    frozen_marker = " [FROZEN]" if frozen_arm is not None else ""
                    print(f"  [{step}] {elapsed:.0f}ms | 关节={np.array2string(joint_pos, precision=2)} | "
                          f"动作={np.array2string(action, precision=3)} | 相机(w/g)={has_w}/{has_g}{grip_marker}{frozen_marker} | {gpu_memory_str()}")

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
    parser = argparse.ArgumentParser(description="UR3 Pi0 推理端")
    parser.add_argument("--mode", type=str, default="cpu", choices=["cpu", "bf16", "fp32"])
    parser.add_argument("--hz", type=int, default=5, help="推理频率 (Hz, 默认 5)")
    parser.add_argument("--warmup", type=int, default=1, help="预热推理次数")
    args = parser.parse_args()

    engine = UR3Pi0Inference(mode=args.mode, hz=args.hz)

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

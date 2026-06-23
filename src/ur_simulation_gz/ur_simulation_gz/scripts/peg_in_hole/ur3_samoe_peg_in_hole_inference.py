#!/usr/bin/env python3
"""
UR3 SA-MOE inference — Conda pi0-env
Loads SA-MOE checkpoint, reads /tmp/ observations, writes actions.
"""
import os, sys, time, argparse
import numpy as np
import torch
from pathlib import Path

from lerobot.policies.sa_moe_pi0 import SAMoEPi0Policy

JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
ACTION_FILE = "/tmp/ur3_action.txt"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"
FORCE_FILE = "/tmp/ur3_force.npy"

TASK = "pick up the peg and insert it into the hole"
ARM_JOINTS = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
              "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
ALL_JOINTS = ARM_JOINTS + ["robotiq_85_left_knuckle_joint"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hz", type=int, default=10)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not (ckpt / "config.json").exists():
        print(f"ERROR: config.json not found at {ckpt}")
        sys.exit(1)

    import json
    with open(ckpt / "config.json") as f:
        ckpt_config = json.load(f)
    use_relative = ckpt_config.get("use_relative_actions", False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SA-MOE from: {ckpt}")
    policy = SAMoEPi0Policy.from_pretrained(str(ckpt))
    policy.eval()
    # Move to CUDA but keep native mixed precision (vision=float32, SA-MOE=bf16)
    policy.to(device=device)
    print(f"Loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f}GB")

    # Load normalizer stats from checkpoint
    from safetensors.torch import load_file
    pre_files = list(ckpt.glob("*preprocessor*normalizer*.safetensors"))
    post_files = list(ckpt.glob("*postprocessor*unnormalizer*.safetensors"))

    if pre_files:
        pre = load_file(str(pre_files[0]))
        state_mean = pre["observation.state.mean"].numpy().astype(np.float32)
        state_std = pre["observation.state.std"].numpy().astype(np.float32)
    else:
        state_mean = np.zeros(7, dtype=np.float32)
        state_std = np.ones(7, dtype=np.float32)

    if post_files:
        post = load_file(str(post_files[0]))
        action_mean = post["action.mean"].numpy().astype(np.float32)
        action_std = post["action.std"].numpy().astype(np.float32)
    else:
        action_mean = np.zeros(7, dtype=np.float32)
        action_std = np.ones(7, dtype=np.float32)
    print(f"State norm: mean={state_mean.round(3)}")

    # Tokenizer (local cache)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/ubuntu/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/local/"
    )
    task_tokens = tokenizer(TASK, return_tensors="pt", padding="max_length",
                            max_length=48, truncation=True)
    task_ids = task_tokens["input_ids"].to(device)
    task_mask = task_tokens["attention_mask"].bool().to(device)

    print(f"Running inference at {args.hz} Hz...")
    period = 1.0 / args.hz

    while not os.path.exists(JOINT_STATE_FILE):
        time.sleep(0.1)
    print("Observations available, starting loop")

    step = 0
    while True:
        t0 = time.perf_counter()
        try:
            # Read observations (retry on empty/corrupt files up to 3 times)
            for retry in range(3):
                try:
                    with open(JOINT_STATE_FILE) as f:
                        raw = f.read().strip()
                    if not raw:
                        raise ValueError("Empty joint state")
                    state = np.array([float(x) for x in raw.split()], dtype=np.float32)
                    if state.shape != (7,):
                        raise ValueError(f"Bad state shape {state.shape}")

                    cam0 = np.load(CAMERA0_FILE)
                    cam1 = np.load(CAMERA1_FILE)
                    force = np.load(FORCE_FILE)

                    if cam0.shape != (224, 224, 3):
                        raise ValueError(f"Bad cam0 shape {cam0.shape}")
                    if cam1.shape != (224, 224, 3):
                        raise ValueError(f"Bad cam1 shape {cam1.shape}")
                    if force.shape != (6,):
                        raise ValueError(f"Bad force shape {force.shape}")
                    break
                except (ValueError, FileNotFoundError, OSError) as e:
                    if retry == 2:
                        raise
                    time.sleep(0.003)  # 3ms retry delay

            # Transpose to CHW for SigLIP [3,224,224]
            cam0 = np.transpose(cam0, (2, 0, 1))
            cam1 = np.transpose(cam1, (2, 0, 1))

            # Normalize state
            state_norm = (state - state_mean) / (state_std + 1e-8)

            # Build batch (no explicit dtype — autocast handles it)
            batch = {
                "observation.state": torch.from_numpy(state_norm).unsqueeze(0).cuda(),
                "observation.force": torch.from_numpy(force).unsqueeze(0).cuda(),
                "observation.images.camera0": torch.from_numpy(cam0).unsqueeze(0).cuda(),
                "observation.images.camera1": torch.from_numpy(cam1).unsqueeze(0).cuda(),
                "observation.language.tokens": task_ids,
                "observation.language.attention_mask": task_mask,
            }

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                actions = policy.predict_action_chunk(batch)

            action = actions[0].cpu().numpy()
            if action.ndim > 1:
                action = action[0]
            # 反归一化
            action_unnorm = action * action_std + action_mean
            # 相对模式: model output = delta → absolute = state + delta
            # 绝对模式: model output = absolute → 直接使用
            if use_relative:
                absolute_action = state + action_unnorm
            else:
                absolute_action = action_unnorm

            # Atomic write action
            tmp = ACTION_FILE + ".tmp"
            with open(tmp, "w") as f:
                f.write(" ".join(f"{a:.6f}" for a in absolute_action))
            os.rename(tmp, ACTION_FILE)

            elapsed = time.perf_counter() - t0
            if step % 10 == 0:
                print(f"  step {step}: action={np.array2string(action, precision=3)} "
                      f"({elapsed*1000:.0f}ms)")

            step += 1
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error step {step}: {e}", flush=True)
            time.sleep(0.1)
            step += 1


if __name__ == "__main__":
    main()

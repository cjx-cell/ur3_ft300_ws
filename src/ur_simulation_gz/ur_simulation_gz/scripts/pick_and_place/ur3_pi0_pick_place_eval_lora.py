#!/usr/bin/env python3
"""评估 LoRA 模型在训练数据上的预测精度

用法:
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate pi0-env
  python3 eval_lora_model.py --model <merged_model_dir> --ckpt <checkpoint_dir>
"""

import argparse, json, torch, numpy as np
from pathlib import Path
from safetensors.torch import load_file
from accelerate import init_empty_weights
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoTokenizer

TASK = "pick up the red cube and place it into the bowl\n"
JOINT_NAMES = ["shldr_pan", "shldr_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model prediction accuracy")
    parser.add_argument("--model", type=Path, required=True, help="merged 模型目录 (含 model.safetensors + config.json)")
    parser.add_argument("--ckpt", type=Path, required=True, help="LoRA checkpoint 目录 (含 normalizer stats)")
    args = parser.parse_args()

    model_dir = args.model
    ckpt_dir = args.ckpt

    print("=" * 70)
    print("LoRA 模型数据集评估")
    print("=" * 70)

    # ── 1. 加载归一化参数 ──
    pre_files = list(ckpt_dir.glob("policy_preprocessor_step*"))
    post_files = list(ckpt_dir.glob("policy_postprocessor_step*"))
    if not pre_files or not post_files:
        print("ERROR: 未找到 normalizer stats 文件，请检查 --ckpt 路径")
        return
    pre = load_file(str(pre_files[0]))
    post = load_file(str(post_files[0]))
    state_mean = pre["observation.state.mean"].numpy().astype(np.float32)
    state_std = pre["observation.state.std"].numpy().astype(np.float32)
    action_mean = post["action.mean"].numpy().astype(np.float32)
    action_std = post["action.std"].numpy().astype(np.float32)
    print(f"归一化参数已加载")

    # ── 2. 加载模型 ──
    print("加载模型...")
    with open(model_dir / "config.json") as f:
        raw = json.load(f)
    input_features = {k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                      for k, v in raw["input_features"].items()}
    output_features = {k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                       for k, v in raw["output_features"].items()}

    config = PI0Config(device="cpu", dtype="bfloat16", empty_cameras=0,
                       input_features=input_features, output_features=output_features)
    config.device = "meta"

    sd = load_file(str(model_dir / "model.safetensors"), device="cpu")
    for k in list(sd.keys()):
        if sd[k].dtype in (torch.float16, torch.float32, torch.bfloat16):
            sd[k] = sd[k].to(dtype=torch.bfloat16)

    with init_empty_weights():
        policy = PI0Policy(config)
    policy.load_state_dict(sd, strict=False, assign=True)
    policy = policy.to_empty(device="cuda")
    policy.eval()
    torch.set_grad_enabled(False)
    print(f"模型就绪 | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/ubuntu/ur3_ft300_ws/ai-models/paligemma_tokenizer", local_files_only=True)  # tokenizer path unchanged

    # ── 3. 加载数据集 ──
    print("加载数据集...")
    ds = LeRobotDataset(repo_id="cjx-cell/ur3_pick_place", video_backend="pyav")
    print(f"数据集: {len(ds)} frames, {ds.meta.total_episodes} episodes")

    # ── 4. 评估 ──
    print(f"\n{'='*70}")
    print("逐 episode 评估 (每 episode 采样 3 帧: start/mid/end)")
    print(f"{'='*70}")

    all_errors = []
    episode_errors = []

    for ep_idx in range(ds.meta.total_episodes):
        ep = ds.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        ep_len = ep["length"]

        offsets = [
            max(0, int(ep_len * 0.05)),
            int(ep_len * 0.50),
            min(ep_len - 1, int(ep_len * 0.90)),
        ]

        ep_errors = []
        for offset in offsets:
            idx = ep_start + offset
            frame = ds[idx]

            state_raw = frame["observation.state"].numpy().astype(np.float32)
            state_norm = (state_raw - state_mean) / (state_std + 1e-8)
            state_t = torch.from_numpy(state_norm).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)

            # Dataset images are already [C,H,W]
            cam0 = frame["observation.images.camera0"].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
            cam1 = frame["observation.images.camera1"].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)

            tokens = tokenizer(TASK, return_tensors="pt", padding="max_length", truncation=True, max_length=48)

            batch = {
                "observation.state": state_t,
                "observation.images.camera0": cam0,
                "observation.images.camera1": cam1,
                OBS_LANGUAGE_TOKENS: tokens["input_ids"].to("cuda"),
                OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].to("cuda").bool(),
            }

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # 用 predict_action_chunk 取第一个 action，避免 temporal ensemble 跨 episode 污染
                    policy.reset()
                    action_chunk = policy.predict_action_chunk(batch)
                    pred = action_chunk[:, 0, :]  # 取第一个动作

            pred_np = pred.cpu().float().numpy().flatten()
            pred_unnorm = pred_np * action_std + action_mean
            gt = frame["action"].numpy()

            abs_err = np.abs(pred_unnorm - gt)
            mae = abs_err.mean()
            ep_errors.append(mae)
            all_errors.append({
                "ep": ep_idx, "offset": offset, "pct": offset / max(1, ep_len),
                "mae": mae, "abs_err": abs_err,
                "pred": pred_unnorm.copy(), "gt": gt.copy()
            })

        episode_errors.append(np.mean(ep_errors))

        if ep_idx < 5 or ep_idx % 10 == 9 or ep_idx == ds.meta.total_episodes - 1:
            phases = ["start", "mid  ", "end  "]
            parts = " | ".join(f"{phases[i]}: {ep_errors[i]:.4f}" for i in range(3))
            print(f"  ep {ep_idx:2d}: avg={np.mean(ep_errors):.4f}  ({parts})")

    # ── 5. 汇总 ──
    print(f"\n{'='*70}")
    print("汇总")
    print(f"{'='*70}")

    mae_array = np.array([e["mae"] for e in all_errors])
    print(f"\n整体 (rad):")
    print(f"  Mean MAE: {mae_array.mean():.4f}")
    print(f"  Median:   {np.median(mae_array):.4f}")
    print(f"  Min:      {mae_array.min():.4f}")
    print(f"  Max:      {mae_array.max():.4f}")
    print(f"  Std:      {mae_array.std():.4f}")

    per_joint_mae = np.array([e["abs_err"] for e in all_errors])
    print(f"\n各关节 MAE (rad):")
    for j in range(7):
        j_mae = per_joint_mae[:, j]
        print(f"  {JOINT_NAMES[j]:12s}: mean={j_mae.mean():.4f}  median={np.median(j_mae):.4f}  "
              f"max={j_mae.max():.4f}  std={j_mae.std():.4f}")

    # ── 6. 结论 ──
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    avg_mae = mae_array.mean()
    if avg_mae < 0.10:
        print(f"✓ 模型精度优秀 (avg MAE={avg_mae:.4f} rad) — 可以上 Gazebo 测试")
    elif avg_mae < 0.15:
        print(f"△ 模型精度良好 (avg MAE={avg_mae:.4f} rad) — 可以试但注意动作偏差")
    elif avg_mae < 0.25:
        print(f"△ 模型精度一般 (avg MAE={avg_mae:.4f} rad) — 建议继续训练")
    else:
        print(f"✗ 模型精度差 (avg MAE={avg_mae:.4f} rad) — 需要更多训练")

    print(f"\nGPU: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")


if __name__ == "__main__":
    main()

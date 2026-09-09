#!/usr/bin/env python3
"""
Single Episode Training Script for LeRobot Diffusion Policy (50M).
Trains Diffusion Policy on lerobot_overfit_1ep (Episode 301),
logs training loss, exports loss curves and checkpoints to experiment and output directories,
and executes automated Action MSE & MAE evaluation.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "/home/ubuntu/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

import shutil

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

def main():
    parser = argparse.ArgumentParser(description="Train LeRobot Diffusion Policy on Single Episode Dataset (Episode 302)")
    parser.add_argument("--steps", type=int, default=2500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dataset_dir", type=str, default="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_overfit_5ep")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/experiments/diffusion_overfit_ep302")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training DiffusionPolicy (50M) on Single Episode (Episode 302)...")
    print(f"   Device:  {device}")
    print(f"   Dataset: {args.dataset_dir}")
    print(f"   Output:  {args.output_dir}")

    output_path = Path(args.output_dir)
    ckpt_dir = output_path / f"checkpoints/{args.steps:06d}/pretrained_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "diffusion_train.log"
    csv_file = output_path / "training_metrics.csv"
    with open(log_file, "w") as f:
        f.write("=== Diffusion Policy Single Episode (Episode 302) Training Log ===\n")

    # 1. Load Single Episode from lerobot_overfit_5ep
    dataset_path = Path("/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_overfit_5ep")
    cfg = DiffusionConfig()
    cfg.use_separate_rgb_encoder_per_camera = True  # Dedicated ResNet-18 per camera (prevents visual feature corruption)

    fps = 10
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, fps),
    }
    for img_key in ["observation.images.camera0", "observation.images.camera1"]:
        delta_timestamps[img_key] = make_delta_timestamps(cfg.observation_delta_indices, fps)

    dataset = LeRobotDataset(
        repo_id="lerobot_overfit_5ep",
        root=dataset_path,
        episodes=[1],  # Single episode 1 (Episode 302: peg=(-0.012, 0.344), hole=(-0.188, 0.341))
        delta_timestamps=delta_timestamps
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 2. Instantiate Policy and Preprocessors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.to(device)
    policy.train()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    print("\n⚡ Starting Diffusion Policy 1-Episode training loop...")
    start_time = time.time()
    metrics = []

    step = 0
    data_iter = iter(dataloader)

    while step < args.steps:
        step += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_prep = preprocessor(batch_dev)

        loss, _ = policy(batch_prep)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        metrics.append({"step": step, "loss": loss_val})

        if step == 1 or step % 50 == 0 or step == args.steps:
            elapsed = time.time() - start_time
            speed = step / elapsed if elapsed > 0 else 0.0
            log_str = f"Step {step:5d} / {args.steps} | Loss: {loss_val:.4f} | Speed: {speed:.1f} steps/s"
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")

    total_time = time.time() - start_time
    print(f"\n✓ Single Episode Training finished in {total_time:.1f} seconds!")

    # 3. Save Checkpoints & Preprocessors
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    policy.config.save_pretrained(ckpt_dir)

    # Save CSV
    df = pd.DataFrame(metrics)
    df.to_csv(csv_file, index=False)

    # Copy artifacts to /home/ubuntu/ur3_ft300_ws/outputs/diffusion_overfit_1ep
    outputs_dir = Path("/home/ubuntu/ur3_ft300_ws/outputs/diffusion_overfit_1ep")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(outputs_dir / "training_metrics.csv", index=False)
    shutil.copy(log_file, outputs_dir / "diffusion_train.log")

    # Render Loss Curve Plot
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(df["step"].to_numpy(), df["loss"].to_numpy(), label="Diffusion Policy (50M) Single Ep Loss", color="#007acc", linewidth=1.5)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Single Episode (Episode 301) Diffusion Policy Training Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        img_path = output_path / "loss_curve_linear.png"
        plt.savefig(img_path, dpi=200)
        plt.savefig(outputs_dir / "loss_curve_linear.png", dpi=200)
        plt.close()
        print(f"✓ Training loss curves saved to {outputs_dir}")
    except Exception as e:
        print(f"⚠ Warning: Failed to render loss curve ({e})")

    # 4. Automatically run Action MSE & MAE Evaluation on Episode 301
    try:
        from eval_diffusion_mse import evaluate_policy
        print("\n🔍 Automatically starting Action MSE & MAE evaluation on Episode 301...")
        evaluate_policy(ckpt_dir, args.dataset_dir, args.output_dir)
    except Exception as e:
        print(f"⚠ Warning: Failed to execute automated MSE evaluation ({e})")

    print("✓ Single Episode DiffusionPolicy training & evaluation complete!")

if __name__ == "__main__":
    main()

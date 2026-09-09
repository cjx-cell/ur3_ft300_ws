#!/usr/bin/env python3
"""
Train Diffusion Policy on trimmed 26-episode dataset (Episode 301+).
- Uses all 26 episodes (no episode filter)
- Separate ResNet-18 per camera
- Proper unnormalization via dataset_stats
"""

import os, sys, time, shutil
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "/home/ubuntu/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors


def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset_dir", type=str,
                        default="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_trimmed_301plus")
    parser.add_argument("--output_dir", type=str,
                        default="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/experiments/diffusion_trimmed_26ep")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'=' * 70}")
    print(f"🚀 Training DiffusionPolicy (50M) on 26 Trimmed Episodes (301+)")
    print(f"   Device:  {device}")
    print(f"   Dataset: {args.dataset_dir}")
    print(f"   Output:  {args.output_dir}")
    print(f"   Steps:   {args.steps}")
    print(f"   Batch:   {args.batch_size}")
    print(f"   LR:      {args.lr}")
    print(f"{'=' * 70}")

    output_path = Path(args.output_dir)
    ckpt_dir = output_path / f"checkpoints/{args.steps:06d}/pretrained_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "train.log"
    csv_file = output_path / "training_metrics.csv"
    with open(log_file, "w") as f:
        f.write("=== Diffusion Policy 26-Episode Trimmed Training Log ===\n")

    # 1. Load dataset (all episodes)
    dataset_path = Path(args.dataset_dir)
    cfg = DiffusionConfig()
    cfg.use_separate_rgb_encoder_per_camera = True

    fps = 10
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, fps),
    }
    for img_key in ["observation.images.camera0", "observation.images.camera1"]:
        delta_timestamps[img_key] = make_delta_timestamps(cfg.observation_delta_indices, fps)

    dataset = LeRobotDataset(
        repo_id="pap_moe/ur3_peg_trimmed_301plus",
        root=dataset_path,
        delta_timestamps=delta_timestamps,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4
    )

    print(f"   Dataset loaded: {len(dataset)} samples across {dataset.meta.total_episodes} episodes")

    # 2. Policy & preprocessors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.to(device)
    policy.train()

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"   Model params: {total_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    # 3. Training loop
    print(f"\n⚡ Training started...")
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

        if step == 1 or step % 100 == 0 or step == args.steps:
            elapsed = time.time() - start_time
            speed = step / elapsed
            log_str = f"Step {step:5d}/{args.steps} | Loss: {loss_val:.4f} | Speed: {speed:.1f} steps/s"
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")

    total_time = time.time() - start_time
    print(f"\n✓ Training finished in {total_time:.1f}s ({total_time/60:.1f} min)")

    # 4. Save checkpoint
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    policy.config.save_pretrained(ckpt_dir)
    print(f"✓ Checkpoint saved to {ckpt_dir}")

    # 5. Save metrics CSV
    df = pd.DataFrame(metrics)
    df.to_csv(csv_file, index=False)

    # 6. Copy to outputs
    outputs_dir = Path("/home/ubuntu/ur3_ft300_ws/outputs/diffusion_trimmed_26ep")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputs_dir / "training_metrics.csv", index=False)
    shutil.copy(log_file, outputs_dir / "train.log")

    # 7. Plot loss curve
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(df["step"], df["loss"], color="#007acc", linewidth=0.8, alpha=0.5, label="Raw")
        window = min(50, len(df))
        ax1.plot(df["step"], df["loss"].rolling(window).mean(), color="#e63946", linewidth=2, label=f"MA-{window}")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("26-Episode Trimmed Diffusion Policy Training Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2.plot(df["step"], df["loss"], color="#007acc", linewidth=0.8, alpha=0.5)
        ax2.plot(df["step"], df["loss"].rolling(window).mean(), color="#e63946", linewidth=2)
        ax2.set_yscale("log")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss (log)")
        ax2.set_title("Log Scale")
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path / "loss_curve.png", dpi=200)
        plt.savefig(outputs_dir / "loss_curve.png", dpi=200)
        plt.close()
        print(f"✓ Loss curve saved to {outputs_dir}/loss_curve.png")
    except Exception as e:
        print(f"⚠ Loss curve plot failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"✅ 26-Episode Trimmed DiffusionPolicy training complete!")
    print(f"   Checkpoint: {ckpt_dir}")
    print(f"   Final Loss: {metrics[-1]['loss']:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

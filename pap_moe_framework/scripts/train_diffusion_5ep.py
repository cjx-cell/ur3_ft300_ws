#!/usr/bin/env python3
"""
Train LeRobot DiffusionPolicy on 5-episode overfit dataset.
Dataset: /home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_overfit_5ep
Output: /home/ubuntu/ur3_ft300_ws/pap_moe_framework/experiments/diffusion_overfit_5ep
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/ubuntu/lerobot/src")

from lerobot.configs import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, 
                        default="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_overfit_5ep")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/experiments/diffusion_overfit_5ep")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training DiffusionPolicy on device: {device}")
    print(f"   Dataset: {args.dataset_dir}")
    print(f"   Output:  {args.output_dir}")

    dataset_path = Path(args.dataset_dir)
    dataset_metadata = LeRobotDatasetMetadata(repo_id="lerobot_overfit_5ep", root=dataset_path)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}

    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        vision_backbone="resnet18",
        optimizer_lr=args.lr,
    )

    policy = DiffusionPolicy(cfg)
    fps = dataset_metadata.fps

    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, fps),
    }
    for img_key in cfg.image_features:
        delta_timestamps[img_key] = make_delta_timestamps(cfg.observation_delta_indices, fps)

    dataset = LeRobotDataset(repo_id="lerobot_overfit_5ep", root=dataset_path, delta_timestamps=delta_timestamps)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)

    policy.train()
    policy.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_path / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"

    step = 0
    start_time = time.time()
    print("⚡ Starting Diffusion Policy training loop...")

    log_file = output_path / "diffusion_train.log"
    metrics_csv = output_path / "training_metrics.csv"

    step_history = []
    loss_history = []

    with open(log_file, "w") as f_log, open(metrics_csv, "w") as f_csv:
        f_csv.write("step,loss,elapsed_sec\n")

        done = False
        while not done:
            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch = preprocessor(batch)
                if "action_is_pad" not in batch:
                    batch["action_is_pad"] = torch.zeros(batch["action"].shape[:2], dtype=torch.bool, device=device)

                loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                loss_val = loss.item()
                elapsed = time.time() - start_time

                step_history.append(step)
                loss_history.append(loss_val)

                f_csv.write(f"{step},{loss_val:.6f},{elapsed:.2f}\n")
                f_csv.flush()

                if step % 50 == 0 or step == 1:
                    fps_rate = step / elapsed
                    msg = f"Step {step:5d} / {args.steps} | Loss: {loss_val:.4f} | Speed: {fps_rate:.1f} steps/s"
                    print(msg)
                    f_log.write(msg + "\n")
                    f_log.flush()

                if step >= args.steps:
                    done = True
                    break

    print(f"✓ Training finished in {time.time() - start_time:.1f} seconds!")
    print(f"Saving checkpoint to: {ckpt_dir}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    cfg.save_pretrained(ckpt_dir)

    # Render training loss curves using matplotlib
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(step_history, loss_history, label="Diffusion Policy Training Loss", color="#1f77b4", linewidth=1.5)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Diffusion Policy (50M) Overfit 5-Episode Training Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        curve_path = output_path / "loss_curve_linear.png"
        plt.savefig(curve_path, dpi=200)
        plt.close()

        # Copy to /home/ubuntu/ur3_ft300_ws/outputs per AGENTS.md workspace rule
        outputs_dir = Path("/home/ubuntu/ur3_ft300_ws/outputs/diffusion_overfit_5ep")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(curve_path, outputs_dir / "loss_curve_linear.png")
        shutil.copy(log_file, outputs_dir / "diffusion_train.log")
        shutil.copy(metrics_csv, outputs_dir / "training_metrics.csv")
        print(f"✓ Training loss curves and logs saved to:")
        print(f"   - Experiment dir: {output_path}")
        print(f"   - Workspace outputs: {outputs_dir}")
    except Exception as e:
        print(f"⚠ Warning: Failed to render loss curves ({e})")

    # Automatically run Action MSE Evaluation on dataset
    try:
        from eval_diffusion_mse import evaluate_policy
        print("\n🔍 Automatically starting dataset Action MSE evaluation...")
        evaluate_policy(ckpt_dir, args.dataset_dir, args.output_dir)
    except Exception as e:
        print(f"⚠ Warning: Failed to execute automated MSE evaluation ({e})")

    print("✓ DiffusionPolicy training, loss curves & MSE evaluation complete!")

if __name__ == "__main__":
    main()

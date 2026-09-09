#!/usr/bin/env python3
"""
Action MSE Evaluation Script for Trained LeRobot DiffusionPolicy.
Evaluates ground-truth vs predicted action MSE across 5 overfit episodes.
Saves json metrics and plots comparison.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

def evaluate_policy(ckpt_dir, dataset_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Starting Action MSE Evaluation for Diffusion Policy...")
    print(f"   Checkpoint: {ckpt_dir}")
    print(f"   Dataset:    {dataset_dir}")

    ckpt_path = Path(ckpt_dir)
    dataset_path = Path(dataset_dir)

    # 1. Load model & processors
    policy = DiffusionPolicy.from_pretrained(ckpt_path)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=ckpt_path)

    policy.to(device)
    policy.eval()

    # 2. Load dataset with delta_timestamps
    fps = 10
    cfg = policy.config
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, fps),
    }
    for img_key in cfg.image_features:
        delta_timestamps[img_key] = make_delta_timestamps(cfg.observation_delta_indices, fps)

    dataset = LeRobotDataset(repo_id="lerobot_overfit_5ep", root=dataset_path, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    gt_actions_list = []
    pred_actions_list = []

    unnorm_step = None
    for step in postprocessor.steps:
        if "Unnormalizer" in type(step).__name__:
            unnorm_step = step
            break

    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            gt_act = batch["action"][:, 0, :].numpy()  # Ground truth action at t (B, 7)
            
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_prep = preprocessor(batch_dev)
            if policy.config.image_features:
                batch_prep["observation.images"] = torch.stack([batch_prep[key] for key in policy.config.image_features], dim=-4)
            
            # Predict action chunk directly using diffusion model
            chunk = policy.diffusion.generate_actions(batch_prep)
            
            # Unnormalize action chunk to raw physical units
            if unnorm_step is not None:
                chunk_post = unnorm_step({"action": chunk.cpu()})["action"]
            else:
                chunk_post = chunk.cpu()

            pred_act = chunk_post[:, 0, :].numpy()  # First action step (B, 7)

            gt_actions_list.append(gt_act)
            pred_actions_list.append(pred_act)

    gt_actions = np.concatenate(gt_actions_list, axis=0)      # (N, 7)
    pred_actions = np.concatenate(pred_actions_list, axis=0)  # (N, 7)

    # 3. Calculate MSE, MAE, and Max Error metrics
    diff = pred_actions - gt_actions
    sq_err = diff ** 2
    abs_err = np.abs(diff)

    per_dim_mse = np.mean(sq_err, axis=0)                     # (7,)
    per_dim_mae = np.mean(abs_err, axis=0)                    # (7,)
    max_abs_err = np.max(abs_err, axis=0)                     # (7,)

    overall_mse = float(np.mean(sq_err))
    overall_rmse = float(np.sqrt(overall_mse))
    overall_mae = float(np.mean(abs_err))
    overall_max_err = float(np.max(abs_err))

    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "gripper_joint"
    ]

    per_dim_results = {}
    for j_name, mse_val, mae_val, max_err in zip(joint_names, per_dim_mse, per_dim_mae, max_abs_err):
        rmse_val = float(np.sqrt(mse_val))
        per_dim_results[j_name] = {
            "mse": float(mse_val),
            "rmse_rad": rmse_val,
            "rmse_deg": float(np.degrees(rmse_val)) if "gripper" not in j_name else rmse_val,
            "mae_rad": float(mae_val),
            "mae_deg": float(np.degrees(mae_val)) if "gripper" not in j_name else float(mae_val),
            "max_error": float(max_err)
        }

    eval_results = {
        "model_name": "DiffusionPolicy (50M)",
        "total_samples": int(len(gt_actions)),
        "overall_mse": overall_mse,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_max_error": overall_max_err,
        "inference_time_sec": float(time.time() - start_time),
        "per_joint_metrics": per_dim_results
    }

    # 4. Save JSON results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "eval_results.json"
    with open(json_file, "w") as f:
        json.dump(eval_results, f, indent=4)

    # Copy to /home/ubuntu/ur3_ft300_ws/outputs per AGENTS.md rule
    outputs_dir = Path("/home/ubuntu/ur3_ft300_ws/outputs/diffusion_overfit_5ep")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    print("\n============================================================")
    print("🎯 Diffusion Policy Action Evaluation Results")
    print("============================================================")
    print(f"Overall Action MSE:  {overall_mse:.6f}")
    print(f"Overall Action RMSE: {overall_rmse:.6f} rad ({np.degrees(overall_rmse):.3f}°)")
    print(f"Overall Action MAE:  {overall_mae:.6f} rad ({np.degrees(overall_mae):.3f}°)")
    print(f"Overall Max Error:   {overall_max_err:.6f}")
    print("------------------------------------------------------------")
    print("Per-Joint Evaluation Metrics:")
    for j_name, metrics in per_dim_results.items():
        if "gripper" in j_name:
            print(f"  - {j_name:30s}: MSE={metrics['mse']:.6f} | RMSE={metrics['rmse_rad']:.4f} | MAE={metrics['mae_rad']:.4f}")
        else:
            print(f"  - {j_name:30s}: MSE={metrics['mse']:.6f} | RMSE={metrics['rmse_deg']:.3f}° | MAE={metrics['mae_deg']:.3f}°")
    print("============================================================\n")

    return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, 
                        default="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/experiments/diffusion_overfit_5ep/checkpoints/005000/pretrained_model")
    parser.add_argument("--dataset_dir", type=str, 
                        default="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/datasets/lerobot_overfit_5ep")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/ubuntu/ur3_ft300_ws/pap_moe_framework/experiments/diffusion_overfit_5ep")
    args = parser.parse_args()

    evaluate_policy(args.ckpt_dir, args.dataset_dir, args.output_dir)

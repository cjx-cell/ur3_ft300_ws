#!/usr/bin/env python3
"""Offline first-window action audit for official LeRobot ACT and DP policies."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "/home/ubuntu/lerobot/src")

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def _delta(indices: list[int] | None, fps: int) -> list[float] | None:
    if indices is None:
        return None
    return [index / fps for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-id", default="local/pap_moe_workspace50_v10_baseline")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    dataset_root = Path(args.dataset).resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if config.type == "act":
        policy = ACTPolicy.from_pretrained(checkpoint)
    elif config.type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(checkpoint)
    else:
        raise ValueError(f"unsupported policy type: {config.type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device).eval()
    metadata = LeRobotDatasetMetadata(repo_id=args.repo_id, root=dataset_root)
    timestamps = {
        "observation.state": _delta(config.observation_delta_indices, metadata.fps),
        "action": _delta(config.action_delta_indices, metadata.fps),
    }
    for key in config.image_features:
        timestamps[key] = _delta(config.observation_delta_indices, metadata.fps)
    timestamps = {key: value for key, value in timestamps.items() if value is not None}
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        delta_timestamps=timestamps,
    )
    sample_count = min(args.max_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=checkpoint
    )

    predicted_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []
    torch.manual_seed(args.seed)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            raw_target = batch["action"].numpy()
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            processed = preprocessor(batch)
            if config.type == "act":
                normalized = policy.predict_action_chunk(processed)
                window = config.n_action_steps
                raw_target = raw_target[:, :window]
            else:
                processed = dict(processed)
                processed["observation.images"] = torch.stack(
                    [processed[key] for key in config.image_features], dim=-4
                )
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + batch_index
                )
                noise = torch.randn(
                    len(raw_target),
                    config.horizon,
                    config.output_features["action"].shape[0],
                    generator=generator,
                    device=device,
                )
                normalized = policy.diffusion.generate_actions(processed, noise=noise)
                window = config.n_action_steps
                start = config.n_obs_steps - 1
                raw_target = raw_target[:, start : start + window]
            predicted = postprocessor(normalized)[:, :window].detach().cpu().numpy()
            predicted_all.append(predicted)
            target_all.append(raw_target)

    predicted = np.concatenate(predicted_all)
    target = np.concatenate(target_all)
    gripper_contract_violation_rate = float(
        np.mean((predicted[..., 6] < 0.0) | (predicted[..., 6] > 0.8))
    )
    predicted[..., 6] = np.clip(predicted[..., 6], 0.0, 0.8)
    error = predicted - target
    per_dim_rmse = np.sqrt(np.mean(np.square(error), axis=(0, 1)))
    per_dim_mae = np.mean(np.abs(error), axis=(0, 1))
    result = {
        "policy_type": config.type,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "samples": int(len(predicted)),
        "executed_window": int(predicted.shape[1]),
        "overall_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "overall_mae": float(np.mean(np.abs(error))),
        "arm_rmse_rad": per_dim_rmse[:6].tolist(),
        "arm_mae_rad": per_dim_mae[:6].tolist(),
        "gripper_rmse_rad": float(per_dim_rmse[6]),
        "gripper_mae_rad": float(per_dim_mae[6]),
        "gripper_contract_violation_rate_before_clip": gripper_contract_violation_rate,
        "prediction_min": predicted.min(axis=(0, 1)).tolist(),
        "prediction_max": predicted.max(axis=(0, 1)).tolist(),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

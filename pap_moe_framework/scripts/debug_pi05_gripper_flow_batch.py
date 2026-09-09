#!/usr/bin/env python3
"""Inspect the unified Pi0.5 flow loss and gradients on one dataset frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import lerobot.policies.pi05.processor_pi05  # noqa: F401
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=24)
    parser.add_argument("--frame", type=int, default=205)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset.resolve()
    policy = PI05Policy.from_pretrained(str(checkpoint), strict=False)
    policy.config.rtc_config = None
    policy.init_rtc_processor()
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )

    repo_id = "pap_moe/pi05_d2_true_baseline"
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=[args.episode],
        delta_timestamps=resolve_delta_timestamps(policy.config, metadata),
        video_backend="torchcodec",
        return_uint8=True,
    )
    sample = dataset[args.frame]
    physical_gripper = sample["action"][:, 6].clone()
    for camera_key in metadata.camera_keys:
        if sample[camera_key].dtype == torch.uint8:
            sample[camera_key] = sample[camera_key].float() / 255.0
    # Match torch DataLoader's batch collation before invoking the processor.
    sample = {
        key: value.unsqueeze(0)
        if isinstance(value, torch.Tensor)
        else [value]
        for key, value in sample.items()
    }
    batch = preprocessor(sample)
    normalized_gripper = batch["action"][0, :, 6].detach().cpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).train()
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    policy.zero_grad(set_to_none=True)
    loss, loss_dict = policy.forward(batch)
    loss.backward()
    gradient = policy.model.action_out_proj.weight.grad.detach().float().cpu()

    result = {
        "episode": args.episode,
        "frame": args.frame,
        "physical_gripper_first10": physical_gripper[:10].tolist(),
        "physical_closed_fraction": float((physical_gripper >= 0.5).float().mean()),
        "normalized_gripper_first10": normalized_gripper[:10].tolist(),
        "normalized_min": float(normalized_gripper.min()),
        "normalized_max": float(normalized_gripper.max()),
        "normalized_positive_fraction": float((normalized_gripper > 0).float().mean()),
        "loss": float(loss.detach().cpu()),
        "loss_per_dim": loss_dict["loss_per_dim"],
        "weighted_loss_per_dim": loss_dict["weighted_loss_per_dim"],
        "effective_weight_per_dim": loss_dict["effective_weight_per_dim"],
        "action_out_proj_grad_norm_per_row": gradient.norm(dim=1).tolist(),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

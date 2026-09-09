#!/usr/bin/env python3
"""Evaluate an original Pi0.5 checkpoint on PAP ablation's matched flow inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPTS_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, action="append", required=True)
    parser.add_argument("--max-frames-per-expert", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.path.insert(0, str(LEROBOT_SRC))
    from eval_pap_moe_expert_ablation import (  # noqa: PLC0415
        EXPERT_NAMES,
        _fixed_flow_inputs,
        _load_episode_once,
        _raw_batch,
        _select_frame_indices,
    )
    import lerobot.policies.pi05.processor_pi05  # noqa: F401, PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: PLC0415

    started = time.perf_counter()
    policy = PI05Policy.from_pretrained(str(args.checkpoint.resolve()), strict=True)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint.resolve())
    )
    device = next(policy.parameters()).device
    action_dim = policy.config.output_features["action"].shape[0]
    losses: list[np.ndarray] = []
    target_routes: list[np.ndarray] = []
    per_episode: list[dict[str, object]] = []

    with torch.inference_mode():
        for episode_number, episode_path in enumerate(args.episode_npz, start=1):
            episode = _load_episode_once(episode_path)
            episode_losses: list[np.ndarray] = []
            episode_routes: list[np.ndarray] = []
            frame_indices = _select_frame_indices(
                episode["stage"], 10, args.max_frames_per_expert
            )
            for start in range(0, len(frame_indices), args.batch_size):
                selected = frame_indices[start : start + args.batch_size]
                raw = _raw_batch(
                    episode,
                    selected,
                    policy.config.chunk_size,
                    wrist_dropout=False,
                    continuous_gripper=True,
                )
                batch = preprocessor(
                    {
                        "observation.state": raw["observation.state"],
                        "observation.images.camera0": raw["observation.images.camera0"],
                        "observation.images.camera1": raw["observation.images.camera1"],
                        "action": raw["action"],
                        "task": raw["task"],
                    }
                )
                images, image_masks = policy._preprocess_images(batch)
                actions = policy.prepare_action(batch)
                route = np.asarray(episode["stage"][selected], dtype=np.float32)
                route /= np.clip(route.sum(axis=-1, keepdims=True), 1e-8, None)
                for seed_index in range(args.num_seeds):
                    noise, time_values = _fixed_flow_inputs(
                        actions.shape,
                        selected,
                        device,
                        args.seed + seed_index * 1_000_003,
                    )
                    per_token = policy.model.forward(
                        images,
                        image_masks,
                        batch["observation.language.tokens"],
                        batch["observation.language.attention_mask"],
                        actions,
                        noise=noise,
                        time=time_values,
                    )
                    loss_values = (
                        per_token[:, :, :action_dim]
                        .mean(dim=(1, 2))
                        .float()
                        .cpu()
                        .numpy()
                    )
                    losses.append(loss_values)
                    episode_losses.append(loss_values)
                    target_routes.append(route)
                    episode_routes.append(route)
            episode_loss_array = np.concatenate(episode_losses)
            per_episode.append(
                {
                    "episode": str(episode_path.resolve()),
                    "sampled_frames": int(len(episode_loss_array) // args.num_seeds),
                    "frame_seed_pairs": int(len(episode_loss_array)),
                    "flow_mse": float(episode_loss_array.mean()),
                }
            )
            print(
                f"Episode {episode_number}/{len(args.episode_npz)}: "
                f"{len(frame_indices)} frames",
                flush=True,
            )

    loss_array = np.concatenate(losses)
    routes = np.concatenate(target_routes)
    dominant = routes.argmax(axis=-1)
    subsets = {}
    for expert_index, name in enumerate(EXPERT_NAMES):
        selected = dominant == expert_index
        subsets[name] = {
            "frame_seed_pairs": int(selected.sum()),
            "mse": None if not selected.any() else float(loss_array[selected].mean()),
        }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "episodes": [str(path.resolve()) for path in args.episode_npz],
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "sampled_frames": int(len(loss_array) // args.num_seeds),
        "flow_mse": float(loss_array.mean()),
        "per_episode": per_episode,
        "dominant_expert_subsets": subsets,
        "evaluation_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

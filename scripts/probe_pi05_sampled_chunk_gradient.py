#!/usr/bin/env python3
"""Check whether Pi0.5 sampled action chunks support bounded direct gradients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPT_DIR = Path(__file__).resolve().parent
TASK = "grasp the peg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--index", type=int, default=2)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--num-steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(0, str(SCRIPT_DIR))
    from eval_pi05_offline_action_chunk import _raw_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )
    import lerobot.policies.pi05.processor_pi05  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PI05Policy.from_pretrained(str(args.checkpoint), strict=False)
    policy.to(device=device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    for step in preprocessor.steps:
        if hasattr(step, "global_task"):
            step.global_task = None

    for parameter in policy.parameters():
        parameter.requires_grad = False
    for parameter in policy.model.action_out_proj.parameters():
        parameter.requires_grad = True

    with np.load(args.corrections) as archive:
        state = archive["state"][args.index].astype(np.float32)
        camera0 = archive["camera0"][args.index].astype(np.float32)
        camera1 = archive["camera1"][args.index].astype(np.float32)
        target = archive["target_action"][args.index].astype(np.float32)
    raw = _raw_observation(state, camera0, camera1, TASK)
    raw[ACTION] = torch.from_numpy(target).unsqueeze(0)
    batch = preprocessor(raw)
    images, img_masks = policy._preprocess_images(batch)  # noqa: SLF001
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    normalized_target = policy.prepare_action(batch)
    sample_impl = policy.model.sample_actions.__wrapped__

    results = []
    for num_steps in args.num_steps:
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(args.seed)
        noise = policy.model.sample_noise(
            (1, policy.config.chunk_size, policy.config.max_action_dim), device
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            prediction = sample_impl(
                policy.model,
                images,
                img_masks,
                tokens,
                masks,
                noise=noise,
                num_steps=num_steps,
            )
            loss = torch.nn.functional.smooth_l1_loss(
                prediction[:, :20, :7], normalized_target[:, :20, :7]
            )
            loss.backward()
            grad_norm = float(torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().float().norm()
                for parameter in policy.model.action_out_proj.parameters()
                if parameter.grad is not None
            ])).item())
            results.append({
                "num_steps": num_steps,
                "success": True,
                "loss": float(loss.detach().item()),
                "gradient_norm": grad_norm,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
                ),
            })
        except torch.cuda.OutOfMemoryError as error:
            results.append({
                "num_steps": num_steps,
                "success": False,
                "error": f"{type(error).__name__}: {error}",
            })
            policy.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "corrections": str(args.corrections.resolve()),
        "index": args.index,
        "seed": args.seed,
        "trainable_module": "model.action_out_proj",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

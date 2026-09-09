#!/usr/bin/env python3
"""Verify that a zero-conditioned PAP-MoE exactly preserves a Pi0.5 backbone.

The Pi0.5 and PAP-MoE policies are loaded sequentially to fit on one GPU. Both
receive the same already-preprocessed observation, action chunk, flow noise,
and flow time. A forward hook captures the actual velocity-field output from
the action projection. PAP-MoE is evaluated with every physical expert masked.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPTS_DIR = Path(__file__).resolve().parent


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _capture_velocity(model, forward_kwargs: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
    projections: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        projections.append(output.detach().float().cpu())

    handle = model.action_out_proj.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model.forward(**forward_kwargs)
    finally:
        handle.remove()
    if not projections:
        raise RuntimeError("action_out_proj hook did not capture an output")
    return projections[-1], projections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-checkpoint", type=Path, required=True)
    parser.add_argument("--pap-init-checkpoint", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.path.insert(0, str(LEROBOT_SRC))
    from eval_pap_moe_expert_ablation import _load_episode_once, _raw_batch  # noqa: PLC0415
    import lerobot.policies.pi05.processor_pi05  # noqa: F401, PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy  # noqa: PLC0415
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: PLC0415

    episode = _load_episode_once(args.episode_npz)
    selected = np.asarray([args.frame], dtype=np.int64)

    pi_policy = PI05Policy.from_pretrained(str(args.pi_checkpoint.resolve()), strict=True)
    pi_policy.eval()
    pi_preprocessor, _ = make_pre_post_processors(
        pi_policy.config, pretrained_path=str(args.pi_checkpoint.resolve())
    )
    raw = _raw_batch(
        episode,
        selected,
        pi_policy.config.chunk_size,
        wrist_dropout=False,
        continuous_gripper=True,
    )
    processed = pi_preprocessor(
        {
            "observation.state": raw["observation.state"],
            "observation.images.camera0": raw["observation.images.camera0"],
            "observation.images.camera1": raw["observation.images.camera1"],
            "action": raw["action"],
            "task": raw["task"],
        }
    )
    pi_device = next(pi_policy.parameters()).device
    images, image_masks = pi_policy._preprocess_images(processed)
    actions = pi_policy.prepare_action(processed)
    common_cpu = {
        "images": _to_cpu(images),
        "img_masks": _to_cpu(image_masks),
        "tokens": _to_cpu(processed["observation.language.tokens"]),
        "masks": _to_cpu(processed["observation.language.attention_mask"]),
        "actions": _to_cpu(actions),
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    noise_cpu = torch.randn(actions.shape, generator=generator, dtype=torch.float32)
    time_cpu = torch.tensor([0.5], dtype=torch.float32)
    pi_kwargs = {key: _to_device(value, pi_device) for key, value in common_cpu.items()}
    pi_kwargs["noise"] = noise_cpu.to(pi_device)
    pi_kwargs["time"] = time_cpu.to(pi_device)
    pi_velocity, pi_projections = _capture_velocity(pi_policy.model, pi_kwargs)
    if len(pi_projections) != 1:
        raise RuntimeError(f"Pi0.5 unexpectedly used action_out_proj {len(pi_projections)} times")

    del pi_kwargs, processed, pi_preprocessor, pi_policy
    gc.collect()
    torch.cuda.empty_cache()

    pap_policy = PAPMoEPolicy.from_pretrained(str(args.pap_init_checkpoint.resolve()))
    pap_policy.eval()
    pap_device = next(pap_policy.parameters()).device
    pap_kwargs = {key: _to_device(value, pap_device) for key, value in common_cpu.items()}
    pap_kwargs.update(
        {
            "force": torch.zeros(1, pap_policy.config.force_dim, device=pap_device),
            "force_fast": torch.zeros(
                1,
                pap_policy.config.fast_force_window_size,
                pap_policy.config.force_dim,
                device=pap_device,
            ),
            "force_slow": torch.zeros(
                1,
                pap_policy.config.slow_force_window_size,
                pap_policy.config.force_dim,
                device=pap_device,
            ),
            "state": torch.zeros(1, pap_policy.config.robot_state_dim, device=pap_device),
            "state_history": torch.zeros(
                1,
                pap_policy.config.state_history_size,
                pap_policy.config.robot_state_dim,
                device=pap_device,
            ),
            "visual_quality": torch.zeros(
                1, pap_policy.config.visual_quality_dim, device=pap_device
            ),
            "expert_mask": torch.zeros(1, pap_policy.config.num_experts, device=pap_device),
            "noise": noise_cpu.to(pap_device),
            "time": time_cpu.to(pap_device),
        }
    )
    pap_velocity, pap_projections = _capture_velocity(pap_policy.model, pap_kwargs)
    if len(pap_projections) != 2:
        raise RuntimeError(
            f"PAP-MoE unexpectedly used action_out_proj {len(pap_projections)} times"
        )

    difference = (pap_velocity - pi_velocity).abs()
    pap_internal_difference = (pap_projections[-1] - pap_projections[-2]).abs()
    result = {
        "pi_checkpoint": str(args.pi_checkpoint.resolve()),
        "pap_init_checkpoint": str(args.pap_init_checkpoint.resolve()),
        "episode": str(args.episode_npz.resolve()),
        "frame": args.frame,
        "seed": args.seed,
        "threshold": args.atol,
        "velocity_shape": list(pi_velocity.shape),
        "pap_zero_condition_internal_max_abs_diff": float(pap_internal_difference.max()),
        "pap_vs_pi05_max_abs_diff": float(difference.max()),
        "pap_vs_pi05_mean_abs_diff": float(difference.mean()),
        "passed": bool(float(difference.max()) <= args.atol),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

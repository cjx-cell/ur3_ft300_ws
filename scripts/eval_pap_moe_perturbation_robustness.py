#!/usr/bin/env python3
"""Audit decoded PAP-MoE action sensitivity outside exact demonstration states.

The test keeps the checkpoint and diffusion noise fixed, perturbs one raw
observation at a time, and compares the first ten *decoded physical actions*.
It separates three deployment paths:

* ``physicsgate_full``: predicted 50-step PhysicsGate route + all experts;
* ``dataset_route_full``: demonstrated 50-step route + all experts;
* ``all_zero``: the same jointly trained action backbone with PAP residuals off.

This is a local robustness diagnostic, not a closed-loop success-rate test.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPTS_DIR = Path(__file__).resolve().parent
EXPERT_NAMES = ["E1_free_motion", "E2_visual_blind", "E3_rigid_contact", "E4_compliant"]
MODES = ("physicsgate_full", "dataset_route_full", "all_zero")


def _clone_raw(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in raw.items()
    }


def _zero_fill_shift(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    shifted = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., :dy, :] = 0
    elif dy < 0:
        shifted[..., dy:, :] = 0
    if dx > 0:
        shifted[..., :, :dx] = 0
    elif dx < 0:
        shifted[..., :, dx:] = 0
    return shifted


def _refresh_visual_quality(raw: dict[str, Any]) -> None:
    """Match the online two-camera quality descriptor after image corruption."""
    cameras = torch.stack(
        [raw["observation.images.camera0"], raw["observation.images.camera1"]], dim=1
    ).float()
    gray = cameras.mean(dim=2)
    finite = torch.isfinite(cameras).all(dim=(1, 2, 3, 4))
    black = (gray <= 0.02).float().mean(dim=(1, 2, 3))
    saturated = (gray >= 0.98).float().mean(dim=(1, 2, 3))
    contrast = gray.std(dim=(-2, -1), correction=0).mean(dim=1)
    valid = finite & (contrast >= 0.01)
    raw["observation.visual_quality"] = torch.stack(
        [black, saturated, contrast, valid.float()], dim=-1
    )


def _perturbations(raw: dict[str, Any], force_scale: np.ndarray, seed: int):
    yield "clean", "clean", 0.0, raw

    direction = torch.tensor([1, -1, 1, -1, 1, -1], dtype=torch.float32)
    for magnitude in (0.01, 0.03, 0.05):
        for sign in (-1.0, 1.0):
            changed = _clone_raw(raw)
            delta = sign * magnitude * direction
            changed["observation.state"][..., :6] += delta
            changed["observation.state_history"][..., :6] += delta
            yield (
                f"state_{sign:+.0f}_{magnitude:.2f}",
                "state_joint_offset_rad",
                magnitude,
                changed,
            )

    wrench_direction = torch.tensor(
        [1.0, -0.65, 0.4, 0.25, -0.15, 0.1], dtype=torch.float32
    )
    wrench_direction /= wrench_direction.square().mean().sqrt()
    scale = torch.from_numpy(force_scale.astype(np.float32))
    for magnitude in (0.25, 0.5, 1.0):
        changed = _clone_raw(raw)
        delta = magnitude * scale * wrench_direction
        for key in ("observation.force", "observation.force_fast", "observation.force_slow"):
            changed[key] += delta
        yield f"force_bias_{magnitude:.2f}sigma", "force_bias_sigma", magnitude, changed

    for level_index, magnitude in enumerate((0.25, 0.5, 1.0)):
        changed = _clone_raw(raw)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 7919 * (level_index + 1))
        for key in ("observation.force", "observation.force_fast", "observation.force_slow"):
            noise = torch.randn(changed[key].shape, generator=generator)
            changed[key] += magnitude * noise * scale
        yield f"force_noise_{magnitude:.2f}sigma", "force_noise_sigma", magnitude, changed

    for pixels in (4, 8, 16):
        changed = _clone_raw(raw)
        for key in ("observation.images.camera0", "observation.images.camera1"):
            changed[key] = _zero_fill_shift(changed[key], pixels, pixels // 2)
        _refresh_visual_quality(changed)
        yield f"image_shift_{pixels}px", "image_shift_pixels", float(pixels), changed

    for darkness in (0.1, 0.25, 0.5):
        changed = _clone_raw(raw)
        for key in ("observation.images.camera0", "observation.images.camera1"):
            changed[key] = (changed[key] * (1.0 - darkness)).clamp(0, 1)
        _refresh_visual_quality(changed)
        yield f"image_dark_{darkness:.2f}", "image_dark_fraction", darkness, changed

    for fraction in (0.1, 0.25, 0.4):
        changed = _clone_raw(raw)
        for key in ("observation.images.camera0", "observation.images.camera1"):
            image = changed[key]
            height, width = image.shape[-2:]
            side_h = max(1, round(height * np.sqrt(fraction)))
            side_w = max(1, round(width * np.sqrt(fraction)))
            y0, x0 = (height - side_h) // 2, (width - side_w) // 2
            image[..., y0 : y0 + side_h, x0 : x0 + side_w] = 0
        _refresh_visual_quality(changed)
        yield f"image_occlusion_{fraction:.2f}", "image_occlusion_fraction", fraction, changed


def _as_numpy(value: torch.Tensor | None) -> np.ndarray | None:
    return None if value is None else value.detach().float().cpu().numpy()


def _route_sequence(episode: dict[str, np.ndarray], index: int, horizon: int) -> np.ndarray:
    positions = np.minimum(index + np.arange(horizon), len(episode["stage"]) - 1)
    route = np.asarray(episode["stage"][positions], dtype=np.float32).copy()
    route /= np.clip(route.sum(axis=-1, keepdims=True), 1e-8, None)
    return route[None]


def _select_anchor(stage: np.ndarray, expert_index: int) -> int:
    candidates = np.flatnonzero(stage.argmax(axis=-1) == expert_index)
    if not len(candidates):
        candidates = np.arange(len(stage))
    return int(candidates[len(candidates) // 2])


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    perturbation_records = [record for record in records if record["family"] != "clean"]
    result: dict[str, Any] = {"by_mode_family_level": {}, "sensitivity_ratios": {}}
    for mode in MODES:
        mode_result: dict[str, Any] = {}
        for family in sorted({record["family"] for record in perturbation_records}):
            family_result: dict[str, Any] = {}
            levels = sorted(
                {record["level"] for record in perturbation_records if record["family"] == family}
            )
            for level in levels:
                selected = [
                    record["modes"][mode]
                    for record in perturbation_records
                    if record["family"] == family and record["level"] == level
                ]
                family_result[str(level)] = {
                    "samples": len(selected),
                    "first10_physical_arm_delta_l2_mean": float(
                        np.mean([item["first10_physical_arm_delta_l2_mean"] for item in selected])
                    ),
                    "first10_physical_arm_delta_l2_max": float(
                        np.max([item["first10_physical_arm_delta_l2_max"] for item in selected])
                    ),
                    "first10_physical_action_mse_to_demo": float(
                        np.mean([item["first10_physical_action_mse_to_demo"] for item in selected])
                    ),
                    "first10_internal_arm_jump_mean": float(
                        np.mean([item["first10_internal_arm_jump_mean"] for item in selected])
                    ),
                    "route_l1_change_mean": float(
                        np.mean([item["route_l1_change_mean"] for item in selected])
                    ),
                    "route_argmax_flip_fraction": float(
                        np.mean([item["route_argmax_flip_fraction"] for item in selected])
                    ),
                    "condition_residual_norm_mean": float(
                        np.mean([item["condition_residual_norm_mean"] for item in selected])
                    ),
                }
            mode_result[family] = family_result
        result["by_mode_family_level"][mode] = mode_result

    for family in sorted({record["family"] for record in perturbation_records}):
        ratios: dict[str, float | None] = {}
        for level in sorted(
            {record["level"] for record in perturbation_records if record["family"] == family}
        ):
            selected = [
                record
                for record in perturbation_records
                if record["family"] == family and record["level"] == level
            ]
            full = np.mean(
                [r["modes"]["physicsgate_full"]["first10_physical_arm_delta_l2_mean"] for r in selected]
            )
            zero = np.mean(
                [r["modes"]["all_zero"]["first10_physical_arm_delta_l2_mean"] for r in selected]
            )
            ratios[str(level)] = None if zero < 1e-12 else float(full / zero)
        result["sensitivity_ratios"][family] = ratios
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--executed-horizon", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.path.insert(0, str(LEROBOT_SRC))
    from eval_pap_moe_expert_ablation import _load_episode_once, _raw_batch  # noqa: PLC0415
    import lerobot.policies.pap_moe.processor_pap_moe  # noqa: F401, PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.perf_counter()
    policy = PAPMoEPolicy.from_pretrained(str(args.checkpoint.resolve()))
    policy.config.train_physicsgate_only = False
    policy.config.train_expert_only = False
    policy.config.train_gate_calibration_only = False
    policy.config.train_conditioner_only = True
    policy.config.train_pap_moe_joint = False
    if policy.config.rtc_config is not None:
        policy.config.rtc_config.enabled = False
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint.resolve())
    )
    device = next(policy.parameters()).device
    horizon = min(args.executed_horizon, policy.config.chunk_size)
    action_dim = policy.config.output_features["action"].shape[0]
    records: list[dict[str, Any]] = []

    # Rotate representative phases over the five scene positions.
    anchor_experts = (0, 2, 3, 0, 2)
    with torch.inference_mode():
        for episode_number, episode_path in enumerate(args.episode_npz):
            episode = _load_episode_once(episode_path)
            expert_index = anchor_experts[episode_number % len(anchor_experts)]
            index = _select_anchor(episode["stage"], expert_index)
            raw_clean = _raw_batch(
                episode,
                np.asarray([index]),
                policy.config.chunk_size,
                wrist_dropout=False,
                continuous_gripper=True,
            )
            all_force = np.concatenate(
                [episode["force"], episode["force_fast"].reshape(-1, 6), episode["force_slow"].reshape(-1, 6)],
                axis=0,
            )
            force_scale = np.maximum(all_force.std(axis=0), 1e-3)
            true_route = torch.from_numpy(
                _route_sequence(episode, index, policy.config.chunk_size)
            ).to(device=device)
            target = np.asarray(raw_clean["action"][0, :horizon], dtype=np.float32)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(args.seed + episode_number * 10007 + index)
            fixed_noise = torch.randn(
                (1, policy.config.chunk_size, policy.config.max_action_dim),
                generator=generator,
                dtype=torch.float32,
            ).to(device)
            clean_outputs: dict[str, dict[str, np.ndarray]] = {}

            for variant_name, family, level, raw in _perturbations(
                raw_clean, force_scale, args.seed + episode_number * 65537 + index
            ):
                batch = preprocessor(raw)
                images, image_masks = policy._preprocess_images(batch)
                common = dict(
                    images=images,
                    img_masks=image_masks,
                    tokens=batch["observation.language.tokens"],
                    masks=batch["observation.language.attention_mask"],
                    force=batch["observation.force"],
                    force_fast=batch["observation.force_fast"],
                    force_slow=batch["observation.force_slow"],
                    state=batch["observation.state"],
                    state_history=batch["observation.state_history"],
                    visual_quality=batch["observation.visual_quality"],
                    noise=fixed_noise,
                )
                mode_outputs: dict[str, Any] = {}
                for mode in MODES:
                    kwargs: dict[str, Any] = {}
                    if mode == "dataset_route_full":
                        kwargs["stage_override"] = true_route
                    kwargs["expert_mask"] = torch.tensor(
                        [0.0, 0.0, 0.0, 0.0] if mode == "all_zero" else [1.0, 1.0, 1.0, 1.0],
                        device=device,
                    )
                    output = policy.model.sample_actions(**common, **kwargs)
                    normalized = output["actions"][..., :action_dim]
                    physical = postprocessor(normalized)[0, :horizon].numpy()
                    route = _as_numpy(output["routing_probs"])
                    if route is not None and route.ndim == 2:
                        route = np.repeat(route[:, None, :], policy.config.chunk_size, axis=1)
                    residual = _as_numpy(output.get("applied_condition_residual_norm"))
                    if residual is None:
                        residual = _as_numpy(output.get("condition_residual_norm"))
                    current = {
                        "normalized": normalized[0, :horizon].float().cpu().numpy(),
                        "physical": physical,
                        "route": route[0, :horizon] if route is not None else np.zeros((horizon, 4)),
                        "residual": np.zeros(1) if residual is None else residual.reshape(-1),
                    }
                    if family == "clean":
                        clean_outputs[mode] = current
                    baseline = clean_outputs[mode]
                    physical_delta = current["physical"] - baseline["physical"]
                    arm_delta = np.linalg.norm(physical_delta[:, :6], axis=-1)
                    route_delta = np.abs(current["route"] - baseline["route"])
                    route_flips = current["route"].argmax(-1) != baseline["route"].argmax(-1)
                    internal_jump = np.linalg.norm(np.diff(current["physical"][:, :6], axis=0), axis=-1)
                    mode_outputs[mode] = {
                        "first10_physical_arm_delta_l2_mean": float(arm_delta.mean()),
                        "first10_physical_arm_delta_l2_max": float(arm_delta.max()),
                        "first10_physical_action_mse_to_demo": float(
                            np.mean((current["physical"] - target) ** 2)
                        ),
                        "first10_normalized_action_delta_l2_mean": float(
                            np.linalg.norm(current["normalized"] - baseline["normalized"], axis=-1).mean()
                        ),
                        "first10_internal_arm_jump_mean": float(internal_jump.mean()),
                        "first10_internal_arm_jump_max": float(internal_jump.max()),
                        "route_l1_change_mean": float(route_delta.mean()),
                        "route_argmax_flip_fraction": float(route_flips.mean()),
                        "condition_residual_norm_mean": float(current["residual"].mean()),
                        "condition_residual_norm_max": float(current["residual"].max()),
                    }
                records.append(
                    {
                        "episode": str(episode_path.resolve()),
                        "frame_index": index,
                        "anchor_expert": EXPERT_NAMES[expert_index],
                        "variant": variant_name,
                        "family": family,
                        "level": level,
                        "modes": mode_outputs,
                    }
                )
                print(
                    f"episode {episode_number + 1}/{len(args.episode_npz)} frame {index}: {variant_name}",
                    flush=True,
                )
            del episode

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "episodes": [str(path.resolve()) for path in args.episode_npz],
        "seed": args.seed,
        "executed_horizon": horizon,
        "fixed_diffusion_noise_per_anchor": True,
        "rtc_enabled": False,
        "modes": {
            "physicsgate_full": "Predicted 50-step PhysicsGate route, all expert residuals enabled.",
            "dataset_route_full": "Demonstrated 50-step route, all expert residuals enabled.",
            "all_zero": "Same jointly trained action backbone, all physical residuals disabled.",
        },
        "metric_units": {
            "physical_arm_delta_l2": "joint radians L2 over six UR3 joints",
            "physical_action_mse_to_demo": "mean squared physical action units; six radians plus gripper radians",
            "force_levels": "multiples of each episode's raw per-channel wrench standard deviation",
        },
        "records": records,
        "summary": _summarize(records),
        "evaluation_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure PAP-MoE action dependence on each physical expert.

The evaluator fixes the flow-matching noise, time, observations, semantic
prediction, and ground-truth soft route for every mask. The only changed value
is the selected expert output token, so loss deltas are attributable to the
physical-conditioning path rather than stochastic sampling or PhysicsGate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
EXPERT_NAMES = ["E1_free_motion", "E2_visual_blind", "E3_rigid_contact", "E4_compliant"]
MASKS = {
    "full": [1.0, 1.0, 1.0, 1.0],
    "all_zero": [0.0, 0.0, 0.0, 0.0],
    "drop_E1": [0.0, 1.0, 1.0, 1.0],
    "drop_E2": [1.0, 0.0, 1.0, 1.0],
    "drop_E3": [1.0, 1.0, 0.0, 1.0],
    "drop_E4": [1.0, 1.0, 1.0, 0.0],
}
OPEN_GRIPPER_PHASES = {
    "release the peg after verification",
    "retract and go back to home",
}


def _visual_quality(camera0: np.ndarray, camera1: np.ndarray) -> np.ndarray:
    cameras = np.stack([camera0, camera1], axis=1).astype(np.float32, copy=False)
    gray = cameras.mean(axis=-1)
    finite = np.isfinite(cameras).all(axis=(1, 2, 3, 4))
    black = np.mean(gray <= 0.02, axis=(1, 2, 3))
    saturated = np.mean(gray >= 0.98, axis=(1, 2, 3))
    contrast = np.mean(np.std(gray, axis=(2, 3)), axis=1)
    valid = finite & (contrast >= 0.01)
    return np.stack([black, saturated, contrast, valid.astype(np.float32)], axis=-1).astype(
        np.float32
    )


def _observed_states(states: np.ndarray) -> np.ndarray:
    result = np.asarray(states, dtype=np.float32).copy()
    result[..., 6] = (result[..., 6] > 0.12).astype(np.float32)
    return result


def _action_chunks(
    actions: np.ndarray,
    tasks: np.ndarray,
    indices: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    positions = indices[:, None] + np.arange(chunk_size)[None, :]
    positions = np.minimum(positions, len(actions) - 1)
    result = np.ascontiguousarray(actions[positions], dtype=np.float32)
    result[..., 6] = (result[..., 6] > 0.12).astype(np.float32)
    task_chunks = np.asarray(tasks, dtype=object)[positions]
    result[np.isin(task_chunks, list(OPEN_GRIPPER_PHASES)), 6] = 0.0
    return result


def _route_chunks(
    stages: np.ndarray,
    indices: np.ndarray,
    chunk_size: int,
    *,
    wrist_dropout: bool,
    all_camera_dropout: bool,
) -> np.ndarray:
    positions = np.minimum(
        indices[:, None] + np.arange(chunk_size)[None, :], len(stages) - 1
    )
    routes = np.ascontiguousarray(stages[positions], dtype=np.float32).copy()
    if all_camera_dropout:
        routes[..., 0] = 0.0
        routes[..., 1] = 1.0
    elif wrist_dropout:
        routes[..., 1] += routes[..., 2] + routes[..., 3]
        routes[..., 2:] = 0.0
    routes /= np.clip(routes.sum(axis=-1, keepdims=True), 1e-8, None)
    return routes


def _raw_batch(
    episode: dict[str, np.ndarray],
    indices: np.ndarray,
    chunk_size: int,
    *,
    wrist_dropout: bool,
    continuous_gripper: bool,
    all_camera_dropout: bool = False,
    visual_history_indices: tuple[int, ...] | None = None,
) -> dict[str, object]:
    camera0_hwc = np.ascontiguousarray(episode["camera0"][indices])
    camera1_hwc = np.ascontiguousarray(episode["camera1"][indices])
    stage = np.asarray(episode["stage"][indices], dtype=np.float32).copy()
    if wrist_dropout and all_camera_dropout:
        raise ValueError("wrist and all-camera dropout are mutually exclusive")
    if all_camera_dropout:
        camera0_hwc = np.zeros_like(camera0_hwc)
        camera1_hwc = np.zeros_like(camera1_hwc)
        visual_quality = _visual_quality(camera0_hwc, camera1_hwc)
        # Match training-time q=0 factorization exactly:
        # [q*E1, 1-q, E3, E4], then normalize. Contact experts remain active
        # alongside E2 instead of being incorrectly collapsed into E2.
        stage[:, 0] = 0.0
        stage[:, 1] = 1.0
        stage /= np.clip(stage.sum(axis=-1, keepdims=True), 1e-8, None)
    elif wrist_dropout:
        camera0_hwc = np.zeros_like(camera0_hwc)
        visual_quality = _visual_quality(camera0_hwc, camera1_hwc)
        stage[:, 1] += stage[:, 2] + stage[:, 3]
        stage[:, 2:] = 0.0
        stage /= np.clip(stage.sum(axis=-1, keepdims=True), 1e-8, None)
    else:
        visual_quality = np.asarray(episode["visual_quality"][indices], dtype=np.float32)
    if visual_history_indices is None:
        camera0 = np.ascontiguousarray(camera0_hwc.transpose(0, 3, 1, 2))
        camera1 = np.ascontiguousarray(camera1_hwc.transpose(0, 3, 1, 2))
        camera_padding = None
    else:
        offsets = np.asarray(visual_history_indices, dtype=np.int64)
        history_positions = indices[:, None] + offsets[None, :]
        camera_padding = history_positions < 0
        history_positions = np.clip(history_positions, 0, len(episode["camera0"]) - 1)
        camera0_history = np.ascontiguousarray(episode["camera0"][history_positions])
        camera1_history = np.ascontiguousarray(episode["camera1"][history_positions])
        if all_camera_dropout:
            camera0_history[...] = 0
            camera1_history[...] = 0
        elif wrist_dropout:
            camera0_history[...] = 0
        camera0 = np.ascontiguousarray(camera0_history.transpose(0, 1, 4, 2, 3))
        camera1 = np.ascontiguousarray(camera1_history.transpose(0, 1, 4, 2, 3))
    observed_state = (
        np.asarray(episode["state"][indices], dtype=np.float32)
        if continuous_gripper
        else _observed_states(episode["state"][indices])
    )
    action_chunks = (
        np.ascontiguousarray(
            episode["action"][
                np.minimum(
                    indices[:, None] + np.arange(chunk_size)[None, :],
                    len(episode["action"]) - 1,
                )
            ],
            dtype=np.float32,
        )
        if continuous_gripper
        else _action_chunks(episode["action"], episode["task"], indices, chunk_size)
    )
    result = {
        "observation.state": torch.from_numpy(observed_state),
        "observation.state_history": torch.from_numpy(episode["state_history"][indices]),
        "observation.force": torch.from_numpy(episode["force"][indices]),
        "observation.force_fast": torch.from_numpy(episode["force_fast"][indices]),
        "observation.force_slow": torch.from_numpy(episode["force_slow"][indices]),
        "observation.visual_quality": torch.from_numpy(visual_quality),
        "observation.stage": torch.from_numpy(stage),
        "observation.images.camera0": torch.from_numpy(camera0),
        "observation.images.camera1": torch.from_numpy(camera1),
        "action": torch.from_numpy(action_chunks),
        "task": [str(value).strip() for value in episode["task"][indices]],
    }
    if camera_padding is not None:
        result["observation.images.camera0_is_pad"] = torch.from_numpy(camera_padding.copy())
        result["observation.images.camera1_is_pad"] = torch.from_numpy(camera_padding.copy())
    return result


def _load_episode_once(episode_path: Path) -> dict[str, np.ndarray]:
    """Decompress each NPZ member once instead of once per evaluation batch."""
    required_keys = (
        "camera0",
        "camera1",
        "state",
        "state_history",
        "force",
        "force_fast",
        "force_slow",
        "visual_quality",
        "stage",
        "action",
        "task",
    )
    with np.load(episode_path, allow_pickle=True) as archive:
        missing = [key for key in required_keys if key not in archive]
        if missing:
            raise KeyError(f"{episode_path} is missing required arrays: {missing}")
        return {key: archive[key] for key in required_keys}


def _select_frame_indices(
    stage: np.ndarray,
    stride: int,
    max_frames_per_expert: int,
) -> np.ndarray:
    if max_frames_per_expert <= 0:
        return np.arange(0, len(stage), stride)

    dominant_expert = stage.argmax(axis=-1)
    selected: list[np.ndarray] = []
    for expert_index in range(len(EXPERT_NAMES)):
        candidates = np.flatnonzero(dominant_expert == expert_index)
        if len(candidates) > max_frames_per_expert:
            positions = np.linspace(0, len(candidates) - 1, max_frames_per_expert)
            candidates = candidates[np.rint(positions).astype(np.int64)]
        selected.append(candidates)
    if not any(len(indices) for indices in selected):
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(selected))


def _fixed_flow_inputs(
    action_shape: torch.Size,
    indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + int(indices[0]))
    noise = torch.randn(action_shape, generator=generator, dtype=torch.float32).to(device)
    fractions = torch.from_numpy((indices % 997).astype(np.float32) / 996.0).to(device)
    time_values = 0.2 + 0.6 * fractions
    return noise, time_values


def _summarize(
    losses: dict[str, list[np.ndarray]],
    targets: list[np.ndarray],
    num_seeds: int,
) -> dict[str, object]:
    target_array = np.concatenate(targets)
    target_ids = target_array.argmax(axis=-1)
    loss_arrays = {name: np.concatenate(values) for name, values in losses.items()}
    full = loss_arrays["full"]
    result: dict[str, object] = {
        "sampled_frames": int(len(full) // num_seeds),
        "evaluated_frame_seed_pairs": int(len(full)),
        "target_mean": target_array.mean(axis=0).tolist(),
        "masks": {},
    }

    def paired_delta_stats(values: np.ndarray, selected: np.ndarray) -> dict[str, object]:
        deltas = values[selected] - full[selected]
        if len(deltas) == 0:
            return {
                "delta_vs_full": None,
                "delta_standard_error": None,
                "delta_ci95": None,
                "worse_than_full_fraction": None,
            }
        mean = float(deltas.mean())
        standard_error = 0.0
        if len(deltas) > 1:
            standard_error = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
        return {
            "delta_vs_full": mean,
            "delta_standard_error": standard_error,
            "delta_ci95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
            "worse_than_full_fraction": float((deltas > 0).mean()),
        }

    for name, values in loss_arrays.items():
        subsets = {}
        for expert_index, expert_name in enumerate(EXPERT_NAMES):
            selected = target_ids == expert_index
            subsets[expert_name] = {
                "frames": int(selected.sum() // num_seeds),
                "frame_seed_pairs": int(selected.sum()),
                "mse": None if not selected.any() else float(values[selected].mean()),
                **paired_delta_stats(values, selected),
            }
        all_frames = np.ones(len(values), dtype=bool)
        result["masks"][name] = {
            "expert_mask": MASKS[name],
            "mse": float(values.mean()),
            **paired_delta_stats(values, all_frames),
            "dominant_expert_subsets": subsets,
        }
    return result


def _summarize_full_condition_details(
    losses: list[np.ndarray],
    route_sequences: list[np.ndarray],
    action_names: list[str],
    executed_horizon: int,
) -> dict[str, object]:
    """Report where the full-condition flow loss occurs inside the action chunk."""
    if not losses:
        return {"available": False, "reason": "full mask was not evaluated"}
    loss = np.concatenate(losses, axis=0)
    routes = np.concatenate(route_sequences, axis=0)
    horizon = loss.shape[1]
    executed_horizon = min(max(1, executed_horizon), horizon)
    route_ids = routes.argmax(axis=-1)
    transition = np.zeros(route_ids.shape, dtype=bool)
    transition[:, 1:] = route_ids[:, 1:] != route_ids[:, :-1]
    stable = ~transition

    def selected_mse(mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        expanded = np.broadcast_to(mask[..., None], loss.shape)
        return float(loss[expanded].mean())

    return {
        "metric": "normalized_flow_matching_mse",
        "full_50_or_configured_chunk_mse": float(loss.mean()),
        "executed_prefix_horizon": executed_horizon,
        "executed_prefix_mse": float(loss[:, :executed_horizon].mean()),
        "per_horizon_mse": loss.mean(axis=(0, 2)).tolist(),
        "per_action_dimension_mse": {
            name: float(value)
            for name, value in zip(action_names, loss.mean(axis=(0, 1)), strict=False)
        },
        "route_transition_steps": int(transition.sum()),
        "route_transition_mse": selected_mse(transition),
        "route_stable_steps": int(stable.sum()),
        "route_stable_mse": selected_mse(stable),
        "notes": (
            "Per-action values are normalized flow-matching losses, not physical-unit "
            "joint-position errors. Route transitions use ground-truth route argmax changes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, action="append", required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--max-frames-per-expert",
        type=int,
        default=0,
        help="If positive, select up to this many dominant frames per expert per episode instead of striding.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument(
        "--routing-source",
        choices=("dataset", "predicted"),
        default="dataset",
        help="Use stored soft routes or the checkpoint's PhysicsGate for action conditioning.",
    )
    parser.add_argument(
        "--mask",
        dest="mask_names",
        action="append",
        choices=tuple(MASKS),
        help="Evaluate only this expert mask (repeatable). Defaults to every mask.",
    )
    parser.add_argument(
        "--wrist-dropout",
        action="store_true",
        help="Black camera0, recompute visual quality, and shift contact target mass to E2.",
    )
    parser.add_argument(
        "--all-camera-dropout",
        action="store_true",
        help="Black both real cameras and apply the training-time q=0 E2/contact cooperative route.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--continuous-gripper",
        action="store_true",
        help="Preserve the Workspace50 continuous 0..0.8 gripper state/action contract.",
    )
    args = parser.parse_args()
    if args.wrist_dropout and args.all_camera_dropout:
        raise ValueError("--wrist-dropout and --all-camera-dropout are mutually exclusive")
    if (
        args.stride < 1
        or args.batch_size < 1
        or args.max_frames_per_expert < 0
        or args.num_seeds < 1
    ):
        raise ValueError(
            "--stride/--batch-size/--num-seeds must be positive and "
            "--max-frames-per-expert non-negative"
        )
    selected_masks = list(dict.fromkeys(args.mask_names or MASKS))
    if "full" not in selected_masks:
        raise ValueError("--mask full is required as the paired comparison baseline")

    sys.path.insert(0, str(LEROBOT_SRC))
    import lerobot.policies.pap_moe.processor_pap_moe  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PAP-MoE action ablation")
    started = time.perf_counter()
    policy = PAPMoEPolicy.from_pretrained(str(args.checkpoint.resolve()))
    if policy.config.physics_expert_architecture != "heterogeneous_v2":
        raise ValueError("Expert ablation requires a heterogeneous_v2 checkpoint")
    # Checkpoints persist their training-stage fast-path flag. Evaluation must
    # always execute the cached-prefix action path, irrespective of which stage
    # produced the weights.
    policy.config.train_physicsgate_only = False
    policy.config.train_expert_only = False
    policy.config.train_gate_calibration_only = False
    policy.config.train_conditioner_only = True
    policy.config.train_pap_moe_joint = False
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint.resolve()))

    losses: dict[str, list[np.ndarray]] = {name: [] for name in selected_masks}
    baseline_losses: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    norm_values: list[np.ndarray] = []
    full_condition_detailed_losses: list[np.ndarray] = []
    full_condition_route_sequences: list[np.ndarray] = []
    per_episode: list[dict[str, object]] = []
    action_dim = policy.config.output_features["action"].shape[0]
    device = next(policy.parameters()).device

    with torch.inference_mode():
        for episode_number, episode_path in enumerate(args.episode_npz, start=1):
            print(f"Loading episode {episode_number}/{len(args.episode_npz)}: {episode_path}", flush=True)
            episode = _load_episode_once(episode_path)
            episode_losses: dict[str, list[np.ndarray]] = {
                name: [] for name in selected_masks
            }
            episode_baseline_losses: list[np.ndarray] = []
            episode_targets: list[np.ndarray] = []
            frame_indices = _select_frame_indices(
                episode["stage"],
                args.stride,
                args.max_frames_per_expert,
            )
            batch_starts = range(0, len(frame_indices), args.batch_size)
            total_batches = (len(frame_indices) + args.batch_size - 1) // args.batch_size
            print(
                f"Evaluating {len(frame_indices)} sampled frames in {total_batches} batches",
                flush=True,
            )
            for batch_number, start in enumerate(batch_starts, start=1):
                selected = frame_indices[start : start + args.batch_size]
                batch = preprocessor(
                    _raw_batch(
                        episode,
                        selected,
                        policy.config.chunk_size,
                        wrist_dropout=args.wrist_dropout,
                        continuous_gripper=args.continuous_gripper,
                        all_camera_dropout=args.all_camera_dropout,
                        visual_history_indices=(
                            tuple(policy.config.visual_memory_history_indices)
                            if policy.config.use_visual_memory
                            else None
                        ),
                    )
                )
                images, image_masks, visual_history, visual_history_padding = (
                    policy._preprocess_pap_images(batch)
                )
                actions = policy.prepare_action(batch)
                route = batch["observation.stage"].float().clamp_min(0)
                route = route / route.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                ground_truth_route_sequence = _route_chunks(
                    episode["stage"],
                    selected,
                    policy.config.chunk_size,
                    wrist_dropout=args.wrist_dropout,
                    all_camera_dropout=args.all_camera_dropout,
                )
                common: dict[str, object] = {
                    "images": images,
                    "img_masks": image_masks,
                    "tokens": batch["observation.language.tokens"],
                    "masks": batch["observation.language.attention_mask"],
                    "actions": actions,
                    "force": batch["observation.force"],
                    "force_fast": batch["observation.force_fast"],
                    "force_slow": batch["observation.force_slow"],
                    "state": batch["observation.state"],
                    "state_history": batch["observation.state_history"],
                    "visual_quality": batch["observation.visual_quality"],
                    "visual_history": visual_history,
                    "visual_history_padding": visual_history_padding,
                }
                if args.routing_source == "dataset":
                    if getattr(policy.config, "action_step_routing", False):
                        common["stage_override"] = torch.from_numpy(
                            _route_chunks(
                                episode["stage"],
                                selected,
                                policy.config.chunk_size,
                                wrist_dropout=args.wrist_dropout,
                                all_camera_dropout=args.all_camera_dropout,
                            )
                        ).to(device=device, dtype=route.dtype)
                    else:
                        common["stage_override"] = route
                for seed_index in range(args.num_seeds):
                    noise, time_values = _fixed_flow_inputs(
                        actions.shape,
                        selected,
                        device,
                        args.seed + seed_index * 1_000_003,
                    )
                    targets.append(route.cpu().numpy())
                    episode_targets.append(route.cpu().numpy())
                    for name in selected_masks:
                        mask = MASKS[name]
                        outputs = policy.model.forward(
                            **common,
                            noise=noise,
                            time=time_values,
                            expert_mask=torch.tensor(mask, device=device),
                        )
                        per_frame = outputs["action_loss"][:, :, :action_dim].mean(dim=(1, 2))
                        loss_values = per_frame.float().cpu().numpy()
                        losses[name].append(loss_values)
                        episode_losses[name].append(loss_values)
                        if name == "full":
                            full_condition_detailed_losses.append(
                                outputs["action_loss"][:, :, :action_dim]
                                .float()
                                .cpu()
                                .numpy()
                            )
                            full_condition_route_sequences.append(
                                ground_truth_route_sequence.copy()
                            )
                            if "baseline_action_loss" in outputs:
                                baseline_per_frame = outputs["baseline_action_loss"][
                                    :, :, :action_dim
                                ].mean(dim=(1, 2))
                                baseline_losses.append(
                                    baseline_per_frame.float().cpu().numpy()
                                )
                                episode_baseline_losses.append(
                                    baseline_per_frame.float().cpu().numpy()
                                )
                            norm_values.append(outputs["expert_token_norms"].float().cpu().numpy())
                if batch_number % 25 == 0 or batch_number == total_batches:
                    print(
                        f"Episode {episode_number}: batch {batch_number}/{total_batches}",
                        flush=True,
                    )
            episode_unconditioned = (
                episode_baseline_losses
                if episode_baseline_losses
                else episode_losses.get("all_zero", [])
            )
            if not episode_unconditioned:
                raise ValueError(
                    "Checkpoint does not expose baseline_action_loss; include --mask all_zero "
                    "to compute the paired unconditioned action-flow baseline"
                )
            episode_summary = _summarize(
                episode_losses, episode_targets, args.num_seeds
            )
            episode_summary.update(
                {
                    "episode": str(episode_path.resolve()),
                    "paired_internal_unconditioned_action_flow_mse": float(
                        np.concatenate(episode_unconditioned).mean()
                    ),
                }
            )
            per_episode.append(episode_summary)
            del episode

    unconditioned_losses = (
        baseline_losses if baseline_losses else losses.get("all_zero", [])
    )
    if not unconditioned_losses:
        raise ValueError(
            "Checkpoint does not expose baseline_action_loss; include --mask all_zero "
            "to compute the paired unconditioned action-flow baseline"
        )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "episodes": [str(path.resolve()) for path in args.episode_npz],
        "stride": args.stride,
        "max_frames_per_expert_per_episode": args.max_frames_per_expert,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "routing_source": args.routing_source,
        "route_sequence_source": (
            "direct_observation_physics_gate"
            if getattr(policy.config, "physics_gate_architecture", "legacy_softmax")
            == "physics_gate_v2"
            and args.routing_source == "predicted"
            else None
        ),
        "temporal_route_action_source": (
            "normalized_dataset_demonstration_action_chunk"
            if getattr(policy.config, "physics_gate_architecture", "legacy_softmax")
            == "temporal_bcm_v2"
            and args.routing_source == "predicted"
            else None
        ),
        "wrist_dropout": args.wrist_dropout,
        "all_camera_dropout": args.all_camera_dropout,
        "evaluated_masks": selected_masks,
        "seeds": [args.seed + index * 1_000_003 for index in range(args.num_seeds)],
        "fixed_time_range": [0.2, 0.8],
        "gripper_contract": (
            {
                "observation": "continuous_physical_radians_0_to_0.8",
                "target": "continuous_physical_radians_0_to_0.8",
            }
            if args.continuous_gripper
            else {
                "observation": "physical_joint_gt_0.12",
                "target": "physical_joint_gt_0.12_with_release_retract_forced_open",
            }
        ),
        "paired_internal_unconditioned_action_flow_mse": float(
            np.concatenate(unconditioned_losses).mean()
        ),
        "metric_notes": {
            "paired_internal_unconditioned_action_flow_mse": (
                "The jointly trained PAP action backbone with all physical "
                "conditions masked. It is not an independently trained Pi0.5 baseline."
            ),
            "baseline_comparison": (
                "Evaluate the qualified Pi0.5 checkpoint separately with "
                "eval_pi05_matched_flow_mse.py under identical seeds."
            ),
            "temporal_route_action_source": (
                "For temporal_bcm_v2, this flow-matching audit conditions future routing on "
                "the normalized demonstration action chunk. It measures intrinsic route/action "
                "fit, not the deployment two-pass draft-action distribution."
            ),
        },
        "expert_names": EXPERT_NAMES,
        "expert_output_norm_mean": np.concatenate(norm_values).mean(axis=0).tolist(),
        "evaluation_seconds": time.perf_counter() - started,
        "full_condition_action_diagnostics": _summarize_full_condition_details(
            full_condition_detailed_losses,
            full_condition_route_sequences,
            list(policy.config.action_feature_names),
            policy.config.n_action_steps,
        ),
        "per_episode": per_episode,
        **_summarize(losses, targets, args.num_seeds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

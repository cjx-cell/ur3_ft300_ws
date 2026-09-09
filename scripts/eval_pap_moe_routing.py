#!/usr/bin/env python3
"""Evaluate PAP-MoE PhysicsGate routing on held-out raw episodes.

The normal pass compares the predicted four-expert distribution with the
stored soft routing prior. The wrist-dropout pass blacks only camera0 but
retains the clean physical target because camera1 remains available. Only the
all-camera-dropout pass activates the E2 blindness target.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
EXPERT_NAMES = ["E1_nominal", "E2_visual_degraded", "E3_contact_recovery", "E4_compliant"]
DEFAULT_GLOBAL_TASK = "pick up the peg and insert it into the hole"


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


def _raw_batch(
    episode: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    dropout_mode: str,
    action_horizon: int,
    visual_history_indices: tuple[int, ...] | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    camera0_hwc = np.ascontiguousarray(episode["camera0"][indices])
    camera1_hwc = np.ascontiguousarray(episode["camera1"][indices])
    stage = np.asarray(episode["stage"][indices], dtype=np.float32).copy()
    if dropout_mode != "normal":
        camera0_hwc = np.zeros_like(camera0_hwc)
        if dropout_mode == "all_camera_dropout":
            camera1_hwc = np.zeros_like(camera1_hwc)
        visual_quality = _visual_quality(camera0_hwc, camera1_hwc)
        if dropout_mode == "all_camera_dropout":
            # Match paired E2 training exactly: dropout has q=0 in
            # [q*E1, 1-q, E3, E4], followed by normalization. Contact experts
            # remain active so E2 can cooperate with E3/E4.
            stage[:, 0] = 0.0
            stage[:, 1] = 1.0
            stage /= np.clip(stage.sum(axis=-1, keepdims=True), 1e-8, None)
    else:
        visual_quality = np.asarray(episode["visual_quality"][indices], dtype=np.float32)

    camera_padding = None
    if visual_history_indices is None:
        camera0 = np.ascontiguousarray(camera0_hwc.transpose(0, 3, 1, 2))
        camera1 = np.ascontiguousarray(camera1_hwc.transpose(0, 3, 1, 2))
    else:
        offsets = np.asarray(visual_history_indices, dtype=np.int64)
        history_positions = indices[:, None] + offsets[None, :]
        camera_padding = history_positions < 0
        history_positions = np.clip(history_positions, 0, len(episode["camera0"]) - 1)
        camera0_history = np.ascontiguousarray(episode["camera0"][history_positions])
        camera1_history = np.ascontiguousarray(episode["camera1"][history_positions])
        if dropout_mode == "all_camera_dropout":
            camera0_history[...] = 0
            camera1_history[...] = 0
        elif dropout_mode == "wrist_dropout":
            camera0_history[...] = 0
        camera0 = np.ascontiguousarray(camera0_history.transpose(0, 1, 4, 2, 3))
        camera1 = np.ascontiguousarray(camera1_history.transpose(0, 1, 4, 2, 3))
    action_indices = np.minimum(
        indices[:, None] + np.arange(action_horizon, dtype=np.int64)[None],
        len(episode["action"]) - 1,
    )
    action_chunk = np.ascontiguousarray(episode["action"][action_indices], dtype=np.float32)
    raw = {
        "observation.state": torch.from_numpy(episode["state"][indices]),
        "observation.state_history": torch.from_numpy(episode["state_history"][indices]),
        "observation.force": torch.from_numpy(episode["force"][indices]),
        "observation.force_fast": torch.from_numpy(episode["force_fast"][indices]),
        "observation.force_slow": torch.from_numpy(episode["force_slow"][indices]),
        "observation.visual_quality": torch.from_numpy(visual_quality),
        "observation.stage": torch.from_numpy(stage),
        "observation.images.camera0": torch.from_numpy(camera0),
        "observation.images.camera1": torch.from_numpy(camera1),
        "action": torch.from_numpy(action_chunk),
        "task": [str(value).strip() for value in episode["task"][indices]],
    }
    if camera_padding is not None:
        raw["observation.images.camera0_is_pad"] = torch.from_numpy(camera_padding.copy())
        raw["observation.images.camera1_is_pad"] = torch.from_numpy(camera_padding.copy())
    return raw, stage


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
    )
    with np.load(episode_path, allow_pickle=True) as archive:
        missing = [key for key in required_keys if key not in archive]
        if missing:
            raise KeyError(f"{episode_path} is missing required arrays: {missing}")
        episode = {key: archive[key] for key in required_keys}
        if "task" in archive:
            episode["task"] = archive["task"]
        else:
            episode["task"] = np.full(
                len(episode["state"]), DEFAULT_GLOBAL_TASK, dtype=object
            )
        return episode


def _future_route_targets(
    episode: dict[str, np.ndarray], indices: np.ndarray, horizon: int, dropout_mode: str
) -> np.ndarray:
    offsets = np.arange(horizon, dtype=np.int64)
    future_indices = np.minimum(indices[:, None] + offsets[None], len(episode["stage"]) - 1)
    targets = np.asarray(episode["stage"][future_indices], dtype=np.float32).copy()
    if dropout_mode == "all_camera_dropout":
        targets[..., 0] = 0.0
        targets[..., 1] = 1.0
        targets /= np.clip(targets.sum(axis=-1, keepdims=True), 1e-8, None)
    return targets


def _summarize_route_sequence(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    target_ids = targets.argmax(axis=-1)
    prediction_ids = predictions.argmax(axis=-1)
    horizon_mae = np.abs(targets - predictions).mean(axis=(0, 2))
    horizon_accuracy = (target_ids == prediction_ids).mean(axis=0)
    transition_mask = target_ids != target_ids[:, :1]
    return {
        "sampled_chunks": int(targets.shape[0]),
        "horizon": int(targets.shape[1]),
        "all_steps_mae": float(np.abs(targets - predictions).mean()),
        "all_steps_argmax_accuracy": float((target_ids == prediction_ids).mean()),
        "executed_first10_mae": float(np.abs(targets[:, :10] - predictions[:, :10]).mean()),
        "executed_first10_argmax_accuracy": float(
            (target_ids[:, :10] == prediction_ids[:, :10]).mean()
        ),
        "transition_steps": int(transition_mask.sum()),
        "transition_step_argmax_accuracy": (
            None
            if not transition_mask.any()
            else float((target_ids[transition_mask] == prediction_ids[transition_mask]).mean())
        ),
        "mae_by_horizon": horizon_mae.tolist(),
        "argmax_accuracy_by_horizon": horizon_accuracy.tolist(),
    }


def _summarize(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    clipped = np.clip(predictions, 1e-8, 1.0)
    confusion = np.zeros((4, 4), dtype=np.int64)
    target_ids = targets.argmax(axis=-1)
    prediction_ids = predictions.argmax(axis=-1)
    for target, prediction in zip(target_ids, prediction_ids, strict=True):
        confusion[target, prediction] += 1
    contact_mask = targets[:, 2:].sum(axis=-1) >= 0.2
    return {
        "sampled_frames": int(len(targets)),
        "soft_cross_entropy": float(np.mean(-np.sum(targets * np.log(clipped), axis=-1))),
        "mean_absolute_error": float(np.mean(np.abs(targets - predictions))),
        "argmax_accuracy": float(np.mean(target_ids == prediction_ids)),
        "contact_subset_frames": int(contact_mask.sum()),
        "contact_subset_argmax_accuracy": (
            None
            if not contact_mask.any()
            else float(np.mean(target_ids[contact_mask] == prediction_ids[contact_mask]))
        ),
        "target_mean": targets.mean(axis=0).tolist(),
        "prediction_mean": predictions.mean(axis=0).tolist(),
        "confusion_matrix_rows_target_cols_prediction": confusion.tolist(),
    }


def _summarize_factors(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    """Report b/c/m quality without treating mobility as defined in free space."""
    absolute_error = np.abs(targets - predictions)
    contact_mask = targets[:, 1] >= 0.2
    binary_targets = targets >= 0.5
    binary_predictions = predictions >= 0.5
    return {
        "factor_names": ["visual_blindness_b", "contact_c", "contact_mobility_m"],
        "target_mean": targets.mean(axis=0).tolist(),
        "prediction_mean": predictions.mean(axis=0).tolist(),
        "mean_absolute_error": absolute_error.mean(axis=0).tolist(),
        "threshold_accuracy": (binary_targets == binary_predictions).mean(axis=0).tolist(),
        "mobility_contact_subset_frames": int(contact_mask.sum()),
        "mobility_contact_subset_mae": (
            None if not contact_mask.any() else float(absolute_error[contact_mask, 2].mean())
        ),
        "mobility_contact_subset_accuracy": (
            None
            if not contact_mask.any()
            else float((binary_targets[contact_mask, 2] == binary_predictions[contact_mask, 2]).mean())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, action="append", required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wrist-dropout",
        action="store_true",
        help="Evaluate camera0 dropout while camera1 remains valid; retain the physical target.",
    )
    parser.add_argument(
        "--all-camera-dropout",
        action="store_true",
        help="Also evaluate simultaneous camera0/camera1 blackout used in online E2 training.",
    )
    args = parser.parse_args()
    if args.stride < 1 or args.batch_size < 1:
        raise ValueError("--stride and --batch-size must be positive")

    sys.path.insert(0, str(LEROBOT_SRC))
    import lerobot.policies.pap_moe.processor_pap_moe  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy
    from lerobot.policies.pap_moe.pap_moe_modules import FactorizedPhysicsGate
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PAP-MoE evaluation")
    started = time.perf_counter()
    policy = PAPMoEPolicy.from_pretrained(str(args.checkpoint))
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))

    modes = ["normal"]
    if args.wrist_dropout:
        modes.append("wrist_dropout")
    if args.all_camera_dropout:
        modes.append("all_camera_dropout")
    mode_targets: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    mode_predictions: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    mode_factor_targets: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    mode_factor_predictions: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    mode_sequence_targets: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    mode_sequence_predictions: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    episodes: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}

    with torch.inference_mode():
        for episode_number, episode_path in enumerate(args.episode_npz, start=1):
            print(f"Loading episode {episode_number}/{len(args.episode_npz)}: {episode_path}", flush=True)
            episode = _load_episode_once(episode_path)
            frame_indices = np.arange(0, len(episode["state"]), args.stride)
            total_batches = (len(frame_indices) + args.batch_size - 1) // args.batch_size
            for mode in modes:
                print(
                    f"Evaluating episode {episode_number} mode={mode}: "
                    f"{len(frame_indices)} frames in {total_batches} batches",
                    flush=True,
                )
                ep_targets: list[np.ndarray] = []
                ep_predictions: list[np.ndarray] = []
                for batch_number, start in enumerate(
                    range(0, len(frame_indices), args.batch_size), start=1
                ):
                    selected = frame_indices[start : start + args.batch_size]
                    raw, targets = _raw_batch(
                        episode,
                        selected,
                        dropout_mode=mode,
                        action_horizon=policy.config.chunk_size,
                        visual_history_indices=(
                            tuple(policy.config.visual_memory_history_indices)
                            if policy.config.use_visual_memory
                            else None
                        ),
                    )
                    batch = preprocessor(raw)
                    images, image_masks, visual_history, visual_history_padding = (
                        policy._preprocess_pap_images(batch)
                    )
                    vlm_tokens, _, _ = policy.model._get_vlm_output(
                        images,
                        image_masks,
                        batch[OBS_LANGUAGE_TOKENS],
                        batch[OBS_LANGUAGE_ATTENTION_MASK],
                        use_cache=False,
                    )
                    force_tokens, proprio_token, quality_token = policy.model._encode_physics_inputs(
                        batch["observation.force"],
                        batch["observation.state"],
                        batch["observation.force_fast"],
                        batch["observation.force_slow"],
                        batch["observation.state_history"],
                        batch["observation.visual_quality"],
                    )
                    visual_memory, memory_age = policy.model._encode_visual_memory(
                        visual_history, visual_history_padding
                    )
                    route_action_tokens = None
                    if (
                        getattr(policy.config, "physics_gate_architecture", "legacy_softmax")
                        == "temporal_bcm_v2"
                    ):
                        normalized_actions = policy.prepare_action(batch)
                        route_action_tokens = policy.model.action_in_proj(
                            normalized_actions.to(
                                dtype=policy.model.action_in_proj.weight.dtype
                            )
                        )
                    pap_outputs = policy.model._forward_pap_moe(
                        force_tokens,
                        vlm_tokens,
                        proprio_token,
                        quality_token,
                        route_action_tokens=route_action_tokens,
                        run_experts=False,
                        visual_memory=visual_memory,
                        memory_age=memory_age,
                    )
                    routing = pap_outputs["stage_probs"]
                    ep_targets.append(targets)
                    ep_predictions.append(routing.float().cpu().numpy())
                    factor_predictions = pap_outputs.get("factor_probs")
                    if factor_predictions is not None:
                        factor_targets = FactorizedPhysicsGate.expert_probs_to_factors(
                            torch.from_numpy(targets)
                        )
                        mode_factor_targets[mode].append(factor_targets.numpy())
                        mode_factor_predictions[mode].append(
                            factor_predictions.float().cpu().numpy()
                        )
                    route_sequence = pap_outputs.get("predicted_route_sequence")
                    if route_sequence is not None:
                        sequence_targets = _future_route_targets(
                            episode, selected, policy.config.chunk_size, mode
                        )
                        mode_sequence_targets[mode].append(sequence_targets)
                        mode_sequence_predictions[mode].append(
                            route_sequence.float().cpu().numpy()
                        )
                    if batch_number % 25 == 0 or batch_number == total_batches:
                        print(
                            f"Episode {episode_number} mode={mode}: "
                            f"batch {batch_number}/{total_batches}",
                            flush=True,
                        )

                targets_array = np.concatenate(ep_targets)
                predictions_array = np.concatenate(ep_predictions)
                mode_targets[mode].append(targets_array)
                mode_predictions[mode].append(predictions_array)
                ep_summary = _summarize(targets_array, predictions_array)
                episodes[mode].append(
                    {
                        "episode": str(episode_path.resolve()),
                        "sampled_frames": ep_summary["sampled_frames"],
                        "soft_cross_entropy": ep_summary["soft_cross_entropy"],
                        "argmax_accuracy": ep_summary["argmax_accuracy"],
                    }
                )
            del episode

    results = {}
    for mode in modes:
        summary = _summarize(
            np.concatenate(mode_targets[mode]), np.concatenate(mode_predictions[mode])
        )
        if mode_factor_predictions[mode]:
            summary["factor_metrics"] = _summarize_factors(
                np.concatenate(mode_factor_targets[mode]),
                np.concatenate(mode_factor_predictions[mode]),
            )
        if mode_sequence_predictions[mode]:
            summary["route_sequence_metrics"] = _summarize_route_sequence(
                np.concatenate(mode_sequence_targets[mode]),
                np.concatenate(mode_sequence_predictions[mode]),
            )
        summary["episodes"] = episodes[mode]
        results[mode] = summary
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "stride": args.stride,
        "batch_size": args.batch_size,
        "expert_names": EXPERT_NAMES,
        "route_sequence_source": (
            "direct_observation_physics_gate"
            if getattr(policy.config, "physics_gate_architecture", "legacy_softmax")
            == "physics_gate_v2"
            else None
        ),
        "temporal_route_action_source": (
            "normalized_dataset_demonstration_action_chunk"
            if getattr(policy.config, "physics_gate_architecture", "legacy_softmax")
            == "temporal_bcm_v2"
            else None
        ),
        "evaluation_seconds": time.perf_counter() - started,
        "modes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

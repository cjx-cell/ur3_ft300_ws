#!/usr/bin/env python3
"""Decode one Pi0.5 checkpoint across semantic phases of multiple raw episodes."""

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
GLOBAL_TASK = "pick up the peg and insert it into the hole"
OPEN_GRIPPER_PHASES = {
    "release the peg after verification",
    "retract and go back to home",
}


def _binarize_observed_gripper(values: np.ndarray) -> np.ndarray:
    """Match online inference, which only observes the physical joint value."""
    result = values.copy().astype(np.float32)
    result[..., 6] = (result[..., 6] > 0.12).astype(np.float32)
    return result


def _binarize_target_gripper(values: np.ndarray, tasks: np.ndarray) -> np.ndarray:
    """Match the v6->LeRobot action-label conversion contract exactly."""
    result = _binarize_observed_gripper(values)
    task_array = np.asarray([str(task) for task in np.atleast_1d(tasks)])
    if result.ndim == 1:
        if task_array.shape != (1,):
            raise ValueError(f"Expected one task for one action, got {tasks}")
        if task_array[0] in OPEN_GRIPPER_PHASES:
            result[6] = 0.0
    else:
        if len(task_array) != len(result):
            raise ValueError(f"Action/task length mismatch: {len(result)} != {len(task_array)}")
        result[np.isin(task_array, list(OPEN_GRIPPER_PHASES)), 6] = 0.0
    return result


def _evaluation_frames(
    tasks: np.ndarray,
    horizon: int,
    frame_stride: int | None,
) -> list[int]:
    """Return phase landmarks, optionally unioned with dense regular samples."""
    task_text = np.asarray([str(value) for value in tasks])
    frames = [0]
    if len(task_text) > 100 and task_text[100] == task_text[0]:
        frames.append(100)
    frames.extend(index for index in range(1, len(task_text)) if task_text[index] != task_text[index - 1])
    if frame_stride is not None:
        frames.extend(range(0, len(task_text) - horizon + 1, frame_stride))
    return sorted({frame for frame in frames if frame + horizon <= len(tasks)})


def _raw_observation(
    state: np.ndarray,
    camera0: np.ndarray,
    camera1: np.ndarray,
    force: np.ndarray,
    force_fast: np.ndarray,
    force_slow: np.ndarray,
    state_history: np.ndarray,
    task: str = GLOBAL_TASK,
) -> dict[str, object]:
    return {
        "observation.state": torch.from_numpy(state),
        # AddBatchDimensionProcessorStep handles standard Pi0.5 keys only;
        # auxiliary temporal observations carry their batch dimension here.
        "observation.force": torch.from_numpy(force).unsqueeze(0),
        "observation.force_fast": torch.from_numpy(force_fast).unsqueeze(0),
        "observation.force_slow": torch.from_numpy(force_slow).unsqueeze(0),
        "observation.state_history": torch.from_numpy(state_history).unsqueeze(0),
        "observation.images.camera0": torch.from_numpy(np.ascontiguousarray(camera0.transpose(2, 0, 1))),
        "observation.images.camera1": torch.from_numpy(np.ascontiguousarray(camera1.transpose(2, 0, 1))),
        "task": task,
    }


def _postprocess_chunk(chunk: torch.Tensor, postprocessor: object) -> torch.Tensor:
    return torch.stack(
        [postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])],
        dim=1,
    )


def _rate_limit_arm_chunk(
    prediction: np.ndarray,
    state: np.ndarray,
    max_step_rad: float,
) -> np.ndarray:
    """Apply the same causal absolute-joint rate limit used at execution."""
    limited = prediction.copy()
    previous = state[:6].copy()
    for index in range(len(limited)):
        limited[index, :6] = previous + np.clip(
            limited[index, :6] - previous,
            -max_step_rad,
            max_step_rad,
        )
        previous = limited[index, :6]
    return limited


def _scale_physical_arm_residual(
    prediction: np.ndarray,
    state: np.ndarray,
    residual_scale: float,
) -> np.ndarray:
    """Scale absolute arm commands around the measured joint state in radians."""
    scaled = prediction.copy()
    scaled[:, :6] = state[None, :6] + residual_scale * (prediction[:, :6] - state[None, :6])
    return scaled


def _evaluate_frame(
    *,
    policy: object,
    preprocessor: object,
    postprocessor: object,
    episode: dict[str, np.ndarray],
    frame: int,
    seed: int,
    num_seeds: int,
    execution_horizon: int,
    physical_arm_residual_scale: float,
    use_frame_task: bool,
) -> dict[str, object]:
    horizon = int(policy.config.chunk_size)
    state = _binarize_observed_gripper(episode["state"][frame])
    target = _binarize_target_gripper(
        episode["action"][frame : frame + horizon],
        episode["task"][frame : frame + horizon],
    )
    batch = preprocessor(
        _raw_observation(
            state,
            episode["camera0"][frame].astype(np.float32),
            episode["camera1"][frame].astype(np.float32),
            episode["force"][frame].astype(np.float32),
            episode["force_fast"][frame].astype(np.float32),
            episode["force_slow"][frame].astype(np.float32),
            episode["state_history"][frame].astype(np.float32),
            str(episode["task"][frame]) if use_frame_task else GLOBAL_TASK,
        )
    )

    predictions = []
    inference_ms = []
    for sample_seed in range(seed, seed + num_seeds):
        torch.manual_seed(sample_seed)
        started = time.perf_counter()
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(batch)
            prediction = _postprocess_chunk(normalized, postprocessor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms.append((time.perf_counter() - started) * 1000.0)
        physical_prediction = prediction[0].float().cpu().numpy()
        predictions.append(
            _scale_physical_arm_residual(
                physical_prediction,
                state,
                physical_arm_residual_scale,
            )
        )

    seed_metrics = []
    for sample_seed, prediction, duration_ms in zip(
        range(seed, seed + num_seeds), predictions, inference_ms, strict=True
    ):
        error = prediction[:, :6] - target[:, :6]
        deltas = np.diff(
            np.concatenate([state[None, :6], prediction[:, :6]], axis=0),
            axis=0,
        )
        seed_metrics.append(
            {
                "seed": sample_seed,
                "inference_ms": duration_ms,
                "executed_arm_mae_rad": float(np.abs(error[:execution_horizon]).mean()),
                "executed_endpoint_arm_mae_rad": float(
                    np.abs(prediction[execution_horizon - 1, :6] - target[execution_horizon - 1, :6]).mean()
                ),
                "full_chunk_arm_mae_rad": float(np.abs(error).mean()),
                "max_predicted_step_rad": float(np.abs(deltas).max()),
                "max_executed_predicted_step_rad": float(np.abs(deltas[:execution_horizon]).max()),
                "executed_gripper_accuracy": float(
                    (
                        (prediction[:execution_horizon, 6] >= 0.5) == (target[:execution_horizon, 6] >= 0.5)
                    ).mean()
                ),
                "full_chunk_gripper_accuracy": float(
                    ((prediction[:, 6] >= 0.5) == (target[:, 6] >= 0.5)).mean()
                ),
            }
        )

    prediction_array = np.stack(predictions)
    ensemble_prediction = prediction_array.mean(axis=0)
    ensemble_error = ensemble_prediction[:, :6] - target[:, :6]
    ensemble_deltas = np.diff(
        np.concatenate([state[None, :6], ensemble_prediction[:, :6]], axis=0),
        axis=0,
    )
    rate_limited_prediction = _rate_limit_arm_chunk(ensemble_prediction, state, max_step_rad=0.12)
    rate_limited_error = rate_limited_prediction[:, :6] - target[:, :6]
    rate_limited_deltas = np.diff(
        np.concatenate([state[None, :6], rate_limited_prediction[:, :6]], axis=0),
        axis=0,
    )
    return {
        "frame": frame,
        "phase": str(episode["task"][frame]),
        "mean_executed_arm_mae_rad": float(np.mean([item["executed_arm_mae_rad"] for item in seed_metrics])),
        "mean_executed_endpoint_arm_mae_rad": float(
            np.mean([item["executed_endpoint_arm_mae_rad"] for item in seed_metrics])
        ),
        "mean_max_predicted_step_rad": float(
            np.mean([item["max_predicted_step_rad"] for item in seed_metrics])
        ),
        "mean_max_executed_predicted_step_rad": float(
            np.mean([item["max_executed_predicted_step_rad"] for item in seed_metrics])
        ),
        "mean_gripper_accuracy": float(np.mean([item["executed_gripper_accuracy"] for item in seed_metrics])),
        "mean_full_chunk_gripper_accuracy": float(
            np.mean([item["full_chunk_gripper_accuracy"] for item in seed_metrics])
        ),
        "prediction_std_mean": float(prediction_array.std(axis=0).mean()),
        "ensemble_executed_arm_mae_rad": float(np.abs(ensemble_error[:execution_horizon]).mean()),
        "ensemble_executed_endpoint_arm_mae_rad": float(
            np.abs(ensemble_prediction[execution_horizon - 1, :6] - target[execution_horizon - 1, :6]).mean()
        ),
        "ensemble_max_executed_predicted_step_rad": float(np.abs(ensemble_deltas[:execution_horizon]).max()),
        "ensemble_gripper_accuracy": float(
            (
                (ensemble_prediction[:execution_horizon, 6] >= 0.5) == (target[:execution_horizon, 6] >= 0.5)
            ).mean()
        ),
        "ensemble_rate_limited_executed_arm_mae_rad": float(
            np.abs(rate_limited_error[:execution_horizon]).mean()
        ),
        "ensemble_rate_limited_executed_endpoint_arm_mae_rad": float(
            np.abs(
                rate_limited_prediction[execution_horizon - 1, :6] - target[execution_horizon - 1, :6]
            ).mean()
        ),
        "ensemble_rate_limited_max_executed_step_rad": float(
            np.abs(rate_limited_deltas[:execution_horizon]).max()
        ),
        "executed_target_arm": target[:execution_horizon, :6].tolist(),
        "executed_ensemble_arm": ensemble_prediction[:execution_horizon, :6].tolist(),
        "executed_rate_limited_ensemble_arm": rate_limited_prediction[:execution_horizon, :6].tolist(),
        "executed_target_gripper": target[:execution_horizon, 6].tolist(),
        "executed_ensemble_gripper": ensemble_prediction[:execution_horizon, 6].tolist(),
        "executed_gripper_by_seed": prediction_array[:, :execution_horizon, 6].tolist(),
        "seed_metrics": seed_metrics,
    }


def _phase_metrics(frame_results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Aggregate repeated representative frames without hiding phase identity."""
    phases: dict[str, list[dict[str, object]]] = {}
    for result in frame_results:
        phases.setdefault(str(result["phase"]), []).append(result)
    return {
        phase: {
            "representative_frame_count": len(results),
            "mean_executed_arm_mae_rad": float(
                np.mean([float(result["mean_executed_arm_mae_rad"]) for result in results])
            ),
            "worst_executed_arm_mae_rad": float(
                np.max([float(result["mean_executed_arm_mae_rad"]) for result in results])
            ),
        }
        for phase, results in phases.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help=(
            "Also evaluate every Nth frame with a complete action horizon; phase starts are always retained."
        ),
    )
    parser.add_argument("--expected-phase-count", type=int, default=8)
    parser.add_argument(
        "--use-frame-task",
        action="store_true",
        help="Prompt the policy with each evaluated frame's semantic task.",
    )
    parser.add_argument(
        "--arm-residual-scale",
        type=float,
        default=None,
        help="Override deterministic arm servo residual gain for calibration.",
    )
    parser.add_argument(
        "--physical-arm-residual-scale",
        type=float,
        default=1.0,
        help=(
            "Scale decoded absolute arm commands around the measured physical joint "
            "state before rate limiting. This is the deployable servo gain in radians."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be positive")
    if args.frame_stride is not None and args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    if args.arm_residual_scale is not None and not 0 < args.arm_residual_scale <= 1:
        raise ValueError("--arm-residual-scale must be in (0, 1]")
    if not 0 < args.physical_arm_residual_scale <= 1:
        raise ValueError("--physical-arm-residual-scale must be in (0, 1]")

    episodes = []
    required = (
        "state",
        "action",
        "camera0",
        "camera1",
        "task",
        "force",
        "force_fast",
        "force_slow",
        "state_history",
    )
    for episode_path in args.episode_npz:
        with np.load(episode_path, allow_pickle=True) as archive:
            episodes.append(
                (
                    episode_path,
                    {key: archive[key] for key in required},
                )
            )

    sys.path.insert(0, str(LEROBOT_SRC))
    import lerobot.policies.pi05.processor_pi05  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    load_started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PI05Policy.from_pretrained(str(args.checkpoint.resolve()), strict=False)
    if args.arm_residual_scale is not None:
        if policy.model.arm_head is None:
            raise ValueError("--arm-residual-scale requires a deterministic arm head")
        policy.config.arm_head_residual_scale = args.arm_residual_scale
        policy.model.arm_head.residual_scale = args.arm_residual_scale
    policy.to(device=device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint.resolve()),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    horizon = int(policy.config.chunk_size)
    execution_horizon = min(int(policy.config.n_action_steps), horizon)
    episode_results = []
    all_frames = []
    for episode_path, episode in episodes:
        frame_results = []
        for frame in _evaluation_frames(episode["task"], horizon, args.frame_stride):
            result = _evaluate_frame(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                episode=episode,
                frame=frame,
                seed=args.seed,
                num_seeds=args.num_seeds,
                execution_horizon=execution_horizon,
                physical_arm_residual_scale=args.physical_arm_residual_scale,
                use_frame_task=args.use_frame_task,
            )
            frame_results.append(result)
            all_frames.append(result)
            print(
                f"{episode_path.name}: frame={frame} phase={result['phase']} "
                f"executed_mae={result['mean_executed_arm_mae_rad']:.5f}",
                flush=True,
            )
        if not frame_results:
            raise ValueError(f"No complete action horizon in {episode_path}")
        episode_results.append(
            {
                "episode": str(episode_path.resolve()),
                "mean_executed_arm_mae_rad": float(
                    np.mean([item["mean_executed_arm_mae_rad"] for item in frame_results])
                ),
                "worst_executed_arm_mae_rad": float(
                    np.max([item["mean_executed_arm_mae_rad"] for item in frame_results])
                ),
                "max_executed_predicted_step_rad": float(
                    np.max([item["mean_max_executed_predicted_step_rad"] for item in frame_results])
                ),
                "mean_gripper_accuracy": float(
                    np.mean([item["mean_gripper_accuracy"] for item in frame_results])
                ),
                "mean_full_chunk_gripper_accuracy": float(
                    np.mean([item["mean_full_chunk_gripper_accuracy"] for item in frame_results])
                ),
                "mean_ensemble_executed_arm_mae_rad": float(
                    np.mean([item["ensemble_executed_arm_mae_rad"] for item in frame_results])
                ),
                "mean_ensemble_rate_limited_executed_arm_mae_rad": float(
                    np.mean([item["ensemble_rate_limited_executed_arm_mae_rad"] for item in frame_results])
                ),
                "max_ensemble_rate_limited_executed_step_rad": float(
                    np.max([item["ensemble_rate_limited_max_executed_step_rad"] for item in frame_results])
                ),
                "mean_ensemble_gripper_accuracy": float(
                    np.mean([item["ensemble_gripper_accuracy"] for item in frame_results])
                ),
                "phase_metrics": _phase_metrics(frame_results),
                "frames": frame_results,
            }
        )

    observed_phases = sorted({phase for episode in episode_results for phase in episode["phase_metrics"]})
    if len(observed_phases) != args.expected_phase_count:
        raise ValueError(
            "Semantic phase coverage mismatch: "
            f"expected={args.expected_phase_count}, "
            f"observed={len(observed_phases)} {observed_phases}"
        )

    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "frame_stride": args.frame_stride,
        "use_frame_task": args.use_frame_task,
        "arm_residual_scale": (
            args.arm_residual_scale
            if args.arm_residual_scale is not None
            else policy.config.arm_head_residual_scale
        ),
        "physical_arm_residual_scale": args.physical_arm_residual_scale,
        "execution_horizon": execution_horizon,
        "device": str(device),
        "load_seconds": load_seconds,
        "gripper_contract": {
            "observation": "physical_joint_gt_0.12",
            "target": "physical_joint_gt_0.12_with_release_retract_forced_open",
        },
        "observed_phases": observed_phases,
        "aggregate_mean_executed_arm_mae_rad": float(
            np.mean([item["mean_executed_arm_mae_rad"] for item in all_frames])
        ),
        "aggregate_worst_executed_arm_mae_rad": float(
            np.max([item["mean_executed_arm_mae_rad"] for item in all_frames])
        ),
        "aggregate_mean_prediction_std": float(np.mean([item["prediction_std_mean"] for item in all_frames])),
        "aggregate_max_executed_predicted_step_rad": float(
            np.max([item["mean_max_executed_predicted_step_rad"] for item in all_frames])
        ),
        "aggregate_mean_gripper_accuracy": float(
            np.mean([item["mean_gripper_accuracy"] for item in all_frames])
        ),
        "aggregate_mean_full_chunk_gripper_accuracy": float(
            np.mean([item["mean_full_chunk_gripper_accuracy"] for item in all_frames])
        ),
        "aggregate_mean_ensemble_executed_arm_mae_rad": float(
            np.mean([item["ensemble_executed_arm_mae_rad"] for item in all_frames])
        ),
        "aggregate_mean_ensemble_rate_limited_executed_arm_mae_rad": float(
            np.mean([item["ensemble_rate_limited_executed_arm_mae_rad"] for item in all_frames])
        ),
        "aggregate_max_ensemble_rate_limited_executed_step_rad": float(
            np.max([item["ensemble_rate_limited_max_executed_step_rad"] for item in all_frames])
        ),
        "aggregate_mean_ensemble_gripper_accuracy": float(
            np.mean([item["ensemble_gripper_accuracy"] for item in all_frames])
        ),
        "episodes": episode_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, args.output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

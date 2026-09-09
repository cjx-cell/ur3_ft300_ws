#!/usr/bin/env python3
"""Compare a Pi0.5 action chunk with a recorded peg-in-hole trajectory."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
DEFAULT_TASK = "pick up the peg and insert it into the hole"
OPEN_GRIPPER_PHASES = {
    "release the peg after verification",
    "retract and go back to home",
}


def _binarize_observed_gripper(values: np.ndarray) -> np.ndarray:
    result = values.copy().astype(np.float32)
    result[..., 6] = (result[..., 6] > 0.12).astype(np.float32)
    return result


def _binarize_target_gripper(
    values: np.ndarray, tasks: np.ndarray
) -> np.ndarray:
    result = _binarize_observed_gripper(values)
    task_array = np.asarray([str(task) for task in np.atleast_1d(tasks)])
    if len(task_array) != len(result):
        raise ValueError(
            f"Action/task length mismatch: {len(result)} != {len(task_array)}"
        )
    result[np.isin(task_array, list(OPEN_GRIPPER_PHASES)), 6] = 0.0
    return result


def _raw_observation(
    state: np.ndarray,
    camera0: np.ndarray,
    camera1: np.ndarray,
    task: str,
) -> dict[str, object]:
    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.camera0": torch.from_numpy(
            np.ascontiguousarray(camera0.transpose(2, 0, 1))
        ),
        "observation.images.camera1": torch.from_numpy(
            np.ascontiguousarray(camera1.transpose(2, 0, 1))
        ),
        "task": task,
    }


def _postprocess_chunk(chunk: torch.Tensor, postprocessor: object) -> torch.Tensor:
    return torch.stack(
        [postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])],
        dim=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Read state/action and a single decoded video frame from a LeRobot dataset.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        help="Episode index used with --dataset-root.",
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--legacy-binary-gripper",
        action="store_true",
        help="Only for pre-v7 datasets whose gripper contract was binary.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Evaluate consecutive seeds in one model-loading process.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the checkpoint's flow-matching integration step count.",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/pi05_offline_eval.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401

    frame = args.frame
    horizon = 50

    load_start = time.perf_counter()
    if args.dataset_root is not None:
        if args.episode_npz is not None:
            raise ValueError("Use either --episode-npz or --dataset-root, not both")
        if args.episode_index is None:
            raise ValueError("--episode-index is required with --dataset-root")

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id="pap_moe/ur3_peg_in_hole_overfit_5ep_relative_h50_global_task",
            root=args.dataset_root,
            episodes=[args.episode_index],
            # TorchCodec can leave decoder worker threads alive after a one-frame
            # CLI evaluation. TorchVision's PyAV backend exits cleanly.
            video_backend="pyav",
        )
        if frame < 0 or frame + horizon > len(dataset):
            raise ValueError(f"Frame {frame} has no complete {horizon}-step target horizon")

        sample = dataset[frame]
        state = sample["observation.state"].detach().cpu().numpy()
        ground_truth = np.stack(
            [
                dataset.reader.hf_dataset[frame + index]["action"]
                .detach()
                .cpu()
                .numpy()
                for index in range(horizon)
            ]
        ).astype(np.float32)
        if args.legacy_binary_gripper:
            state = _binarize_observed_gripper(state[None])[0]
            ground_truth = _binarize_observed_gripper(ground_truth)
        camera0 = (
            sample["observation.images.camera0"]
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            .astype(np.float32)
        )
        camera1 = (
            sample["observation.images.camera1"]
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            .astype(np.float32)
        )
        episode_source = f"{args.dataset_root.resolve()}#episode={args.episode_index}"
        del sample, dataset
        gc.collect()
    else:
        if args.episode_npz is None:
            raise ValueError("Either --episode-npz or --dataset-root is required")
        episode = np.load(args.episode_npz, allow_pickle=True)
        if frame < 0 or frame + horizon > len(episode["action"]):
            raise ValueError(f"Frame {frame} has no complete {horizon}-step target horizon")

        state = episode["state"][frame].astype(np.float32)
        ground_truth = episode["action"][frame : frame + horizon].astype(
            np.float32
        )
        if args.legacy_binary_gripper:
            state = _binarize_observed_gripper(state[None])[0]
            ground_truth = _binarize_target_gripper(
                ground_truth,
                episode["task"][frame : frame + horizon],
            )
        camera0 = episode["camera0"][frame].astype(np.float32)
        camera1 = episode["camera1"][frame].astype(np.float32)
        episode_source = str(args.episode_npz.resolve())
        del episode
        gc.collect()

    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter_config_path = args.checkpoint / "adapter_config.json"
    if adapter_config_path.is_file():
        from peft import PeftModel

        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_path = Path(adapter_config["base_model_name_or_path"])
        policy_config = PreTrainedConfig.from_pretrained(str(args.checkpoint))
        policy = PI05Policy.from_pretrained(
            str(base_model_path), config=policy_config, strict=True
        )
        policy = PeftModel.from_pretrained(
            policy, str(args.checkpoint), is_trainable=False
        )
    else:
        policy = PI05Policy.from_pretrained(str(args.checkpoint), strict=True)
    policy.to(device=device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint),
    )
    load_s = time.perf_counter() - load_start

    batch = preprocessor(_raw_observation(state, camera0, camera1, DEFAULT_TASK))
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1")
    predicted_samples = []
    inference_times = []
    predict_kwargs = {}
    if args.num_inference_steps is not None:
        predict_kwargs["num_steps"] = args.num_inference_steps
    for sample_seed in range(args.seed, args.seed + args.num_seeds):
        torch.manual_seed(sample_seed)
        infer_start = time.perf_counter()
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(batch, **predict_kwargs)
            predicted = _postprocess_chunk(normalized, postprocessor)
        inference_times.append(time.perf_counter() - infer_start)
        predicted_samples.append(predicted[0].float().cpu().numpy())
    predicted_np = predicted_samples[0]
    inference_s = inference_times[0]
    execution_horizon = min(
        int(getattr(policy.config, "n_action_steps", horizon)),
        horizon,
    )

    arm_error = predicted_np[:, :6] - ground_truth[:, :6]
    predicted_deltas = np.diff(
        np.concatenate([state[None, :6], predicted_np[:, :6]], axis=0), axis=0
    )
    target_deltas = np.diff(
        np.concatenate([state[None, :6], ground_truth[:, :6]], axis=0), axis=0
    )
    gripper_error = predicted_np[:, 6] - ground_truth[:, 6]
    endpoint_mask = np.logical_or(
        ground_truth[:, 6] <= 1.0e-3,
        ground_truth[:, 6] >= 0.8 - 1.0e-3,
    )

    seed_metrics = []
    for sample_index, sample_prediction in enumerate(predicted_samples):
        sample_error = sample_prediction[:, :6] - ground_truth[:, :6]
        sample_deltas = np.diff(
            np.concatenate([state[None, :6], sample_prediction[:, :6]], axis=0),
            axis=0,
        )
        seed_metrics.append(
            {
                "seed": args.seed + sample_index,
                "inference_s": inference_times[sample_index],
                "arm_mae_rad": float(np.mean(np.abs(sample_error))),
                "endpoint_arm_mae_rad": float(
                    np.mean(
                        np.abs(
                            sample_prediction[-1, :6] - ground_truth[-1, :6]
                        )
                    )
                ),
                "executed_arm_mae_rad": float(
                    np.mean(np.abs(sample_error[:execution_horizon]))
                ),
                "executed_endpoint_arm_mae_rad": float(
                    np.mean(
                        np.abs(
                            sample_prediction[execution_horizon - 1, :6]
                            - ground_truth[execution_horizon - 1, :6]
                        )
                    )
                ),
                "first_action_arm_mae_rad": float(
                    np.mean(np.abs(sample_prediction[0, :6] - ground_truth[0, :6]))
                ),
                "max_predicted_step_rad": float(np.max(np.abs(sample_deltas))),
                "gripper_predicted_min_rad": float(np.min(sample_prediction[:, 6])),
                "gripper_predicted_max_rad": float(np.max(sample_prediction[:, 6])),
                "predicted_last_joint_1": float(sample_prediction[-1, 0]),
                "predicted_last_joint_6": float(sample_prediction[-1, 5]),
            }
        )
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "episode_npz": episode_source,
        "frame": frame,
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "num_inference_steps": (
            args.num_inference_steps
            if args.num_inference_steps is not None
            else policy.config.num_inference_steps
        ),
        "execution_horizon": execution_horizon,
        "device": str(device),
        "load_s": load_s,
        "inference_s": inference_s,
        "arm_mae_rad": float(np.mean(np.abs(arm_error))),
        "arm_rmse_rad": float(np.sqrt(np.mean(np.square(arm_error)))),
        "first_action_arm_mae_rad": float(
            np.mean(np.abs(predicted_np[0, :6] - ground_truth[0, :6]))
        ),
        "endpoint_arm_mae_rad": float(
            np.mean(np.abs(predicted_np[-1, :6] - ground_truth[-1, :6]))
        ),
        "executed_arm_mae_rad": float(
            np.mean(np.abs(arm_error[:execution_horizon]))
        ),
        "executed_endpoint_arm_mae_rad": float(
            np.mean(
                np.abs(
                    predicted_np[execution_horizon - 1, :6]
                    - ground_truth[execution_horizon - 1, :6]
                )
            )
        ),
        "delta_arm_mae_rad": float(
            np.mean(np.abs(predicted_deltas - target_deltas))
        ),
        "max_predicted_step_rad": float(np.max(np.abs(predicted_deltas))),
        "max_target_step_rad": float(np.max(np.abs(target_deltas))),
        "gripper_mae_rad": float(np.mean(np.abs(gripper_error))),
        "gripper_rmse_rad": float(np.sqrt(np.mean(np.square(gripper_error)))),
        "gripper_predicted_min_rad": float(np.min(predicted_np[:, 6])),
        "gripper_predicted_max_rad": float(np.max(predicted_np[:, 6])),
        "gripper_endpoint_accuracy": (
            float(
                np.mean(
                    (predicted_np[endpoint_mask, 6] >= 0.4)
                    == (ground_truth[endpoint_mask, 6] >= 0.4)
                )
            )
            if np.any(endpoint_mask)
            else None
        ),
        "state": state.tolist(),
        "predicted_first": predicted_np[0].tolist(),
        "target_first": ground_truth[0].tolist(),
        "predicted_last": predicted_np[-1].tolist(),
        "target_last": ground_truth[-1].tolist(),
        "seed_metrics": seed_metrics,
        "seed_arm_mae_mean_rad": float(
            np.mean([item["arm_mae_rad"] for item in seed_metrics])
        ),
        "seed_arm_mae_std_rad": float(
            np.std([item["arm_mae_rad"] for item in seed_metrics])
        ),
        "seed_endpoint_mae_mean_rad": float(
            np.mean([item["endpoint_arm_mae_rad"] for item in seed_metrics])
        ),
        "seed_executed_mae_mean_rad": float(
            np.mean([item["executed_arm_mae_rad"] for item in seed_metrics])
        ),
        "seed_executed_endpoint_mae_mean_rad": float(
            np.mean(
                [item["executed_endpoint_arm_mae_rad"] for item in seed_metrics]
            )
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        state=state,
        predicted=predicted_np,
        target=ground_truth,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rank Pi0.5 PEFT checkpoints on identical recorded action chunks.

The large Pi0.5 base model is loaded once.  All LoRA adapters are then attached
to that base and selected in turn, avoiding six expensive base-model reloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
DEFAULT_TASK = "pick up the peg and insert it into the hole"


def _adapter_name(checkpoint: Path) -> str:
    """Return a PEFT-safe name unique across different training runs."""
    run_name = checkpoint.parents[2].name
    step_name = checkpoint.parent.name
    return f"{run_name}__{step_name}".replace("-", "_").replace(".", "_")


def _raw_observation(
    state: np.ndarray, camera0: np.ndarray, camera1: np.ndarray
) -> dict[str, object]:
    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.camera0": torch.from_numpy(
            np.ascontiguousarray(camera0.transpose(2, 0, 1))
        ),
        "observation.images.camera1": torch.from_numpy(
            np.ascontiguousarray(camera1.transpose(2, 0, 1))
        ),
        "task": DEFAULT_TASK,
    }


def _postprocess_chunk(chunk: torch.Tensor, postprocessor: object) -> torch.Tensor:
    return torch.stack(
        [postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])],
        dim=1,
    )


def _load_eval_samples(dataset_root: Path, episode_index: int, frames: list[int]):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id="pap_moe/ur3_peg_in_hole_exactfit_pilot30",
        root=dataset_root,
        episodes=[episode_index],
        video_backend="pyav",
    )
    horizon = 50
    samples = []
    for frame in frames:
        if frame < 0 or frame + horizon > len(dataset):
            raise ValueError(
                f"Frame {frame} has no complete {horizon}-step horizon; "
                f"selected episode has {len(dataset)} frames"
            )
        sample = dataset[frame]
        state = sample["observation.state"].detach().cpu().numpy().astype(np.float32)
        target = np.stack(
            [
                dataset.reader.hf_dataset[frame + offset]["action"]
                .detach()
                .cpu()
                .numpy()
                for offset in range(horizon)
            ]
        ).astype(np.float32)
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
        samples.append((frame, state, camera0, camera1, target))
    return samples


def _metrics(
    predicted: np.ndarray, target: np.ndarray, state: np.ndarray, execution_horizon: int
) -> dict:
    arm_error = predicted[:, :6] - target[:, :6]
    predicted_delta = np.diff(
        np.concatenate([state[None, :6], predicted[:, :6]], axis=0), axis=0
    )
    endpoint_mask = np.logical_or(target[:, 6] <= 1.0e-3, target[:, 6] >= 0.799)
    result = {
        "arm_mae_rad": float(np.mean(np.abs(arm_error))),
        "executed_arm_mae_rad": float(
            np.mean(np.abs(arm_error[:execution_horizon]))
        ),
        "first_arm_mae_rad": float(np.mean(np.abs(arm_error[0]))),
        "endpoint_arm_mae_rad": float(np.mean(np.abs(arm_error[-1]))),
        "max_predicted_step_rad": float(np.max(np.abs(predicted_delta))),
        "executed_max_predicted_step_rad": float(
            np.max(np.abs(predicted_delta[:execution_horizon]))
        ),
        "gripper_mae_rad": float(np.mean(np.abs(predicted[:, 6] - target[:, 6]))),
        "gripper_predicted_min_rad": float(np.min(predicted[:, 6])),
        "gripper_predicted_max_rad": float(np.max(predicted[:, 6])),
        "gripper_endpoint_accuracy": None,
    }
    if np.any(endpoint_mask):
        result["gripper_endpoint_accuracy"] = float(
            np.mean(
                (predicted[endpoint_mask, 6] >= 0.4)
                == (target[endpoint_mask, 6] >= 0.4)
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        default=[],
        help="Additional explicit PEFT checkpoint to compare; may be repeated.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--frames", type=int, nargs="+", default=[0, 80, 115, 130, 170, 210, 230]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1")
    checkpoints = sorted(
        path / "pretrained_model"
        for path in args.checkpoints_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]")
        if (path / "pretrained_model" / "adapter_model.safetensors").is_file()
    )
    checkpoints.extend(path.resolve() for path in args.checkpoint)
    checkpoints = list(dict.fromkeys(checkpoints))
    if not checkpoints:
        raise FileNotFoundError(f"No numeric PEFT checkpoints under {args.checkpoints_root}")
    if args.execution_horizon < 1 or args.execution_horizon > 50:
        raise ValueError("--execution-horizon must be within [1, 50]")
    missing = [path for path in checkpoints if not (path / "adapter_model.safetensors").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PEFT adapter checkpoints: {missing}")

    sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401
    from peft import PeftModel

    samples = _load_eval_samples(args.dataset_root, args.episode_index, args.frames)
    first_checkpoint = checkpoints[0]
    adapter_config = json.loads(
        (first_checkpoint / "adapter_config.json").read_text(encoding="utf-8")
    )
    base_model_path = Path(adapter_config["base_model_name_or_path"])
    policy_config = PreTrainedConfig.from_pretrained(str(first_checkpoint))

    load_start = time.perf_counter()
    base_policy = PI05Policy.from_pretrained(
        str(base_model_path), config=policy_config, strict=True
    )
    # PEFT normalizes ``modules_to_save`` tensor keys to the built-in
    # ``default`` adapter name when serializing. Loading a single hybrid
    # checkpoint under a renamed adapter silently misses the fully-trained
    # action expert/projection tensors. Keep ``default`` for the single-model
    # audit; custom names remain useful for ordinary multi-LoRA comparison.
    single_checkpoint = len(checkpoints) == 1
    first_name = "default" if single_checkpoint else _adapter_name(checkpoints[0])
    policy = PeftModel.from_pretrained(
        base_policy,
        str(first_checkpoint),
        adapter_name=first_name,
        is_trainable=False,
    )
    for checkpoint in checkpoints[1:]:
        policy.load_adapter(str(checkpoint), adapter_name=_adapter_name(checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device=device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config, pretrained_path=str(first_checkpoint)
    )
    load_s = time.perf_counter() - load_start

    details = []
    summaries = []
    for checkpoint in checkpoints:
        adapter_name = "default" if single_checkpoint else _adapter_name(checkpoint)
        policy.set_adapter(adapter_name)
        adapter_rows = []
        for frame, state, camera0, camera1, target in samples:
            batch = preprocessor(_raw_observation(state, camera0, camera1))
            for seed in range(args.seed, args.seed + args.num_seeds):
                torch.manual_seed(seed)
                infer_start = time.perf_counter()
                with torch.inference_mode():
                    normalized = policy.predict_action_chunk(batch)
                    predicted = _postprocess_chunk(normalized, postprocessor)
                row = {
                    "checkpoint": adapter_name,
                    "frame": frame,
                    "seed": seed,
                    "inference_s": time.perf_counter() - infer_start,
                    **_metrics(
                        predicted[0].float().cpu().numpy(), target, state,
                        args.execution_horizon,
                    ),
                }
                details.append(row)
                adapter_rows.append(row)
        summaries.append(
            {
                "checkpoint": adapter_name,
                "arm_mae_rad": float(np.mean([r["arm_mae_rad"] for r in adapter_rows])),
                "executed_arm_mae_rad": float(
                    np.mean([r["executed_arm_mae_rad"] for r in adapter_rows])
                ),
                "first_arm_mae_rad": float(
                    np.mean([r["first_arm_mae_rad"] for r in adapter_rows])
                ),
                "endpoint_arm_mae_rad": float(
                    np.mean([r["endpoint_arm_mae_rad"] for r in adapter_rows])
                ),
                "gripper_mae_rad": float(
                    np.mean([r["gripper_mae_rad"] for r in adapter_rows])
                ),
                "gripper_endpoint_accuracy": float(
                    np.mean(
                        [
                            r["gripper_endpoint_accuracy"]
                            for r in adapter_rows
                            if r["gripper_endpoint_accuracy"] is not None
                        ]
                    )
                ),
                "max_predicted_step_rad": float(
                    np.max([r["max_predicted_step_rad"] for r in adapter_rows])
                ),
                "executed_max_predicted_step_rad": float(
                    np.max([r["executed_max_predicted_step_rad"] for r in adapter_rows])
                ),
                "mean_inference_s": float(
                    np.mean([r["inference_s"] for r in adapter_rows])
                ),
            }
        )

    summaries.sort(key=lambda item: (item["arm_mae_rad"], item["gripper_mae_rad"]))
    output = {
        "dataset_root": str(args.dataset_root.resolve()),
        "episode_index": args.episode_index,
        "frames": args.frames,
        "seeds": list(range(args.seed, args.seed + args.num_seeds)),
        "execution_horizon": args.execution_horizon,
        "device": str(device),
        "base_model": str(base_model_path),
        "load_s": load_s,
        "ranking": summaries,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"load_s": load_s, "ranking": summaries}, indent=2))


if __name__ == "__main__":
    main()

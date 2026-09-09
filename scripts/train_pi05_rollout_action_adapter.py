#!/usr/bin/env python3
"""Fit a tiny Pi0.5 action-output adapter on rollout corrections with anchors."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPT_DIR = Path(__file__).resolve().parent
TASK = "pick up the peg and insert it into the hole"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--episode-npz", type=Path, required=True)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--execution-horizon", type=int, default=20)
    parser.add_argument("--lora-r", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(0, str(SCRIPT_DIR))
    from eval_pi05_offline_action_chunk import (
        _binarize_observed_gripper,
        _binarize_target_gripper,
        _postprocess_chunk,
        _raw_observation,
    )
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import ACTION
    import lerobot.policies.pi05.processor_pi05  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PI05Policy.from_pretrained(str(args.checkpoint), strict=False)
    policy.to(device=device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    if args.lora_r > 0:
        policy = policy.wrap_with_peft(
            peft_cli_overrides={
                "method_type": "lora",
                "r": args.lora_r,
                "lora_alpha": args.lora_r,
                "lora_dropout": 0.0,
            }
        )
        policy.to(device=device)
        trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        trainable_module_name = f"pi05_default_action_lora_rank_{args.lora_r}"
    else:
        for parameter in policy.parameters():
            parameter.requires_grad = False
        output_projection = policy.model.action_out_proj
        for parameter in output_projection.parameters():
            parameter.requires_grad = True
        trainable_parameters = list(output_projection.parameters())
        trainable_module_name = "model.action_out_proj"

    with np.load(args.corrections) as archive:
        corrections = {key: archive[key] for key in archive.files}
    with np.load(args.episode_npz, allow_pickle=True) as archive:
        demo = {
            key: archive[key]
            for key in ("state", "action", "camera0", "camera1", "task")
        }

    def processed(state, camera0, camera1, target):
        raw = _raw_observation(state.astype(np.float32), camera0.astype(np.float32),
                               camera1.astype(np.float32), TASK)
        raw[ACTION] = torch.from_numpy(target.astype(np.float32)).unsqueeze(0)
        return preprocessor(raw)

    replay_batches = [
        processed(
            corrections["state"][index],
            corrections["camera0"][index],
            corrections["camera1"][index],
            corrections["target_action"][index],
        )
        for index in range(len(corrections["state"]))
    ]

    task_text = np.asarray([str(value) for value in demo["task"]])
    phase_starts = [0] + (np.flatnonzero(task_text[1:] != task_text[:-1]) + 1).tolist()
    anchor_frames = sorted(
        set(corrections["demo_frame"].astype(int).tolist())
        | set(phase_starts)
        | set(range(0, len(task_text) - policy.config.chunk_size + 1, 80))
    )
    anchor_frames = [
        frame for frame in anchor_frames
        if frame + policy.config.chunk_size <= len(task_text)
    ]
    anchor_targets = []
    anchor_batches = []
    for frame in anchor_frames:
        target = _binarize_target_gripper(
            demo["action"][frame : frame + policy.config.chunk_size],
            demo["task"][frame : frame + policy.config.chunk_size],
        ).astype(np.float32)
        anchor_targets.append(target)
        anchor_batches.append(
            processed(
                _binarize_observed_gripper(demo["state"][frame]),
                demo["camera0"][frame], demo["camera1"][frame], target,
            )
        )

    horizon = args.execution_horizon
    priority_indices = np.flatnonzero(corrections["alignment_l2_rad"] >= 0.25).tolist()
    replay_schedule = list(range(len(replay_batches)))
    for index in priority_indices:
        replay_schedule.extend([index] * 4)

    def evaluate(step: int) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        policy.eval()
        replay_errors = []
        replay_endpoint_errors = []
        replay_steps = []
        anchor_errors = []
        anchor_steps = []
        with torch.inference_mode():
            for index, batch in enumerate(replay_batches):
                torch.manual_seed(args.seed)
                prediction = _postprocess_chunk(
                    policy.predict_action_chunk(batch), postprocessor
                )[0].float().cpu().numpy()
                target = corrections["target_action"][index]
                replay_errors.append(float(np.abs(prediction[:horizon, :6] - target[:horizon, :6]).mean()))
                replay_endpoint_errors.append(float(np.abs(prediction[horizon - 1, :6] - target[horizon - 1, :6]).mean()))
                replay_steps.append(float(np.max(np.abs(np.diff(
                    np.concatenate([corrections["state"][index, None, :6], prediction[:horizon, :6]], axis=0),
                    axis=0,
                )))))
            for frame, target, batch in zip(anchor_frames, anchor_targets, anchor_batches, strict=True):
                torch.manual_seed(args.seed)
                prediction = _postprocess_chunk(
                    policy.predict_action_chunk(batch), postprocessor
                )[0].float().cpu().numpy()
                anchor_errors.append(float(np.abs(prediction[:horizon, :6] - target[:horizon, :6]).mean()))
                state = _binarize_observed_gripper(demo["state"][frame])
                anchor_steps.append(float(np.max(np.abs(np.diff(
                    np.concatenate([state[None, :6], prediction[:horizon, :6]], axis=0), axis=0
                )))))
        metrics = {
            "step": float(step),
            "replay_arm_mae_rad": float(np.mean(replay_errors)),
            "replay_worst_arm_mae_rad": float(np.max(replay_errors)),
            "replay_endpoint_arm_mae_rad": float(np.mean(replay_endpoint_errors)),
            "replay_max_step_rad": float(np.max(replay_steps)),
            "anchor_arm_mae_rad": float(np.mean(anchor_errors)),
            "anchor_worst_arm_mae_rad": float(np.max(anchor_errors)),
            "anchor_max_step_rad": float(np.max(anchor_steps)),
            "priority_replay_arm_mae_rad": float(
                np.mean([replay_errors[index] for index in priority_indices])
            ),
            "priority_replay_worst_arm_mae_rad": float(
                np.max([replay_errors[index] for index in priority_indices])
            ),
        }
        snapshot = {
            name: value.detach().cpu().clone()
            for name, value in policy.named_parameters()
            if value.requires_grad
        }
        return metrics, snapshot

    baseline, baseline_snapshot = evaluate(0)
    history = [baseline]
    candidates = [(baseline, baseline_snapshot)]
    print(f"step=0 {json.dumps(baseline)}", flush=True)

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.0)
    policy.train()
    for step in range(1, args.steps + 1):
        replay_index = replay_schedule[(step - 1) % len(replay_schedule)]
        matched_frame = int(corrections["demo_frame"][replay_index])
        matched_anchor_index = anchor_frames.index(matched_frame)
        global_anchor_index = (step * 7) % len(anchor_batches)

        torch.manual_seed(args.seed + step)
        replay_loss, _ = policy(replay_batches[replay_index])
        torch.manual_seed(args.seed + step)
        matched_anchor_loss, _ = policy(anchor_batches[matched_anchor_index])
        torch.manual_seed(args.seed + step + 100_003)
        global_anchor_loss, _ = policy(anchor_batches[global_anchor_index])
        loss = replay_loss + 2.0 * matched_anchor_loss + global_anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            metrics, snapshot = evaluate(step)
            metrics["training_loss"] = float(loss.detach().item())
            history.append(metrics)
            candidates.append((metrics, snapshot))
            print(f"step={step} {json.dumps(metrics)}", flush=True)
            policy.train()

    anchor_limit = max(baseline["anchor_arm_mae_rad"] * 1.10, baseline["anchor_arm_mae_rad"] + 0.005)
    qualified = [
        candidate for candidate in candidates[1:]
        if candidate[0]["anchor_arm_mae_rad"] <= anchor_limit
        and candidate[0]["replay_arm_mae_rad"] <= baseline["replay_arm_mae_rad"] * 0.90
        and candidate[0]["priority_replay_arm_mae_rad"]
        <= baseline["priority_replay_arm_mae_rad"] * 0.80
        and candidate[0]["replay_worst_arm_mae_rad"]
        <= baseline["replay_worst_arm_mae_rad"] * 1.05
        and candidate[0]["replay_max_step_rad"] <= baseline["replay_max_step_rad"] * 1.05
        and candidate[0]["anchor_max_step_rad"] <= baseline["anchor_max_step_rad"] * 1.05
    ]
    selected = min(qualified, key=lambda value: value[0]["replay_arm_mae_rad"]) if qualified else None
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "corrections": str(args.corrections.resolve()),
        "trainable_module": trainable_module_name,
        "trainable_parameters": int(sum(p.numel() for p in trainable_parameters)),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "anchor_frames": anchor_frames,
        "priority_replay_indices": priority_indices,
        "priority_replay_schedule_multiplier": 5,
        "baseline": baseline,
        "anchor_limit_rad": anchor_limit,
        "history": history,
        "qualified": bool(qualified),
        "selected": selected[0] if selected else None,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if selected is not None:
        parameter_by_name = dict(policy.named_parameters())
        for name, value in selected[1].items():
            parameter_by_name[name].data.copy_(value.to(parameter_by_name[name].device))
        if args.lora_r > 0:
            args.output_adapter.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(args.output_adapter)
        else:
            args.output_adapter.parent.mkdir(parents=True, exist_ok=True)
            save_file(selected[1], str(args.output_adapter), metadata={
                "source_checkpoint": str(args.checkpoint.resolve()),
                "selected_step": str(int(selected[0]["step"])),
            })
        print(f"QUALIFIED adapter saved to {args.output_adapter}", flush=True)
    else:
        print("REJECTED: no step passed replay/anchor/safety gates", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

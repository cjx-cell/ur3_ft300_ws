#!/usr/bin/env python3
"""Directly optimize a Pi0.5 sampled chunk on critical rollout states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPT_DIR = Path(__file__).resolve().parent
TASK = "grasp the peg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument(
        "--corrections",
        type=Path,
        nargs="+",
        required=True,
        help="One or more correction packs concatenated in the given order.",
    )
    parser.add_argument("--episode-npz", type=Path, required=True)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, default=None)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument(
        "--train-action-out-with-lora",
        action="store_true",
        help=(
            "Train and save a full adapter-local action_out_proj alongside LoRA. "
            "The projection is stored separately and must only be activated while "
            "that adapter is active."
        ),
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=20,
        help="Optimize and gate the prefix that deployment executes before replanning.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument(
        "--executed-gripper-weight",
        type=float,
        default=0.0,
        help="Extra gripper imitation weight inside the actually executed prefix.",
    )
    parser.add_argument("--anchor-weight", type=float, default=2.0)
    parser.add_argument("--arm-delta-weight", type=float, default=0.5)
    parser.add_argument("--endpoint-weight", type=float, default=0.0)
    parser.add_argument(
        "--first-action-weight",
        type=float,
        default=1.0,
        help="Extra imitation weight on the first executed arm action.",
    )
    parser.add_argument("--max-relative-step", type=float, default=1.05)
    parser.add_argument(
        "--max-mean-arm-mae-rad",
        type=float,
        default=None,
        help="Optional absolute mean-MAE gate for precision stages.",
    )
    parser.add_argument(
        "--max-worst-arm-mae-rad",
        type=float,
        default=None,
        help="Optional absolute worst-sample MAE gate for precision stages.",
    )
    parser.add_argument(
        "--max-critical-step-rad",
        type=float,
        default=None,
        help="Optional absolute raw predicted-step gate before deployment clamps.",
    )
    parser.add_argument(
        "--min-mean-relative-improvement",
        type=float,
        default=0.05,
        help="Required fractional improvement of mean MAE under --relative-gates.",
    )
    parser.add_argument(
        "--min-worst-relative-improvement",
        type=float,
        default=0.0,
        help="Required fractional improvement of worst-case MAE under --relative-gates.",
    )
    parser.add_argument("--max-anchor-drift", type=float, default=0.03)
    parser.add_argument("--close-sample-weight", type=int, default=1)
    parser.add_argument(
        "--require-future-close",
        action="store_true",
        help="Require a close in the full chunk even when it lies beyond the executed horizon.",
    )
    parser.add_argument(
        "--require-executed-close",
        action="store_true",
        help="Require recall of target close actions inside the executed prefix.",
    )
    parser.add_argument(
        "--relative-gates",
        action="store_true",
        help="Gate a broad roll-in pack by relative improvement from its own baseline.",
    )
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--indices", type=int, nargs="+", default=[2, 3])
    args = parser.parse_args()

    if args.train_action_out_with_lora and args.lora_r <= 0:
        parser.error("--train-action-out-with-lora requires --lora-r > 0")
    for name in (
        "max_mean_arm_mae_rad",
        "max_worst_arm_mae_rad",
        "max_critical_step_rad",
    ):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")

    torch.manual_seed(args.seed)
    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(0, str(SCRIPT_DIR))
    from eval_pi05_offline_action_chunk import (
        _binarize_observed_gripper,
        _binarize_target_gripper,
        _postprocess_chunk,
        _raw_observation,
    )
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    import lerobot.policies.pi05.processor_pi05  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_lora = args.initial_adapter is not None and args.initial_adapter.is_dir()
    # Match the deployment/offline-evaluation loading contract exactly: a PEFT
    # adapter owns the effective policy config, while its weights are applied to
    # the explicitly supplied frozen base checkpoint.
    policy_config = (
        PreTrainedConfig.from_pretrained(str(args.initial_adapter))
        if initial_lora
        else None
    )
    policy = PI05Policy.from_pretrained(
        str(args.checkpoint), config=policy_config, strict=policy_config is not None
    )
    policy.to(device=device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    if not 1 <= args.execution_horizon <= policy.config.chunk_size:
        raise ValueError("--execution-horizon must be within the policy chunk size")
    execution_horizon = args.execution_horizon
    endpoint_index = execution_horizon - 1
    dynamic_steps = 0
    for step in preprocessor.steps:
        if hasattr(step, "global_task"):
            step.global_task = None
            dynamic_steps += 1
    if dynamic_steps != 1:
        raise RuntimeError(f"Expected one dynamic-task processor step, found {dynamic_steps}")

    if args.initial_adapter is not None and not initial_lora:
        from safetensors.torch import load_file

        adapter = load_file(str(args.initial_adapter.resolve()), device="cpu")
        parameters = dict(policy.named_parameters())
        expected = {"model.action_out_proj.weight", "model.action_out_proj.bias"}
        if set(adapter) != expected:
            raise ValueError(f"Unexpected initial adapter keys: {sorted(adapter)}")
        with torch.no_grad():
            for name, value in adapter.items():
                if value.shape != parameters[name].shape:
                    raise ValueError(f"Initial adapter shape mismatch for {name}")
                parameters[name].copy_(value.to(dtype=parameters[name].dtype))

    if args.lora_r > 0:
        if initial_lora:
            from peft import PeftModel

            trainable_model = PeftModel.from_pretrained(
                policy, str(args.initial_adapter.resolve()), is_trainable=True
            )
            loaded_ranks = {
                int(config.r) for config in trainable_model.peft_config.values()
            }
            if loaded_ranks != {args.lora_r}:
                raise ValueError(
                    f"Initial LoRA rank {sorted(loaded_ranks)} does not match --lora-r {args.lora_r}"
                )
            for config in trainable_model.peft_config.values():
                config.base_model_name_or_path = str(args.checkpoint.resolve())
        else:
            # PEFT derives adapter provenance from ``name_or_path``.  The loaded
            # checkpoint config can retain the training run's older pretrained
            # path, so make the explicit CLI checkpoint authoritative.
            policy.name_or_path = str(args.checkpoint.resolve())
            policy.config.pretrained_path = str(args.checkpoint.resolve())
            peft_overrides = {
                "method_type": "lora",
                "r": args.lora_r,
                "lora_alpha": args.lora_r,
                "lora_dropout": 0.0,
            }
            trainable_model = policy.wrap_with_peft(peft_cli_overrides=peft_overrides)
        trainable_model.to(device=device)
        trainable_module = f"pi05_default_action_lora_rank_{args.lora_r}"
        if args.train_action_out_with_lora:
            action_out = trainable_model.base_model.model.model.action_out_proj
            action_out_base = getattr(action_out, "base_layer", action_out)
            for parameter in action_out_base.parameters():
                parameter.requires_grad_(True)
            if initial_lora:
                from safetensors.torch import load_file

                local_projection_path = args.initial_adapter / "adapter_local_action_out.safetensors"
                if not local_projection_path.is_file():
                    raise FileNotFoundError(
                        "Initial adapter is missing adapter-local action_out projection: "
                        f"{local_projection_path}"
                    )
                local_projection = load_file(str(local_projection_path), device="cpu")
                expected = {"weight", "bias"}
                if set(local_projection) != expected:
                    raise ValueError(
                        f"Unexpected adapter-local action_out keys: {sorted(local_projection)}"
                    )
                with torch.no_grad():
                    for name, value in local_projection.items():
                        parameter = getattr(action_out_base, name)
                        if parameter.shape != value.shape:
                            raise ValueError(
                                f"Adapter-local action_out shape mismatch for {name}: "
                                f"{value.shape} != {parameter.shape}"
                            )
                        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            trainable_module += "_plus_adapter_local_action_out_proj"
    else:
        for parameter in policy.parameters():
            parameter.requires_grad = False
        for parameter in policy.model.action_out_proj.parameters():
            parameter.requires_grad = True
        trainable_model = policy
        trainable_module = "model.action_out_proj"
    trainable = [parameter for parameter in trainable_model.parameters() if parameter.requires_grad]

    correction_packs = []
    for correction_path in args.corrections:
        with np.load(correction_path) as archive:
            correction_packs.append({key: archive[key] for key in archive.files})
    expected_keys = set(correction_packs[0])
    for correction_path, pack in zip(args.corrections[1:], correction_packs[1:], strict=True):
        if set(pack) != expected_keys:
            raise ValueError(
                f"Correction pack {correction_path} keys do not match the first pack"
            )
    corrections = {
        key: np.concatenate([pack[key] for pack in correction_packs], axis=0)
        for key in expected_keys
    }
    sample_count = len(corrections["state"])
    invalid_indices = [index for index in args.indices if not 0 <= index < sample_count]
    if invalid_indices:
        raise IndexError(
            f"Correction indices {invalid_indices} outside concatenated sample count {sample_count}"
        )
    with np.load(args.episode_npz, allow_pickle=True) as archive:
        demo = {key: archive[key] for key in ("state", "action", "camera0", "camera1", "task")}

    def processed(state: np.ndarray, camera0: np.ndarray, camera1: np.ndarray, target: np.ndarray):
        raw = _raw_observation(
            state.astype(np.float32), camera0.astype(np.float32), camera1.astype(np.float32), args.task
        )
        raw[ACTION] = torch.from_numpy(target.astype(np.float32)).unsqueeze(0)
        return preprocessor(raw)

    replay_batches = {}
    anchor_batches = {}
    for index in args.indices:
        replay_batches[index] = processed(
            corrections["state"][index], corrections["camera0"][index],
            corrections["camera1"][index], corrections["target_action"][index],
        )
        frame = int(corrections["demo_frame"][index])
        target = _binarize_target_gripper(
            demo["action"][frame : frame + policy.config.chunk_size],
            demo["task"][frame : frame + policy.config.chunk_size],
        )
        anchor_batches[index] = processed(
            _binarize_observed_gripper(demo["state"][frame]), demo["camera0"][frame],
            demo["camera1"][frame], target,
        )

    def fixed_noise() -> torch.Tensor:
        torch.manual_seed(args.seed)
        return policy.model.sample_noise(
            (1, policy.config.chunk_size, policy.config.max_action_dim), device
        )

    sample_impl = policy.model.sample_actions.__wrapped__

    def sampled_normalized(batch: dict[str, torch.Tensor], *, gradients: bool) -> torch.Tensor:
        images, img_masks = policy._preprocess_images(batch)  # noqa: SLF001
        context = torch.enable_grad() if gradients else torch.no_grad()
        with context:
            sampled = sample_impl(
                policy.model, images, img_masks,
                batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
                noise=fixed_noise(), num_steps=policy.config.num_inference_steps,
            )
            # Match PI05Policy.predict_action_chunk, which removes the internal
            # max_action_dim padding before postprocessing.
            return sampled[..., : policy.config.output_features[ACTION].shape[0]]

    # Never optimize through a sampler that differs from the public deployment
    # path.  The fixed noise makes this an exact numerical contract rather than
    # a statistical comparison.
    sampler_contract_max_abs_diff = 0.0
    with torch.inference_mode():
        for index in args.indices:
            expected = trainable_model.predict_action_chunk(
                replay_batches[index],
                noise=fixed_noise(),
                num_steps=policy.config.num_inference_steps,
            )
            actual = sampled_normalized(replay_batches[index], gradients=False)
            sampler_contract_max_abs_diff = max(
                sampler_contract_max_abs_diff,
                float(torch.max(torch.abs(expected - actual)).item()),
            )
    if sampler_contract_max_abs_diff > 1.0e-5:
        raise RuntimeError(
            "Differentiable sampler does not match predict_action_chunk: "
            f"max_abs_diff={sampler_contract_max_abs_diff:.8g}"
        )

    # Frozen source sampled chunks are the explicit anti-forgetting targets.
    with torch.no_grad():
        source_anchor = {
            index: sampled_normalized(batch, gradients=False).detach()
            for index, batch in anchor_batches.items()
        }

    def evaluate(step: int) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        errors, endpoint_errors, max_steps = [], [], []
        close_recalls, close_values = [], []
        executed_close_recalls, executed_close_values = [], []
        false_closes, anchor_drifts = [], []
        policy.eval()
        with torch.inference_mode():
            for index in args.indices:
                normalized = sampled_normalized(replay_batches[index], gradients=False)
                prediction = _postprocess_chunk(normalized, postprocessor)[0].float().cpu().numpy()
                target = corrections["target_action"][index]
                errors.append(float(np.abs(
                    prediction[:execution_horizon, :6] - target[:execution_horizon, :6]
                ).mean()))
                endpoint_errors.append(float(np.abs(
                    prediction[endpoint_index, :6] - target[endpoint_index, :6]
                ).mean()))
                deltas = np.diff(np.concatenate([
                    corrections["state"][index, None, :6],
                    prediction[:execution_horizon, :6],
                ], axis=0), axis=0)
                max_steps.append(float(np.abs(deltas).max()))
                close_mask = target[:, 6] >= 0.5
                executed_close_mask = target[:execution_horizon, 6] >= 0.5
                open_execution_mask = target[:execution_horizon, 6] < 0.5
                false_closes.extend(
                    (prediction[:execution_horizon, 6][open_execution_mask] >= 0.5)
                    .astype(np.float32).tolist()
                )
                if close_mask.any():
                    close_recalls.append(float(np.mean(prediction[close_mask, 6] >= 0.5)))
                    close_values.extend(prediction[close_mask, 6].tolist())
                if executed_close_mask.any():
                    executed_prediction = prediction[:execution_horizon, 6]
                    executed_close_recalls.append(float(np.mean(
                        executed_prediction[executed_close_mask] >= 0.5
                    )))
                    executed_close_values.extend(
                        executed_prediction[executed_close_mask].tolist()
                    )
                anchor_prediction = sampled_normalized(anchor_batches[index], gradients=False)
                anchor_drifts.append(float(torch.mean(torch.abs(
                    anchor_prediction[:, :execution_horizon, :7]
                    - source_anchor[index][:, :execution_horizon, :7]
                )).item()))
        metrics = {
            "step": step,
            "critical_mean_arm_mae_rad": float(np.mean(errors)),
            "critical_worst_arm_mae_rad": float(np.max(errors)),
            "critical_endpoint_arm_mae_rad": float(np.mean(endpoint_errors)),
            "critical_max_step_rad": float(np.max(max_steps)),
            "future_close_recall": float(np.mean(close_recalls)) if close_recalls else 1.0,
            "future_close_mean_action": float(np.mean(close_values)) if close_values else 1.0,
            "future_close_max_action": float(np.max(close_values)) if close_values else 1.0,
            "executed_close_recall": (
                float(np.mean(executed_close_recalls)) if executed_close_recalls else 0.0
            ),
            "executed_close_mean_action": (
                float(np.mean(executed_close_values)) if executed_close_values else 0.0
            ),
            "executed_close_max_action": (
                float(np.max(executed_close_values)) if executed_close_values else 0.0
            ),
            "executed_false_close_rate": float(np.mean(false_closes)) if false_closes else 0.0,
            "matched_anchor_normalized_drift": float(np.mean(anchor_drifts)),
        }
        snapshot = {
            name: value.detach().cpu().clone()
            for name, value in trainable_model.named_parameters() if value.requires_grad
        }
        return metrics, snapshot

    baseline, baseline_snapshot = evaluate(0)
    history = [baseline]
    candidates = [(baseline, baseline_snapshot)]
    print(f"step=0 {json.dumps(baseline)}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    replay_schedule = []
    for index in args.indices:
        has_close_target = bool(np.any(corrections["target_action"][index, :, 6] >= 0.5))
        replay_schedule.extend([index] * (args.close_sample_weight if has_close_target else 1))

    for step in range(1, args.steps + 1):
        index = replay_schedule[(step - 1) % len(replay_schedule)]
        # Keep dropout/inference behavior identical to deployment while allowing
        # gradients through the explicitly unwrapped sampler.
        policy.eval()
        replay_prediction = sampled_normalized(replay_batches[index], gradients=True)
        replay_target = policy.prepare_action(replay_batches[index])
        arm_loss = torch.nn.functional.smooth_l1_loss(
            replay_prediction[:, :execution_horizon, :6],
            replay_target[:, :execution_horizon, :6],
        )
        arm_delta_loss = torch.nn.functional.smooth_l1_loss(
            torch.diff(replay_prediction[:, :execution_horizon, :6], dim=1),
            torch.diff(replay_target[:, :execution_horizon, :6], dim=1),
        )
        first_action_loss = torch.nn.functional.smooth_l1_loss(
            replay_prediction[:, 0, :6], replay_target[:, 0, :6]
        )
        endpoint_loss = torch.nn.functional.smooth_l1_loss(
            replay_prediction[:, endpoint_index, :6], replay_target[:, endpoint_index, :6]
        )
        gripper_loss = torch.nn.functional.smooth_l1_loss(
            replay_prediction[:, :, 6], replay_target[:, :, 6]
        )
        executed_gripper_loss = torch.nn.functional.smooth_l1_loss(
            replay_prediction[:, :execution_horizon, 6],
            replay_target[:, :execution_horizon, 6],
        )
        anchor_prediction = sampled_normalized(anchor_batches[index], gradients=True)
        anchor_loss = torch.nn.functional.smooth_l1_loss(
            anchor_prediction[:, :execution_horizon, :7],
            source_anchor[index][:, :execution_horizon, :7],
        )
        loss = (
            arm_loss
            + args.arm_delta_weight * arm_delta_loss
            + args.first_action_weight * first_action_loss
            + args.endpoint_weight * endpoint_loss
            + args.gripper_weight * gripper_loss
            + args.executed_gripper_weight * executed_gripper_loss
            + args.anchor_weight * anchor_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            metrics, snapshot = evaluate(step)
            metrics.update({
                "training_loss": float(loss.detach().item()),
                "arm_loss": float(arm_loss.detach().item()),
                "first_action_loss": float(first_action_loss.detach().item()),
                "endpoint_loss": float(endpoint_loss.detach().item()),
                "gripper_loss": float(gripper_loss.detach().item()),
                "executed_gripper_loss": float(executed_gripper_loss.detach().item()),
                "anchor_loss": float(anchor_loss.detach().item()),
                "gradient_norm": float(grad_norm.detach().item()),
            })
            history.append(metrics)
            candidates.append((metrics, snapshot))
            print(f"step={step} {json.dumps(metrics)}", flush=True)

    def passes_gates(metrics: dict[str, float]) -> bool:
        if args.relative_gates:
            action_ok = (
                metrics["critical_mean_arm_mae_rad"]
                <= baseline["critical_mean_arm_mae_rad"]
                * (1.0 - args.min_mean_relative_improvement)
                and metrics["critical_worst_arm_mae_rad"]
                <= baseline["critical_worst_arm_mae_rad"]
                * (1.0 - args.min_worst_relative_improvement)
                and metrics["critical_max_step_rad"]
                <= baseline["critical_max_step_rad"] * args.max_relative_step
            )
        else:
            action_ok = (
                metrics["critical_mean_arm_mae_rad"] < 0.1576
                and metrics["critical_worst_arm_mae_rad"] <= 0.2078
                and metrics["critical_max_step_rad"] <= 0.1402
            )
        absolute_ok = (
            (
                args.max_mean_arm_mae_rad is None
                or metrics["critical_mean_arm_mae_rad"] <= args.max_mean_arm_mae_rad
            )
            and (
                args.max_worst_arm_mae_rad is None
                or metrics["critical_worst_arm_mae_rad"] <= args.max_worst_arm_mae_rad
            )
            and (
                args.max_critical_step_rad is None
                or metrics["critical_max_step_rad"] <= args.max_critical_step_rad
            )
        )
        return (
            action_ok
            and absolute_ok
            and (not args.require_future_close or metrics["future_close_recall"] > 0.0)
            and (not args.require_executed_close or metrics["executed_close_recall"] > 0.0)
            and metrics["executed_false_close_rate"] == 0.0
            and metrics["matched_anchor_normalized_drift"] <= args.max_anchor_drift
        )

    qualified = [candidate for candidate in candidates[1:] if passes_gates(candidate[0])]
    selected = min(qualified, key=lambda item: item[0]["critical_mean_arm_mae_rad"]) if qualified else None
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "task": args.task,
        "initial_adapter": str(args.initial_adapter.resolve()) if args.initial_adapter else None,
        "corrections": [str(path.resolve()) for path in args.corrections],
        "indices": args.indices,
        "seed": args.seed,
        "num_inference_steps": policy.config.num_inference_steps,
        "sampler_contract_max_abs_diff": sampler_contract_max_abs_diff,
        "execution_horizon": execution_horizon,
        "trainable_module": trainable_module,
        "train_action_out_with_lora": args.train_action_out_with_lora,
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "gripper_weight": args.gripper_weight,
        "executed_gripper_weight": args.executed_gripper_weight,
        "anchor_weight": args.anchor_weight,
        "arm_delta_weight": args.arm_delta_weight,
        "first_action_weight": args.first_action_weight,
        "endpoint_weight": args.endpoint_weight,
        "close_sample_weight": args.close_sample_weight,
        "require_future_close": args.require_future_close,
        "require_executed_close": args.require_executed_close,
        "relative_gates": args.relative_gates,
        "max_relative_step": args.max_relative_step,
        "max_mean_arm_mae_rad": args.max_mean_arm_mae_rad,
        "max_worst_arm_mae_rad": args.max_worst_arm_mae_rad,
        "max_critical_step_rad": args.max_critical_step_rad,
        "min_worst_relative_improvement": args.min_worst_relative_improvement,
        "min_mean_relative_improvement": args.min_mean_relative_improvement,
        "max_anchor_drift": args.max_anchor_drift,
        "loss_contract": {
            "arm_imitation_horizon": execution_horizon,
            "gripper_imitation_horizon": policy.config.chunk_size,
            "matched_source_anchor_weight": args.anchor_weight,
            "arm_delta_weight": args.arm_delta_weight,
            "first_action_weight": args.first_action_weight,
            "endpoint_weight": args.endpoint_weight,
        },
        "baseline": baseline,
        "history": history,
        "qualified": bool(qualified),
        "selected": selected[0] if selected else None,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if selected is not None:
        parameter_by_name = dict(trainable_model.named_parameters())
        for name, value in selected[1].items():
            parameter_by_name[name].data.copy_(value.to(parameter_by_name[name].device))
        if args.lora_r > 0:
            args.output_adapter.mkdir(parents=True, exist_ok=True)
            trainable_model.save_pretrained(args.output_adapter)
            if args.train_action_out_with_lora:
                action_out = trainable_model.base_model.model.model.action_out_proj
                action_out_base = getattr(action_out, "base_layer", action_out)
                save_file(
                    {
                        "weight": action_out_base.weight.detach().cpu().contiguous(),
                        "bias": action_out_base.bias.detach().cpu().contiguous(),
                    },
                    str(args.output_adapter / "adapter_local_action_out.safetensors"),
                    metadata={
                        "source_checkpoint": str(args.checkpoint.resolve()),
                        "selected_step": str(selected[0]["step"]),
                        "scope": "adapter_local",
                    },
                )
        else:
            args.output_adapter.parent.mkdir(parents=True, exist_ok=True)
            save_file(selected[1], str(args.output_adapter), metadata={
                "source_checkpoint": str(args.checkpoint.resolve()),
                "selected_step": str(selected[0]["step"]),
                "training_method": "differentiable_sampled_chunk",
            })
        print(f"QUALIFIED adapter saved to {args.output_adapter}", flush=True)
    else:
        print("REJECTED: no sampled-chunk step passed all gates", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

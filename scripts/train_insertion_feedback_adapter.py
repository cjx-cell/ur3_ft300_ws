#!/usr/bin/env python3
"""Train a deterministic learned insertion vector field from raw recovery.

The adapter consumes only robot joint feedback and an FT300 force-rise scalar.
It never consumes peg/hole pose and is not a replay table: noisy off-trajectory
states are supervised to return to the next expert state on the recovery path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pap_moe_framework.rollout_recovery.schema import load_and_validate
from pap_moe_framework.insertion_feedback_adapter import InsertionFeedbackAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--first-raw-frame", type=int, default=567)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-delta-rad", type=float, default=0.04)
    parser.add_argument("--noise-std-rad", type=float, default=0.012)
    parser.add_argument(
        "--noise-scales",
        type=float,
        nargs=6,
        default=(1.0, 0.7, 0.7, 0.7, 0.25, 1.0),
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("refusing to overwrite output/report")
    if min(
        args.hidden_dim,
        args.steps,
        args.batch_size,
    ) < 1 or min(args.max_delta_rad, args.noise_std_rad, args.learning_rate) <= 0.0:
        parser.error("training sizes, limits and rates must be positive")

    summary = load_and_validate(args.episode)
    if summary.recovery_phase != "insertion":
        parser.error("feedback adapter requires a validated insertion recovery")
    with np.load(args.episode, allow_pickle=False) as data:
        mode = data["control_mode"]
        expert_indices = np.flatnonzero(mode == 1)
        action_times = data["expert_action_timestamp"][expert_indices]
        _, first = np.unique(np.round(action_times, 6), return_index=True)
        unique_indices = expert_indices[np.sort(first)]
        unique_indices = unique_indices[unique_indices >= args.first_raw_frame]
        path = data["expert_action"][unique_indices, :6].astype(np.float32)
    if len(path) < 30:
        parser.error(f"only {len(path)} unique expert points remain after start frame")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InsertionFeedbackAdapter(args.hidden_dim, args.max_delta_rad).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Hold out regularly interleaved points so the metric measures continuous
    # interpolation rather than memorization of exact recovery timestamps.
    base_indices = np.arange(len(path) - 1)
    test_mask = base_indices % 7 == 0
    train_indices = base_indices[~test_mask]
    test_indices = base_indices[test_mask]
    noise_scale = np.asarray(args.noise_scales, dtype=np.float32)
    if noise_scale.shape != (6,) or not np.isfinite(noise_scale).all() or np.any(noise_scale <= 0):
        parser.error("noise-scales must contain six positive finite values")

    def make_batch(indices: np.ndarray, *, noisy: bool) -> tuple[torch.Tensor, torch.Tensor]:
        current = path[indices].copy()
        if noisy:
            noise = rng.normal(size=current.shape).astype(np.float32)
            noise *= args.noise_std_rad * noise_scale[None, :]
            current += noise
        target_delta = path[indices + 1] - current
        # The last recovery state is a learned terminal attractor.  Without
        # terminal supervision a locally accurate vector field can continue
        # past the socket after reaching the end of the demonstration.
        if noisy and len(indices) >= 5:
            terminal = np.arange(len(indices)) % 5 == 1
            terminal_noise = rng.normal(size=(int(terminal.sum()), 6)).astype(np.float32)
            terminal_noise *= args.noise_std_rad * noise_scale[None, :]
            current[terminal] = path[-1] + terminal_noise
            target_delta[terminal] = path[-1] - current[terminal]
        # Force-rise feature 0 denotes free insertion. Under sustained contact,
        # supervise a robust terminal attractor instead of holding a deflected
        # state. Real contact can bend wrist joints by far more than nominal
        # demonstration noise; preserving that deflection caused online OOD.
        force_feature = np.zeros((len(indices), 1), dtype=np.float32)
        if noisy and len(indices) >= 4:
            contact = np.arange(len(indices)) % 5 == 0
            contact &= ~terminal
            contact_noise = rng.normal(size=(int(contact.sum()), 6)).astype(np.float32)
            contact_noise *= args.noise_std_rad * noise_scale[None, :]
            current[contact] = path[-1] + contact_noise
            force_feature[contact, 0] = rng.uniform(0.6, 1.0, int(contact.sum()))
            target_delta[contact] = path[-1] - current[contact]
        goal_error = path[-1][None, :] - current
        features = np.concatenate([current, goal_error, force_feature], axis=1)
        return (
            torch.from_numpy(features).to(device),
            torch.from_numpy(target_delta).to(device),
        )

    def audit_closed_loop(force_feature: float = 0.0) -> tuple[list[float], float]:
        starts = [0, len(path) // 4, len(path) // 2]
        errors: list[float] = []
        max_step = 0.0
        with torch.no_grad():
            for start in starts:
                q = path[start].copy()
                q += args.noise_std_rad * noise_scale
                for _ in range((len(path) - start) + 100):
                    feature = torch.from_numpy(
                        np.r_[q, path[-1] - q, np.float32(force_feature)][None].astype(np.float32)
                    ).to(device)
                    delta = model(feature)[0].cpu().numpy()
                    max_step = max(max_step, float(np.max(np.abs(delta))))
                    q += delta
                errors.append(float(np.max(np.abs(q - path[-1]))))
        return errors, max_step

    history: list[dict[str, float]] = []
    candidates: list[tuple[dict[str, float], dict[str, torch.Tensor]]] = []
    model.train()
    for step in range(1, args.steps + 1):
        chosen = rng.choice(train_indices, size=args.batch_size, replace=True)
        features, target = make_batch(chosen, noisy=True)
        predicted = model(features)
        loss = torch.nn.functional.smooth_l1_loss(predicted, target, beta=0.005)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                test_features, test_target = make_batch(test_indices, noisy=False)
                test_prediction = model(test_features)
                error = torch.abs(test_prediction - test_target)
                contact_features = test_features.clone()
                contact_features[:, 12] = 1.0
                contact_delta = torch.abs(model(contact_features))
            metrics = {
                "step": step,
                "train_loss": float(loss.item()),
                "test_mean_abs_delta_error_rad": float(error.mean().item()),
                "test_max_abs_delta_error_rad": float(error.max().item()),
                "test_max_predicted_step_rad": float(test_prediction.abs().max().item()),
                "contact_response_max_delta_rad": float(contact_delta.max().item()),
                "gradient_norm": float(grad_norm.item()),
            }
            closed_loop_errors, closed_loop_max_step = audit_closed_loop()
            metrics["closed_loop_max_final_error_rad"] = max(closed_loop_errors)
            metrics["closed_loop_max_step_rad"] = closed_loop_max_step
            contact_errors = []
            contact_steps = []
            for force_feature in (0.6, 0.8, 1.0):
                errors_at_force, step_at_force = audit_closed_loop(force_feature)
                contact_errors.extend(errors_at_force)
                contact_steps.append(step_at_force)
            metrics["contact_closed_loop_max_final_error_rad"] = max(contact_errors)
            metrics["contact_closed_loop_max_step_rad"] = max(contact_steps)
            history.append(metrics)
            candidates.append(
                (
                    metrics,
                    {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                )
            )
            print(json.dumps(metrics), flush=True)
            model.train()

    def qualifies(metrics: dict[str, float]) -> bool:
        return bool(
            metrics["test_mean_abs_delta_error_rad"] <= 0.002
            and metrics["test_max_abs_delta_error_rad"] <= 0.012
            and metrics["contact_response_max_delta_rad"] <= args.max_delta_rad + 1e-6
            and metrics["closed_loop_max_step_rad"] <= args.max_delta_rad + 1e-6
            and metrics["closed_loop_max_final_error_rad"] <= 0.025
            and metrics["contact_closed_loop_max_final_error_rad"] <= 0.025
        )

    qualified_candidates = [candidate for candidate in candidates if qualifies(candidate[0])]
    selected = (
        min(qualified_candidates, key=lambda candidate: candidate[0]["closed_loop_max_final_error_rad"])
        if qualified_candidates
        else None
    )
    qualified = selected is not None
    report = {
        "source_episode": str(args.episode.resolve()),
        "source_schema_valid": True,
        "input_contract": (
            "joint_state_6d_plus_learned_terminal_joint_error_6d_plus_"
            "normalized_ft300_force_rise"
        ),
        "online_object_truth": False,
        "first_raw_frame": args.first_raw_frame,
        "expert_path_points": int(len(path)),
        "hidden_dim": args.hidden_dim,
        "max_delta_rad": args.max_delta_rad,
        "noise_std_rad": args.noise_std_rad,
        "noise_scales": noise_scale.tolist(),
        "history": history,
        "selected": None if selected is None else selected[0],
        "qualified": qualified,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not qualified:
        print("REJECTED insertion feedback adapter", flush=True)
        return 2
    model.load_state_dict(selected[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "hidden_dim": args.hidden_dim,
            "max_delta_rad": args.max_delta_rad,
            "force_baseline_n": 1.33,
            "force_full_scale_n": 10.0,
            "terminal_arm_state": torch.from_numpy(path[-1].copy()),
            "source_episode": str(args.episode.resolve()),
            "online_object_truth": False,
        },
        args.output,
    )
    print(f"QUALIFIED insertion feedback adapter saved to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

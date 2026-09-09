#!/usr/bin/env python3
"""Add a learned high-force terminal stabilizer to a qualified free-insertion adapter."""

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

from pap_moe_framework.insertion_feedback_adapter import InsertionFeedbackAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--failure-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise-std-rad", type=float, default=0.06)
    parser.add_argument("--contact-force-start-n", type=float, default=5.0)
    parser.add_argument("--contact-force-full-n", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("refusing to overwrite output/report")
    if args.contact_force_full_n <= args.contact_force_start_n:
        parser.error("contact-force-full-n must exceed contact-force-start-n")

    base = torch.load(args.base_adapter, map_location="cpu", weights_only=False)
    if base.get("online_object_truth") is not False:
        parser.error("base adapter must explicitly forbid online object truth")
    terminal = np.asarray(base["terminal_arm_state"], dtype=np.float32)
    trace_paths = sorted(args.failure_trace.glob("chunk_*.npz"))
    trace_states = []
    for path in trace_paths:
        with np.load(path, allow_pickle=False) as trace:
            if bool(trace.get("insertion_feedback_active", False)):
                trace_states.append(np.asarray(trace["state"][:6], dtype=np.float32))
    if len(trace_states) < 10:
        parser.error("failure trace has too few feedback-active states")
    trace_states = np.stack(trace_states)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InsertionFeedbackAdapter(args.hidden_dim, float(base["max_delta_rad"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scale = np.asarray([1.0, 0.8, 0.8, 0.8, 1.2, 1.0], dtype=np.float32)

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        count_trace = args.batch_size // 3
        chosen = trace_states[rng.integers(0, len(trace_states), size=count_trace)].copy()
        synthetic = terminal + rng.normal(
            size=(args.batch_size - count_trace, 6)
        ).astype(np.float32) * args.noise_std_rad * scale
        current = np.concatenate([chosen, synthetic], axis=0)
        current += rng.normal(size=current.shape).astype(np.float32) * 0.003
        force = rng.uniform(0.5, 1.0, size=(len(current), 1)).astype(np.float32)
        features = np.concatenate([current, terminal[None] - current, force], axis=1)
        target = terminal[None] - current
        return torch.from_numpy(features).to(device), torch.from_numpy(target).to(device)

    def audit(starts: np.ndarray) -> tuple[float, float]:
        max_error = 0.0
        max_step = 0.0
        with torch.no_grad():
            for initial in starts:
                q = initial.copy()
                for _ in range(150):
                    feature = np.r_[q, terminal - q, np.float32(1.0)][None].astype(np.float32)
                    delta = model(torch.from_numpy(feature).to(device))[0].cpu().numpy()
                    q += delta
                    max_step = max(max_step, float(np.max(np.abs(delta))))
                max_error = max(max_error, float(np.max(np.abs(q - terminal))))
        return max_error, max_step

    history = []
    candidates = []
    audit_starts = np.concatenate(
        [trace_states[:: max(1, len(trace_states) // 8)], terminal[None] + args.noise_std_rad * scale],
        axis=0,
    )
    for step in range(1, args.steps + 1):
        features, target = batch()
        prediction = model(features)
        loss = torch.nn.functional.smooth_l1_loss(prediction, target, beta=0.01)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            final_error, max_step = audit(audit_starts)
            metrics = {
                "step": step,
                "train_loss": float(loss.item()),
                "audit_max_final_error_rad": final_error,
                "audit_max_step_rad": max_step,
                "gradient_norm": float(gradient_norm.item()),
            }
            history.append(metrics)
            candidates.append((metrics, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
            print(json.dumps(metrics), flush=True)
            model.train()

    qualified = [x for x in candidates if x[0]["audit_max_final_error_rad"] <= 0.01]
    selected = min(qualified, key=lambda x: x[0]["audit_max_final_error_rad"]) if qualified else None
    report = {
        "base_adapter": str(args.base_adapter.resolve()),
        "failure_trace": str(args.failure_trace.resolve()),
        "input_contract": "joint_state_6d_plus_terminal_error_6d_plus_normalized_ft300_force_rise",
        "online_object_truth": False,
        "contact_force_start_n": args.contact_force_start_n,
        "contact_force_full_n": args.contact_force_full_n,
        "trace_states": len(trace_states),
        "history": history,
        "selected": None if selected is None else selected[0],
        "qualified": selected is not None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if selected is None:
        print("REJECTED contact stabilizer", flush=True)
        return 2

    combined = dict(base)
    combined["contact_state_dict"] = selected[1]
    combined["contact_hidden_dim"] = args.hidden_dim
    combined["contact_force_start_n"] = args.contact_force_start_n
    combined["contact_force_full_n"] = args.contact_force_full_n
    combined["contact_failure_trace"] = str(args.failure_trace.resolve())
    combined["online_object_truth"] = False
    torch.save(combined, args.output)
    print(f"QUALIFIED combined adapter saved to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

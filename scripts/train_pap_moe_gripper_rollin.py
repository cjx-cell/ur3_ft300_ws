#!/usr/bin/env python3
"""Train only PAP-MoE's gripper head with explicit open roll-in supervision."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

LEROBOT_SRC = Path("/home/ubuntu/lerobot/src")
SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--success-episode", type=Path, required=True)
    parser.add_argument("--recovery-episode", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--open-weight", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13001)
    args = parser.parse_args()
    if args.output_checkpoint.exists():
        raise FileExistsError(args.output_checkpoint)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.path[:0] = [str(LEROBOT_SRC), str(SCRIPT_DIR)]
    import lerobot.policies.pap_moe.processor_pap_moe  # noqa: F401
    from eval_pap_moe_offline_action_chunk import _raw_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy

    device = torch.device("cuda")
    policy = PAPMoEPolicy.from_pretrained(str(args.source_checkpoint.resolve()))
    policy.to(device).eval()
    if policy.model.gripper_head is None:
        raise RuntimeError("Source checkpoint has no deterministic gripper head")
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for parameter in policy.model.gripper_head.parameters():
        parameter.requires_grad = True
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=str(args.source_checkpoint.resolve())
    )
    chunk_size = int(policy.config.chunk_size)

    inputs_visual: list[torch.Tensor] = []
    inputs_sensor: list[torch.Tensor] = []
    inputs_semantic: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    domains: list[str] = []

    def cache(episode: dict[str, np.ndarray], frame: int, target: np.ndarray, domain: str) -> None:
        captured: list[tuple[torch.Tensor, ...]] = []

        def hook(_module: object, values: tuple[torch.Tensor, ...]) -> None:
            captured.append(tuple(value.detach().float() for value in values))

        handle = policy.model.gripper_head.register_forward_pre_hook(hook)
        try:
            batch = preprocessor(_raw_observation(episode, frame))
            images, masks = policy._preprocess_images(batch)  # noqa: SLF001
            with torch.inference_mode():
                policy.model.predict_gripper_logits(
                    images,
                    masks,
                    batch["observation.language.tokens"],
                    batch["observation.language.attention_mask"],
                    batch["observation.force"],
                    batch["observation.force_fast"],
                    batch["observation.force_slow"],
                    batch["observation.state_history"],
                )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one gripper invocation, got {len(captured)}")
        expected_inputs = 3 if policy.model.gripper_head.semantic is not None else 2
        if len(captured[0]) != expected_inputs:
            raise RuntimeError(
                "Gripper-head input contract mismatch: "
                f"expected {expected_inputs}, captured {len(captured[0])}"
            )
        inputs_visual.append(captured[0][0])
        inputs_sensor.append(captured[0][1])
        if expected_inputs == 3:
            inputs_semantic.append(captured[0][2])
        targets.append(torch.from_numpy(target.astype(np.float32))[None].to(device))
        domains.append(domain)

    required = (
        "state", "state_history", "force", "force_fast", "force_slow",
        "visual_quality", "camera0", "camera1", "stage",
    )
    with np.load(args.success_episode, allow_pickle=True) as archive:
        success = {key: archive[key] for key in required}
        success["task"] = archive["task"]
        success_action = archive["action"]
        # Retain the complete pre-grasp/grasp/lift behavior; later insertion
        # phases are not needed to calibrate the initial close decision.
        for frame in range(0, min(220, len(success_action) - chunk_size + 1), args.frame_stride):
            target = (success_action[frame : frame + chunk_size, 6] > 0.12)
            cache(success, frame, target, "success")

    with np.load(args.recovery_episode, allow_pickle=True) as archive:
        recovery = {key: archive[key] for key in required}
        recovery["task"] = archive["semantic_subtask"]
        expert_action = archive["expert_action"]
        takeover = int(np.flatnonzero(archive["intervention_mask"])[0])
        for frame in range(0, takeover, args.frame_stride):
            cache(recovery, frame, np.zeros(chunk_size, dtype=bool), "rollin_open")
        for frame in range(takeover, len(expert_action), args.frame_stride):
            end = min(frame + chunk_size, len(expert_action))
            target = expert_action[frame:end, 6] > 0.12
            target = np.pad(target, (0, chunk_size - len(target)), mode="edge")
            cache(recovery, frame, target, "expert")

    visual = torch.cat(inputs_visual)
    sensor = torch.cat(inputs_sensor)
    semantic = torch.cat(inputs_semantic) if inputs_semantic else None
    target = torch.cat(targets)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    optimizer = torch.optim.AdamW(policy.model.gripper_head.parameters(), lr=args.learning_rate)
    policy.model.gripper_head.train()
    batch_size = 32
    history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(target), (batch_size,), generator=generator, device=device)
        semantic_batch = semantic[indices] if semantic is not None else None
        logits = policy.model.gripper_head(
            visual[indices], sensor[indices], semantic_batch
        )
        weights = torch.where(target[indices] > 0.5, 1.0, args.open_weight)
        loss = (F.binary_cross_entropy_with_logits(logits, target[indices], reduction="none") * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.model.gripper_head.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            with torch.inference_mode():
                probability = torch.sigmoid(
                    policy.model.gripper_head(visual, sensor, semantic)
                )
                prediction = probability >= args.threshold
                accuracy = (prediction == target.bool()).float().mean().item()
                rollin = torch.tensor([name == "rollin_open" for name in domains], device=device)
                rollin_false_close = prediction[rollin].float().mean().item()
            item = {"step": step, "loss": loss.item(), "accuracy": accuracy,
                    "rollin_false_close": rollin_false_close}
            history.append(item)
            print(item, flush=True)

    policy.model.gripper_head.eval()
    policy.config.gripper_head_probability_threshold = args.threshold
    policy.config.train_gripper_head_only = True
    args.output_checkpoint.mkdir(parents=True)
    policy.save_pretrained(args.output_checkpoint)
    for pattern in ("policy_preprocessor*", "policy_postprocessor*", "train_config.json"):
        for source in args.source_checkpoint.glob(pattern):
            shutil.copy2(source, args.output_checkpoint / source.name)
    report = {"source": str(args.source_checkpoint), "takeover": takeover,
              "samples": len(target), "domains": {name: domains.count(name) for name in set(domains)},
              "semantic_context_dim": int(policy.config.gripper_head_semantic_context_dim),
              "threshold": args.threshold, "open_weight": args.open_weight, "history": history}
    (args.output_checkpoint / "gripper_rollin_training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

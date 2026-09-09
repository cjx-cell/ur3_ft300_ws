#!/usr/bin/env python3
"""PAP-MoE inference process for the UR3 peg-in-hole task.

This process is intentionally policy-specific. It reads the latest observation
snapshot produced by ``ur3_pap_moe_peg_in_hole_ros_side.py``, predicts the
policy's 50-step action horizon, and atomically publishes the first 10 actions.

The trajectory time base is owned by the ROS-side process:

- dataset / action rate: 10 Hz
- published execution prefix: 10 actions
- trajectory point interval: 100 ms
- vision-language replanning interval: 1 second
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import policy_action_exchange as action_exchange
from ur3_peg_in_hole_inference_common import (
    DemonstrationStateRecovery,
    postprocess_action_chunk,
    validate_action_semantics,
)

LEROBOT_SRC = "/home/ubuntu/lerobot/src"
PAP_MOE_FRAMEWORK_SCRIPTS = (
    "/home/ubuntu/ur3_ft300_ws/pap_moe_framework/scripts"
)

JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"
FORCE_FILE = "/tmp/ur3_force.npy"
FORCE_FAST_FILE = "/tmp/ur3_force_fast.npy"
FORCE_SLOW_FILE = "/tmp/ur3_force_slow.npy"
STATE_HISTORY_FILE = "/tmp/ur3_state_history.npy"
VISUAL_QUALITY_FILE = "/tmp/ur3_visual_quality.npy"
OBSERVATION_META_FILE = "/tmp/ur3_pap_moe_observation_meta.json"
ACTION_FILE = "/tmp/ur3_action.txt"
ACTION_CHUNK_FILE = "/tmp/ur3_action_chunk.npy"
ACTION_CHUNK_TMP_FILE = "/tmp/ur3_action_chunk_tmp.npy"
ACTION_CHUNK_META_FILE = "/tmp/ur3_action_chunk_meta.json"
READY_FILE = "/tmp/ur3_inference_ready.txt"

TASK = "pick up the peg and insert it into the hole"
EXPECTED_STATE_DIM = 7
EXPECTED_FORCE_DIM = 6
EXPECTED_FORCE_FAST_SHAPE = (64, 6)
EXPECTED_FORCE_SLOW_SHAPE = (50, 6)
EXPECTED_STATE_HISTORY_SHAPE = (10, 7)
EXPECTED_VISUAL_QUALITY_SHAPE = (4,)
EXPECTED_ACTION_DIM = 7
EXPECTED_IMAGE_SHAPE = (224, 224, 3)
PREDICTED_ACTION_STEPS = 50
EXECUTED_ACTION_STEPS = 10
EXPERT_NAMES = (
    "E1_free_motion",
    "E2_visual_blind",
    "E3_rigid_contact",
    "E4_compliant",
)

DEFAULT_CHECKPOINT = (
    "/home/ubuntu/ur3_ft300_ws/outputs/train/"
    "pap_moe_v9_conditioner_20260808_021028/checkpoints/001000/pretrained_model"
)
DEFAULT_METRICS_FILE = "/tmp/ur3_pap_moe_inference_metrics.csv"


def _atomic_write_text(path: str, text: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        file.write(text)
    os.replace(tmp_path, path)


import policy_diagnostic_trace as diagnostic_trace


def _load_observation() -> dict[str, np.ndarray | dict]:
    with open(JOINT_STATE_FILE, encoding="utf-8") as file:
        raw_state = file.read().strip()
    if not raw_state:
        raise ValueError("Joint-state snapshot is empty")

    state = np.asarray([float(value) for value in raw_state.split()], dtype=np.float32)
    camera0 = np.load(CAMERA0_FILE)
    camera1 = np.load(CAMERA1_FILE)
    force = np.load(FORCE_FILE)
    force_fast = np.load(FORCE_FAST_FILE)
    force_slow = np.load(FORCE_SLOW_FILE)
    state_history = np.load(STATE_HISTORY_FILE)
    visual_quality = np.load(VISUAL_QUALITY_FILE)
    with open(OBSERVATION_META_FILE, encoding="utf-8") as file:
        metadata = json.load(file)

    if state.shape != (EXPECTED_STATE_DIM,):
        raise ValueError(
            f"Invalid state shape {state.shape}; expected {(EXPECTED_STATE_DIM,)}"
        )
    if camera0.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"Invalid camera0 shape {camera0.shape}; expected {EXPECTED_IMAGE_SHAPE}"
        )
    if camera1.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"Invalid camera1 shape {camera1.shape}; expected {EXPECTED_IMAGE_SHAPE}"
        )
    if force.shape != (EXPECTED_FORCE_DIM,):
        raise ValueError(
            f"Invalid force shape {force.shape}; expected {(EXPECTED_FORCE_DIM,)}"
        )
    expected_shapes = {
        "force_fast": (force_fast, EXPECTED_FORCE_FAST_SHAPE),
        "force_slow": (force_slow, EXPECTED_FORCE_SLOW_SHAPE),
        "state_history": (state_history, EXPECTED_STATE_HISTORY_SHAPE),
        "visual_quality": (visual_quality, EXPECTED_VISUAL_QUALITY_SHAPE),
    }
    for name, (array, shape) in expected_shapes.items():
        if array.shape != shape:
            raise ValueError(f"Invalid {name} shape {array.shape}; expected {shape}")
    arrays = (
        state,
        camera0,
        camera1,
        force,
        force_fast,
        force_slow,
        state_history,
        visual_quality,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Observation contains NaN or Inf")
    if metadata.get("schema") != "pap_moe_v6":
        raise ValueError(f"Unexpected observation schema: {metadata.get('schema')!r}")
    if metadata.get("force_reference_mode") != "per_sample_phase_payload_bias_v2":
        raise ValueError(
            "Unexpected force reference mode: "
            f"{metadata.get('force_reference_mode')!r}"
        )
    if not bool(metadata.get("empty_force_bias_valid", False)):
        raise ValueError("PAP-MoE empty-tool force bias is not calibrated yet")
    expected_gripper_units = "continuous_radians_0_0.8"
    if metadata.get("state_history_gripper_units") != expected_gripper_units:
        raise ValueError(
            "Unexpected state-history gripper units: "
            f"{metadata.get('state_history_gripper_units')!r}"
        )
    required_valid = {
        "force_fast_valid": EXPECTED_FORCE_FAST_SHAPE[0],
        "force_slow_valid": EXPECTED_FORCE_SLOW_SHAPE[0],
        "state_history_valid": EXPECTED_STATE_HISTORY_SHAPE[0],
    }
    incomplete = {
        key: int(metadata.get(key, 0))
        for key, expected in required_valid.items()
        if int(metadata.get(key, 0)) < expected
    }
    if incomplete:
        raise ValueError(f"PAP-MoE history buffers are not warm yet: {incomplete}")

    # Match the v9 dataset exactly: current state and history both contain the
    # measured Robotiq knuckle joint in physical radians (0=open, 0.8=closed).
    state = state.copy()
    state[6] = np.clip(state[6], 0.0, 0.8)

    return {
        "state": state,
        "camera0": camera0,
        "camera1": camera1,
        "force": force,
        "force_fast": force_fast,
        "force_slow": force_slow,
        "state_history": state_history,
        "visual_quality": visual_quality,
        "metadata": metadata,
    }


def _raw_observation(
    observation: dict[str, np.ndarray | dict],
) -> dict[str, object]:
    state = observation["state"]
    camera0 = observation["camera0"]
    camera1 = observation["camera1"]
    force = observation["force"]
    camera0_chw = np.ascontiguousarray(np.transpose(camera0, (2, 0, 1)))
    camera1_chw = np.ascontiguousarray(np.transpose(camera1, (2, 0, 1)))

    return {
        "observation.state": torch.from_numpy(state),
        "observation.force": torch.from_numpy(force).unsqueeze(0),
        "observation.force_fast": torch.from_numpy(observation["force_fast"]).unsqueeze(
            0
        ),
        "observation.force_slow": torch.from_numpy(observation["force_slow"]).unsqueeze(
            0
        ),
        "observation.state_history": torch.from_numpy(
            observation["state_history"]
        ).unsqueeze(0),
        "observation.visual_quality": torch.from_numpy(
            observation["visual_quality"]
        ).unsqueeze(0),
        "observation.images.camera0": torch.from_numpy(camera0_chw),
        "observation.images.camera1": torch.from_numpy(camera1_chw),
        "task": TASK,
    }


def _compute_analytic_route(prior, observation: dict[str, np.ndarray | dict]) -> np.ndarray:
    """Replay the latest 10 Hz window through the collection-time routing prior.

    Inference replans once per second, whereas routing labels were generated at
    the 10 Hz policy rate. Replaying the non-overlapping state/slow-force window
    keeps the prior's contact memory and hysteresis in the same time base as
    training instead of accidentally making them ten times slower.
    """
    state_history = np.asarray(observation["state_history"], dtype=np.float32)
    force_slow = np.asarray(observation["force_slow"], dtype=np.float32)
    force_fast = np.asarray(observation["force_fast"], dtype=np.float32)
    visual_quality = np.asarray(observation["visual_quality"], dtype=np.float32)

    # force_slow is 10 Hz over five seconds; only replay the latest one-second
    # interval paired with the 10-element state history.
    slow_tail = force_slow[-len(state_history) :]
    velocities = np.zeros(len(state_history), dtype=np.float32)
    if len(state_history) > 1:
        velocities[1:] = (
            np.linalg.norm(np.diff(state_history[:, :6], axis=0), axis=1) * 10.0
        )

    black_fraction, saturated_fraction, _contrast, valid = visual_quality
    if valid < 0.5 or black_fraction >= 0.95:
        degraded = True
        degradation_type = "dropout"
        glare_gain = 1.0
    elif saturated_fraction >= 0.85:
        degraded = True
        degradation_type = "glare"
        # Invert the collection prior's visual-loss mapping conservatively.
        glare_gain = float(np.clip(1.0 + 5.0 * saturated_fraction, 1.0, 6.0))
    else:
        degraded = False
        degradation_type = "normal"
        glare_gain = 1.0

    route = None
    for index, (force_value, state_value, velocity) in enumerate(
        zip(slow_tail, state_history, velocities, strict=True)
    ):
        is_latest = index == len(state_history) - 1
        route = prior.compute(
            np.asarray(observation["force"] if is_latest else force_value),
            force_fast if is_latest else None,
            tool0_z=None,
            cam_degraded=degraded,
            cam_degradation_type=degradation_type,
            cam_glare_gain=glare_gain,
            gripper_joint_val=float(state_value[6]),
            joint_vel_norm=float(velocity),
        )
    if route is None or route.shape != (len(EXPERT_NAMES),) or not np.isfinite(route).all():
        raise ValueError(f"Invalid analytic routing result: {route}")
    return route.astype(np.float32, copy=False)


def _compute_contact_oracle_route(observation: dict[str, np.ndarray | dict]) -> np.ndarray:
    """Map privileged Gazebo contact truth to diagnostic expert weights.

    The normal-vision centroids are measured from Workspace50 dominant-route
    subsets. They keep the diagnostic inside the action model's training
    support: in particular, Workspace50 never contains a pure E4 target.
    This route is an ablation oracle, not a deployable policy observation or a
    proposed task-general routing prior.
    """
    metadata = observation.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("contact_oracle requires observation metadata")
    truth = metadata.get("contact_truth")
    if not isinstance(truth, dict) or not truth.get("valid", False):
        raise ValueError("contact_oracle requires valid Gazebo contact truth")
    visual_quality = np.asarray(observation["visual_quality"], dtype=np.float32)
    visual_valid = bool(visual_quality.shape == (4,) and visual_quality[3] >= 0.5)
    if not visual_valid:
        return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    centroids = {
        "free": np.asarray([0.9827, 0.0, 0.0024, 0.0148], dtype=np.float32),
        "rigid": np.asarray([0.0255, 0.0, 0.9126, 0.0619], dtype=np.float32),
        "movable": np.asarray([0.0772, 0.0, 0.1511, 0.7717], dtype=np.float32),
    }
    active = []
    if bool(truth.get("hole", False)):
        active.append(centroids["rigid"])
    if bool(truth.get("gripper", False)):
        active.append(centroids["movable"])
    route = centroids["free"].copy() if not active else np.mean(active, axis=0)
    route /= route.sum()
    return route


def _initialize_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "chunk_id",
                "wall_time_s",
                "observation_mtime_s",
                "inference_ms",
                "predicted_steps",
                "published_steps",
                "force_l2",
                "expert_id",
                "expert_name",
                "route_E1",
                "route_E2",
                "route_E3",
                "route_E4",
                "gate_E1",
                "gate_E2",
                "gate_E3",
                "gate_E4",
                "factor_blindness_b",
                "factor_contact_c",
                "factor_mobility_m",
                "routing_confidence",
                "condition_residual_norm",
                "applied_condition_residual_norm",
                "scale_E1",
                "scale_E2",
                "scale_E3",
                "scale_E4",
                "token_norm_E1",
                "token_norm_E2",
                "token_norm_E3",
                "token_norm_E4",
                "rtc_enabled",
                "boundary_jump_arm_l2",
                "replanning_disagreement_arm_l2",
            ]
        )


def _append_metrics(
    path: Path,
    *,
    chunk_id: int,
    observation_mtime: float,
    inference_ms: float,
    force_l2: float,
    expert_id: int,
    expert_probs: np.ndarray,
    gate_probs: np.ndarray,
    factor_probs: np.ndarray | None,
    routing_confidence: float,
    condition_residual_norm: float | None,
    applied_condition_residual_norm: float | None,
    expert_conditioning_scales: np.ndarray | None,
    expert_token_norms: np.ndarray,
    executed_action_steps: int,
    rtc_enabled: bool,
    boundary_jump: float | None,
    replanning_disagreement: float | None,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                chunk_id,
                f"{time.time():.6f}",
                f"{observation_mtime:.6f}",
                f"{inference_ms:.3f}",
                PREDICTED_ACTION_STEPS,
                executed_action_steps,
                f"{force_l2:.6f}",
                expert_id,
                EXPERT_NAMES[expert_id],
                *(f"{probability:.6f}" for probability in expert_probs),
                *(f"{probability:.6f}" for probability in gate_probs),
                *(
                    ["", "", ""]
                    if factor_probs is None
                    else [f"{probability:.6f}" for probability in factor_probs]
                ),
                f"{routing_confidence:.6f}",
                "" if condition_residual_norm is None else f"{condition_residual_norm:.6f}",
                (
                    ""
                    if applied_condition_residual_norm is None
                    else f"{applied_condition_residual_norm:.6f}"
                ),
                *(
                    ["", "", "", ""]
                    if expert_conditioning_scales is None
                    else [f"{scale:.6f}" for scale in expert_conditioning_scales]
                ),
                *(f"{norm:.6f}" for norm in expert_token_norms),
                int(rtc_enabled),
                "" if boundary_jump is None else f"{boundary_jump:.6f}",
                (
                    ""
                    if replanning_disagreement is None
                    else f"{replanning_disagreement:.6f}"
                ),
            ]
        )
        file.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--pi05-backbone-checkpoint",
        type=Path,
        help="Restore the frozen shared Pi0.5 backbone at its original mixed precision.",
    )
    parser.add_argument("--metrics-file", default=DEFAULT_METRICS_FILE)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed the flow-matching RNG once before the live episode.",
    )
    parser.add_argument(
        "--fixed-noise-per-replan",
        action="store_true",
        help="Diagnostic mode: reset flow-matching noise to --seed at every replan.",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=EXECUTED_ACTION_STEPS,
        help="Publish this many steps from each 50-step chunk; default remains 10.",
    )
    parser.add_argument(
        "--expert-mask",
        default="1,1,1,1",
        help="Four comma-separated expert multipliers. Use 0,0,0,0 for the semantic baseline.",
    )
    parser.add_argument(
        "--routing-source",
        choices=("physicsgate", "analytic", "contact_oracle", "fixed_e1"),
        default="physicsgate",
        help=(
            "Use the learned PhysicsGate, the observable collection-time soft "
            "routing prior, the privileged Gazebo contact oracle, or the fixed-E1 "
            "diagnostic route. For an action-step "
            "checkpoint, an override is repeated over the complete action chunk; "
            "no unavailable future simulator state is used."
        ),
    )
    parser.add_argument(
        "--action-conditioning-scale",
        type=float,
        default=None,
        help=(
            "Optional inference override in [0, 1] for the PAP action residual. "
            "Zero preserves the exact Pi0.5 action path while retaining routing diagnostics."
        ),
    )
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="Enable synchronous Real-Time Chunking guidance between replans.",
    )
    parser.add_argument("--rtc-execution-horizon", type=int, default=10)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=("EXP", "LINEAR", "ONES", "ZEROS"),
        default="EXP",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--demo-recovery-episode", type=Path)
    parser.add_argument("--demo-recovery-after-chunks", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.execute_steps <= PREDICTED_ACTION_STEPS:
        raise ValueError(
            f"--execute-steps must be in [1, {PREDICTED_ACTION_STEPS}]"
        )
    expert_mask_values = np.fromstring(args.expert_mask, sep=",", dtype=np.float32)
    if expert_mask_values.shape != (len(EXPERT_NAMES),) or not np.isfinite(
        expert_mask_values
    ).all():
        raise ValueError("--expert-mask must contain four finite comma-separated values")
    if np.any((expert_mask_values < 0.0) | (expert_mask_values > 1.0)):
        raise ValueError("--expert-mask values must lie in [0, 1]")
    # An all-zero mask is the deliberate Pi0.5-identity ablation. The current
    # ActionTokenConditioner explicitly zeros fully masked attention rows, so
    # this path is finite and exactly bypasses the physical residual.
    if args.action_conditioning_scale is not None and not (
        np.isfinite(args.action_conditioning_scale)
        and 0.0 <= args.action_conditioning_scale <= 1.0
    ):
        raise ValueError("--action-conditioning-scale must be finite and in [0, 1]")
    if args.rtc_execution_horizon < 1:
        raise ValueError("--rtc-execution-horizon must be positive")
    if not np.isfinite(args.rtc_max_guidance_weight) or args.rtc_max_guidance_weight <= 0:
        raise ValueError("--rtc-max-guidance-weight must be finite and positive")
    if args.rtc and args.rtc_execution_horizon > PREDICTED_ACTION_STEPS - args.execute_steps:
        raise ValueError(
            "--rtc-execution-horizon exceeds the previous chunk remainder: "
            f"{args.rtc_execution_horizon} > {PREDICTED_ACTION_STEPS - args.execute_steps}"
        )

    checkpoint = Path(args.checkpoint)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"PAP-MoE config not found: {config_path}")

    with config_path.open(encoding="utf-8") as file:
        checkpoint_config = json.load(file)

    policy_type = checkpoint_config.get("type")
    if policy_type != "pap_moe":
        raise ValueError(
            f"{checkpoint} is type={policy_type!r}; "
            "ur3_pap_moe_peg_in_hole_inference.py only accepts type='pap_moe'"
        )

    chunk_size = int(checkpoint_config.get("chunk_size", PREDICTED_ACTION_STEPS))
    if chunk_size != PREDICTED_ACTION_STEPS:
        raise ValueError(
            "PAP-MoE checkpoint chunk_size="
            f"{chunk_size}; expected {PREDICTED_ACTION_STEPS}"
        )
    if checkpoint_config.get("use_relative_actions", True):
        raise ValueError(
            "PAP-MoE v6 requires absolute actions (use_relative_actions=false)"
        )
    required_features = {
        "observation.force_fast": [64, 6],
        "observation.force_slow": [50, 6],
        "observation.state_history": [10, 7],
        "observation.visual_quality": [4],
    }
    configured_features = checkpoint_config.get("input_features", {})
    missing_features = {
        key: shape
        for key, shape in required_features.items()
        if configured_features.get(key, {}).get("shape") != shape
    }
    if missing_features:
        raise ValueError(
            f"Checkpoint does not satisfy the PAP-MoE v6 schema: {missing_features}"
        )
    try:
        validate_action_semantics(checkpoint, checkpoint_config)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Checkpoint validation failed: {error}") from None
    if args.validate_only:
        print(f"Checkpoint validation passed: {checkpoint}")
        return

    sys.path.insert(0, LEROBOT_SRC)
    import lerobot.policies.pi05.processor_pi05  # noqa: F401
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPolicy
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    sys.path.insert(0, PAP_MOE_FRAMEWORK_SCRIPTS)
    from pap_moe_routing_prior import PhysicsRoutingPrior

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    print(f"Loading PAP-MoE checkpoint: {checkpoint}", flush=True)
    policy = PAPMoEPolicy.from_pretrained(str(checkpoint), strict=False)
    if args.action_conditioning_scale is not None:
        policy.config.action_conditioning_scale = args.action_conditioning_scale
        print(
            "PAP action conditioning scale overridden to "
            f"{args.action_conditioning_scale:.3f}",
            flush=True,
        )
    if args.pi05_backbone_checkpoint is not None:
        from safetensors import safe_open

        source_file = args.pi05_backbone_checkpoint / "model.safetensors"
        if not source_file.is_file():
            raise FileNotFoundError(f"Pi0.5 backbone weights not found: {source_file}")
        target_state = policy.state_dict()
        restored = 0
        lora_projections = (
            ".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.",
            ".gate_proj.", ".up_proj.", ".down_proj.",
        )
        with safe_open(source_file, framework="pt", device="cpu") as source:
            for source_key in source.keys():
                target_key = source_key
                if (
                    source_key.endswith((".weight", ".bias"))
                    and any(name in source_key for name in lora_projections)
                ):
                    prefix, suffix = source_key.rsplit(".", 1)
                    target_key = f"{prefix}.base.{suffix}"
                target = target_state.get(target_key)
                tensor = source.get_tensor(source_key)
                if target is None or target.shape != tensor.shape:
                    continue
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                restored += 1
        print(
            f"Restored {restored} shared tensors from Pi0.5 backbone: "
            f"{args.pi05_backbone_checkpoint}",
            flush=True,
        )
    if args.rtc:
        policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=RTCAttentionSchedule(
                args.rtc_prefix_attention_schedule
            ),
        )
        policy.init_rtc_processor()
    policy.to(device=device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
    )
    expert_mask = torch.from_numpy(expert_mask_values).to(device=device)
    rtc_action_mask = torch.tensor(
        [1.0] * (EXPECTED_ACTION_DIM - 1) + [0.0], device=device
    )

    metrics_path = Path(args.metrics_file)
    _initialize_metrics(metrics_path)

    # Compile the complete PAP path before the ROS-side process spawns the
    # slender peg. Without this warmup, the first live inference takes 6--7 s
    # and repeatedly times out even though steady-state inference is ~0.55 s.
    warmup_state = np.asarray(
        [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0, 0.0],
        dtype=np.float32,
    )
    warmup_observation = {
        "state": warmup_state,
        "camera0": np.zeros(EXPECTED_IMAGE_SHAPE, dtype=np.float32),
        "camera1": np.zeros(EXPECTED_IMAGE_SHAPE, dtype=np.float32),
        "force": np.zeros(EXPECTED_FORCE_DIM, dtype=np.float32),
        "force_fast": np.zeros(EXPECTED_FORCE_FAST_SHAPE, dtype=np.float32),
        "force_slow": np.zeros(EXPECTED_FORCE_SLOW_SHAPE, dtype=np.float32),
        "state_history": np.tile(
            np.asarray(
                [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0, 0.100],
                dtype=np.float32,
            ),
            (EXPECTED_STATE_HISTORY_SHAPE[0], 1),
        ),
        "visual_quality": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "metadata": {},
    }
    warmup_batch = preprocessor(_raw_observation(warmup_observation))
    warmup_batch["expert_mask"] = expert_mask
    torch.manual_seed(args.seed)
    warmup_start = time.perf_counter()
    inference_context = torch.no_grad if args.rtc else torch.inference_mode
    with inference_context():
        warmup_chunk = policy.predict_action_chunk(warmup_batch)
        postprocess_action_chunk(warmup_chunk, postprocessor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(
        "PAP-MoE warmup complete in "
        f"{(time.perf_counter() - warmup_start) * 1000.0:.1f} ms.",
        flush=True,
    )
    torch.manual_seed(args.seed)

    ready_metadata = {
        "policy_type": "pap_moe",
        "checkpoint": str(checkpoint),
        "predicted_action_steps": PREDICTED_ACTION_STEPS,
        "executed_action_steps": args.execute_steps,
        "action_dt_s": 0.1,
        "observation_schema": "pap_moe_v6",
        "inference_seed": args.seed,
        "fixed_noise_per_replan": args.fixed_noise_per_replan,
        "expert_mask": expert_mask_values.tolist(),
        "routing_source": args.routing_source,
        "action_conditioning_scale": policy.config.action_conditioning_scale,
        "gripper_action_mode": "continuous_radians_0_0.8",
        "state_gripper_mode": "continuous_radians_0_0.8",
        "state_history_gripper_mode": "continuous_radians_0_0.8",
        "rtc_enabled": args.rtc,
        "rtc_execution_horizon": args.rtc_execution_horizon if args.rtc else None,
        "rtc_max_guidance_weight": args.rtc_max_guidance_weight if args.rtc else None,
        "rtc_prefix_attention_schedule": (
            args.rtc_prefix_attention_schedule if args.rtc else None
        ),
        "rtc_inference_delay_steps": 0 if args.rtc else None,
        "rtc_action_dimensions": "arm_only" if args.rtc else None,
    }
    _atomic_write_text(
        READY_FILE, json.dumps(ready_metadata, ensure_ascii=False) + "\n"
    )

    print(
        f"PAP-MoE inference ready: predict 50 steps, publish {args.execute_steps} steps, "
        "ROS execution dt=100 ms",
        flush=True,
    )

    while not os.path.exists(JOINT_STATE_FILE):
        time.sleep(0.01)

    chunk_id = 0
    # Process the snapshot that released the wait above.  Using its current
    # mtime here drops observation 0 and creates a needless 3 s ROS timeout.
    last_observation_mtime = None
    previous_normalized_leftover = None
    previous_physical_chunk = None
    analytic_prior = PhysicsRoutingPrior() if args.routing_source == "analytic" else None
    demo_recovery = (
        DemonstrationStateRecovery(
            args.demo_recovery_episode, PREDICTED_ACTION_STEPS, args.execute_steps
        )
        if args.demo_recovery_episode is not None
        else None
    )

    while True:
        try:
            observation_mtime = os.path.getmtime(JOINT_STATE_FILE)
            if observation_mtime == last_observation_mtime:
                time.sleep(0.001)
                continue
            last_observation_mtime = observation_mtime
            request_id = action_exchange.observation_id(JOINT_STATE_FILE)

            observation = None
            for retry in range(3):
                try:
                    observation = _load_observation()
                    break
                except (OSError, ValueError):
                    if retry == 2:
                        raise
                    time.sleep(0.003)

            if action_exchange.enabled() and request_id != action_exchange.observation_id(JOINT_STATE_FILE):
                raise RuntimeError("Observation changed during load; refusing unpaired inference")
            batch = preprocessor(_raw_observation(observation))
            batch["expert_mask"] = expert_mask
            analytic_route = None
            if analytic_prior is not None:
                analytic_route = _compute_analytic_route(analytic_prior, observation)
            elif args.routing_source == "contact_oracle":
                analytic_route = _compute_contact_oracle_route(observation)
            elif args.routing_source == "fixed_e1":
                analytic_route = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if analytic_route is not None:
                route_override = torch.from_numpy(analytic_route).unsqueeze(0).to(device=device)
                if getattr(policy.config, "action_step_routing", False):
                    # A live controller cannot know the physical state of future
                    # observations. Repeat the current observable teacher route
                    # over the chunk so that analytic/fixed_e1 genuinely bypasses
                    # the route forecaster while remaining deployable.
                    route_override = route_override.unsqueeze(1).expand(
                        -1, policy.config.chunk_size, -1
                    )
                batch["stage_override"] = route_override

            inference_start = time.perf_counter()
            use_demo_recovery = (
                demo_recovery is not None
                and chunk_id >= args.demo_recovery_after_chunks
            )
            recovery_index = None
            recovery_distance = None
            if use_demo_recovery:
                recovery_chunk, recovery_index, recovery_distance = demo_recovery.action_chunk(
                    np.asarray(observation["state"], dtype=np.float32)
                )
                predicted_chunk = torch.from_numpy(recovery_chunk[None])
                normalized_chunk = None
            else:
                with inference_context():
                    if args.fixed_noise_per_replan:
                        torch.manual_seed(args.seed)
                    predict_kwargs = {}
                    if args.rtc:
                        predict_kwargs = {
                            "prev_chunk_left_over": previous_normalized_leftover,
                            "inference_delay": 0,
                            "execution_horizon": args.rtc_execution_horizon,
                            "rtc_action_mask": rtc_action_mask,
                        }
                    diagnostic_rng = None
                    if diagnostic_trace.enabled():
                        diagnostic_rng = {
                            "cpu": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                            "previous_leftover": previous_normalized_leftover,
                        }
                    normalized_chunk = policy.predict_action_chunk(batch, **predict_kwargs)
                    predicted_chunk = postprocess_action_chunk(
                        normalized_chunk, postprocessor
                    )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            gate_probs = policy.last_stage_probs[0].float().numpy()
            routing_output = policy.last_routing_probs[0].float().numpy()
            # Block-level PAP checkpoints expose [4], while action-step routing
            # exposes [50,4].  The complete sequence has already conditioned
            # the 50 action tokens; online status/CSV rows describe the current
            # replan instant and therefore use horizon zero.
            expert_probs = (
                routing_output[0] if routing_output.ndim == 2 else routing_output
            )
            factor_probs = (
                None
                if policy.last_factor_probs is None
                else policy.last_factor_probs[0].float().numpy()
            )
            confidence_output = policy.last_routing_confidence[0].float().reshape(-1)
            routing_confidence = float(confidence_output[0].item())
            condition_residual_norm = (
                None
                if policy.last_condition_residual_norm is None
                else float(policy.last_condition_residual_norm[0])
            )
            applied_condition_residual_norm = (
                None
                if policy.last_applied_condition_residual_norm is None
                else float(policy.last_applied_condition_residual_norm[0])
            )
            expert_conditioning_scales = (
                None
                if policy.last_expert_conditioning_scales is None
                else policy.last_expert_conditioning_scales.float().numpy()
            )
            expert_token_norms = policy.last_expert_token_norms[0].float().numpy()
            if expert_probs.shape != (len(EXPERT_NAMES),):
                raise ValueError(
                    f"Invalid expert route shape {expert_probs.shape}; "
                    f"expected {(len(EXPERT_NAMES),)}"
                )
            expert_id = int(expert_probs.argmax())

            if predicted_chunk.ndim != 3:
                raise ValueError(
                    "Invalid policy output shape "
                    f"{tuple(predicted_chunk.shape)}; expected [B, 50, A]"
                )
            if predicted_chunk.shape[1] != PREDICTED_ACTION_STEPS:
                raise ValueError(
                    f"Policy returned {predicted_chunk.shape[1]} steps; "
                    f"expected {PREDICTED_ACTION_STEPS}"
                )
            if predicted_chunk.shape[2] != EXPECTED_ACTION_DIM:
                raise ValueError(
                    f"Policy returned action_dim={predicted_chunk.shape[2]}; "
                    f"expected {EXPECTED_ACTION_DIM}"
                )

            execution_chunk = (
                predicted_chunk[0, : args.execute_steps]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
            if not np.isfinite(execution_chunk).all():
                raise ValueError("Predicted action chunk contains NaN or Inf")
            # The v9 action label is a physical Robotiq knuckle angle. Preserve
            # the learned continuous trajectory and only clamp to its hardware
            # range; the ROS-side controller applies the same contract.
            execution_chunk[:, 6] = np.clip(execution_chunk[:, 6], 0.0, 0.8)

            if use_demo_recovery:
                execution_chunk = recovery_chunk[: args.execute_steps].copy()

            physical_chunk = predicted_chunk[0].detach().to(
                device="cpu", dtype=torch.float32
            ).numpy()
            boundary_jump = None
            replanning_disagreement = None
            if previous_physical_chunk is not None:
                boundary_jump = float(
                    np.linalg.norm(
                        physical_chunk[0, :6]
                        - previous_physical_chunk[args.execute_steps - 1, :6]
                    )
                )
                replanning_disagreement = float(
                    np.linalg.norm(
                        physical_chunk[0, :6]
                        - previous_physical_chunk[args.execute_steps, :6]
                    )
                )
            if args.rtc and normalized_chunk is not None:
                previous_normalized_leftover = normalized_chunk[
                    :, args.execute_steps:
                ].detach().clone()
            elif use_demo_recovery:
                previous_normalized_leftover = None
            previous_physical_chunk = physical_chunk.copy()

            np.save(ACTION_CHUNK_TMP_FILE, execution_chunk)
            os.replace(ACTION_CHUNK_TMP_FILE, ACTION_CHUNK_FILE)
            action_exchange.publish_reply(request_id, execution_chunk)
            _atomic_write_text(
                ACTION_FILE,
                " ".join(f"{value:.6f}" for value in execution_chunk[0]) + "\n",
            )

            chunk_metadata = {
                "chunk_id": chunk_id,
                "observation_mtime_s": observation_mtime,
                "published_time_s": time.time(),
                "inference_ms": inference_ms,
                "predicted_steps": PREDICTED_ACTION_STEPS,
                "published_steps": args.execute_steps,
                "action_dt_s": 0.1,
                "expert_id": expert_id,
                "expert_name": EXPERT_NAMES[expert_id],
                "expert_probs": expert_probs.tolist(),
                "gate_probs": gate_probs.tolist(),
                "factor_probs_bcm": None if factor_probs is None else factor_probs.tolist(),
                "routing_confidence": routing_confidence,
                "condition_residual_norm": condition_residual_norm,
                "applied_condition_residual_norm": applied_condition_residual_norm,
                "expert_conditioning_scales": (
                    None
                    if expert_conditioning_scales is None
                    else expert_conditioning_scales.tolist()
                ),
                "expert_token_norms": expert_token_norms.tolist(),
                "routing_source": args.routing_source,
                "rtc_enabled": args.rtc,
                "boundary_jump_arm_l2": boundary_jump,
                "replanning_disagreement_arm_l2": replanning_disagreement,
            }
            _atomic_write_text(
                ACTION_CHUNK_META_FILE,
                json.dumps(chunk_metadata, ensure_ascii=False) + "\n",
            )
            _append_metrics(
                metrics_path,
                chunk_id=chunk_id,
                observation_mtime=observation_mtime,
                inference_ms=inference_ms,
                force_l2=float(np.linalg.norm(observation["force"])),
                expert_id=expert_id,
                expert_probs=expert_probs,
                gate_probs=gate_probs,
                factor_probs=factor_probs,
                routing_confidence=routing_confidence,
                condition_residual_norm=condition_residual_norm,
                applied_condition_residual_norm=applied_condition_residual_norm,
                expert_conditioning_scales=expert_conditioning_scales,
                expert_token_norms=expert_token_norms,
                executed_action_steps=args.execute_steps,
                rtc_enabled=args.rtc,
                boundary_jump=boundary_jump,
                replanning_disagreement=replanning_disagreement,
            )

            print(
                f"[PAP-MoE chunk {chunk_id:05d}] "
                f"inference={inference_ms:.1f} ms, "
                f"force_l2={np.linalg.norm(observation['force']):.2f}, "
                f"expert={EXPERT_NAMES[expert_id]} ({expert_probs[expert_id]:.3f}), "
                f"route={np.round(expert_probs, 3)}, "
                f"gate={np.round(gate_probs, 3)}, "
                f"bcm={None if factor_probs is None else np.round(factor_probs, 3)}, "
                f"confidence={routing_confidence:.3f}, "
                f"residual={condition_residual_norm}/{applied_condition_residual_norm}, "
                f"routing={args.routing_source}, "
                f"rtc={args.rtc}, "
                f"boundary_jump={boundary_jump}, "
                f"replan_disagreement={replanning_disagreement}, "
                f"first_action={np.round(execution_chunk[0], 4)}",
                flush=True,
            )
            if diagnostic_trace.enabled():
                diagnostic_trace.snapshot(
                    chunk_id, observation=observation, processed=batch,
                    normalized_action=normalized_chunk, physical_action=physical_chunk,
                    published_action=execution_chunk, route_sequence=routing_output,
                    expert_token_norms=expert_token_norms, metadata=chunk_metadata,
                    rng=diagnostic_rng if not use_demo_recovery else None,
                )
            chunk_id += 1

        except KeyboardInterrupt:
            print("Stopping PAP-MoE inference.", flush=True)
            break
        except Exception as error:
            print(f"PAP-MoE inference error: {error}", file=sys.stderr, flush=True)
            time.sleep(0.05)


if __name__ == "__main__":
    main()

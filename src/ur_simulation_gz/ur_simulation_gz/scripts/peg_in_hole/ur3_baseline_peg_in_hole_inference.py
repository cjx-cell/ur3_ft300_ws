#!/usr/bin/env python3
"""Pi0.5 baseline inference process for the UR3 peg-in-hole task."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import policy_action_exchange as action_exchange
_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
from pap_moe_framework.insertion_feedback_adapter import (  # noqa: E402
    InsertionFeedbackAdapter,
    build_feedback_chunk,
)
from ur3_peg_in_hole_inference_common import (
    DemonstrationStateRecovery,
    postprocess_action_chunk,
    validate_action_semantics,
)

LEROBOT_SRC = "/home/ubuntu/lerobot/src"

JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"
FORCE_FILE = "/tmp/ur3_force.npy"
FORCE_FAST_FILE = "/tmp/ur3_force_fast.npy"
FORCE_SLOW_FILE = "/tmp/ur3_force_slow.npy"
STATE_HISTORY_FILE = "/tmp/ur3_state_history.npy"
OBSERVATION_META_FILE = "/tmp/ur3_pap_moe_observation_meta.json"
ACTION_FILE = "/tmp/ur3_action.txt"
ACTION_CHUNK_FILE = "/tmp/ur3_action_chunk.npy"
ACTION_CHUNK_TMP_FILE = "/tmp/ur3_action_chunk_tmp.npy"
READY_FILE = "/tmp/ur3_inference_ready.txt"

TASK = "pick up the peg and insert it into the hole"
EXPECTED_STATE_DIM = 7
EXPECTED_ACTION_DIM = 7
EXPECTED_IMAGE_SHAPE = (224, 224, 3)
EXPECTED_FORCE_SHAPE = (6,)
EXPECTED_FORCE_FAST_SHAPE = (64, 6)
EXPECTED_FORCE_SLOW_SHAPE = (50, 6)
EXPECTED_STATE_HISTORY_SHAPE = (10, 7)
PHYSICAL_GRIPPER_CLOSED_THRESHOLD = 0.35
PHYSICAL_GRIPPER_CLOSED_POSITION_RAD = 0.629
DEFAULT_CHECKPOINT = (
    "/home/ubuntu/ur3_ft300_ws/outputs/train/"
    "pi05_relative_h50_global_task_rebalanced_10k/"
    "checkpoints/010000/pretrained_model"
)


def _load_action_quantile_bounds(checkpoint: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the exact action normalizer used by the checkpoint."""
    from safetensors.torch import load_file

    state_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    state = load_file(str(state_path), device="cpu")
    q01 = state.get("action.q01")
    q99 = state.get("action.q99")
    if q01 is None or q99 is None or q01.shape != (EXPECTED_ACTION_DIM,) or q99.shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(f"Invalid action quantile state in {state_path}")
    if not torch.all(q99[:6] > q01[:6]):
        raise ValueError(f"Degenerate arm action quantiles in {state_path}")
    return q01.to(torch.float32), q99.to(torch.float32)


def _synchronize_rtc_arm_chunk(
    normalized_chunk: torch.Tensor,
    physical_chunk: np.ndarray,
    q01: torch.Tensor,
    q99: torch.Tensor,
) -> torch.Tensor:
    """Replace RTC's arm plan with the exact causally rate-limited plan.

    RTC guides the next replan with the normalized leftover.  If ROS executes
    a rate-limited physical prefix while RTC retains the raw policy chunk, its
    continuity constraint is anchored to actions that never happened.
    """
    if normalized_chunk.ndim != 3 or normalized_chunk.shape[0] != 1:
        raise ValueError(f"Unexpected normalized RTC chunk shape {tuple(normalized_chunk.shape)}")
    if physical_chunk.shape != (normalized_chunk.shape[1], EXPECTED_ACTION_DIM):
        raise ValueError(
            f"Physical/normalized chunk mismatch: {physical_chunk.shape} vs "
            f"{tuple(normalized_chunk.shape)}"
        )
    result = normalized_chunk.clone()
    physical_arm = torch.as_tensor(
        physical_chunk[:, :6], device=result.device, dtype=result.dtype
    )
    lo = q01[:6].to(device=result.device, dtype=result.dtype)
    hi = q99[:6].to(device=result.device, dtype=result.dtype)
    result[0, :, :6] = 2.0 * (physical_arm - lo) / (hi - lo) - 1.0
    return result


def _atomic_write_text(path: str, text: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        file.write(text)
    os.replace(tmp_path, path)


def _load_observation(
    requires_release_sensors: bool,
    state_gripper_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float]:
    with open(JOINT_STATE_FILE, encoding="utf-8") as file:
        raw_state = file.read().strip()
    if not raw_state:
        raise ValueError("Joint-state snapshot is empty")

    state = np.asarray([float(value) for value in raw_state.split()], dtype=np.float32)
    camera0 = np.load(CAMERA0_FILE)
    camera1 = np.load(CAMERA1_FILE)

    if state.shape != (EXPECTED_STATE_DIM,):
        raise ValueError(f"Invalid state shape {state.shape}; expected {(EXPECTED_STATE_DIM,)}")
    if camera0.shape != EXPECTED_IMAGE_SHAPE or camera1.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"Invalid image shapes camera0={camera0.shape}, camera1={camera1.shape}; "
            f"expected {EXPECTED_IMAGE_SHAPE}"
        )
    if not all(np.isfinite(array).all() for array in (state, camera0, camera1)):
        raise ValueError("Observation contains NaN or Inf")

    measured_gripper = float(state[6])
    if state_gripper_mode == "binary_threshold_0.12":
        state[6] = 1.0 if state[6] > 0.12 else 0.0
    elif state_gripper_mode == "continuous_0_1_closed_0.629rad":
        state[6] = np.clip(
            state[6] / PHYSICAL_GRIPPER_CLOSED_POSITION_RAD, 0.0, 1.0
        )
    elif state_gripper_mode == "continuous_radians_0_0.8":
        # v8 stores the measured driver joint in physical radians.  A
        # contact-stalled angle is a valid observation, not a semantic bit.
        state[6] = np.clip(state[6], 0.0, 0.8)
    else:
        raise ValueError(f"Unsupported state-gripper mode: {state_gripper_mode}")

    release_sensors: dict[str, np.ndarray] = {}
    if requires_release_sensors:
        sensor_paths = {
            "force": (FORCE_FILE, EXPECTED_FORCE_SHAPE),
            "force_fast": (FORCE_FAST_FILE, EXPECTED_FORCE_FAST_SHAPE),
            "force_slow": (FORCE_SLOW_FILE, EXPECTED_FORCE_SLOW_SHAPE),
            "state_history": (STATE_HISTORY_FILE, EXPECTED_STATE_HISTORY_SHAPE),
        }
        for name, (path, expected_shape) in sensor_paths.items():
            value = np.load(path).astype(np.float32)
            if value.shape != expected_shape:
                raise ValueError(f"Invalid {name} shape {value.shape}; expected {expected_shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
            release_sensors[name] = value
        with open(OBSERVATION_META_FILE, encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("force_reference_mode") != "per_sample_phase_payload_bias_v2":
            raise ValueError("Release sensors use an unexpected force reference")
        if not bool(metadata.get("empty_force_bias_valid", False)):
            raise ValueError("Release sensor empty-tool force bias is not ready")
        if metadata.get("state_history_gripper_units") != "v6_analog_0.100_open_0.629_closed":
            raise ValueError("Release state-history gripper units do not match training")
        for key, minimum in (
            ("force_fast_valid", EXPECTED_FORCE_FAST_SHAPE[0]),
            ("force_slow_valid", EXPECTED_FORCE_SLOW_SHAPE[0]),
            ("state_history_valid", EXPECTED_STATE_HISTORY_SHAPE[0]),
        ):
            if int(metadata.get(key, 0)) < minimum:
                raise ValueError(f"Release sensor history is not ready: {key}")
        release_sensors["observation_simulation_time_s"] = np.asarray(
            metadata.get("simulation_time_s", np.nan), dtype=np.float64
        )

    return state, camera0, camera1, release_sensors, measured_gripper


def _raw_observation(
    state: np.ndarray,
    camera0: np.ndarray,
    camera1: np.ndarray,
    release_sensors: dict[str, np.ndarray] | None = None,
    task: str = TASK,
) -> dict[str, object]:
    observation = {
        "observation.state": torch.from_numpy(state),
        "observation.images.camera0": torch.from_numpy(
            np.ascontiguousarray(np.transpose(camera0, (2, 0, 1)))
        ),
        "observation.images.camera1": torch.from_numpy(
            np.ascontiguousarray(np.transpose(camera1, (2, 0, 1)))
        ),
        "task": task,
    }
    if release_sensors:
        observation.update(
            {
                "observation.force": torch.from_numpy(release_sensors["force"]).unsqueeze(0),
                "observation.force_fast": torch.from_numpy(release_sensors["force_fast"]).unsqueeze(0),
                "observation.force_slow": torch.from_numpy(release_sensors["force_slow"]).unsqueeze(0),
                "observation.state_history": torch.from_numpy(release_sensors["state_history"]).unsqueeze(0),
            }
        )
    return observation


def _rate_limit_arm_chunk(
    action_chunk: np.ndarray,
    current_state: np.ndarray,
    max_step_rad: float,
) -> np.ndarray:
    """Causally limit absolute arm targets while leaving the gripper intact."""
    limited = action_chunk.copy()
    if max_step_rad <= 0:
        return limited
    previous = current_state[:6].astype(np.float32, copy=True)
    for index in range(len(limited)):
        delta = np.clip(limited[index, :6] - previous, -max_step_rad, max_step_rad)
        limited[index, :6] = previous + delta
        previous = limited[index, :6]
    return limited


def _scale_physical_arm_residual(
    action_chunk: np.ndarray,
    current_state: np.ndarray,
    residual_scale: float,
) -> np.ndarray:
    """Scale decoded absolute targets around the measured arm state in radians."""
    scaled = action_chunk.copy()
    scaled[:, :6] = current_state[None, :6] + residual_scale * (action_chunk[:, :6] - current_state[None, :6])
    return scaled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--state-gripper-mode",
        choices=(
            "binary_threshold_0.12",
            "continuous_0_1_closed_0.629rad",
            "continuous_radians_0_0.8",
        ),
        default="continuous_radians_0_0.8",
        help="Online observation contract used when the checkpoint was trained.",
    )
    parser.add_argument(
        "--action-out-adapter",
        type=Path,
        default=None,
        help="Optional safetensors delta containing only model.action_out_proj parameters.",
    )
    parser.add_argument(
        "--lora-adapter",
        type=Path,
        default=None,
        help="Optional PEFT adapter directory applied after --action-out-adapter.",
    )
    parser.add_argument("--stage-lora-adapter", type=Path, default=None)
    parser.add_argument("--stage-lora-reference", type=float, nargs=6, default=None)
    parser.add_argument("--stage-lora-threshold", type=float, default=0.15)
    parser.add_argument("--final-stage-lora-adapter", type=Path, default=None)
    parser.add_argument("--final-stage-lora-reference", type=float, nargs=6, default=None)
    parser.add_argument("--final-stage-lora-threshold", type=float, default=0.15)
    parser.add_argument("--grasp-stage-lora-adapter", type=Path, default=None)
    parser.add_argument("--grasp-stage-lora-reference", type=float, nargs=6, default=None)
    parser.add_argument("--grasp-stage-lora-threshold", type=float, default=0.10)
    parser.add_argument("--post-grasp-lora-adapter", type=Path, default=None)
    parser.add_argument("--post-grasp-lora-reference", type=float, nargs=6, default=None)
    parser.add_argument("--post-grasp-lora-threshold", type=float, default=0.15)
    parser.add_argument("--insertion-lora-adapter", type=Path, default=None)
    parser.add_argument("--insertion-lora-reference", type=float, nargs=6, default=None)
    parser.add_argument("--insertion-lora-threshold", type=float, default=0.10)
    parser.add_argument("--contact-lora-adapter", type=Path, default=None)
    parser.add_argument("--contact-lora-reference", type=float, nargs=6, default=None)
    parser.add_argument("--contact-lora-threshold", type=float, default=0.10)
    parser.add_argument(
        "--insertion-feedback-adapter",
        type=Path,
        default=None,
        help=(
            "Optional learned joint/FT300 feedback policy activated only after "
            "the contact-stage FSM latches. It must declare online_object_truth=false."
        ),
    )
    parser.add_argument("--gripper-close-reference", type=float, nargs=6, default=None)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.0)
    parser.add_argument("--gripper-open-reference", type=float, nargs=6, default=None)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.0)
    parser.add_argument("--hold-arm-on-close-transition", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--demo-recovery-episode", type=Path)
    parser.add_argument("--demo-recovery-after-chunks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fixed-noise-per-replan",
        action="store_true",
        help=(
            "Reset flow-matching noise to --seed at every replan. This is a "
            "deterministic diagnostic for separating sampling jitter from an "
            "incorrect observation-conditioned action fixed point."
        ),
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Average this many independently sampled physical action chunks.",
    )
    parser.add_argument(
        "--max-arm-step-rad",
        type=float,
        default=0.0,
        help="Causal per-joint absolute-target limit; zero disables it.",
    )
    parser.add_argument(
        "--physical-arm-residual-scale",
        type=float,
        default=1.0,
        help=(
            "Scale decoded absolute arm targets around the measured physical "
            "joint state before causal rate limiting."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Optionally archive the first online observations and executed chunks.",
    )
    parser.add_argument(
        "--trace-first-chunks",
        type=int,
        default=0,
        help="Number of initial replans to archive under --trace-dir.",
    )
    parser.add_argument(
        "--trace-multimodal",
        action="store_true",
        help=(
            "Archive FT300 fast/slow windows and state history for offline shadow "
            "evaluation. These fields are not added to the Pi0.5 policy input."
        ),
    )
    parser.add_argument(
        "--hold-gripper-per-replan",
        action="store_true",
        help=(
            "Use only the chunk's first binary gripper decision and hold it "
            "through the execution horizon, while retaining all arm targets."
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
    parser.add_argument(
        "--rtc-consumed-steps",
        type=int,
        default=None,
        help="Actions actually consumed by ROS before replanning; defaults to n_action_steps.",
    )
    parser.add_argument("--task-prompt", default=TASK)
    parser.add_argument(
        "--enable-dynamic-task-prompt",
        action="store_true",
        help="Disable the saved fixed-global-task override and use --task-prompt.",
    )
    args = parser.parse_args()
    if args.ensemble_size < 1:
        raise ValueError("--ensemble-size must be at least 1")
    if not np.isfinite(args.max_arm_step_rad) or args.max_arm_step_rad < 0:
        raise ValueError("--max-arm-step-rad must be finite and non-negative")
    if not np.isfinite(args.physical_arm_residual_scale) or not 0 < args.physical_arm_residual_scale <= 1:
        raise ValueError("--physical-arm-residual-scale must be finite and in (0, 1]")
    if args.trace_first_chunks < 0:
        raise ValueError("--trace-first-chunks must be non-negative")
    if args.trace_first_chunks and args.trace_dir is None:
        raise ValueError("--trace-first-chunks requires --trace-dir")
    if (args.stage_lora_adapter is None) != (args.stage_lora_reference is None):
        raise ValueError("stage LoRA adapter and six-joint reference must be provided together")
    if args.stage_lora_adapter is not None and args.lora_adapter is None:
        raise ValueError("--stage-lora-adapter requires a primary --lora-adapter")
    if not np.isfinite(args.stage_lora_threshold) or args.stage_lora_threshold <= 0:
        raise ValueError("--stage-lora-threshold must be finite and positive")
    if (args.grasp_stage_lora_adapter is None) != (args.grasp_stage_lora_reference is None):
        raise ValueError("grasp-stage LoRA adapter and six-joint reference must be provided together")
    if args.grasp_stage_lora_adapter is not None and args.final_stage_lora_adapter is None:
        raise ValueError("--grasp-stage-lora-adapter requires --final-stage-lora-adapter")
    if not np.isfinite(args.grasp_stage_lora_threshold) or args.grasp_stage_lora_threshold <= 0:
        raise ValueError("--grasp-stage-lora-threshold must be finite and positive")
    if (args.insertion_lora_adapter is None) != (args.insertion_lora_reference is None):
        raise ValueError("insertion LoRA adapter and six-joint reference must be provided together")
    if args.insertion_lora_adapter is not None and args.post_grasp_lora_adapter is None:
        raise ValueError("--insertion-lora-adapter requires --post-grasp-lora-adapter")
    if not np.isfinite(args.insertion_lora_threshold) or args.insertion_lora_threshold <= 0:
        raise ValueError("--insertion-lora-threshold must be finite and positive")
    if (args.contact_lora_adapter is None) != (args.contact_lora_reference is None):
        raise ValueError("contact LoRA adapter and six-joint reference must be provided together")
    if args.contact_lora_adapter is not None and args.insertion_lora_adapter is None:
        raise ValueError("--contact-lora-adapter requires --insertion-lora-adapter")
    if not np.isfinite(args.contact_lora_threshold) or args.contact_lora_threshold <= 0:
        raise ValueError("--contact-lora-threshold must be finite and positive")
    if args.rtc_execution_horizon < 1:
        raise ValueError("--rtc-execution-horizon must be positive")
    if not np.isfinite(args.rtc_max_guidance_weight) or args.rtc_max_guidance_weight <= 0:
        raise ValueError("--rtc-max-guidance-weight must be finite and positive")
    if args.rtc and args.ensemble_size != 1:
        raise ValueError("RTC currently requires --ensemble-size=1")
    if args.rtc and (
        args.physical_arm_residual_scale != 1.0
        or args.hold_gripper_per_replan
    ):
        raise ValueError(
            "RTC comparison requires unit physical residual scaling and no "
            "per-replan gripper hold"
        )
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

    # Never let the ROS side mistake metadata from an older inference process
    # for readiness of the checkpoint selected in this invocation.
    if not args.validate_only and os.path.exists(READY_FILE):
        os.remove(READY_FILE)

    checkpoint = Path(args.checkpoint)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Pi0.5 config not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        checkpoint_config = json.load(file)
    if checkpoint_config.get("type") != "pi05":
        raise ValueError(f"{checkpoint} is type={checkpoint_config.get('type')!r}; expected 'pi05'")
    try:
        validate_action_semantics(checkpoint, checkpoint_config)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Checkpoint validation failed: {error}") from None
    if args.validate_only:
        print(f"Checkpoint validation passed: {checkpoint}")
        return

    requires_release_sensors = bool(checkpoint_config.get("use_release_gripper_override", False))
    trace_requires_release_sensors = bool(
        args.trace_multimodal and args.trace_dir is not None and args.trace_first_chunks > 0
    )

    predicted_action_steps = int(checkpoint_config.get("chunk_size", 50))
    executed_action_steps = int(checkpoint_config.get("n_action_steps", predicted_action_steps))
    # The online runner may intentionally evaluate a longer prefix than the
    # checkpoint's training-time n_action_steps.  The policy still predicts
    # the full chunk; this explicit value is the number ROS actually executes
    # before asking for a new observation.  Previously the override was only
    # honored with demonstration recovery, which made the same pure-policy
    # experiment fail its readiness contract before control started.
    if args.rtc_consumed_steps is not None:
        executed_action_steps = args.rtc_consumed_steps
    if not 1 <= executed_action_steps <= predicted_action_steps:
        raise ValueError(
            "Invalid checkpoint action horizon: "
            f"n_action_steps={executed_action_steps}, "
            f"chunk_size={predicted_action_steps}"
        )
    rtc_consumed_steps = (
        executed_action_steps if args.rtc_consumed_steps is None else args.rtc_consumed_steps
    )
    if args.rtc and not 1 <= rtc_consumed_steps < predicted_action_steps:
        raise ValueError(
            f"--rtc-consumed-steps must be in [1, {predicted_action_steps - 1}]"
        )
    if args.rtc and args.rtc_execution_horizon > predicted_action_steps - rtc_consumed_steps:
        raise ValueError(
            "--rtc-execution-horizon exceeds the previous chunk remainder: "
            f"{args.rtc_execution_horizon} > {predicted_action_steps - rtc_consumed_steps}"
        )

    sys.path.insert(0, LEROBOT_SRC)
    import lerobot.policies.pi05.processor_pi05  # noqa: F401
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    print(f"Validating Pi0.5 checkpoint: {checkpoint}", flush=True)
    # Native LeRobot PEFT checkpoints contain an adapter rather than a duplicate
    # 8.8 GB base model. Reconstruct the exact saved Pi0.5 config, strict-load
    # its declared base, then attach the adapter through PEFT's official path.
    adapter_config_path = checkpoint / "adapter_config.json"
    if adapter_config_path.is_file():
        from peft import PeftModel

        with adapter_config_path.open(encoding="utf-8") as file:
            adapter_config = json.load(file)
        base_model_path = Path(adapter_config["base_model_name_or_path"])
        if not (base_model_path / "model.safetensors").is_file():
            raise FileNotFoundError(f"PEFT base model is missing: {base_model_path}")
        policy_config = PreTrainedConfig.from_pretrained(str(checkpoint))
        policy = PI05Policy.from_pretrained(
            str(base_model_path), config=policy_config, strict=True
        )
        policy = PeftModel.from_pretrained(
            policy, str(checkpoint), is_trainable=False
        )
        print(
            f"Loaded native LeRobot PEFT adapter: {checkpoint} "
            f"(base={base_model_path})",
            flush=True,
        )
    else:
        # A baseline checkpoint must match the declared Pi0.5 architecture
        # exactly. Silent missing/unexpected tensors make the comparison invalid.
        policy = PI05Policy.from_pretrained(str(checkpoint), strict=True)
    if args.action_out_adapter is not None:
        from safetensors.torch import load_file

        adapter_path = args.action_out_adapter.resolve()
        if not adapter_path.is_file():
            raise FileNotFoundError(f"Action-output adapter not found: {adapter_path}")
        adapter = load_file(str(adapter_path), device="cpu")
        expected_names = {
            "model.action_out_proj.weight",
            "model.action_out_proj.bias",
        }
        if set(adapter) != expected_names:
            raise ValueError(
                f"Unexpected action-output adapter keys: {sorted(adapter)}; "
                f"expected={sorted(expected_names)}"
            )
        parameters = dict(policy.named_parameters())
        with torch.no_grad():
            for name, value in adapter.items():
                if value.shape != parameters[name].shape:
                    raise ValueError(
                        f"Adapter shape mismatch for {name}: {value.shape} != {parameters[name].shape}"
                    )
                parameters[name].copy_(value.to(dtype=parameters[name].dtype))
        print(f"Loaded action-output adapter: {adapter_path}", flush=True)
    grasp_stage_local_action_out_path = None
    grasp_stage_local_action_out = None
    post_grasp_local_action_out_path = None
    post_grasp_local_action_out = None
    insertion_local_action_out_path = None
    insertion_local_action_out = None
    contact_local_action_out_path = None
    contact_local_action_out = None
    base_action_out = None
    base_action_out_state = None
    if args.lora_adapter is not None:
        from peft import PeftModel

        lora_path = args.lora_adapter.resolve()
        for required in (lora_path / "adapter_config.json", lora_path / "adapter_model.safetensors"):
            if not required.is_file():
                raise FileNotFoundError(f"LoRA adapter file not found: {required}")
        policy = PeftModel.from_pretrained(policy, str(lora_path), is_trainable=False)
        print(f"Loaded LoRA adapter: {lora_path}", flush=True)
        if args.stage_lora_adapter is not None:
            stage_lora_path = args.stage_lora_adapter.resolve()
            for required in (
                stage_lora_path / "adapter_config.json",
                stage_lora_path / "adapter_model.safetensors",
            ):
                if not required.is_file():
                    raise FileNotFoundError(f"Stage LoRA adapter file not found: {required}")
            policy.load_adapter(str(stage_lora_path), adapter_name="stage")
            policy.set_adapter("default")
            print(f"Loaded state-routed stage LoRA adapter: {stage_lora_path}", flush=True)
            if args.final_stage_lora_adapter is not None:
                final_path = args.final_stage_lora_adapter.resolve()
                for required in (
                    final_path / "adapter_config.json",
                    final_path / "adapter_model.safetensors",
                ):
                    if not required.is_file():
                        raise FileNotFoundError(f"Final-stage LoRA file not found: {required}")
                policy.load_adapter(str(final_path), adapter_name="final_stage")
                policy.set_adapter("default")
                print(f"Loaded final state-routed LoRA adapter: {final_path}", flush=True)
            if args.grasp_stage_lora_adapter is not None:
                grasp_stage_path = args.grasp_stage_lora_adapter.resolve()
                for required in (
                    grasp_stage_path / "adapter_config.json",
                    grasp_stage_path / "adapter_model.safetensors",
                ):
                    if not required.is_file():
                        raise FileNotFoundError(f"Grasp-stage LoRA file not found: {required}")
                policy.load_adapter(str(grasp_stage_path), adapter_name="grasp_stage")
                policy.set_adapter("default")
                print(f"Loaded grasp state-routed LoRA adapter: {grasp_stage_path}", flush=True)
                local_projection_path = grasp_stage_path / "adapter_local_action_out.safetensors"
                if local_projection_path.is_file():
                    from safetensors.torch import load_file

                    grasp_stage_local_action_out = load_file(
                        str(local_projection_path), device="cpu"
                    )
                    expected = {"weight", "bias"}
                    if set(grasp_stage_local_action_out) != expected:
                        raise ValueError(
                            "Unexpected grasp-stage adapter-local action_out keys: "
                            f"{sorted(grasp_stage_local_action_out)}"
                        )
                    action_out = policy.base_model.model.model.action_out_proj
                    base_action_out = getattr(action_out, "base_layer", action_out)
                    base_action_out_state = {
                        name: getattr(base_action_out, name).detach().clone()
                        for name in expected
                    }
                    for name, value in grasp_stage_local_action_out.items():
                        parameter = getattr(base_action_out, name)
                        if parameter.shape != value.shape:
                            raise ValueError(
                                f"Grasp-stage local action_out shape mismatch for {name}: "
                                f"{value.shape} != {parameter.shape}"
                            )
                        grasp_stage_local_action_out[name] = value.to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    grasp_stage_local_action_out_path = local_projection_path.resolve()
                    print(
                        "Loaded grasp-stage adapter-local action_out projection: "
                        f"{grasp_stage_local_action_out_path}",
                        flush=True,
                    )
            if args.post_grasp_lora_adapter is not None:
                post_grasp_path = args.post_grasp_lora_adapter.resolve()
                for required in (
                    post_grasp_path / "adapter_config.json",
                    post_grasp_path / "adapter_model.safetensors",
                ):
                    if not required.is_file():
                        raise FileNotFoundError(f"Post-grasp LoRA file not found: {required}")
                policy.load_adapter(str(post_grasp_path), adapter_name="post_grasp")
                policy.set_adapter("default")
                print(f"Loaded post-grasp LoRA adapter: {post_grasp_path}", flush=True)
                local_projection_path = post_grasp_path / "adapter_local_action_out.safetensors"
                if local_projection_path.is_file():
                    from safetensors.torch import load_file

                    post_grasp_local_action_out = load_file(
                        str(local_projection_path), device="cpu"
                    )
                    expected = {"weight", "bias"}
                    if set(post_grasp_local_action_out) != expected:
                        raise ValueError(
                            "Unexpected post-grasp adapter-local action_out keys: "
                            f"{sorted(post_grasp_local_action_out)}"
                        )
                    action_out = policy.base_model.model.model.action_out_proj
                    base_action_out = getattr(action_out, "base_layer", action_out)
                    if base_action_out_state is None:
                        base_action_out_state = {
                            name: getattr(base_action_out, name).detach().clone()
                            for name in expected
                        }
                    for name, value in post_grasp_local_action_out.items():
                        parameter = getattr(base_action_out, name)
                        if parameter.shape != value.shape:
                            raise ValueError(
                                f"Post-grasp local action_out shape mismatch for {name}: "
                                f"{value.shape} != {parameter.shape}"
                            )
                        post_grasp_local_action_out[name] = value.to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    post_grasp_local_action_out_path = local_projection_path.resolve()
                    print(
                        "Loaded post-grasp adapter-local action_out projection: "
                        f"{post_grasp_local_action_out_path}",
                        flush=True,
                    )
            if args.insertion_lora_adapter is not None:
                insertion_path = args.insertion_lora_adapter.resolve()
                for required in (
                    insertion_path / "adapter_config.json",
                    insertion_path / "adapter_model.safetensors",
                ):
                    if not required.is_file():
                        raise FileNotFoundError(f"Insertion LoRA file not found: {required}")
                policy.load_adapter(str(insertion_path), adapter_name="insertion")
                policy.set_adapter("default")
                print(f"Loaded insertion LoRA adapter: {insertion_path}", flush=True)
                local_projection_path = insertion_path / "adapter_local_action_out.safetensors"
                if local_projection_path.is_file():
                    from safetensors.torch import load_file

                    insertion_local_action_out = load_file(
                        str(local_projection_path), device="cpu"
                    )
                    expected = {"weight", "bias"}
                    if set(insertion_local_action_out) != expected:
                        raise ValueError(
                            "Unexpected insertion adapter-local action_out keys: "
                            f"{sorted(insertion_local_action_out)}"
                        )
                    action_out = policy.base_model.model.model.action_out_proj
                    base_action_out = getattr(action_out, "base_layer", action_out)
                    if base_action_out_state is None:
                        base_action_out_state = {
                            name: getattr(base_action_out, name).detach().clone()
                            for name in expected
                        }
                    for name, value in insertion_local_action_out.items():
                        parameter = getattr(base_action_out, name)
                        if parameter.shape != value.shape:
                            raise ValueError(
                                f"Insertion local action_out shape mismatch for {name}: "
                                f"{value.shape} != {parameter.shape}"
                            )
                        insertion_local_action_out[name] = value.to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    insertion_local_action_out_path = local_projection_path.resolve()
                    print(
                        "Loaded insertion adapter-local action_out projection: "
                        f"{insertion_local_action_out_path}",
                        flush=True,
                    )
            if args.contact_lora_adapter is not None:
                contact_path = args.contact_lora_adapter.resolve()
                for required in (
                    contact_path / "adapter_config.json",
                    contact_path / "adapter_model.safetensors",
                ):
                    if not required.is_file():
                        raise FileNotFoundError(f"Contact LoRA file not found: {required}")
                policy.load_adapter(str(contact_path), adapter_name="contact")
                policy.set_adapter("default")
                print(f"Loaded contact LoRA adapter: {contact_path}", flush=True)
                local_projection_path = contact_path / "adapter_local_action_out.safetensors"
                if local_projection_path.is_file():
                    from safetensors.torch import load_file

                    contact_local_action_out = load_file(
                        str(local_projection_path), device="cpu"
                    )
                    expected = {"weight", "bias"}
                    if set(contact_local_action_out) != expected:
                        raise ValueError(
                            "Unexpected contact adapter-local action_out keys: "
                            f"{sorted(contact_local_action_out)}"
                        )
                    action_out = policy.base_model.model.model.action_out_proj
                    base_action_out = getattr(action_out, "base_layer", action_out)
                    if base_action_out_state is None:
                        base_action_out_state = {
                            name: getattr(base_action_out, name).detach().clone()
                            for name in expected
                        }
                    for name, value in contact_local_action_out.items():
                        parameter = getattr(base_action_out, name)
                        if parameter.shape != value.shape:
                            raise ValueError(
                                f"Contact local action_out shape mismatch for {name}: "
                                f"{value.shape} != {parameter.shape}"
                            )
                        contact_local_action_out[name] = value.to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    contact_local_action_out_path = local_projection_path.resolve()
                    print(
                        "Loaded contact adapter-local action_out projection: "
                        f"{contact_local_action_out_path}",
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
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
    )
    rtc_action_q01 = rtc_action_q99 = None
    if args.rtc and args.max_arm_step_rad > 0:
        rtc_action_q01, rtc_action_q99 = _load_action_quantile_bounds(checkpoint)
    if args.enable_dynamic_task_prompt:
        dynamic_prompt_steps = 0
        for step in preprocessor.steps:
            if hasattr(step, "global_task"):
                step.global_task = None
                dynamic_prompt_steps += 1
        if dynamic_prompt_steps != 1:
            raise ValueError(
                "Expected exactly one saved global-task override step, "
                f"found {dynamic_prompt_steps}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device=device)
    policy.eval()
    insertion_feedback_model = None
    insertion_feedback_terminal = None
    insertion_feedback_force_baseline_n = None
    insertion_feedback_force_full_scale_n = None
    insertion_contact_model = None
    insertion_contact_force_start_n = 0.0
    insertion_contact_force_full_n = 0.0
    if args.insertion_feedback_adapter is not None:
        feedback_path = args.insertion_feedback_adapter.resolve()
        if not feedback_path.is_file():
            raise FileNotFoundError(f"Insertion feedback adapter not found: {feedback_path}")
        feedback = torch.load(feedback_path, map_location="cpu", weights_only=False)
        if feedback.get("online_object_truth") is not False:
            raise ValueError("Insertion feedback adapter must explicitly prohibit object truth")
        insertion_feedback_model = InsertionFeedbackAdapter(
            hidden_dim=int(feedback["hidden_dim"]),
            max_delta_rad=float(feedback["max_delta_rad"]),
        )
        insertion_feedback_model.load_state_dict(feedback["state_dict"])
        insertion_feedback_model.to(device=device).eval()
        insertion_feedback_terminal = np.asarray(
            feedback["terminal_arm_state"], dtype=np.float32
        )
        if insertion_feedback_terminal.shape != (6,):
            raise ValueError("Insertion feedback terminal state must be [6]")
        insertion_feedback_force_baseline_n = float(feedback["force_baseline_n"])
        insertion_feedback_force_full_scale_n = float(feedback["force_full_scale_n"])
        if "contact_state_dict" in feedback:
            insertion_contact_model = InsertionFeedbackAdapter(
                hidden_dim=int(feedback["contact_hidden_dim"]),
                max_delta_rad=float(feedback["max_delta_rad"]),
            )
            insertion_contact_model.load_state_dict(feedback["contact_state_dict"])
            insertion_contact_model.to(device=device).eval()
            insertion_contact_force_start_n = float(feedback["contact_force_start_n"])
            insertion_contact_force_full_n = float(feedback["contact_force_full_n"])
        print(
            "Loaded learned insertion feedback adapter (joint/FT300 only, no object truth): "
            f"{feedback_path}",
            flush=True,
        )
        if insertion_contact_model is not None:
            print(
                "Loaded learned high-force contact stabilizer: "
                f"blend {insertion_contact_force_start_n:.1f}--"
                f"{insertion_contact_force_full_n:.1f} N",
                flush=True,
            )
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    # RTC remains arm-only.  A 7-D experiment showed that constraining the
    # continuous gripper to a still-inaccurate long-horizon leftover can force
    # an early close before the replanned arm reaches the peg.  Gripper
    # continuity must therefore be learned in the executed prefix rather than
    # imposed from an inaccurate 50-step plan.
    rtc_action_mask = torch.tensor(
        [1.0] * (EXPECTED_ACTION_DIM - 1) + [0.0], device=device
    )

    # Complete one full inference path before declaring readiness.  The first
    # CUDA pass on this workstation takes several seconds; without this warmup
    # an upright, slender peg can topple while the ROS side waits for action 0.
    warmup_state = np.asarray(
        [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0, 0.0],
        dtype=np.float32,
    )
    if args.state_gripper_mode == "continuous_0_1_closed_0.629rad":
        warmup_state[6] = 0.100 / PHYSICAL_GRIPPER_CLOSED_POSITION_RAD
    warmup_image = np.zeros(EXPECTED_IMAGE_SHAPE, dtype=np.float32)
    warmup_release_sensors = None
    if requires_release_sensors:
        warmup_history_state = warmup_state.copy()
        # The frame-level policy state is binary, while the auxiliary history
        # retains the v6 dataset's analog gripper convention.
        warmup_history_state[6] = 0.100
        warmup_release_sensors = {
            "force": np.zeros(EXPECTED_FORCE_SHAPE, dtype=np.float32),
            "force_fast": np.zeros(EXPECTED_FORCE_FAST_SHAPE, dtype=np.float32),
            "force_slow": np.zeros(EXPECTED_FORCE_SLOW_SHAPE, dtype=np.float32),
            "state_history": np.tile(warmup_history_state, (EXPECTED_STATE_HISTORY_SHAPE[0], 1)),
        }
    warmup_batch = preprocessor(
        _raw_observation(
            warmup_state,
            warmup_image,
            warmup_image,
            warmup_release_sensors,
            args.task_prompt,
        )
    )
    warmup_start = time.perf_counter()
    inference_context = torch.no_grad if args.rtc else torch.inference_mode
    with inference_context():
        warmup_chunk = policy.predict_action_chunk(warmup_batch)
        postprocess_action_chunk(warmup_chunk, postprocessor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(
        f"Pi0.5 warmup complete in {(time.perf_counter() - warmup_start) * 1000.0:.1f} ms.",
        flush=True,
    )
    # Warmup must not consume the evaluation's requested stochastic seed.
    torch.manual_seed(args.seed)

    ready_metadata = {
        "policy_type": "pi05",
        "checkpoint": str(checkpoint),
        "predicted_action_steps": predicted_action_steps,
        "executed_action_steps": executed_action_steps,
        "action_dt_s": 0.1,
        "seed": args.seed,
        "fixed_noise_per_replan": args.fixed_noise_per_replan,
        "ensemble_size": args.ensemble_size,
        "physical_arm_residual_scale": args.physical_arm_residual_scale,
        "max_arm_step_rad": args.max_arm_step_rad,
        "trace_first_chunks": args.trace_first_chunks,
        "trace_multimodal": args.trace_multimodal,
        "trace_multimodal_affects_policy_input": False,
        "action_postprocessing": (
            "physical_mean_then_measured_state_residual_scale_then_causal_arm_limit_then_continuous_gripper"
            if args.state_gripper_mode == "continuous_radians_0_0.8"
            else "physical_mean_then_measured_state_residual_scale_then_causal_arm_limit_then_binary_gripper"
        ),
        "gripper_action_mode": (
            "continuous_radians_0_0.8"
            if args.state_gripper_mode == "continuous_radians_0_0.8"
            else (
                "binary_executed_prefix_hold_per_replan"
                if args.hold_gripper_per_replan
                else "binary_threshold_0.5"
            )
        ),
        "state_gripper_mode": args.state_gripper_mode,
        "physical_gripper_closed_threshold": PHYSICAL_GRIPPER_CLOSED_THRESHOLD,
        "release_gripper_override": requires_release_sensors,
        "release_head_probability_threshold": checkpoint_config.get("release_head_probability_threshold"),
        "rtc_enabled": args.rtc,
        "rtc_execution_horizon": args.rtc_execution_horizon if args.rtc else None,
        "rtc_max_guidance_weight": args.rtc_max_guidance_weight if args.rtc else None,
        "rtc_prefix_attention_schedule": (
            args.rtc_prefix_attention_schedule if args.rtc else None
        ),
        "rtc_consumed_steps": rtc_consumed_steps if args.rtc else None,
        "rtc_inference_delay_steps": 0 if args.rtc else None,
        "rtc_action_dimensions": "arm_only" if args.rtc else None,
        "task_prompt": args.task_prompt,
        "dynamic_task_prompt_enabled": args.enable_dynamic_task_prompt,
        "action_out_adapter": (
            str(args.action_out_adapter.resolve()) if args.action_out_adapter is not None else None
        ),
        "lora_adapter": str(args.lora_adapter.resolve()) if args.lora_adapter is not None else None,
        "stage_lora_adapter": (
            str(args.stage_lora_adapter.resolve()) if args.stage_lora_adapter is not None else None
        ),
        "stage_lora_reference": args.stage_lora_reference,
        "stage_lora_threshold": args.stage_lora_threshold,
        "final_stage_lora_adapter": (
            str(args.final_stage_lora_adapter.resolve())
            if args.final_stage_lora_adapter is not None else None
        ),
        "final_stage_lora_reference": args.final_stage_lora_reference,
        "final_stage_lora_threshold": args.final_stage_lora_threshold,
        "grasp_stage_lora_adapter": (
            str(args.grasp_stage_lora_adapter.resolve())
            if args.grasp_stage_lora_adapter is not None else None
        ),
        "grasp_stage_lora_reference": args.grasp_stage_lora_reference,
        "grasp_stage_lora_threshold": args.grasp_stage_lora_threshold,
        "grasp_stage_local_action_out": (
            str(grasp_stage_local_action_out_path)
            if grasp_stage_local_action_out_path is not None else None
        ),
        "post_grasp_lora_adapter": (
            str(args.post_grasp_lora_adapter.resolve())
            if args.post_grasp_lora_adapter is not None else None
        ),
        "post_grasp_lora_reference": args.post_grasp_lora_reference,
        "post_grasp_lora_threshold": args.post_grasp_lora_threshold,
        "post_grasp_local_action_out": (
            str(post_grasp_local_action_out_path)
            if post_grasp_local_action_out_path is not None else None
        ),
        "insertion_lora_adapter": (
            str(args.insertion_lora_adapter.resolve())
            if args.insertion_lora_adapter is not None else None
        ),
        "insertion_lora_reference": args.insertion_lora_reference,
        "insertion_lora_threshold": args.insertion_lora_threshold,
        "insertion_local_action_out": (
            str(insertion_local_action_out_path)
            if insertion_local_action_out_path is not None else None
        ),
        "contact_lora_adapter": (
            str(args.contact_lora_adapter.resolve())
            if args.contact_lora_adapter is not None else None
        ),
        "contact_lora_reference": args.contact_lora_reference,
        "contact_lora_threshold": args.contact_lora_threshold,
        "contact_local_action_out": (
            str(contact_local_action_out_path)
            if contact_local_action_out_path is not None else None
        ),
        "insertion_feedback_adapter": (
            str(args.insertion_feedback_adapter.resolve())
            if args.insertion_feedback_adapter is not None else None
        ),
        "insertion_feedback_online_object_truth": False,
        "gripper_close_reference": args.gripper_close_reference,
        "gripper_close_threshold": args.gripper_close_threshold,
        "hold_arm_on_close_transition": args.hold_arm_on_close_transition,
    }
    _atomic_write_text(READY_FILE, json.dumps(ready_metadata) + "\n")
    print("Pi0.5 inference ready.", flush=True)

    while not os.path.exists(JOINT_STATE_FILE):
        time.sleep(0.01)
    # Process the snapshot that released the wait above.  Initializing this to
    # the file's current mtime drops observation 0 and forces a needless ROS
    # timeout before the next snapshot is written.
    last_observation_mtime = None
    chunk_id = 0
    previous_normalized_leftover = None
    previous_physical_chunk = None
    demo_recovery = (
        DemonstrationStateRecovery(
            args.demo_recovery_episode, predicted_action_steps, executed_action_steps
        )
        if args.demo_recovery_episode is not None
        else None
    )

    active_lora_adapter = "default"
    stage_lora_started = False
    stage_lora_completed = False
    final_stage_started = False
    grasp_stage_started = False
    post_grasp_started = False
    insertion_started = False
    contact_started = False
    stage_lora_reference = (
        np.asarray(args.stage_lora_reference, dtype=np.float32)
        if args.stage_lora_reference is not None
        else None
    )
    final_stage_reference = (
        np.asarray(args.final_stage_lora_reference, dtype=np.float32)
        if args.final_stage_lora_reference is not None
        else None
    )
    grasp_stage_reference = (
        np.asarray(args.grasp_stage_lora_reference, dtype=np.float32)
        if args.grasp_stage_lora_reference is not None else None
    )
    post_grasp_reference = (
        np.asarray(args.post_grasp_lora_reference, dtype=np.float32)
        if args.post_grasp_lora_reference is not None else None
    )
    insertion_reference = (
        np.asarray(args.insertion_lora_reference, dtype=np.float32)
        if args.insertion_lora_reference is not None else None
    )
    contact_reference = (
        np.asarray(args.contact_lora_reference, dtype=np.float32)
        if args.contact_lora_reference is not None else None
    )
    while True:
        try:
            observation_mtime = os.path.getmtime(JOINT_STATE_FILE)
            if observation_mtime == last_observation_mtime:
                time.sleep(0.001)
                continue
            last_observation_mtime = observation_mtime
            request_id = action_exchange.observation_id(JOINT_STATE_FILE)

            state, camera0, camera1, release_sensors, measured_gripper = _load_observation(
                requires_release_sensors or trace_requires_release_sensors,
                args.state_gripper_mode,
            )
            if action_exchange.enabled() and request_id != action_exchange.observation_id(JOINT_STATE_FILE):
                raise RuntimeError("Observation changed during load; refusing unpaired inference")
            measured_gripper_closed = measured_gripper >= PHYSICAL_GRIPPER_CLOSED_THRESHOLD
            if stage_lora_reference is not None:
                distance = float(np.linalg.norm(state[:6] - stage_lora_reference))
                final_distance = (
                    float(np.linalg.norm(state[:6] - final_stage_reference))
                    if final_stage_reference is not None else None
                )
                grasp_distance = (
                    float(np.linalg.norm(state[:6] - grasp_stage_reference))
                    if grasp_stage_reference is not None else None
                )
                # A tiny simulated joint drift above the dataset's 0.12
                # binarization threshold is not a completed physical grasp.
                # Keep stage routing latched until the fingers have moved well
                # into the 0.100-open / 0.629-closed physical range.
                if measured_gripper_closed:
                    stage_lora_completed = True
                elif distance <= args.stage_lora_threshold:
                    stage_lora_started = True
                if (
                    stage_lora_started
                    and final_distance is not None
                    and final_distance <= args.final_stage_lora_threshold
                ):
                    final_stage_started = True
                if (
                    final_stage_started
                    and grasp_distance is not None
                    and grasp_distance <= args.grasp_stage_lora_threshold
                ):
                    grasp_stage_started = True
                post_grasp_distance = (
                    float(np.linalg.norm(state[:6] - post_grasp_reference))
                    if post_grasp_reference is not None else None
                )
                if (
                    stage_lora_completed
                    and post_grasp_distance is not None
                    and post_grasp_distance <= args.post_grasp_lora_threshold
                ):
                    post_grasp_started = True
                insertion_distance = (
                    float(np.linalg.norm(state[:6] - insertion_reference))
                    if insertion_reference is not None else None
                )
                if (
                    post_grasp_started
                    and insertion_distance is not None
                    and insertion_distance <= args.insertion_lora_threshold
                ):
                    insertion_started = True
                contact_distance = (
                    float(np.linalg.norm(state[:6] - contact_reference))
                    if contact_reference is not None else None
                )
                if (
                    insertion_started
                    and contact_distance is not None
                    and contact_distance <= args.contact_lora_threshold
                ):
                    contact_started = True
                if stage_lora_completed:
                    desired_adapter = (
                        "contact"
                        if args.contact_lora_adapter is not None and contact_started
                        else (
                            "insertion"
                            if args.insertion_lora_adapter is not None and insertion_started
                            else (
                                "post_grasp"
                                if args.post_grasp_lora_adapter is not None and post_grasp_started
                                else "default"
                            )
                        )
                    )
                elif grasp_stage_started:
                    desired_adapter = "grasp_stage"
                elif final_stage_started:
                    desired_adapter = "final_stage"
                elif stage_lora_started:
                    desired_adapter = "stage"
                else:
                    desired_adapter = "default"
                if desired_adapter != active_lora_adapter:
                    using_local_action_out = False
                    if base_action_out is not None:
                        if desired_adapter == "grasp_stage":
                            projection_state = grasp_stage_local_action_out
                        elif desired_adapter == "post_grasp":
                            projection_state = post_grasp_local_action_out
                        elif desired_adapter == "insertion":
                            projection_state = insertion_local_action_out
                        elif desired_adapter == "contact":
                            projection_state = contact_local_action_out
                        else:
                            projection_state = base_action_out_state
                        if projection_state is None:
                            projection_state = base_action_out_state
                        using_local_action_out = projection_state is not base_action_out_state
                        with torch.no_grad():
                            for name, value in projection_state.items():
                                getattr(base_action_out, name).copy_(value)
                    policy.set_adapter(desired_adapter)
                    active_lora_adapter = desired_adapter
                    print(
                        f"State-routed LoRA switched to {desired_adapter}: "
                        f"joint_l2={distance:.4f}, final_l2={final_distance}, "
                        f"grasp_l2={grasp_distance}, "
                        f"post_grasp_l2={post_grasp_distance}, "
                        f"insertion_l2={insertion_distance}, "
                        f"contact_l2={contact_distance}, "
                        f"gripper_binary={state[6]:.1f}, gripper_physical={measured_gripper:.3f}",
                        f", local_action_out={using_local_action_out}",
                        flush=True,
                    )
            batch = preprocessor(
                _raw_observation(
                    state,
                    camera0,
                    camera1,
                    release_sensors if requires_release_sensors else None,
                    args.task_prompt,
                )
            )

            inference_start = time.perf_counter()
            use_demo_recovery = (
                demo_recovery is not None
                and chunk_id >= args.demo_recovery_after_chunks
            )
            if use_demo_recovery:
                predicted_action_chunk, _, _ = demo_recovery.action_chunk(state)
                normalized_chunk = None
            else:
                with inference_context():
                    sampled_chunks = []
                    for ensemble_index in range(args.ensemble_size):
                        if args.fixed_noise_per_replan:
                            torch.manual_seed(args.seed + ensemble_index)
                        predict_kwargs = {}
                        if args.rtc:
                            predict_kwargs = {
                                "prev_chunk_left_over": previous_normalized_leftover,
                                "inference_delay": 0,
                                "execution_horizon": args.rtc_execution_horizon,
                                "rtc_action_mask": rtc_action_mask,
                            }
                        normalized_chunk = policy.predict_action_chunk(batch, **predict_kwargs)
                        sampled_chunks.append(
                            postprocess_action_chunk(normalized_chunk, postprocessor)
                        )
                    action_chunk = torch.stack(sampled_chunks, dim=0).mean(dim=0)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            if not use_demo_recovery and (
                action_chunk.shape[0] != 1
                or action_chunk.shape[2] != EXPECTED_ACTION_DIM
            ):
                raise ValueError(f"Invalid postprocessed action shape {tuple(action_chunk.shape)}")
            if not use_demo_recovery and action_chunk.shape[1] != predicted_action_steps:
                raise ValueError(
                    f"Policy returned {action_chunk.shape[1]} actions; "
                    f"checkpoint declares chunk_size={predicted_action_steps}"
                )

            if not use_demo_recovery:
                predicted_action_chunk = action_chunk[0].detach().to(
                    device="cpu", dtype=torch.float32
                ).numpy()
                predicted_action_chunk = _scale_physical_arm_residual(
                    predicted_action_chunk, state, args.physical_arm_residual_scale,
                )
                predicted_action_chunk = _rate_limit_arm_chunk(
                    predicted_action_chunk, state, args.max_arm_step_rad,
                )
                if args.rtc and args.max_arm_step_rad > 0:
                    normalized_chunk = _synchronize_rtc_arm_chunk(
                        normalized_chunk,
                        predicted_action_chunk,
                        rtc_action_q01,
                        rtc_action_q99,
                    )
                if insertion_feedback_model is not None and contact_started:
                    force_norm = float(np.linalg.norm(np.load(FORCE_FILE)[:3]))
                    feedback_arm = build_feedback_chunk(
                        insertion_feedback_model,
                        state[:6],
                        insertion_feedback_terminal,
                        contact_model=insertion_contact_model,
                        contact_force_start_n=insertion_contact_force_start_n,
                        contact_force_full_n=insertion_contact_force_full_n,
                        force_norm_n=force_norm,
                        force_baseline_n=insertion_feedback_force_baseline_n,
                        force_full_scale_n=insertion_feedback_force_full_scale_n,
                        chunk_size=predicted_action_steps,
                        device=device,
                    )
                    predicted_action_chunk[:, :6] = feedback_arm
                    predicted_action_chunk[:, 6] = 1.0
            if not np.isfinite(predicted_action_chunk).all():
                raise ValueError("Postprocessed action chunk contains NaN or Inf")
            continuous_gripper = args.state_gripper_mode == "continuous_radians_0_0.8"
            if continuous_gripper:
                # Exact v8 train/inference contract: keep physical radians all
                # the way to FollowJointTrajectory.
                predicted_action_chunk[:, 6] = np.clip(
                    predicted_action_chunk[:, 6], 0.0, 0.8
                )
            else:
                predicted_action_chunk[:, 6] = (
                    predicted_action_chunk[:, 6] >= 0.5
                ).astype(np.float32)
            execution_chunk = predicted_action_chunk[:executed_action_steps].copy()
            if args.hold_gripper_per_replan and not use_demo_recovery and not continuous_gripper:
                # A semantic transition anywhere in the actually executed
                # prefix is the chunk-level decision.  Using only sample zero
                # can indefinitely discard a close predicted a few frames
                # later at an otherwise stable grasp pose.
                # Gripper closure is a discrete semantic decision, but actions
                # beyond the receding-horizon execution prefix are not part of
                # the current control decision.  Pulling a later close out of
                # the full 50-step prediction makes a correctly open descent
                # prefix close in mid-air when only 10 steps are executed.
                execution_chunk[:, 6] = float(np.max(execution_chunk[:, 6]))
            close_gate_distance = None
            close_gate_blocked = False
            if args.gripper_close_reference is not None and not use_demo_recovery:
                close_gate_distance = float(np.linalg.norm(
                    state[:6] - np.asarray(args.gripper_close_reference, dtype=np.float32)
                ))
                close_gate_blocked = (
                    not measured_gripper_closed
                    and close_gate_distance > args.gripper_close_threshold
                    and bool(np.any(execution_chunk[:, 6] >= 0.5))
                )
                if close_gate_blocked:
                    execution_chunk[:, 6] = 0.0
            hold_arm_on_close = (
                args.hold_arm_on_close_transition
                and not use_demo_recovery
                and not measured_gripper_closed
                and not close_gate_blocked
                and bool(np.any(execution_chunk[:, 6] >= 0.5))
            )
            if hold_arm_on_close:
                execution_chunk[:, :6] = state[None, :6]

            open_gate_distance = None
            open_gate_blocked = False
            if args.gripper_open_reference is not None and not use_demo_recovery:
                open_gate_distance = float(np.linalg.norm(
                    state[:6] - np.asarray(args.gripper_open_reference, dtype=np.float32)
                ))
                open_gate_blocked = (
                    measured_gripper_closed
                    and open_gate_distance > args.gripper_open_threshold
                    and bool(np.any(execution_chunk[:, 6] < 0.5))
                )
                if open_gate_blocked:
                    execution_chunk[:, 6] = 1.0

            boundary_jump = None
            replanning_disagreement = None
            if previous_physical_chunk is not None:
                boundary_jump = float(
                    np.linalg.norm(
                        predicted_action_chunk[0, :6]
                        - previous_physical_chunk[rtc_consumed_steps - 1, :6]
                    )
                )
                # With the official Pi0.5 50/50 contract the complete previous
                # chunk has been consumed, so there is no unexecuted sample at
                # index 50 to compare against.  This diagnostic is only defined
                # for receding-horizon prefixes shorter than the prediction.
                if rtc_consumed_steps < len(previous_physical_chunk):
                    replanning_disagreement = float(
                        np.linalg.norm(
                            predicted_action_chunk[0, :6]
                            - previous_physical_chunk[rtc_consumed_steps, :6]
                        )
                    )
            if args.rtc and normalized_chunk is not None:
                previous_normalized_leftover = normalized_chunk[
                    :, rtc_consumed_steps:
                ].detach().clone()
            elif use_demo_recovery:
                previous_normalized_leftover = None
            previous_physical_chunk = predicted_action_chunk.copy()

            if args.trace_dir is not None and chunk_id < args.trace_first_chunks:
                trace_values = {
                    "state": state,
                    "camera0": camera0,
                    "camera1": camera1,
                    # Shadow-evaluation contract: this is the hand-written
                    # route that actually selected the deployed adapter.  A
                    # learned PAP-MoE head may be evaluated against it later,
                    # but never changes the action in this baseline rollout.
                    "fsm_route": np.asarray(active_lora_adapter),
                    "fsm_stage_started": np.asarray(stage_lora_started),
                    "fsm_final_stage_started": np.asarray(final_stage_started),
                    "fsm_grasp_stage_started": np.asarray(grasp_stage_started),
                    "fsm_post_grasp_started": np.asarray(post_grasp_started),
                    "fsm_insertion_started": np.asarray(insertion_started),
                    "fsm_contact_started": np.asarray(contact_started),
                    "measured_gripper_physical": np.asarray(
                        measured_gripper, dtype=np.float32
                    ),
                    "predicted_action_chunk": predicted_action_chunk,
                    "execution_chunk": execution_chunk,
                    "gripper_close_gate_distance": np.asarray(
                        np.nan if close_gate_distance is None else close_gate_distance,
                        dtype=np.float32,
                    ),
                    "gripper_close_gate_blocked": np.asarray(close_gate_blocked),
                    "hold_arm_on_close_transition": np.asarray(hold_arm_on_close),
                    "gripper_open_gate_distance": np.asarray(
                        np.nan if open_gate_distance is None else open_gate_distance,
                        dtype=np.float32,
                    ),
                    "gripper_open_gate_blocked": np.asarray(open_gate_blocked),
                    "chunk_id": np.asarray(chunk_id, dtype=np.int64),
                    "insertion_feedback_active": np.asarray(
                        insertion_feedback_model is not None and contact_started
                    ),
                }
                trace_values.update(release_sensors)
                np.savez_compressed(
                    args.trace_dir / f"chunk_{chunk_id:05d}.npz",
                    **trace_values,
                )

            np.save(ACTION_CHUNK_TMP_FILE, execution_chunk)
            os.replace(ACTION_CHUNK_TMP_FILE, ACTION_CHUNK_FILE)
            action_exchange.publish_reply(request_id, execution_chunk)
            _atomic_write_text(
                ACTION_FILE,
                " ".join(f"{value:.6f}" for value in execution_chunk[0]) + "\n",
            )
            print(
                f"[Pi0.5 chunk {chunk_id:05d}] inference={inference_ms:.1f} ms, "
                f"ensemble={args.ensemble_size}, "
                f"physical_gain={args.physical_arm_residual_scale:.3f}, "
                f"limit={args.max_arm_step_rad:.3f}, "
                f"rtc={args.rtc}, "
                f"boundary_jump={boundary_jump}, "
                f"replan_disagreement={replanning_disagreement}, "
                f"close_gate_distance={close_gate_distance}, "
                f"close_gate_blocked={close_gate_blocked}, "
                f"hold_arm_on_close={hold_arm_on_close}, "
                f"open_gate_distance={open_gate_distance}, "
                f"open_gate_blocked={open_gate_blocked}, "
                f"first_action={np.round(execution_chunk[0], 4)}",
                flush=True,
            )
            chunk_id += 1

        except KeyboardInterrupt:
            print("Stopping Pi0.5 inference.", flush=True)
            break
        except Exception as error:
            print(f"Pi0.5 inference error: {error}", file=sys.stderr, flush=True)
            time.sleep(0.05)


if __name__ == "__main__":
    main()

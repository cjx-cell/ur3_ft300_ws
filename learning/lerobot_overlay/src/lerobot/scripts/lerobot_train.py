#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import dataclasses
import json
import logging
import math
import os
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, SemanticPairedRecoverySampler, make_dataset
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)


def _reject_ambiguous_preprocessor_overrides(
    pretrained_path: str | os.PathLike[str], overrides: dict[str, Any]
) -> None:
    """Prevent one registry-name override from mutating several distinct steps."""
    config_path = Path(pretrained_path) / "policy_preprocessor.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    counts = Counter(step.get("registry_name") for step in config.get("steps", []))
    ambiguous = sorted(name for name in overrides if counts[name] > 1)
    if ambiguous:
        raise ValueError(
            "Pretrained processor overrides are ambiguous because these registry "
            f"names occur more than once: {ambiguous}. Rebuild the processor from "
            "the current policy config and dataset statistics by setting "
            "LEROBOT_REBUILD_PROCESSORS=1. Applying one override to every matching "
            "step can silently double-normalize unrelated features."
        )


def _apply_pap_moe_visual_degradation(
    batch: dict[str, Any], policy_config: Any
) -> torch.Tensor | None:
    """Create paired blind-view PAP samples without changing robot targets.

    Both real cameras are degraded synchronously. The clean routing prior
    ``[E1, 0, E3, E4]`` is transformed with the same factorized rule used by
    data collection: ``[q*E1, 1-q, E3, E4]`` followed by normalization. This
    preserves E2/E3 and E2/E4 cooperation during blind contact.

    Returns the physical 0..1 routing target so the caller can restore it
    after the generic STATE normalizer has processed sensor inputs.
    """
    target_key = "observation.physics_gate_target"
    if getattr(policy_config, "type", None) != "pap_moe" or target_key not in batch:
        return None
    probability = float(
        getattr(policy_config, "visual_degradation_training_probability", 0.0)
    )
    target = batch[target_key].to(dtype=torch.float32).clone()
    if probability <= 0.0:
        return target

    camera_keys = sorted(
        key
        for key in batch
        if key.startswith("observation.images.camera")
        and not key.endswith("_is_pad")
        and isinstance(batch[key], torch.Tensor)
    )
    if len(camera_keys) != 2:
        raise ValueError(
            "PAP-MoE paired visual degradation requires exactly two real "
            f"policy cameras, found {camera_keys}"
        )
    batch_size = target.shape[0]
    device = target.device
    degraded = torch.rand(batch_size, device=device) < probability
    if not bool(degraded.any()):
        return target

    dropout_fraction = float(
        getattr(policy_config, "visual_degradation_dropout_fraction", 0.5)
    )
    dropout = degraded & (torch.rand(batch_size, device=device) < dropout_fraction)
    glare = degraded & ~dropout
    gain_min = float(getattr(policy_config, "visual_degradation_glare_gain_min", 2.0))
    gain_max = float(getattr(policy_config, "visual_degradation_glare_gain_max", 6.0))
    gains = gain_min + torch.rand(batch_size, device=device) * (gain_max - gain_min)

    degraded_cameras = []
    for key in camera_keys:
        image = batch[key]
        if image.ndim not in (4, 5):
            raise ValueError(f"{key} must be BCHW or BTCHW, got shape {tuple(image.shape)}")
        image = image.clone()
        current = image if image.ndim == 4 else image[:, -1]
        current[dropout] = 0.0
        if bool(glare.any()):
            current[glare] = torch.clamp(
                current[glare] * gains[glare, None, None, None], 0.0, 1.0
            )
        if image.ndim == 5:
            image[:, -1] = current
        batch[key] = image
        degraded_cameras.append(current)

    # Recompute the collection quality vector:
    # [black_fraction, saturated_fraction, contrast, valid].
    cameras = torch.stack(degraded_cameras, dim=1)
    gray = cameras.mean(dim=2)
    contrast = gray.std(dim=(2, 3), correction=0).mean(dim=1)
    quality = torch.stack(
        [
            (gray <= 0.02).to(torch.float32).mean(dim=(1, 2, 3)),
            (gray >= 0.98).to(torch.float32).mean(dim=(1, 2, 3)),
            contrast,
            (contrast >= 0.01).to(torch.float32),
        ],
        dim=1,
    )
    quality_key = "observation.visual_quality"
    if quality_key not in batch:
        raise ValueError("PAP-MoE paired visual degradation requires observation.visual_quality")
    batch[quality_key] = batch[quality_key].clone()
    batch[quality_key][degraded] = quality[degraded]

    # Dropout has q=0. Glare follows the collection prior:
    # visual_loss=clip((gain-1)/5), q=1-visual_loss.
    q = torch.ones(batch_size, device=device, dtype=torch.float32)
    q[dropout] = 0.0
    q[glare] = 1.0 - torch.clamp((gains[glare] - 1.0) / 5.0, 0.0, 1.0)
    q_expanded = q.reshape(batch_size, *([1] * (target.ndim - 2)))
    transformed = target.clone()
    transformed[..., 0] = q_expanded * target[..., 0]
    transformed[..., 1] = 1.0 - q_expanded
    transformed[..., 2:] = target[..., 2:]
    transformed = transformed / transformed.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target[degraded] = transformed[degraded]
    batch[target_key] = target
    return target


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator

    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # PAP-MoE-specific expert/gate dashboard.  Do not attach it to baseline
    # policies: zero-filled expert metrics make a Pi0/Pi0.5 run look as though
    # it contains a PhysicsGate when it does not.
    realtime_logger = None
    if is_main_process and cfg.policy.type == "pap_moe":
        artifact_png = "/home/ubuntu/.gemini/antigravity-cli/brain/91e24590-9cdf-4030-bd64-c2787228dc86/stage2_realtime_dashboard.png"
        from lerobot.utils.realtime_logger import RealtimePAPMoELogger

        realtime_logger = RealtimePAPMoELogger(str(cfg.output_dir), artifact_png_path=artifact_png)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        # A resumed/continued PEFT checkpoint is already reconstructed as a
        # PeftModel by make_policy(). Wrapping it again creates a nested second
        # adapter, drops the optimizer parameter identity, and produces an
        # unusable checkpoint. Reactivate the loaded adapter for training
        # instead. Fresh base policies still follow the normal wrapping path.
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            adapter_name = policy.active_adapter
            if isinstance(adapter_name, list):
                if len(adapter_name) != 1:
                    raise ValueError(
                        "PEFT continuation requires exactly one active adapter, "
                        f"got {adapter_name}"
                    )
                adapter_name = adapter_name[0]
            policy.set_adapter(adapter_name, inference_mode=False)
            logging.info(
                "Using already-loaded PEFT adapter '%s' for continuation; "
                "skipping a second PEFT wrap.",
                adapter_name,
            )
        else:
            logging.info("Using PEFT! Wrapping model.")
            # Convert CLI peft config to dict for overrides
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Set metrics CSV output directory for SA-MOE / Pi0
    if hasattr(policy, "_log_metrics_csv"):
        policy._metrics_dir = str(cfg.output_dir)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    processor_pretrained_path = cfg.policy.pretrained_path
    rebuild_processors = os.environ.get("LEROBOT_REBUILD_PROCESSORS", "0") == "1"
    preserve_pretrained_processor_stats = (
        os.environ.get("LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS", "0") == "1"
    )
    if rebuild_processors and preserve_pretrained_processor_stats:
        raise ValueError(
            "LEROBOT_REBUILD_PROCESSORS=1 and "
            "LEROBOT_PRESERVE_PRETRAINED_PROCESSOR_STATS=1 are mutually exclusive"
        )
    if rebuild_processors:
        logging.info(
            "LEROBOT_REBUILD_PROCESSORS=1: building processors from the current "
            "policy config and dataset statistics."
        )
        processor_pretrained_path = None
    if (
        getattr(cfg.policy, "use_relative_actions", False)
        and processor_pretrained_path is not None
        and not cfg.resume
    ):
        logging.warning(
            "use_relative_actions=true with pretrained processors can skip relative transforms if "
            "the checkpoint processors do not define them. Building processors from current policy config."
        )
        processor_pretrained_path = None

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if not (processor_pretrained_path and preserve_pretrained_processor_stats):
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
        }
        if not preserve_pretrained_processor_stats:
            processor_kwargs["preprocessor_overrides"]["normalizer_processor"] = {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        if not preserve_pretrained_processor_stats:
            postprocessor_kwargs["postprocessor_overrides"] = {
                "unnormalizer_processor": {
                    "stats": dataset.meta.stats,
                    "features": policy.config.output_features,
                    "norm_map": policy.config.normalization_mapping,
                },
            }
        else:
            logging.info(
                "Preserving pretrained processor normalization statistics; "
                "only device/rename overrides are applied."
            )
        _reject_ambiguous_preprocessor_overrides(
            processor_pretrained_path,
            processor_kwargs["preprocessor_overrides"],
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=processor_pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if os.environ.get("PAP_MOE_VERIFY_CORRECTED_CONTRACT") == "1":
        from lerobot.policies.pap_moe.experiment_contract import verify_contract

        verify_contract(dataset.root, policy, preprocessor, postprocessor,
                        Path(cfg.output_dir) / "verified_processors")

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
        resumed_lr_override = os.environ.get("LEROBOT_RESUME_LR_OVERRIDE")
        if resumed_lr_override is not None:
            target_lr = float(resumed_lr_override)
            if not math.isfinite(target_lr) or target_lr <= 0.0:
                raise ValueError("LEROBOT_RESUME_LR_OVERRIDE must be a finite positive number")
            for param_group in optimizer.param_groups:
                param_group["lr"] = target_lr
                param_group["initial_lr"] = target_lr
            if lr_scheduler is not None:
                lr_scheduler.base_lrs = [target_lr] * len(optimizer.param_groups)
                lr_scheduler._last_lr = [target_lr] * len(optimizer.param_groups)
            logging.info(
                "Overrode resumed optimizer/scheduler learning rate to %.3e.", target_lr
            )

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataset_sample_weights = None
    if cfg.dataset_sample_weight_key is not None:
        import numpy as np

        key = cfg.dataset_sample_weight_key
        if key not in dataset.hf_dataset.column_names:
            raise ValueError(f"dataset sample-weight column is missing: {key}")
        dataset_sample_weights = np.asarray(dataset.hf_dataset[key], dtype=np.float64).reshape(-1)
        if len(dataset_sample_weights) != len(dataset) or not np.isfinite(dataset_sample_weights).all():
            raise ValueError(f"invalid dataset sample weights in {key}")
        if np.any(dataset_sample_weights < 0.0) or not np.any(dataset_sample_weights > 0.0):
            raise ValueError(f"dataset sample weights in {key} must be non-negative with positive mass")

    # Skill-Progress supervision can be intentionally unavailable on legacy
    # recovery frames. Exclude those frames at the sampler boundary instead of
    # spending optimizer steps on batches with no target signal.
    if getattr(cfg.policy, "train_skill_progress_only", False):
        import numpy as np

        validity_key = "skill_progress_valid"
        if validity_key not in dataset.hf_dataset.column_names:
            raise ValueError(f"Skill-Progress validity column is missing: {validity_key}")
        valid_progress = np.asarray(dataset.hf_dataset[validity_key], dtype=bool).reshape(-1)
        if len(valid_progress) != len(dataset) or not np.any(valid_progress):
            raise ValueError("Skill-Progress dataset contains no valid supervision")
        if dataset_sample_weights is None:
            dataset_sample_weights = valid_progress.astype(np.float64)
        else:
            dataset_sample_weights *= valid_progress
        logging.info(
            "Skill-Progress sampling excludes invalid targets: valid=%d/%d",
            int(valid_progress.sum()),
            len(valid_progress),
        )

    # Local progress transitions are narrow action decision boundaries (for
    # example, precise alignment -> descent).  Make them boostable during
    # ordinary action-only BC without exposing the annotation to the policy.
    # Apply a symmetric window because an action chunk sampled just before the
    # transition must already contain the first actions of the next phase.
    if (
        cfg.skill_progress_transition_sampling_weight > 1.0
        and cfg.skill_progress_transition_sampling_window > 0
    ):
        import numpy as np

        phase_key = "skill_progress_phase"
        if phase_key not in dataset.hf_dataset.column_names:
            raise ValueError(
                f"Skill-progress transition sampling requires column: {phase_key}"
            )
        phases = np.asarray(dataset.hf_dataset[phase_key], dtype=np.int64).reshape(-1)
        episode_indices = np.asarray(
            dataset.hf_dataset["episode_index"], dtype=np.int64
        ).reshape(-1)
        if len(phases) != len(dataset) or len(episode_indices) != len(dataset):
            raise ValueError("invalid skill-progress transition annotations")
        valid = np.ones(len(phases), dtype=bool)
        validity_key = "skill_progress_valid"
        if validity_key in dataset.hf_dataset.column_names:
            valid = np.asarray(
                dataset.hf_dataset[validity_key], dtype=bool
            ).reshape(-1)
            if len(valid) != len(dataset):
                raise ValueError("invalid skill-progress validity annotations")
        transition_condition = (
            (episode_indices[1:] == episode_indices[:-1])
            & valid[1:]
            & valid[:-1]
            & (phases[1:] != phases[:-1])
        )
        from_phase = cfg.skill_progress_transition_from_phase
        to_phase = cfg.skill_progress_transition_to_phase
        if from_phase is not None:
            transition_condition &= (phases[:-1] == from_phase) & (phases[1:] == to_phase)
        transitions = np.flatnonzero(transition_condition) + 1
        transition_mask = np.zeros(len(phases), dtype=bool)
        window = cfg.skill_progress_transition_sampling_window
        for transition in transitions:
            episode = episode_indices[transition]
            start = max(0, int(transition) - window)
            stop = min(len(phases), int(transition) + window + 1)
            transition_mask[start:stop] |= (
                (episode_indices[start:stop] == episode) & valid[start:stop]
            )
        if dataset_sample_weights is None:
            dataset_sample_weights = np.ones(len(phases), dtype=np.float64)
        else:
            dataset_sample_weights = dataset_sample_weights.copy()
        dataset_sample_weights[transition_mask] *= (
            cfg.skill_progress_transition_sampling_weight
        )
        logging.info(
            "Skill-progress transition sampling boost: "
            f"±{window} frames around {len(transitions)} transitions, "
            f"filter={from_phase}->{to_phase}, "
            f"weight={cfg.skill_progress_transition_sampling_weight:.2f}, "
            f"frames={int(transition_mask.sum())}/{len(phases)}"
        )

    # Gripper transitions are narrow decision boundaries and must also be
    # boostable during ordinary dataset-weighted action training.  Previously
    # this option was nested inside the stage/semantic balancing branch, so it
    # silently did nothing for the action-adapter path that deliberately keeps
    # the empirical phase distribution.
    if (
        cfg.gripper_transition_sampling_weight > 1.0
        and cfg.gripper_transition_sampling_window > 0
    ):
        import numpy as np

        gripper_index = getattr(cfg.policy, "gripper_action_index", None)
        if gripper_index is None:
            raise ValueError(
                "gripper transition sampling requires policy.gripper_action_index"
            )
        actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
        episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
        if actions.ndim != 2 or not 0 <= gripper_index < actions.shape[1]:
            raise ValueError(
                f"Invalid gripper action index {gripper_index} for action shape {actions.shape}"
            )
        closed = actions[:, gripper_index] >= 0.5
        transitions = np.flatnonzero(
            (episode_indices[1:] == episode_indices[:-1])
            & (closed[1:] != closed[:-1])
        ) + 1
        transition_mask = np.zeros(len(actions), dtype=bool)
        window = cfg.gripper_transition_sampling_window
        for transition in transitions:
            episode = episode_indices[transition]
            start = max(0, int(transition) - window)
            stop = min(len(actions), int(transition) + window + 1)
            transition_mask[start:stop] |= episode_indices[start:stop] == episode
        if dataset_sample_weights is None:
            dataset_sample_weights = np.ones(len(actions), dtype=np.float64)
        else:
            dataset_sample_weights = dataset_sample_weights.copy()
        dataset_sample_weights[transition_mask] *= cfg.gripper_transition_sampling_weight
        logging.info(
            "Gripper-transition sampling boost: "
            f"±{window} frames around {len(transitions)} transitions, "
            f"weight={cfg.gripper_transition_sampling_weight:.2f}, "
            f"frames={int(transition_mask.sum())}/{len(actions)}"
        )

    # Startup frames are another narrow decision region (for example, an
    # initially open gripper). Compose this boost with D1 and transition
    # weights instead of placing it in a mutually exclusive sampler branch.
    if cfg.initial_frame_sampling_weight > 1.0 and cfg.initial_frame_sampling_count > 0:
        import numpy as np

        frame_indices = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
        initial_mask = frame_indices < cfg.initial_frame_sampling_count
        if dataset_sample_weights is None:
            dataset_sample_weights = np.ones(len(frame_indices), dtype=np.float64)
        else:
            dataset_sample_weights = dataset_sample_weights.copy()
        dataset_sample_weights[initial_mask] *= cfg.initial_frame_sampling_weight
        weighted_fraction = float(
            dataset_sample_weights[initial_mask].sum() / dataset_sample_weights.sum()
        )
        logging.info(
            "Initial-frame sampling boost: "
            f"frame_index < {cfg.initial_frame_sampling_count}, "
            f"weight={cfg.initial_frame_sampling_weight:.2f}, "
            f"frames={int(initial_mask.sum())}/{len(frame_indices)}, "
            f"combined expected sample share={weighted_fraction:.3f}"
        )

    # Target the start of expert recovery, not the start of the stored episode.
    # Full rollout-recovery episodes intentionally retain a zero-weight policy
    # prefix, so frame_index < N would emphasize the wrong observations. Find
    # each recovery episode's first positive D1 weight and boost only its first
    # N positive observations. Keep the multiplicative boost outside paired
    # sampling; paired sampling uses D1 weights only as an eligibility mask.
    if (
        not cfg.semantic_paired_recovery_sampling
        and cfg.semantic_recovery_start_sampling_count > 0
        and cfg.semantic_episode_sampling_weight > 1.0
    ):
        import numpy as np

        recovery_episode_start = cfg.semantic_episode_sampling_start_index
        if recovery_episode_start is None:
            raise ValueError(
                "recovery-start boost requires semantic_episode_sampling_start_index"
            )
        episode_indices = np.asarray(
            dataset.hf_dataset["episode_index"], dtype=np.int64
        )
        if dataset_sample_weights is None:
            dataset_sample_weights = np.ones(len(episode_indices), dtype=np.float64)
        else:
            dataset_sample_weights = dataset_sample_weights.copy()
        positive_before_boost = dataset_sample_weights > 0.0
        recovery_start_mask = np.zeros(len(episode_indices), dtype=bool)
        recovery_episodes = np.unique(
            episode_indices[episode_indices >= recovery_episode_start]
        )
        for episode_index in recovery_episodes:
            positive_indices = np.flatnonzero(
                (episode_indices == episode_index) & positive_before_boost
            )
            recovery_start_mask[
                positive_indices[: cfg.semantic_recovery_start_sampling_count]
            ] = True
        dataset_sample_weights[recovery_start_mask] *= (
            cfg.semantic_episode_sampling_weight
        )
        weighted_fraction = float(
            dataset_sample_weights[recovery_start_mask].sum()
            / dataset_sample_weights.sum()
        )
        logging.info(
            "Recovery-start sampling boost: "
            f"episode_index >= {recovery_episode_start}, first positive "
            f"{cfg.semantic_recovery_start_sampling_count} observations, "
            f"weight={cfg.semantic_episode_sampling_weight:.2f}, "
            f"episodes={len(recovery_episodes)}, "
            f"frames={int(recovery_start_mask.sum())}/{len(episode_indices)}, "
            f"combined expected sample share={weighted_fraction:.3f}"
        )

    if cfg.semantic_paired_recovery_sampling:
        anchor_stage_aliases = {}
        task_table = dataset.meta.tasks
        coarse_grasp = "grasp the peg"
        fine_grasp = [
            "approach the peg",
            "descend onto the peg",
            "stabilize over the peg",
            "close the gripper on the peg",
            "lift the grasped peg",
        ]
        if coarse_grasp in task_table.index and all(
            task in task_table.index for task in fine_grasp
        ):
            anchor_stage_aliases[int(task_table.loc[coarse_grasp].task_index)] = [
                int(task_table.loc[task].task_index) for task in fine_grasp
            ]
        sampler = SemanticPairedRecoverySampler(
            dataset.hf_dataset["task_index"],
            dataset.hf_dataset["episode_index"],
            cfg.semantic_episode_sampling_start_index,
            seed=cfg.seed or 0,
            frame_indices=dataset.hf_dataset["frame_index"],
            recovery_start_count=cfg.semantic_recovery_start_sampling_count,
            anchor_stage_aliases=anchor_stage_aliases,
            eligible_mask=(
                None
                if dataset_sample_weights is None
                else np.asarray(dataset_sample_weights) > 0.0
            ),
        )
        shuffle = False
        logging.info(
            "Paired recovery sampling enabled: every recovery frame is followed "
            "by a same-semantic-stage anchor frame; samples/epoch=%d",
            len(sampler),
        )
    # Balanced sampling: standard policies can balance semantic task phases;
    # PAP-MoE stages balance physical expert labels.
    elif (
        cfg.skill_progress_balanced_sampling
        or cfg.semantic_task_balanced_sampling
        or getattr(cfg.policy, "stage_balanced_sampling", False)
    ):
        import numpy as np

        skill_progress_balancing = cfg.skill_progress_balanced_sampling
        semantic_balancing = cfg.semantic_task_balanced_sampling
        probability_balancing = skill_progress_balancing or semantic_balancing
        soft_sample_weights = None
        if skill_progress_balancing:
            if semantic_balancing:
                raise ValueError(
                    "skill_progress_balanced_sampling and "
                    "semantic_task_balanced_sampling are mutually exclusive"
                )
            label_key = "skill_progress_phase"
            if label_key not in dataset.hf_dataset.column_names:
                raise ValueError(f"Skill-Progress phase column is missing: {label_key}")
            stages = np.asarray(dataset.hf_dataset[label_key], dtype=int)
            if stages.ndim != 1:
                raise ValueError(f"{label_key} must be 1-D, got shape {stages.shape}")
            num_stages = int(getattr(cfg.policy, "num_skill_progress", 0))
            if num_stages < 1 or np.any((stages < 0) | (stages >= num_stages)):
                raise ValueError(f"{label_key} values must be in [0, {num_stages - 1}]")
            logging.info(
                "Skill-Progress-balanced sampling enabled "
                f"({num_stages} shared local phases)..."
            )
        elif semantic_balancing:
            stages = np.asarray(dataset.hf_dataset["task_index"], dtype=int)
            if stages.ndim != 1:
                raise ValueError(f"task_index must be 1-D, got shape {stages.shape}")
            inferred_stage_count = int(stages.max()) + 1
            num_stages = cfg.semantic_task_count or inferred_stage_count
            if np.any((stages < 0) | (stages >= num_stages)):
                raise ValueError(f"task_index values must be in [0, {num_stages - 1}]")
            logging.info(
                "Semantic-task-balanced sampling enabled "
                f"(deterministic round-robin, {num_stages} task phases)..."
            )
        else:
            num_stages = getattr(cfg.policy, "num_experts", 5)
            physics_target_key = (
                "observation.physics_gate_target"
                if "observation.physics_gate_target" in dataset.hf_dataset.column_names
                else "observation.stage"
            )
            if physics_target_key not in dataset.hf_dataset.column_names:
                raise ValueError(
                    "Physical routing target column is missing: expected "
                    "observation.physics_gate_target (PAP-MoE) or "
                    "observation.stage (legacy SAMoE)"
                )
            logging.info(
                "Stage-balanced sampling enabled "
                f"(deterministic round-robin, {num_stages} physical stages, "
                f"target={physics_target_key})..."
            )
            stage_vectors = np.asarray(dataset.hf_dataset[physics_target_key])
            if stage_vectors.ndim == 2:
                if stage_vectors.shape[1] != num_stages:
                    raise ValueError(
                        f"{physics_target_key} width {stage_vectors.shape[1]} "
                        f"!= num_experts {num_stages}"
                    )
                if np.any(stage_vectors < -1e-6) or not np.allclose(
                    stage_vectors.sum(axis=1), 1.0, atol=1e-3
                ):
                    raise ValueError(
                        f"{physics_target_key} soft labels must be probabilities"
                    )
                # Balance continuous expert mass instead of collapsing soft
                # supervision with argmax.  This lets mixed E3/E4 contact
                # frames teach both experts and avoids duplicating a handful
                # of E4-dominant frames thousands of times.
                if dataset_sample_weights is None:
                    expert_mass = stage_vectors.sum(axis=0)
                else:
                    expert_mass = (stage_vectors * dataset_sample_weights[:, None]).sum(axis=0)
                active_experts = expert_mass > 1e-6
                inverse_mass = np.zeros_like(expert_mass, dtype=np.float64)
                inverse_mass[active_experts] = 1.0 / expert_mass[active_experts]
                soft_sample_weights = stage_vectors @ inverse_mass
                soft_sample_weights /= max(float(soft_sample_weights.mean()), 1e-12)
                stages = np.argmax(stage_vectors, axis=1).astype(int)
            elif stage_vectors.ndim == 1:
                stages = np.rint(stage_vectors).astype(int).clip(0, num_stages - 1)
            else:
                raise ValueError(
                    f"{physics_target_key} must be a hard-label vector or an "
                    f"(N, E) soft-label matrix, got shape {stage_vectors.shape}"
                )

        # Group frame indices by stage
        stage_indices = [np.where(stages == s)[0] for s in range(num_stages)]
        class_counts = [len(si) for si in stage_indices]
        active_stage_indices = [si for si in stage_indices if len(si) > 0]
        max_count = max(len(si) for si in active_stage_indices)

        if probability_balancing:
            # Equalize total probability mass across semantic phases while
            # still allowing an independent boost for the closed-loop-critical
            # beginning of each episode.
            inverse_counts = np.zeros(num_stages, dtype=np.float64)
            for stage_index, count in enumerate(class_counts):
                stage_mask = stages == stage_index
                mass = (
                    float(count)
                    if dataset_sample_weights is None
                    else float(dataset_sample_weights[stage_mask].sum())
                )
                if mass > 0:
                    inverse_counts[stage_index] = 1.0 / mass
            soft_sample_weights = inverse_counts[stages]
            if dataset_sample_weights is not None:
                # D1 weights are the inner measure; phase balancing operates
                # on that weighted mass, not on zero-weight chunk carriers.
                soft_sample_weights *= dataset_sample_weights
            if (
                cfg.semantic_episode_sampling_start_index is not None
                and cfg.semantic_episode_sampling_weight > 1.0
            ):
                episode_indices = np.asarray(
                    dataset.hf_dataset["episode_index"], dtype=np.int64
                )
                episode_mask = (
                    episode_indices >= cfg.semantic_episode_sampling_start_index
                )
                soft_sample_weights[episode_mask] *= (
                    cfg.semantic_episode_sampling_weight
                )
                logging.info(
                    "  Combined appended-episode boost: "
                    f"episode_index >= {cfg.semantic_episode_sampling_start_index}, "
                    f"weight={cfg.semantic_episode_sampling_weight:.2f}, "
                    f"frames={int(episode_mask.sum())}/{len(stages)}"
                )
            if cfg.semantic_phase_start_sampling_weight > 1.0 and cfg.semantic_phase_start_sampling_count > 0:
                episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
                phase_age = np.zeros(len(stages), dtype=np.int64)
                for index in range(1, len(stages)):
                    if (
                        episode_indices[index] != episode_indices[index - 1]
                        or stages[index] != stages[index - 1]
                    ):
                        phase_age[index] = 0
                    else:
                        phase_age[index] = phase_age[index - 1] + 1
                phase_start_mask = phase_age < cfg.semantic_phase_start_sampling_count
                soft_sample_weights[phase_start_mask] *= cfg.semantic_phase_start_sampling_weight
                logging.info(
                    "  Combined semantic phase-start boost: "
                    f"first {cfg.semantic_phase_start_sampling_count} frames, "
                    f"weight={cfg.semantic_phase_start_sampling_weight:.2f}, "
                    f"frames={int(phase_start_mask.sum())}/{len(stages)}"
                )
            # Boundary/startup boosts redistribute probability *within* each
            # semantic phase. Renormalize per phase so the outer class balance
            # remains exact instead of accidentally favoring short phases.
            for stage_index in range(num_stages):
                stage_mask = stages == stage_index
                stage_mass = float(soft_sample_weights[stage_mask].sum())
                if stage_mass > 0.0:
                    soft_sample_weights[stage_mask] /= stage_mass
            soft_sample_weights /= max(float(soft_sample_weights.mean()), 1e-12)

        logging.info(f"  Stage distribution: {dict(enumerate(class_counts))}")
        if soft_sample_weights is None:
            logging.info(
                f"  Balanced active stages to: {max_count} each "
                f"(active={len(active_stage_indices)}, "
                f"total={len(active_stage_indices) * max_count}/epoch)"
            )

        class BalancedStageSampler(torch.utils.data.Sampler):
            """Deterministic round-robin sampler: each epoch has equal stage representation.

            On each __iter__ call (new epoch):
            1. Oversamples each stage to max_count
            2. Shuffles within each stage's block
            3. Interleaves: [s0_0, s1_0, s2_0, s3_0, s4_0, s0_1, s1_1, ...]
            """

            def __init__(self, stage_indices, max_count):
                self.stage_indices = stage_indices
                self.max_count = max_count

            def __len__(self):
                return len(self.stage_indices) * self.max_count

            def __iter__(self):
                rng = np.random.RandomState()
                balanced = []
                for indices in self.stage_indices:
                    n_full = self.max_count // len(indices)
                    n_rem = self.max_count % len(indices)
                    oversampled = np.concatenate(
                        [
                            np.tile(indices, n_full),
                            rng.choice(indices, n_rem, replace=False),
                        ]
                    )
                    rng.shuffle(oversampled)
                    balanced.append(oversampled)
                # Interleave: round-robin across stages
                interleaved = np.column_stack(balanced).ravel()
                return iter(interleaved.tolist())

        if soft_sample_weights is not None:
            if dataset_sample_weights is not None and not probability_balancing:
                soft_sample_weights *= dataset_sample_weights
                if not np.any(soft_sample_weights > 0.0):
                    raise ValueError("combined stage and dataset sample weights have zero mass")
            generator = torch.Generator().manual_seed(cfg.seed)
            weighted_num_samples = (
                int(np.count_nonzero(dataset_sample_weights))
                if dataset_sample_weights is not None
                else (
                    len(active_stage_indices) * max_count
                    if probability_balancing
                    else len(soft_sample_weights)
                )
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.from_numpy(soft_sample_weights),
                num_samples=weighted_num_samples,
                replacement=True,
                generator=generator,
            )
            if probability_balancing:
                expected_mass = np.bincount(
                    stages,
                    weights=soft_sample_weights,
                    minlength=num_stages,
                ).astype(np.float64)
            else:
                expected_mass = (stage_vectors * soft_sample_weights[:, None]).sum(axis=0)
            expected_mass /= max(float(expected_mass.sum()), 1e-12)
            logging.info(
                "  Weighted sampling enabled; "
                f"expected class mass={expected_mass.round(3).tolist()}, "
                f"samples/epoch={weighted_num_samples}"
            )
        elif dataset_sample_weights is None:
            sampler = BalancedStageSampler(active_stage_indices, max_count)
        else:
            inverse_counts = np.zeros(num_stages, dtype=np.float64)
            for stage_index, count in enumerate(class_counts):
                stage_mask = stages == stage_index
                mass = float(dataset_sample_weights[stage_mask].sum())
                if mass > 0:
                    inverse_counts[stage_index] = 1.0 / mass
            combined_weights = inverse_counts[stages] * dataset_sample_weights
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.from_numpy(combined_weights),
                num_samples=int(np.count_nonzero(dataset_sample_weights)),
                replacement=True,
                generator=torch.Generator().manual_seed(cfg.seed),
            )
            logging.info(
                "Combined hard-stage balance with dataset weights from %s; samples/epoch=%d",
                cfg.dataset_sample_weight_key,
                len(sampler),
            )
        shuffle = False
    elif dataset_sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(dataset_sample_weights),
            num_samples=int(np.count_nonzero(dataset_sample_weights)),
            replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        shuffle = False
        logging.info(
            "Dataset-weighted sampling enabled from %s; positive=%d/%d, samples/epoch=%d",
            cfg.dataset_sample_weight_key,
            int(np.count_nonzero(dataset_sample_weights)),
            len(dataset_sample_weights),
            len(sampler),
        )
    elif cfg.initial_frame_sampling_weight > 1.0 and cfg.initial_frame_sampling_count > 0:
        import numpy as np

        frame_indices = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
        sample_weights = np.ones(len(frame_indices), dtype=np.float64)
        initial_mask = frame_indices < cfg.initial_frame_sampling_count
        sample_weights[initial_mask] = cfg.initial_frame_sampling_weight
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
        weighted_fraction = float(sample_weights[initial_mask].sum() / sample_weights.sum())
        logging.info(
            "Initial-frame weighted sampling enabled: "
            f"frame_index < {cfg.initial_frame_sampling_count}, "
            f"weight={cfg.initial_frame_sampling_weight:.2f}, "
            f"frames={int(initial_mask.sum())}/{len(frame_indices)}, "
            f"expected sample share={weighted_fraction:.3f}"
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        physical_routing_target = _apply_pap_moe_visual_degradation(
            batch, accelerator.unwrap_model(policy, keep_fp32_wrapper=True).config
        )
        batch = preprocessor(batch)
        # Routing targets are probabilities, not robot state. Keep them in
        # physical 0..1 space even though the generic processor normalizes
        # other STATE features such as joints, force, and visual quality.
        if physical_routing_target is not None:
            batch["observation.physics_gate_target"] = physical_routing_target.to(
                device=batch["observation.state"].device
            )
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if realtime_logger and output_dict:
            lr_val = optimizer.param_groups[0]["lr"]
            realtime_logger.update(step, output_dict, lr_val)

        if is_log_step:
            logging.info(train_tracker)
            if realtime_logger:
                realtime_logger.log_and_plot(step, cfg.steps)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()

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


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import PreTrainedPolicy


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    # PAP joint training defines separate action/physics learning rates. Do not
    # silently discard that contract when explicit optimizer presets are used.
    policy_cfg = cfg.policy
    pap_joint = getattr(policy_cfg, "type", None) == "pap_moe" and (
        getattr(policy_cfg, "train_expert_action_joint", False)
        or getattr(policy_cfg, "train_physicsgate_action_joint", False)
    )
    if pap_joint and not cfg.use_policy_training_preset:
        raise ValueError(
            "PAP-MoE joint training requires use_policy_training_preset=true to preserve "
            "get_optim_params() action/physics parameter groups. Refusing silent single-LR training."
        )
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    if cfg.optimizer is None:
        raise ValueError("Optimizer config is required but not provided in TrainPipelineConfig")
    optimizer = cfg.optimizer.build(params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler

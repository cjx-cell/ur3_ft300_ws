from types import SimpleNamespace

import pytest
import torch

from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.optimizers import AdamWConfig


class GroupedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.physics = torch.nn.Parameter(torch.ones(2))
        self.action = torch.nn.Parameter(torch.ones(3))

    def get_optim_params(self):
        return [{"params": [self.physics]}, {"params": [self.action], "lr": 2.5e-6}]


def config(preset, policy_type="pap_moe", joint=True):
    return SimpleNamespace(
        policy=SimpleNamespace(type=policy_type, train_expert_action_joint=joint),
        use_policy_training_preset=preset,
        optimizer=AdamWConfig(lr=2.5e-5), scheduler=None, steps=10,
    )


def test_pap_joint_rejects_silent_single_group():
    with pytest.raises(ValueError, match="Refusing silent single-LR"):
        make_optimizer_and_scheduler(config(False), GroupedPolicy())


def test_pap_joint_keeps_actual_lr_and_membership():
    policy = GroupedPolicy()
    optimizer, _ = make_optimizer_and_scheduler(config(True), policy)
    assert [g["lr"] for g in optimizer.param_groups] == [2.5e-5, 2.5e-6]
    assert optimizer.param_groups[0]["params"][0] is policy.physics
    assert optimizer.param_groups[1]["params"][0] is policy.action


@pytest.mark.parametrize("policy_type,joint", [("pi05", True), ("pap_moe", False)])
def test_other_training_contracts_unchanged(policy_type, joint):
    optimizer, _ = make_optimizer_and_scheduler(config(False, policy_type, joint), GroupedPolicy())
    assert len(optimizer.param_groups) == 1

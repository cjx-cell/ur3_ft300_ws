from types import SimpleNamespace as NS

import pytest
import torch

from lerobot.policies.pap_moe.configuration_pap_moe import PAPMoEConfig
from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPi05Model


def test_stationary_history_is_zero_despite_different_current_normalization():
    node = NS(config=NS(expert_motion_target='history_window_delta_v2', robot_state_dim=7))
    state = torch.full((1, 7), -.419001852)
    history = torch.full((1, 10, 7), .030344087)
    descriptor = PAPMoEPi05Model._expert_motion_descriptor(node, state, history)
    assert descriptor.count_nonzero() == 0
    history[:, -1] += .25
    torch.testing.assert_close(PAPMoEPi05Model._expert_motion_descriptor(node, state, history), torch.full((1, 7), .25))


def test_legacy_objective_remains_reproducible():
    cfg = PAPMoEConfig()
    assert cfg.expert_motion_target == 'legacy_cross_normalized'
    assert cfg.visual_memory_teacher_policy == 'legacy_soft_weight'
    node = NS(config=cfg)
    state, history = torch.randn(2, 7), torch.randn(2, 10, 7)
    torch.testing.assert_close(PAPMoEPi05Model._expert_motion_descriptor(node, state, history), state - history[:, 0])


def test_clean_only_teacher_excludes_glare_and_missing_history_gradients():
    memory = torch.tensor([[1., 0.], [1., 0.], [1., 0.]], requires_grad=True)
    teacher = torch.tensor([[0., 1.], [0., 1.], [0., 1.]], requires_grad=True)
    routes = torch.tensor([[1., 0., 0., 0.], [.8, .2, 0., 0.], [1., 0., 0., 0.]])
    loss = PAPMoEPi05Model._compute_visual_memory_distillation_loss(
        memory, torch.tensor([[0.], [0.], [1.]]), teacher, routes, clean_only=True)
    loss.backward()
    assert memory.grad[0].abs().sum() > 0
    assert memory.grad[1:].count_nonzero() == 0
    assert teacher.grad is None
    with pytest.raises(ValueError):
        PAPMoEPi05Model._compute_visual_memory_distillation_loss(memory, torch.zeros(3, 1), teacher, None, clean_only=True)


def test_all_corrupt_teachers_have_zero_finite_loss():
    memory = torch.randn(3, 8, requires_grad=True)
    loss = PAPMoEPi05Model._compute_visual_memory_distillation_loss(
        memory, torch.zeros(3, 1), torch.randn(3, 8), torch.tensor([[0., .5, .5, 0.]]).expand(3, -1), clean_only=True)
    assert loss.item() == 0
    loss.backward()
    assert memory.grad.count_nonzero() == 0

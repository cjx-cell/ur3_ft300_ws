from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.pap_moe.configuration_pap_moe import PAPMoEConfig
from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPi05Model
from lerobot.policies.pap_moe.pap_moe_modules import FactorizedPhysicsGate, VisualHistoryEncoder


def test_factor_inverse_small_blindness_and_boundaries():
    b = torch.tensor([0., 1e-9, 1e-8, 1e-7, 1e-6, .1, .5, 1.])
    c = torch.tensor([0., .1, .5, .8, 1.])
    grid = torch.cartesian_prod(b, c, torch.tensor([0., .3, 1.]))
    probs = FactorizedPhysicsGate.factors_to_expert_probs(grid)
    recovered = FactorizedPhysicsGate.expert_probs_to_factors(probs)
    torch.testing.assert_close(recovered[:, :2], grid[:, :2], atol=2e-6, rtol=2e-5)
    active = grid[:, 1] > 0
    torch.testing.assert_close(recovered[active, 2], grid[active, 2])
    assert torch.isfinite(recovered).all()


def test_legacy_inverse_remains_explicitly_available():
    p = FactorizedPhysicsGate.factors_to_expert_probs(torch.tensor([[1e-7, .8, .3]]))
    legacy = FactorizedPhysicsGate.expert_probs_to_factors(p, stable=False)
    assert legacy[0, 1].item() == pytest.approx(.89406967)


def test_prefix_pool_excludes_invalid_tokens_and_gradients():
    dummy = SimpleNamespace(config=SimpleNamespace(mask_invalid_prefix_tokens=True))
    x = torch.tensor([[[1., 3.], [3., 5.], [100., 200.]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    y = PAPMoEPi05Model._pool_physics_prefix(dummy, x, mask)
    torch.testing.assert_close(y, torch.tensor([[2., 4.]]))
    y.sum().backward()
    assert x.grad[0, 2].count_nonzero() == 0
    all_pad = PAPMoEPi05Model._pool_physics_prefix(dummy, x, torch.zeros_like(mask))
    assert all_pad.count_nonzero() == 0
    with pytest.raises(ValueError):
        PAPMoEPi05Model._pool_physics_prefix(dummy, x, None)


def test_old_config_keeps_pooling_behavior():
    cfg = PAPMoEConfig()
    assert not cfg.mask_invalid_prefix_tokens
    assert not cfg.mask_invalid_history_cameras
    dummy = SimpleNamespace(config=cfg)
    x = torch.randn(2, 7, 4)
    assert torch.equal(PAPMoEPi05Model._pool_physics_prefix(dummy, x, None), x.mean(1))


def test_history_ignores_only_invalid_camera_and_preserves_valid_camera_gradient():
    torch.manual_seed(3)
    encoder = VisualHistoryEncoder(16, 8, mask_invalid_cameras=True).eval()
    first = torch.randn(1, 3, 3, 16, 16, requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    padding = [torch.zeros(1, 3, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)]
    original, age = encoder([first, second], padding)
    changed, _ = encoder([first, second * 100 + 50], padding)
    torch.testing.assert_close(original, changed, atol=0, rtol=0)
    original.square().sum().backward()
    assert first.grad.abs().sum() > 0
    assert second.grad.count_nonzero() == 0
    no_memory, no_age = encoder([first, second], [torch.ones_like(padding[0])] * 2)
    assert no_memory.count_nonzero() == 0
    assert torch.equal(no_age, torch.ones_like(age))


def test_history_all_valid_matches_legacy_exactly():
    old = VisualHistoryEncoder(16, 8).eval()
    new = VisualHistoryEncoder(16, 8, mask_invalid_cameras=True).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    images = [torch.randn(2, 3, 3, 16, 16) for _ in range(2)]
    pads = [torch.zeros(2, 3, dtype=torch.bool) for _ in range(2)]
    for a, b in zip(old(images, pads), new(images, pads), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)

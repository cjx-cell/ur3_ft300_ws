import pytest
import torch
from torch.nn import functional as F

from lerobot.policies.pap_moe.pap_moe_modules import E1FusionMLP


def test_default_matches_legacy_exactly():
    layer = E1FusionMLP(8, 16, 'joint', 'early_sum')
    legacy = torch.nn.Sequential(*list(layer.children()))
    x = torch.randn(3,24)
    assert torch.equal(layer(x),legacy(x))


def test_late_sum_uses_same_parameters_and_explicit_formula():
    a = E1FusionMLP(8,16,'joint','early_sum')
    b = E1FusionMLP(8,16,'joint','late_sum')
    b.load_state_dict(a.state_dict(),strict=True)
    assert list(a.state_dict()) == list(b.state_dict())
    x=torch.randn(3,24,requires_grad=True)
    zs=[F.linear(v,w,b[1].bias/3) for v,w in zip(b[0](x).chunk(3,-1),b[1].weight.chunk(3,-1))]
    torch.testing.assert_close(sum(zs),b[1](b[0](x)))
    torch.testing.assert_close(b(x),b[3](sum(F.gelu(z) for z in zs)))
    assert not torch.allclose(a(x),b(x))
    b(x).square().sum().backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in b.parameters())
    assert all(v.abs().sum()>0 for v in x.grad.chunk(3,-1))


def test_invalid_mode_rejected():
    with pytest.raises(ValueError): E1FusionMLP(8,16,'joint','other')

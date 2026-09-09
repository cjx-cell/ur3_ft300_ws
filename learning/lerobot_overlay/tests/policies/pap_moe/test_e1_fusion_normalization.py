import pytest
import torch
from lerobot.policies.pap_moe.pap_moe_modules import E1FusionLayerNorm, FreeMotionExpert


def test_joint_is_exact_legacy_layernorm():
    old, new = torch.nn.LayerNorm(24), E1FusionLayerNorm(8)
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(3, 24)
    assert torch.equal(old(x), new(x))


def test_branchwise_isolates_statistics_and_gradient():
    layer = E1FusionLayerNorm(8, 'branchwise')
    x = torch.randn(2, 24, requires_grad=True)
    altered = x.detach().clone()
    altered[:, 8:16] = altered[:, 8:16]*100+20
    assert torch.equal(layer(x)[:, :8], layer(altered)[:, :8])
    (layer(x)[:, :8] * torch.randn(2, 8)).sum().backward()
    assert x.grad[:, 8:].count_nonzero() == 0
    assert x.grad[:, :8].abs().sum() > 0


def test_affine_mapping_and_parameter_compatibility():
    a, b = FreeMotionExpert(16, 8, 7, 6), FreeMotionExpert(16, 8, 7, 6, 'branchwise')
    b.load_state_dict(a.state_dict(), strict=True)
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(a.state_dict()[k], v) for k, v in b.state_dict().items())
    layer = b.out[0]
    with torch.no_grad():
        layer.weight.copy_(torch.arange(24)+1)
        layer.bias.copy_(torch.arange(24))
    x = torch.randn(2, 24)
    expected = torch.cat([torch.nn.functional.layer_norm(x[:, i:i+8], (8,),
        layer.weight[i:i+8], layer.bias[i:i+8], layer.eps) for i in [0,8,16]], -1)
    torch.testing.assert_close(layer(x), expected)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError): E1FusionLayerNorm(8, 'unknown')


def test_branchwise_preserves_input_dtype():
    layer = E1FusionLayerNorm(8, 'branchwise')
    assert layer(torch.randn(2, 24, dtype=torch.bfloat16)).dtype == torch.bfloat16

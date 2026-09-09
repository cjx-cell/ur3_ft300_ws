#!/usr/bin/env python3
"""CPU-only reproducible contract probes; does not modify model code or weights."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/home/ubuntu/lerobot/src')
from lerobot.policies.pap_moe.pap_moe_modules import ActionTokenConditioner, FactorizedPhysicsGate


def main():
    torch.manual_seed(0)
    result = {}
    factors = torch.tensor([[1e-7, .8, .3], [1e-6, .5, .5], [.3, .7, .6], [0., .7, .6]])
    probs = FactorizedPhysicsGate.factors_to_expert_probs(factors)
    recovered = FactorizedPhysicsGate.expert_probs_to_factors(probs)
    product = probs[:, 1] * probs[:, 2:].sum(-1)
    stable_z = 2 / (1 + (1 - 4 * product).clamp_min(0).sqrt())
    stable_contact = probs[:, 2:].sum(-1) * stable_z
    result['factor_inverse'] = dict(input=factors.tolist(), recovered=recovered.tolist(),
        max_abs_error=(factors-recovered).abs().max().item(),
        candidate_stable_contact=stable_contact.tolist())
    c = ActionTokenConditioner(16, 16, nhead=4, zero_init_output=False)
    x = torch.randn(1, 3, 16, requires_grad=True)
    z = torch.randn(1, 4, 16, requires_grad=True)
    s = torch.zeros(4, requires_grad=True)
    w = torch.tensor([[[1., 0., 0., 0.]]]).expand(1, 3, 4)
    y = c(x, z, w, condition_token_scales=s.tanh(), residual_max_norm=1)
    y.square().mean().backward()
    result['initial_gradient'] = dict(expert_token_gradient_max=z.grad.abs().max().item(),
        scale_gradient_max=s.grad.abs().max().item(),
        note='Action-only test at initialization, not a permanent gradient blockage; auxiliary losses omitted.')
    with torch.no_grad():
        c.cross_attn.in_proj_bias.fill_(.1)
        c.cross_attn.out_proj.bias.fill_(.1)
        zero_route = c(x, z, torch.zeros_like(w), condition_token_scales=s.tanh())
        zero_scale = c(x, z, w, condition_token_scales=s.tanh())
    result['attention_bias'] = dict(zero_route_delta_max=(zero_route-x).abs().max().item(),
        active_route_zero_scale_delta_max=(zero_scale-x).abs().max().item(),
        note='Synthetic nonzero biases demonstrate representational behavior, not measured checkpoint bias magnitude.')
    target = Path('/home/ubuntu/ur3_ft300_ws/artifacts/pap_framework_contract_audit_20260905.json')
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

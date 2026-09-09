import torch
import torch.nn as nn

from lerobot.policies.pap_moe.pap_moe_modules import (
    ActionTokenConditioner,
    CompliantInsertionExpert,
    FactorizedPhysicsGate,
    FactorizedRouteForecaster,
    FreeMotionExpert,
    HeterogeneousPhysicsExperts,
    LegacyPhysicsGate,
    MultiScaleForceEncoder,
    PhysicsGate,
    PhysicsExpertAuxiliaryHeads,
    ProprioHistoryEncoder,
    RigidContactExpert,
    SoftExpertRouter,
    VisualBlindForceExpert,
    VisualHistoryEncoder,
    soft_target_cross_entropy,
)


def test_multiscale_force_and_gate_shapes():
    batch, dim, hidden = 2, 64, 32
    encoder = MultiScaleForceEncoder(6, hidden, dim)
    force_tokens = encoder(
        torch.randn(batch, 6),
        torch.randn(batch, 16, 6),
        torch.randn(batch, 20, 6),
    )
    assert set(force_tokens) == {"current", "fast", "slow", "fused"}
    assert all(token.shape == (batch, dim) for token in force_tokens.values())

    gate = LegacyPhysicsGate(dim, num_experts=4, hidden_dim=hidden, nhead=4)
    probabilities, logits, indices = gate(torch.randn(batch, 7, dim), force_tokens["fast"])
    assert probabilities.shape == logits.shape == (batch, 4)
    assert indices.shape == (batch,)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(batch))


def test_factorized_gate_maps_blindness_contact_and_mobility_to_experts():
    factors = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # normal vision, free motion
            [1.0, 0.0, 0.0],  # blind, free motion
            [0.0, 1.0, 0.0],  # rigid contact
            [0.0, 1.0, 1.0],  # compliant/mobile contact
            [1.0, 1.0, 0.0],  # blind rigid contact: E2 + E3
        ]
    )
    probabilities = FactorizedPhysicsGate.factors_to_expert_probs(factors)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(5))
    torch.testing.assert_close(probabilities[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(probabilities[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(probabilities[2], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(probabilities[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(probabilities[4], torch.tensor([0.0, 0.5, 0.5, 0.0]))
    reconstructed = FactorizedPhysicsGate.expert_probs_to_factors(probabilities)
    torch.testing.assert_close(reconstructed, factors, atol=1e-6, rtol=1e-6)

    gate = FactorizedPhysicsGate(d_model=64, hidden_dim=32, nhead=4)
    routed, logits, indices = gate(torch.randn(2, 6, 64), torch.randn(2, 64))
    assert routed.shape == logits.shape == (2, 4)
    assert gate.last_factor_probs is not None
    assert gate.last_factor_probs.shape == (2, 3)
    assert indices.shape == (2,)


def test_factorized_route_forecaster_keeps_current_route_and_forecasts_horizon():
    torch.manual_seed(31)
    forecaster = FactorizedRouteForecaster(d_model=64, hidden_dim=32, horizon=5, nhead=4)
    current = torch.tensor([[0.2, 0.4, 0.7], [0.0, 0.1, 0.3]])
    factors, routes = forecaster(torch.randn(2, 6, 64), current)
    assert factors.shape == (2, 5, 3)
    assert routes.shape == (2, 5, 4)
    torch.testing.assert_close(factors[:, 0], current)
    torch.testing.assert_close(
        routes[:, 0], FactorizedPhysicsGate.factors_to_expert_probs(current)
    )
    torch.testing.assert_close(routes.sum(dim=-1), torch.ones(2, 5))


def test_physics_gate_predicts_observation_conditioned_50_step_routes():
    torch.manual_seed(37)
    gate = PhysicsGate(
        d_model=64,
        hidden_dim=32,
        horizon=5,
        nhead=4,
    )
    context = torch.randn(2, 6, 64)
    fast_force = torch.randn(2, 64)
    changed_context = context.clone()
    changed_context[:, 0] = -changed_context[:, 0]

    current_routes, _, _ = gate(context, fast_force)
    factors_a = gate.last_predicted_factor_sequence.clone()
    routes_a = gate.last_predicted_route_sequence.clone()
    gate(changed_context, fast_force)
    factors_b = gate.last_predicted_factor_sequence

    assert factors_a.shape == (2, 5, 3)
    assert routes_a.shape == (2, 5, 4)
    torch.testing.assert_close(routes_a[:, 0], current_routes)
    torch.testing.assert_close(routes_a.sum(dim=-1), torch.ones(2, 5))
    # All future factors are predicted from observation context, with no
    # candidate/draft action input.
    assert not torch.allclose(factors_a[:, 1:], factors_b[:, 1:])

    factors_a[:, 1:].sum().backward()
    assert gate.sequence_context_proj[1].weight.grad is not None
    assert gate.step_queries.grad is not None


def test_visual_blind_expert_is_hard_isolated_from_visual_input():
    torch.manual_seed(7)
    batch, dim, hidden = 2, 64, 32
    experts = HeterogeneousPhysicsExperts(dim, hidden).eval()
    state = torch.randn(batch, 7)
    state_history = torch.randn(batch, 10, 7)
    current_force = torch.randn(batch, 6)
    fast_force = torch.randn(batch, 16, 6)
    slow_force = torch.randn(batch, 20, 6)
    quality = torch.randn(batch, 4)
    visual_a = torch.randn(batch, dim)
    visual_b = visual_a + 100.0

    args = (state, current_force, fast_force, slow_force, state_history, quality)
    output_a = experts(visual_a, *args)
    output_b = experts(visual_b, *args)

    assert not torch.allclose(output_a[:, 0], output_b[:, 0])
    torch.testing.assert_close(output_a[:, 1], output_b[:, 1])
    torch.testing.assert_close(output_a[:, 2], output_b[:, 2])
    torch.testing.assert_close(output_a[:, 3], output_b[:, 3])


def test_visual_history_encoder_masks_episode_padding_and_reports_age():
    torch.manual_seed(43)
    encoder = VisualHistoryEncoder(d_model=32, hidden_dim=16)
    histories = [torch.rand(2, 3, 3, 32, 32) for _ in range(2)]
    padding = [
        torch.tensor([[True, False, False], [True, True, True]]) for _ in range(2)
    ]

    token, age = encoder(histories, padding)

    assert token.shape == (2, 32)
    assert age.shape == (2, 1)
    assert torch.count_nonzero(token[0]).item() > 0
    torch.testing.assert_close(token[1], torch.zeros(32))
    torch.testing.assert_close(age[:, 0], torch.tensor([0.0, 1.0]))


def test_e2_uses_history_memory_but_never_current_visual_token():
    torch.manual_seed(47)
    experts = HeterogeneousPhysicsExperts(d_model=32, hidden_dim=16).eval()
    batch = 2
    visual = torch.randn(batch, 32)
    common = (
        torch.randn(batch, 7),
        torch.randn(batch, 6),
        torch.randn(batch, 8, 6),
        torch.randn(batch, 10, 6),
        torch.randn(batch, 10, 7),
        torch.randn(batch, 4),
    )
    memory_a = torch.randn(batch, 32)
    # Use a genuinely different pattern. A constant offset is intentionally
    # removed by the expert's LayerNorm and is therefore not a valid contrast.
    memory_b = torch.randn(batch, 32)
    output_a = experts(visual, *common, visual_memory=memory_a)
    output_visual_changed = experts(visual + 100.0, *common, visual_memory=memory_a)
    output_memory_changed = experts(visual, *common, visual_memory=memory_b)

    assert not torch.allclose(output_a[:, 0], output_visual_changed[:, 0])
    torch.testing.assert_close(output_a[:, 1], output_visual_changed[:, 1])
    assert not torch.allclose(output_a[:, 1], output_memory_changed[:, 1])
    torch.testing.assert_close(output_a[:, 0], output_memory_changed[:, 0])
    torch.testing.assert_close(output_a[:, 2], output_memory_changed[:, 2])
    torch.testing.assert_close(output_a[:, 3], output_memory_changed[:, 3])


def test_physics_experts_are_structurally_heterogeneous_and_all_receive_gradients():
    torch.manual_seed(11)
    batch, dim, hidden = 2, 64, 32
    experts = HeterogeneousPhysicsExperts(dim, hidden)
    assert isinstance(experts.free_load, FreeMotionExpert)
    assert isinstance(experts.visual_blind, VisualBlindForceExpert)
    assert isinstance(experts.rigid_micro, RigidContactExpert)
    assert isinstance(experts.compliant, CompliantInsertionExpert)

    outputs = experts(
        torch.randn(batch, dim),
        torch.randn(batch, 7),
        torch.randn(batch, 6),
        torch.randn(batch, 16, 6),
        torch.randn(batch, 20, 6),
        torch.randn(batch, 10, 7),
        torch.randn(batch, 4),
    )
    assert outputs.shape == (batch, 4, dim)
    routing = torch.full((batch, 4), 0.25)
    semantic = torch.randn(batch, 1, dim)
    conditioner = ActionTokenConditioner(dim, action_dim=32, nhead=4)
    # The production conditioner starts as an identity residual. After its
    # output projection receives updates, action loss must reach every expert.
    nn.init.xavier_uniform_(conditioner.cross_attn.out_proj.weight)
    action_tokens = torch.randn(batch, 10, 32)
    condition_weights = torch.cat([torch.ones(batch, 1), routing], dim=1)
    conditioned = conditioner(
        action_tokens,
        torch.cat([semantic, outputs], dim=1),
        condition_weights,
    )
    conditioned.square().mean().backward()
    for expert in (
        experts.free_load,
        experts.visual_blind,
        experts.rigid_micro,
        experts.compliant,
    ):
        assert any(parameter.grad is not None for parameter in expert.parameters())


def test_physics_expert_auxiliary_heads_decode_distinct_descriptor_shapes():
    heads = PhysicsExpertAuxiliaryHeads(
        d_model=32, hidden_dim=16, state_dim=7, force_dim=6, quality_dim=4
    )
    expert_tokens = torch.randn(3, 4, 32, requires_grad=True)
    predictions = heads(expert_tokens)

    assert predictions["E1"].shape == (3, 13)
    assert predictions["E2"].shape == (3, 17)
    assert predictions["E3"].shape == (3, 19)
    assert predictions["E4"].shape == (3, 19)

    sum(value.square().mean() for value in predictions.values()).backward()
    assert expert_tokens.grad is not None
    assert torch.count_nonzero(expert_tokens.grad).item() > 0


def test_soft_routing_and_action_conditioner_are_differentiable():
    batch, dim = 2, 64
    expert_tokens = torch.randn(batch, 4, dim, requires_grad=True)
    routing = torch.softmax(torch.randn(batch, 4), dim=-1)
    weighted, fused = SoftExpertRouter()(expert_tokens, routing)
    assert weighted.shape == (batch, 4, dim)
    assert fused.shape == (batch, dim)

    conditioner = ActionTokenConditioner(dim, action_dim=32, nhead=4)
    action_tokens = torch.randn(batch, 10, 32, requires_grad=True)
    output = conditioner(action_tokens, weighted)
    assert output.shape == action_tokens.shape
    # Zero-initialized output projection preserves the pretrained action path.
    torch.testing.assert_close(output, action_tokens)
    output.square().mean().backward()
    assert action_tokens.grad is not None


def test_conditioner_applies_soft_routes_after_projection_and_masks_zero_routes():
    torch.manual_seed(13)
    conditioner = ActionTokenConditioner(condition_dim=16, action_dim=16, nhead=4)
    tokens = torch.randn(1, 5, 16)
    weights = torch.tensor([[1.0, 1.0, 0.1, 0.0, 0.5]])

    unweighted, _ = conditioner._project_condition_tokens(tokens)
    weighted, padding_mask = conditioner._project_condition_tokens(tokens, weights)

    torch.testing.assert_close(weighted, unweighted * weights.unsqueeze(-1))
    assert padding_mask.tolist() == [[False, False, False, True, False]]


def test_conditioner_all_zero_routes_is_exact_identity():
    torch.manual_seed(17)
    conditioner = ActionTokenConditioner(condition_dim=16, action_dim=16, nhead=4)
    nn.init.normal_(conditioner.cross_attn.out_proj.weight, std=0.1)
    nn.init.normal_(conditioner.cross_attn.out_proj.bias, std=0.1)
    action_tokens = torch.randn(2, 5, 16)
    condition_tokens = torch.randn(2, 4, 16)
    zero_weights = torch.zeros(2, 4)

    output = conditioner(action_tokens, condition_tokens, zero_weights)

    torch.testing.assert_close(output, action_tokens)
    assert torch.isfinite(output).all()


def test_conditioner_supports_per_action_step_routes():
    torch.manual_seed(19)
    conditioner = ActionTokenConditioner(
        condition_dim=16, action_dim=16, nhead=4, zero_init_output=False
    )
    actions = torch.randn(2, 5, 16)
    conditions = torch.randn(2, 4, 16)
    routes = torch.softmax(torch.randn(2, 5, 4), dim=-1)
    routes[1, 3] = 0

    output = conditioner(actions, conditions, routes)

    assert output.shape == actions.shape
    torch.testing.assert_close(output[1, 3], actions[1, 3])
    assert not torch.allclose(output[0, 0], output[0, 1])
    assert torch.isfinite(output).all()


def test_conditioner_zero_expert_scales_is_identity_and_scales_receive_gradient():
    torch.manual_seed(23)
    conditioner = ActionTokenConditioner(
        condition_dim=16,
        action_dim=16,
        nhead=4,
        zero_init_output=False,
    )
    action_tokens = torch.randn(2, 5, 16)
    condition_tokens = torch.randn(2, 4, 16)
    routing = torch.softmax(torch.randn(2, 4), dim=-1)
    scales = torch.zeros(4, requires_grad=True)

    output = conditioner(
        action_tokens,
        condition_tokens,
        routing,
        condition_token_scales=torch.tanh(scales),
        residual_max_norm=0.25,
    )
    torch.testing.assert_close(output, action_tokens)
    output.square().mean().backward()
    assert scales.grad is not None
    assert torch.isfinite(scales.grad).all()


def test_conditioner_bounds_each_action_token_residual():
    torch.manual_seed(29)
    conditioner = ActionTokenConditioner(
        condition_dim=16,
        action_dim=16,
        nhead=4,
        zero_init_output=False,
    )
    nn.init.normal_(conditioner.cross_attn.out_proj.weight, std=1.0)
    action_tokens = torch.randn(2, 5, 16)
    output = conditioner(
        action_tokens,
        torch.randn(2, 4, 16),
        torch.full((2, 4), 0.25),
        condition_token_scales=torch.ones(4),
        residual_max_norm=0.2,
    )
    residual_norm = (output - action_tokens).norm(dim=-1)
    assert bool((residual_norm <= 0.20001).all())


def test_proprio_history_and_soft_target_loss():
    encoder = ProprioHistoryEncoder(state_dim=7, hidden_dim=16, output_dim=32)
    token = encoder(torch.randn(3, 7), torch.randn(3, 10, 7))
    assert token.shape == (3, 32)

    logits = torch.randn(3, 4, requires_grad=True)
    targets = torch.tensor([[0.7, 0.3, 0.0, 0.0], [0.0, 0.2, 0.5, 0.3], [0.1, 0.1, 0.1, 0.7]])
    loss = soft_target_cross_entropy(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None

    rows = soft_target_cross_entropy(logits.detach(), targets, reduction="none")
    assert rows.shape == (3,)
    torch.testing.assert_close(rows.mean(), soft_target_cross_entropy(logits.detach(), targets))

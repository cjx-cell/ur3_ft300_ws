from types import SimpleNamespace

import torch
import torch.nn as nn

from lerobot.policies.pap_moe.modeling_pap_moe import (
    PAPMoEPi05Model,
    PAPMoEPolicy,
    _filter_legacy_gate_outputs_for_factorized_gate,
)
from lerobot.policies.pap_moe.pap_moe_modules import (
    ActionTokenConditioner,
    LegacyTokenPhysicsExperts,
    PhysicsExpertAuxiliaryHeads,
    PhysicsGate,
    SoftExpertRouter,
)


class _DummyPAPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.force_encoder = nn.Linear(2, 2)
        self.proprio_encoder = nn.Linear(2, 2)
        self.visual_quality_encoder = nn.Linear(2, 2)
        self.visual_history_encoder = None
        self.physics_gate = nn.Linear(2, 2)
        self.expert_library = nn.Linear(2, 2)
        self.expert_auxiliary_heads = nn.Linear(2, 2)
        self.action_conditioner = nn.Linear(2, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.time_mlp_in = nn.Linear(2, 2)
        self.time_mlp_out = nn.Linear(2, 2)
        self.backbone = nn.Linear(2, 2)
        self.lora_probe = nn.Parameter(torch.zeros(1))
        self.gemma_expert_lora_A = nn.Parameter(torch.zeros(1))
        self.gemma_expert_lora_B = nn.Parameter(torch.zeros(1))
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert.lm_head = nn.Linear(2, 2)


def _config(mode: str) -> SimpleNamespace:
    names = (
        "train_expert_action_joint", "train_physicsgate_action_joint", "train_expert_only", "train_physicsgate_only", "train_gate_calibration_only",
        "train_conditioner_only", "train_action_adapter_only", "train_pap_moe_joint",
    )
    return SimpleNamespace(
        **{name: name == mode for name in names},
        train_arm_head_only=False,
        train_gripper_head_only=False,
        action_adapter_train_lora=True,
    )


def _trainable_roots(mode: str) -> set[str]:
    model = _DummyPAPModel()
    PAPMoEPi05Model._apply_training_stage(model, _config(mode))
    return {
        name.split(".", maxsplit=1)[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def test_physicsgate_stage_isolation():
    assert _trainable_roots("train_physicsgate_only") == {
        "force_encoder", "proprio_encoder", "visual_quality_encoder", "physics_gate"
    }


def test_expert_stage_isolation():
    assert _trainable_roots("train_expert_only") == {
        "expert_library", "expert_auxiliary_heads", "action_conditioner"
    }


def test_expert_action_joint_trains_both_sides_of_conditioning_interface():
    assert _trainable_roots("train_expert_action_joint") == {
        "expert_library",
        "expert_auxiliary_heads",
        "action_conditioner",
        "paligemma_with_expert",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    }


def test_physicsgate_action_joint_trains_complete_prediction_aligned_policy():
    assert _trainable_roots("train_physicsgate_action_joint") == {
        "force_encoder", "proprio_encoder", "visual_quality_encoder", "physics_gate",
        "expert_library", "action_conditioner", "paligemma_with_expert",
        "expert_auxiliary_heads",
        "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
    }


def test_physical_conditioning_can_preserve_unconditioned_gripper_velocity():
    model = SimpleNamespace(
        config=SimpleNamespace(
            condition_gripper_with_physical_experts=False,
            gripper_action_index=2,
        )
    )
    baseline = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    conditioned = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])

    output = PAPMoEPi05Model._preserve_unconditioned_gripper_velocity(
        model, baseline, conditioned
    )

    torch.testing.assert_close(output, torch.tensor([[[10.0, 20.0, 3.0, 40.0]]]))
    torch.testing.assert_close(conditioned, torch.tensor([[[10.0, 20.0, 30.0, 40.0]]]))


def _visual_memory_policy_stub():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            use_visual_memory=True,
            visual_memory_history_indices=(-30, -20, -10, 0),
            image_features=("observation.images.wrist", "observation.images.global"),
        ),
        _online_visual_history={},
    )

    def preprocess_images(batch):
        keys = policy.config.image_features
        return [batch[key] for key in keys], [torch.ones(batch[key].shape[0]) for key in keys]

    policy._preprocess_images = preprocess_images
    return policy


def test_visual_memory_training_keeps_only_current_frame_for_pi05():
    policy = _visual_memory_policy_stub()
    wrist = torch.rand(2, 4, 3, 8, 8)
    global_image = torch.rand(2, 4, 3, 8, 8)
    wrist_pad = torch.tensor([[True, False, False, False], [True, True, False, False]])
    batch = {
        "observation.images.wrist": wrist,
        "observation.images.global": global_image,
        "observation.images.wrist_is_pad": wrist_pad,
    }

    images, _, histories, history_padding = PAPMoEPolicy._preprocess_pap_images(policy, batch)

    torch.testing.assert_close(images[0], wrist[:, -1])
    torch.testing.assert_close(images[1], global_image[:, -1])
    torch.testing.assert_close(histories[0], wrist[:, :-1])
    torch.testing.assert_close(histories[1], global_image[:, :-1])
    torch.testing.assert_close(history_padding[0], wrist_pad[:, :-1])
    assert not history_padding[1].any()


def test_visual_memory_online_queue_uses_previous_queries_only():
    policy = _visual_memory_policy_stub()
    first = {
        key: torch.rand(1, 3, 8, 8) for key in policy.config.image_features
    }
    second = {
        key: torch.rand(1, 3, 8, 8) for key in policy.config.image_features
    }

    _, _, first_history, first_padding = PAPMoEPolicy._preprocess_pap_images(
        policy, first, update_online_memory=True
    )
    _, _, second_history, second_padding = PAPMoEPolicy._preprocess_pap_images(
        policy, second, update_online_memory=True
    )

    assert first_padding[0].all()
    assert second_padding[0][0, :-1].all()
    assert not second_padding[0][0, -1]
    torch.testing.assert_close(second_history[0][:, -1], first[policy.config.image_features[0]])
    assert len(policy._online_visual_history[policy.config.image_features[0]]) == 2


def test_visual_memory_distillation_excludes_missing_history_and_blind_teacher():
    memory = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    teacher = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    age = torch.tensor([[0.0], [1.0], [0.0]])
    routes = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )

    loss = PAPMoEPi05Model._compute_visual_memory_distillation_loss(
        memory, age, teacher, routes
    )

    # Only the first sample has both valid history and a clean current view.
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_legacy_physical_conditioning_can_still_modify_gripper_velocity():
    model = SimpleNamespace(
        config=SimpleNamespace(
            condition_gripper_with_physical_experts=True,
            gripper_action_index=2,
        )
    )
    baseline = torch.zeros(1, 1, 4)
    conditioned = torch.ones(1, 1, 4)

    output = PAPMoEPi05Model._preserve_unconditioned_gripper_velocity(
        model, baseline, conditioned
    )

    assert output is conditioned


def test_action_input_fusion_mode_is_explicit_and_legacy_default_is_false():
    legacy = SimpleNamespace(
        config=SimpleNamespace(physical_fusion_architecture="late_output_v1")
    )
    vnext = SimpleNamespace(
        config=SimpleNamespace(physical_fusion_architecture="action_input_tokens_v2")
    )

    assert not PAPMoEPi05Model._uses_action_input_fusion(legacy)
    assert PAPMoEPi05Model._uses_action_input_fusion(vnext)


def test_vnext_training_and_inference_share_action_input_fusion_entry_point():
    torch.manual_seed(41)
    conditioner = ActionTokenConditioner(
        condition_dim=8,
        action_dim=8,
        nhead=2,
        zero_init_output=False,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            physical_fusion_architecture="action_input_tokens_v2",
            bounded_action_conditioning=False,
            action_conditioning_scale=1.0,
        ),
        action_conditioner=conditioner,
    )
    model._uses_action_input_fusion = PAPMoEPi05Model._uses_action_input_fusion.__get__(model)
    model._apply_physical_conditioning = PAPMoEPi05Model._apply_physical_conditioning.__get__(model)

    action_inputs = torch.randn(2, 5, 8, requires_grad=True)
    experts = torch.randn(2, 4, 8, requires_grad=True)
    routes = torch.softmax(torch.randn(2, 5, 4), dim=-1)
    pap_result = {
        "conditioning_tokens": experts,
        "conditioning_weights": routes,
        "routing_confidence": torch.ones(2, 5),
    }

    conditioned, diagnostics = PAPMoEPi05Model._fuse_physics_into_action_input(
        model, action_inputs, pap_result
    )
    assert conditioned.shape == action_inputs.shape
    assert not torch.allclose(conditioned, action_inputs)
    assert diagnostics["applied_condition_residual_norm"].shape == (2,)

    conditioned.square().mean().backward()
    assert action_inputs.grad is not None
    assert experts.grad is not None


def test_expert_representation_loss_trains_physical_descriptors_not_actions():
    model = SimpleNamespace(
        expert_auxiliary_heads=PhysicsExpertAuxiliaryHeads(
            d_model=16, hidden_dim=8, state_dim=7, force_dim=6, quality_dim=4
        ),
        config=SimpleNamespace(robot_state_dim=7, visual_quality_dim=4,
                               expert_motion_target="history_window_delta_v2"),
    )
    expert_tokens = torch.randn(4, 4, 16, requires_grad=True)
    routes = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    state = torch.randn(4, 7)
    history = torch.randn(4, 10, 7)
    force = torch.randn(4, 6)
    fast = torch.randn(4, 64, 6)
    slow = torch.randn(4, 50, 6)
    quality = torch.randn(4, 4)

    model._expert_motion_descriptor = lambda state, history: PAPMoEPi05Model._expert_motion_descriptor(model, state, history)
    loss, per_expert = PAPMoEPi05Model._compute_expert_representation_loss(
        model, expert_tokens, routes, state, history, force, fast, slow, quality
    )
    assert loss.ndim == 0
    assert set(per_expert) == {"E1", "E2", "E3", "E4"}
    loss.backward()
    assert expert_tokens.grad is not None
    assert torch.count_nonzero(expert_tokens.grad).item() > 0


def test_expert_action_joint_uses_lower_lr_for_pretrained_action_generator():
    model = _DummyPAPModel()
    stage_config = _config("train_expert_action_joint")
    PAPMoEPi05Model._apply_training_stage(model, stage_config)
    policy = SimpleNamespace(
        model=model,
        config=SimpleNamespace(
            train_expert_action_joint=True,
            pap_moe_optimizer_lr=2.5e-5,
            joint_action_expert_lr_scale=0.1,
        ),
        parameters=model.parameters,
    )
    groups = PAPMoEPolicy.get_optim_params(policy)
    assert len(groups) == 2
    assert "lr" not in groups[0]
    assert groups[1]["lr"] == 2.5e-6
    assert sum(parameter.numel() for parameter in groups[0]["params"]) > 0
    assert sum(parameter.numel() for parameter in groups[1]["params"]) > 0


def test_gate_calibration_stage_isolation():
    assert _trainable_roots("train_gate_calibration_only") == {
        "force_encoder", "proprio_encoder", "visual_quality_encoder", "physics_gate"
    }


def test_frozen_action_conditioner_propagates_action_loss_to_routing_weights():
    conditioner = ActionTokenConditioner(condition_dim=8, action_dim=8, nhead=2)
    nn.init.normal_(conditioner.cross_attn.out_proj.weight, std=0.1)
    for parameter in conditioner.parameters():
        parameter.requires_grad = False

    action_tokens = torch.randn(2, 5, 8)
    expert_tokens = torch.randn(2, 4, 8)
    routing_logits = torch.randn(2, 4, requires_grad=True)
    routing_probs = routing_logits.softmax(dim=-1)
    output = conditioner(action_tokens, expert_tokens, routing_probs)
    output.square().mean().backward()

    assert routing_logits.grad is not None
    assert torch.isfinite(routing_logits.grad).all()
    assert routing_logits.grad.abs().sum() > 0
    assert not any(parameter.grad is not None for parameter in conditioner.parameters())


def test_conditioner_stage_isolation():
    assert _trainable_roots("train_conditioner_only") == {"action_conditioner"}


def test_action_adapter_stage_isolation():
    assert _trainable_roots("train_action_adapter_only") == {
        "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
        "gemma_expert_lora_A", "gemma_expert_lora_B",
    }


def test_projection_only_action_adapter_freezes_expert_lora():
    model = _DummyPAPModel()
    config = _config("train_action_adapter_only")
    config.action_adapter_train_lora = False
    PAPMoEPi05Model._apply_training_stage(model, config)
    assert {
        name.split(".", maxsplit=1)[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    } == {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}


def test_action_adapter_source_snapshot_only_copies_trainable_parameters():
    trainable = nn.Parameter(torch.tensor([1.0]))
    frozen = nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    holder = SimpleNamespace(
        model=nn.ParameterDict({"trainable": trainable, "frozen": frozen}),
        _action_adapter_source_parameters=None,
    )
    snapshot = PAPMoEPolicy._get_action_adapter_source_parameters(holder)
    assert set(snapshot) == {"trainable"}
    trainable.data.add_(3.0)
    assert snapshot["trainable"].item() == 1.0


def test_conditioning_interface_contains_only_routed_physics_experts():
    batch, dim, hidden = 2, 16, 8
    model = SimpleNamespace(
        force_encoder=nn.Linear(1, 1),
        physics_gate=PhysicsGate(dim, num_experts=4, hidden_dim=hidden, nhead=2),
        expert_library=LegacyTokenPhysicsExperts(dim, hidden),
        soft_router=SoftExpertRouter(),
        config=SimpleNamespace(
            num_experts=4,
            physics_expert_architecture="legacy_token_adapters",
            conditioner_routing_mode="post_projection_v2",
        ),
    )
    force_tokens = {name: torch.randn(batch, dim) for name in ("current", "fast", "slow", "fused")}
    model._pool_physics_prefix = lambda features, mask: PAPMoEPi05Model._pool_physics_prefix(
        model, features, mask
    )
    outputs = PAPMoEPi05Model._forward_pap_moe(
        model,
        force_tokens,
        torch.randn(batch, 3, dim),
        torch.randn(batch, dim),
        torch.randn(batch, dim),
        stage_override=torch.full((batch, 4), 0.25),
        expert_mask=torch.zeros(4),
    )
    assert outputs["conditioning_tokens"].shape == (batch, 4, dim)
    assert outputs["conditioning_weights"].shape == (batch, 4)
    torch.testing.assert_close(outputs["conditioning_weights"], torch.zeros(batch, 4))


def test_joint_ablation_trainable_roots():
    assert _trainable_roots("train_pap_moe_joint") == {
        "force_encoder", "proprio_encoder", "visual_quality_encoder", "physics_gate",
        "expert_library", "expert_auxiliary_heads", "action_conditioner",
        "action_in_proj", "action_out_proj",
        "time_mlp_in", "time_mlp_out", "lora_probe", "gemma_expert_lora_A",
        "gemma_expert_lora_B",
    }


def test_factorized_gate_migration_skips_only_legacy_output_heads():
    state_dict = {
        "model.physics_gate.context_head.3.weight": torch.randn(4, 256),
        "model.physics_gate.context_head.3.bias": torch.randn(4),
        "model.physics_gate.fast_force_head.3.weight": torch.randn(4, 256),
        "model.physics_gate.fast_force_head.3.bias": torch.randn(4),
        "model.physics_gate.context_head.1.weight": torch.randn(256, 1024),
        "model.expert_library.adapters.0.weight": torch.randn(8, 8),
        "model.action_in_proj.weight": torch.randn(8, 8),
    }

    filtered, skipped = _filter_legacy_gate_outputs_for_factorized_gate(
        state_dict, "factorized_bcm_v1"
    )

    assert set(skipped) == {
        "model.physics_gate.context_head.3.weight",
        "model.physics_gate.context_head.3.bias",
        "model.physics_gate.fast_force_head.3.weight",
        "model.physics_gate.fast_force_head.3.bias",
    }
    assert set(filtered) == {
        "model.physics_gate.context_head.1.weight",
        "model.expert_library.adapters.0.weight",
        "model.action_in_proj.weight",
    }
    assert filtered["model.expert_library.adapters.0.weight"] is state_dict[
        "model.expert_library.adapters.0.weight"
    ]


def test_factorized_gate_checkpoint_keeps_its_three_output_heads():
    state_dict = {
        "model.physics_gate.context_head.3.weight": torch.randn(3, 256),
        "model.physics_gate.context_head.3.bias": torch.randn(3),
    }
    filtered, skipped = _filter_legacy_gate_outputs_for_factorized_gate(
        state_dict, "factorized_bcm_v1"
    )
    assert filtered is state_dict
    assert skipped == []

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Test script to verify PI0.5 (pi05) support in PI0 policy"""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.factory import make_policy_config  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    make_pi05_pre_post_processors,  # noqa: E402
)
from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    ContextualGripperHead,
    DeterministicArmHead,
    _clip_normalized_arm_chunk,
    _compute_gripper_head_loss,
    _compute_weighted_action_loss,
    _mask_state_history_channel,
    _pool_last_valid_prefix,
    _pool_mean_valid_prefix,
    _summarize_release_observations,
)
from lerobot.utils.random_utils import set_seed


def test_fix_state_dict_keys_supports_both_vision_tower_layouts():
    class PolicyLayout:
        def __init__(self, expected_key: str):
            self.expected_key = expected_key

        def state_dict(self):
            return {self.expected_key: torch.empty(0)}

    legacy = "model.paligemma.model.vision_tower.embeddings.weight"
    wrapped = "model.paligemma.model.vision_tower.vision_model.embeddings.weight"
    value = torch.ones(1)

    upgraded = PI05Policy._fix_pytorch_state_dict_keys(
        PolicyLayout(wrapped), {legacy: value}, None
    )
    assert list(upgraded) == [wrapped]

    downgraded = PI05Policy._fix_pytorch_state_dict_keys(
        PolicyLayout(legacy), {wrapped: value}, None
    )
    assert list(downgraded) == [legacy]


def test_controlled_ablation_sensor_mode_preserves_graph_and_masks_values():
    class Holder:
        pass

    holder = Holder()
    holder.config = Holder()
    value = torch.randn(2, 6)
    batch = {"observation.force": value}
    holder.config.controlled_ablation_sensor_mode = "vsf"
    assert PI05Policy._controlled_sensor(holder, batch, "observation.force") is value
    holder.config.controlled_ablation_sensor_mode = "vs"
    masked = PI05Policy._controlled_sensor(holder, batch, "observation.force")
    torch.testing.assert_close(masked, torch.zeros_like(value))
from tests.utils import require_cuda, require_hf_token  # noqa: E402


def test_gripper_action_loss_weighting_is_normalized_and_open_aware():
    losses = torch.ones(2, 2, 3)
    actions = torch.tensor(
        [
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]
    )

    per_sample, weights = _compute_weighted_action_loss(
        losses,
        actions,
        gripper_action_index=2,
        gripper_loss_weight=4.0,
        gripper_open_loss_weight=2.0,
        gripper_open_threshold_normalized=0.0,
    )

    torch.testing.assert_close(weights[0, :, 2], torch.tensor([8.0, 4.0]))
    torch.testing.assert_close(weights[1, :, 2], torch.tensor([4.0, 4.0]))
    # Constant element losses remain one after normalization, regardless of
    # the target-dependent weights.
    torch.testing.assert_close(per_sample, torch.ones(2))


def test_gripper_action_loss_default_matches_unweighted_mean():
    losses = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    per_sample, weights = _compute_weighted_action_loss(
        losses,
        torch.zeros_like(losses),
        gripper_action_index=None,
        gripper_loss_weight=1.0,
        gripper_open_loss_weight=1.0,
        gripper_open_threshold_normalized=0.0,
    )
    torch.testing.assert_close(per_sample, losses.mean(dim=(1, 2)))
    torch.testing.assert_close(weights, torch.ones_like(losses))


def test_action_prefix_loss_weighting_targets_executed_horizon_and_is_normalized():
    losses = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    per_sample, weights = _compute_weighted_action_loss(
        losses,
        torch.zeros_like(losses),
        gripper_action_index=None,
        gripper_loss_weight=1.0,
        gripper_open_loss_weight=1.0,
        gripper_open_threshold_normalized=0.0,
        action_prefix_loss_horizon=2,
        action_prefix_loss_weight=4.0,
    )

    torch.testing.assert_close(weights.flatten(), torch.tensor([4.0, 4.0, 1.0, 1.0]))
    torch.testing.assert_close(per_sample, torch.tensor([(4.0 + 8.0 + 3.0 + 4.0) / 10.0]))


def test_action_prefix_and_gripper_weights_compose_multiplicatively():
    losses = torch.ones(1, 2, 2)
    actions = torch.tensor([[[0.0, -1.0], [0.0, 1.0]]])
    per_sample, weights = _compute_weighted_action_loss(
        losses,
        actions,
        gripper_action_index=1,
        gripper_loss_weight=3.0,
        gripper_open_loss_weight=2.0,
        gripper_open_threshold_normalized=0.0,
        action_prefix_loss_horizon=1,
        action_prefix_loss_weight=4.0,
    )

    torch.testing.assert_close(weights[0], torch.tensor([[4.0, 24.0], [1.0, 3.0]]))
    torch.testing.assert_close(per_sample, torch.ones(1))


def test_gripper_head_pooling_skips_internal_padding_gaps():
    hidden = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    masks = torch.tensor(
        [
            [True, True, False, False, True, False],
            [False, True, False, True, False, True],
        ]
    )
    pooled = _pool_last_valid_prefix(hidden, masks)
    torch.testing.assert_close(pooled[0], hidden[0, 4])
    torch.testing.assert_close(pooled[1], hidden[1, 5])
    mean_pooled = _pool_mean_valid_prefix(hidden, masks)
    torch.testing.assert_close(mean_pooled[0], hidden[0, [0, 1, 4]].mean(dim=0))
    torch.testing.assert_close(mean_pooled[1], hidden[1, [1, 3, 5]].mean(dim=0))


def test_gripper_head_loss_is_open_weighted_and_scale_normalized():
    logits = torch.zeros(2, 2)
    normalized_targets = torch.tensor([[-1.0, 1.0], [1.0, 1.0]])
    per_sample, target_closed, weights = _compute_gripper_head_loss(
        logits,
        normalized_targets,
        open_threshold_normalized=0.0,
        open_loss_weight=3.0,
        loss_weight=2.0,
    )
    torch.testing.assert_close(target_closed, torch.tensor([[False, True], [True, True]]))
    torch.testing.assert_close(weights, torch.tensor([[3.0, 1.0], [1.0, 1.0]]))
    # Every zero logit has BCE log(2); normalization removes class-weight scale,
    # while the explicit overall loss weight remains.
    torch.testing.assert_close(per_sample, torch.full((2,), 2.0 * torch.log(torch.tensor(2.0))))


def test_gripper_head_config_requires_explicit_action_index():
    with pytest.raises(ValueError, match="requires gripper_action_index"):
        PI05Config(use_deterministic_gripper_head=True)
    with pytest.raises(ValueError, match="requires use_deterministic_gripper_head"):
        PI05Config(train_gripper_head_only=True)


def test_release_sensor_summary_uses_force_and_proprio_history():
    force = torch.tensor([[1.0, 2.0]])
    force_fast = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    force_slow = torch.tensor([[[2.0, 4.0], [4.0, 8.0]]])
    state_history = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]]])
    summary = _summarize_release_observations(
        force,
        force_fast,
        force_slow,
        state_history,
        force_dim=2,
        state_dim=3,
    )
    assert summary.shape == (1, 24)
    # Last proprio state and last-first motion delta are retained explicitly.
    torch.testing.assert_close(summary[0, 12:15], state_history[0, -1])
    torch.testing.assert_close(summary[0, -3:], state_history[0, -1] - state_history[0, 0])


def test_gripper_history_mask_removes_causal_shortcut_without_mutating_input():
    history = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7)
    original = history.clone()
    masked = _mask_state_history_channel(history, 6)
    torch.testing.assert_close(history, original)
    torch.testing.assert_close(masked[..., :6], history[..., :6])
    torch.testing.assert_close(masked[..., 6], torch.zeros_like(masked[..., 6]))


def test_release_override_config_is_conservative_and_adds_sensor_features():
    config = PI05Config(
        gripper_action_index=6,
        use_release_gripper_override=True,
        max_action_dim=7,
        max_state_dim=7,
    )
    config.validate_features()
    assert config.release_head_probability_threshold > 0.5
    assert config.input_features["observation.force"].shape == (6,)
    assert config.input_features["observation.force_fast"].shape == (64, 6)
    assert config.input_features["observation.force_slow"].shape == (50, 6)
    assert config.input_features["observation.state_history"].shape == (10, 7)
    with pytest.raises(ValueError, match="mutually exclusive"):
        PI05Config(
            gripper_action_index=6,
            use_deterministic_gripper_head=True,
            use_release_gripper_override=True,
        )


def test_contextual_gripper_can_compose_with_release_override_and_adds_sensors():
    config = PI05Config(
        gripper_action_index=6,
        use_deterministic_gripper_head=True,
        gripper_head_use_sensor_context=True,
        use_release_gripper_override=True,
        max_action_dim=7,
        max_state_dim=7,
    )
    config.validate_features()
    assert config.input_features["observation.force"].shape == (6,)
    assert config.input_features["observation.state_history"].shape == (10, 7)

    head = ContextualGripperHead(visual_dim=8, sensor_dim=6, hidden_dim=12, chunk_size=5)
    logits = head(torch.zeros(2, 8), torch.zeros(2, 6))
    assert logits.shape == (2, 5)

    semantic_head = ContextualGripperHead(
        visual_dim=8,
        sensor_dim=6,
        hidden_dim=12,
        chunk_size=5,
        semantic_context_dim=4,
    )
    context = torch.randn(2, 4)
    # The new branch is zero initialized, so enabling it preserves the source
    # checkpoint until semantic-only gripper training updates its projection.
    assert torch.allclose(
        semantic_head(torch.zeros(2, 8), torch.zeros(2, 6), context),
        semantic_head(torch.zeros(2, 8), torch.zeros(2, 6), torch.zeros_like(context)),
    )
    with pytest.raises(ValueError, match="requires semantic_context"):
        semantic_head(torch.zeros(2, 8), torch.zeros(2, 6))


def test_deterministic_arm_head_predicts_state_residual_chunk():
    head = DeterministicArmHead(
        visual_dim=8,
        sensor_dim=6,
        hidden_dim=12,
        chunk_size=5,
        action_dim=3,
        residual_scale=0.5,
    )
    with torch.no_grad():
        head.regressor[-1].weight.zero_()
        head.regressor[-1].bias.zero_()
    state = torch.tensor([[0.1, -0.2, 0.3, 0.4]])
    result = head(torch.zeros(1, 8), torch.zeros(1, 6), state)
    assert result.shape == (1, 5, 3)
    torch.testing.assert_close(result, state[:, None, :3].expand(-1, 5, -1))


def test_deterministic_arm_head_semantic_context_is_optional_and_noop_initialized():
    head = DeterministicArmHead(
        visual_dim=8,
        sensor_dim=6,
        hidden_dim=12,
        chunk_size=5,
        action_dim=3,
        residual_scale=0.5,
        semantic_context_dim=4,
    )
    state = torch.tensor([[0.1, -0.2, 0.3, 0.4]])
    with pytest.raises(ValueError, match="requires semantic_context"):
        head(torch.zeros(1, 8), torch.zeros(1, 6), state)
    result = head(
        torch.zeros(1, 8),
        torch.zeros(1, 6),
        state,
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    )
    # Both the new semantic projection and the existing residual output layer
    # initialize to zero, so enabling the feature cannot jump the robot.
    torch.testing.assert_close(result, state[:, None, :3].expand(-1, 5, -1))


def test_optional_arm_chunk_clip_preserves_quantile_tails_when_disabled():
    chunk = torch.tensor([[[-1.4, 0.2, 1.3]]])
    torch.testing.assert_close(_clip_normalized_arm_chunk(chunk, None), chunk)
    torch.testing.assert_close(
        _clip_normalized_arm_chunk(chunk, 1.0),
        torch.tensor([[[-1.0, 0.2, 1.0]]]),
    )


def test_arm_head_config_requires_explicit_training_mode_and_action_width():
    with pytest.raises(ValueError, match="requires use_deterministic_arm_head"):
        PI05Config(train_arm_head_only=True)
    with pytest.raises(ValueError, match="arm_head_residual_scale"):
        PI05Config(arm_head_residual_scale=0.0)
    with pytest.raises(ValueError, match="arm_head_history_noise_std"):
        PI05Config(arm_head_history_noise_std=-0.01)
    with pytest.raises(ValueError, match="arm_head_force_noise_std"):
        PI05Config(arm_head_force_noise_std=-0.01)
    with pytest.raises(ValueError, match="arm_head_normalized_output_clip"):
        PI05Config(arm_head_normalized_output_clip=0.0)
    with pytest.raises(ValueError, match="arm_head_semantic_context_dim"):
        PI05Config(arm_head_semantic_context_dim=-1)
    config = PI05Config(
        use_deterministic_arm_head=True,
        max_action_dim=7,
        max_state_dim=7,
    )
    config.validate_features()
    assert config.input_features["observation.state_history"].shape == (10, 7)


@require_cuda
@require_hf_token
def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),
        ),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    assert config.tokenizer_max_length == 200, (
        f"Expected tokenizer_max_length=200 for pi05, got {config.tokenizer_max_length}"
    )

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": torch.zeros(14),
            "max": torch.ones(14),
            "q01": torch.zeros(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": torch.zeros(7),
            "max": torch.ones(7),
            "q01": torch.zeros(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = PI05Policy(config)
    # Test forward pass with dummy data
    batch_size = 1
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
    device = config.device
    batch = {
        "observation.state": torch.randn(batch_size, 14, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
            print(f"Action: {action}")
        print(f"Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise

    # Verify pi05 model components exist
    # Check that time_mlp layers exist (for AdaRMS conditioning)
    assert hasattr(policy.model, "time_mlp_in"), "Missing time_mlp_in layer for pi05"
    assert hasattr(policy.model, "time_mlp_out"), "Missing time_mlp_out layer for pi05"

    # Check that action_time_mlp layers don't exist (pi0 only)
    assert not hasattr(policy.model, "action_time_mlp_in"), "action_time_mlp_in should not exist in pi05 mode"
    assert not hasattr(policy.model, "action_time_mlp_out"), (
        "action_time_mlp_out should not exist in pi05 mode"
    )

    # Check that state_proj doesn't exist in pi05 mode
    assert not hasattr(policy.model, "state_proj"), "state_proj should not exist in pi05 mode"

    # Check AdaRMS configuration in the underlying model
    adarms_config = policy.model.paligemma_with_expert.paligemma.config.text_config.use_adarms
    assert adarms_config == False, f"PaliGemma should not use AdaRMS, got {adarms_config}"  # noqa: E712

    adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
    assert adarms_expert_config == True, (  # noqa: E712
        f"Action expert should use AdaRMS in pi05, got {adarms_expert_config}"
    )

    # Pi0.5 consumes the expert hidden states directly; its inherited language
    # vocabulary head is not part of action generation and must not be optimized.
    assert all(
        not parameter.requires_grad
        for parameter in policy.model.paligemma_with_expert.gemma_expert.lm_head.parameters()
    )


@require_cuda
@require_hf_token
def test_config_creation():
    """Test policy config creation through factory."""
    try:
        config = make_policy_config(
            policy_type="pi0",
            max_action_dim=7,
            max_state_dim=14,
        )
        print("Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
        print(f"  PaliGemma variant: {config.paligemma_variant}")
        print(f"  Action expert variant: {config.action_expert_variant}")
    except Exception as e:
        print(f"Config creation failed: {e}")
        raise

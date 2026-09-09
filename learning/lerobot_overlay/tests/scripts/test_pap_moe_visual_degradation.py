from types import SimpleNamespace

import torch

from lerobot.scripts.lerobot_train import _apply_pap_moe_visual_degradation


def test_paired_dropout_relabels_free_and_contact_samples() -> None:
    batch = {
        "observation.images.camera0": torch.ones(2, 3, 4, 4),
        "observation.images.camera1": torch.full((2, 3, 4, 4), 0.5),
        "observation.visual_quality": torch.zeros(2, 4),
        "observation.physics_gate_target": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.6, 0.4]]
        ),
    }
    config = SimpleNamespace(
        type="pap_moe",
        visual_degradation_training_probability=1.0,
        visual_degradation_dropout_fraction=1.0,
        visual_degradation_glare_gain_min=2.0,
        visual_degradation_glare_gain_max=6.0,
    )

    target = _apply_pap_moe_visual_degradation(batch, config)

    assert target is not None
    assert torch.count_nonzero(batch["observation.images.camera0"]) == 0
    assert torch.count_nonzero(batch["observation.images.camera1"]) == 0
    torch.testing.assert_close(
        batch["observation.visual_quality"],
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(target[0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    # Blind rigid/compliant contact remains cooperative instead of becoming
    # an E2-only one-hot target.
    torch.testing.assert_close(target[1], torch.tensor([0.0, 0.5, 0.3, 0.2]))


def test_paired_dropout_relabels_every_future_route_without_mixing_time() -> None:
    route = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.6, 0.4], [0.2, 0.0, 0.8, 0.0]]]
    )
    batch = {
        "observation.images.camera0": torch.ones(1, 3, 4, 4),
        "observation.images.camera1": torch.ones(1, 3, 4, 4),
        "observation.visual_quality": torch.zeros(1, 4),
        "observation.physics_gate_target": route,
    }
    config = SimpleNamespace(
        type="pap_moe",
        visual_degradation_training_probability=1.0,
        visual_degradation_dropout_fraction=1.0,
        visual_degradation_glare_gain_min=2.0,
        visual_degradation_glare_gain_max=6.0,
    )

    target = _apply_pap_moe_visual_degradation(batch, config)

    assert target is not None
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(1, 3))
    torch.testing.assert_close(target[0, 0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(target[0, 1], torch.tensor([0.0, 0.5, 0.3, 0.2]))
    # Full blindness removes E1. With b=1, c=.8, m=0 the raw
    # weights are [0, 1, .8, 0], normalized by 1.8 (not by 2).
    torch.testing.assert_close(target[0, 2], torch.tensor([0.0, 1.0 / 1.8, 0.8 / 1.8, 0.0]))


def test_visual_degradation_changes_only_current_frame_of_memory_sequence() -> None:
    history = torch.rand(2, 4, 3, 8, 8)
    batch = {
        "observation.images.camera0": history.clone(),
        "observation.images.camera1": history.clone(),
        "observation.visual_quality": torch.zeros(2, 4),
        "observation.physics_gate_target": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
    }
    config = SimpleNamespace(
        type="pap_moe",
        visual_degradation_training_probability=1.0,
        visual_degradation_dropout_fraction=1.0,
        visual_degradation_glare_gain_min=2.0,
        visual_degradation_glare_gain_max=6.0,
    )

    _apply_pap_moe_visual_degradation(batch, config)

    torch.testing.assert_close(batch["observation.images.camera0"][:, :-1], history[:, :-1])
    assert torch.count_nonzero(batch["observation.images.camera0"][:, -1]) == 0


def test_camera_history_padding_masks_are_not_counted_as_cameras() -> None:
    history = torch.rand(2, 4, 3, 8, 8)
    batch = {
        "observation.images.camera0": history.clone(),
        "observation.images.camera0_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        "observation.images.camera1": history.clone(),
        "observation.images.camera1_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        "observation.visual_quality": torch.zeros(2, 4),
        "observation.physics_gate_target": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.6, 0.4]]
        ),
    }
    config = SimpleNamespace(
        type="pap_moe",
        visual_degradation_training_probability=1.0,
        visual_degradation_dropout_fraction=1.0,
        visual_degradation_glare_gain_min=2.0,
        visual_degradation_glare_gain_max=6.0,
    )

    target = _apply_pap_moe_visual_degradation(batch, config)

    assert target is not None
    assert torch.all(target[:, 1] > 0)

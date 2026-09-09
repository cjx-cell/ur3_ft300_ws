"""Focused tests for PAP-MoE Real-Time Chunking integration."""

from types import MethodType, SimpleNamespace

import torch
from torch import nn

from lerobot.policies.pap_moe.modeling_pap_moe import PAPMoEPi05Model


class _RecordingRTCProcessor:
    def __init__(self):
        self.call = None

    def denoise_step(self, **kwargs):
        self.call = kwargs
        return kwargs["original_denoise_step_partial"](kwargs["x_t"])


def _model_with_rtc(enabled: bool):
    model = PAPMoEPi05Model.__new__(PAPMoEPi05Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(rtc_config=SimpleNamespace(enabled=enabled))
    model.rtc_processor = _RecordingRTCProcessor()

    def fake_denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        *,
        conditioning_tokens,
        conditioning_weights,
    ):
        del (
            self,
            prefix_pad_masks,
            past_key_values,
            timestep,
            conditioning_tokens,
            conditioning_weights,
        )
        return x_t + 2.0

    model.denoise_step = MethodType(fake_denoise_step, model)
    return model


def _run_step(model, *, inference_delay=None):
    x_t = torch.zeros(1, 5, 3)
    placeholder = torch.zeros(1)
    result = model._denoise_action_step(
        x_t=x_t,
        timestep=torch.ones(1),
        time=0.5,
        prefix_pad_masks=placeholder,
        past_key_values=None,
        conditioning_tokens=placeholder,
        conditioning_weights=None,
        prev_chunk_left_over=torch.ones_like(x_t),
        inference_delay=inference_delay,
        execution_horizon=3,
        rtc_action_mask=torch.tensor([1.0, 1.0, 0.0]),
    )
    return x_t, result


def test_pap_moe_denoising_uses_rtc_when_enabled():
    model = _model_with_rtc(enabled=True)

    x_t, result = _run_step(model, inference_delay=2)

    assert torch.equal(result, x_t + 2.0)
    assert model.rtc_processor.call is not None
    assert model.rtc_processor.call["inference_delay"] == 2
    assert model.rtc_processor.call["execution_horizon"] == 3
    assert torch.equal(
        model.rtc_processor.call["action_mask"], torch.tensor([1.0, 1.0, 0.0])
    )


def test_pap_moe_denoising_defaults_sync_rtc_delay_to_zero():
    model = _model_with_rtc(enabled=True)

    _run_step(model)

    assert model.rtc_processor.call["inference_delay"] == 0


def test_pap_moe_denoising_bypasses_rtc_when_disabled():
    model = _model_with_rtc(enabled=False)

    x_t, result = _run_step(model, inference_delay=2)

    assert torch.equal(result, x_t + 2.0)
    assert model.rtc_processor.call is None

#!/usr/bin/env python
"""
Pre/post processor factory for PAP-MoE policy.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_pap_moe import (
    DEFAULT_GLOBAL_TASK,
    OBS_FORCE,
    OBS_FORCE_FAST,
    OBS_FORCE_SLOW,
    OBS_PHYSICS_GATE_TARGET,
    PAPMoEConfig,
)


@ProcessorStepRegistry.register(name="pap_moe_prepare_state_tokenizer_processor_step")
@dataclass
class PapMoePrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Tokenize one global task instruction while retaining state discretization.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    global_task: str = DEFAULT_GLOBAL_TASK

    def get_config(self) -> dict[str, Any]:
        return {
            "max_state_dim": self.max_state_dim,
            "task_key": self.task_key,
            "global_task": self.global_task,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PAP-MoE")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        state = deepcopy(state)

        # Discretize state into 256 bins
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]
        full_prompts = []
        for i in range(len(tasks)):
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {self.global_task}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        complementary_data[self.task_key] = full_prompts
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pap_moe_pre_post_processors(
    config: PAPMoEConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Constructs pre-processor and post-processor pipelines for the PAP-MoE policy."""

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    force_feature_keys = (OBS_FORCE, OBS_FORCE_FAST, OBS_FORCE_SLOW)
    identity_observation_feature_keys = set(config.identity_observation_feature_keys)
    model_features = {
        key: feature
        for key, feature in config.input_features.items()
        if key not in force_feature_keys
        and key not in identity_observation_feature_keys
    } | config.output_features
    force_features = {
        key: config.input_features[key] for key in force_feature_keys if key in config.input_features
    }

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features=model_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        NormalizerProcessorStep(
            features=force_features,
            norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
            stats=dataset_stats,
        ),
        # Expose only the global task to the VLM tokenizer. Frame-level task
        # annotations may remain in the dataset but PAP-MoE does not consume them.
        PapMoePrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim, global_task=config.global_task
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(
            enabled=config.use_relative_actions,
            relative_step=relative_step,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

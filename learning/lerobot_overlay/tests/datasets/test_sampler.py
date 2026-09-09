#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from datasets import Dataset  # noqa: E402

from lerobot.datasets.io_utils import (
    hf_transform_to_torch,
)
from lerobot.datasets.sampler import EpisodeAwareSampler, SemanticPairedRecoverySampler


def test_semantic_paired_recovery_sampler_matches_stage_and_source_group():
    stages = [0, 1, 0, 1, 0, 1]
    episodes = [0, 0, 1, 1, 24, 24]
    sampler = SemanticPairedRecoverySampler(stages, episodes, 24, seed=7)
    indices = list(sampler)
    assert len(indices) == 4
    for recovery, anchor in zip(indices[::2], indices[1::2], strict=True):
        assert episodes[recovery] >= 24
        assert episodes[anchor] < 24
        assert stages[recovery] == stages[anchor]


def test_semantic_paired_recovery_sampler_prioritizes_episode_starts():
    sampler = SemanticPairedRecoverySampler(
        [0, 0, 0, 0, 0, 0], [0, 0, 24, 24, 25, 25], 24, seed=3,
        frame_indices=[0, 1, 0, 10, 0, 10], recovery_start_count=2,
    )
    recovery = list(sampler)[::2]
    assert set(recovery[:2]) == {2, 4}


def test_semantic_paired_recovery_sampler_supports_coarse_to_fine_anchor_alias():
    stages = [8, 7, 2, 4, 3, 12, 12]
    episodes = [0, 0, 0, 0, 0, 24, 24]
    sampler = SemanticPairedRecoverySampler(
        stages,
        episodes,
        24,
        seed=4,
        anchor_stage_aliases={12: [8, 7, 2, 4, 3]},
    )
    indices = list(sampler)
    assert len(indices) == 4
    for recovery, anchor in zip(indices[::2], indices[1::2], strict=True):
        assert stages[recovery] == 12
        assert stages[anchor] in {8, 7, 2, 4, 3}
        assert episodes[anchor] < 24


def test_semantic_paired_recovery_sampler_excludes_zero_weight_carriers():
    stages = [0, 0, 0, 0, 0, 0]
    episodes = [0, 0, 24, 24, 25, 25]
    sampler = SemanticPairedRecoverySampler(
        stages,
        episodes,
        24,
        seed=5,
        eligible_mask=[False, True, False, True, False, True],
    )
    indices = list(sampler)
    assert set(indices[::2]) == {3, 5}
    assert set(indices[1::2]) == {1}


def calculate_episode_data_index(hf_dataset: Dataset) -> dict[str, torch.Tensor]:
    """Calculate episode data index for testing. Returns {"from": Tensor, "to": Tensor}."""
    episode_data_index: dict[str, list[int]] = {"from": [], "to": []}
    current_episode = None
    if len(hf_dataset) == 0:
        return {"from": torch.tensor([]), "to": torch.tensor([])}
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            episode_data_index["from"].append(idx)
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            current_episode = episode_idx
    episode_data_index["to"].append(idx + 1)
    return {k: torch.tensor(v) for k, v in episode_data_index.items()}


def test_drop_n_first_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], drop_n_first_frames=1)
    assert sampler.indices == [1, 4, 5]
    assert len(sampler) == 3
    assert list(sampler) == [1, 4, 5]


def test_drop_n_last_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], drop_n_last_frames=1)
    assert sampler.indices == [0, 3, 4]
    assert len(sampler) == 3
    assert list(sampler) == [0, 3, 4]


def test_episode_indices_to_use():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(
        episode_data_index["from"], episode_data_index["to"], episode_indices_to_use=[0, 2]
    )
    assert sampler.indices == [0, 1, 3, 4, 5]
    assert len(sampler) == 5
    assert list(sampler) == [0, 1, 3, 4, 5]


def test_shuffle():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], shuffle=False)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert list(sampler) == [0, 1, 2, 3, 4, 5]
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], shuffle=True)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert set(sampler) == {0, 1, 2, 3, 4, 5}


def test_negative_drop_first_frames_raises():
    with pytest.raises(ValueError, match="drop_n_first_frames must be >= 0"):
        EpisodeAwareSampler([0], [10], drop_n_first_frames=-1)


def test_negative_drop_last_frames_raises():
    with pytest.raises(ValueError, match="drop_n_last_frames must be >= 0"):
        EpisodeAwareSampler([0], [10], drop_n_last_frames=-1)


def test_all_episodes_dropped_raises():
    # All episodes have 1 frame, drop_n_first_frames=1 removes all
    with pytest.raises(ValueError, match="No valid frames remain"):
        EpisodeAwareSampler([0, 1, 2], [1, 2, 3], drop_n_first_frames=1)


def test_partial_episode_drop_warns(caplog):
    # Episode 0: 1 frame (dropped), Episode 1: 5 frames (kept)
    with caplog.at_level(logging.WARNING, logger="lerobot.datasets.sampler"):
        sampler = EpisodeAwareSampler([0, 1], [1, 6], drop_n_first_frames=1)
    # Episode 0 is skipped (1 frame, drop 1), Episode 1 keeps frames 2-5
    assert sampler.indices == [2, 3, 4, 5]
    assert "Episode 0" in caplog.text

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
from collections.abc import Iterator

import torch
import numpy as np

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                ep_length = end_index - start_index
                if drop_n_first_frames + drop_n_last_frames >= ep_length:
                    logger.warning(
                        "Episode %d has %d frames but drop_n_first_frames=%d and "
                        "drop_n_last_frames=%d removes all frames. Skipping.",
                        episode_idx,
                        ep_length,
                        drop_n_first_frames,
                        drop_n_last_frames,
                    )
                    continue
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        if not indices:
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class SemanticPairedRecoverySampler:
    """Yield recovery/anchor pairs matched by semantic task index."""

    def __init__(
        self, stages, episode_indices, recovery_episode_start: int, seed: int = 0,
        frame_indices=None, recovery_start_count: int = 0,
        anchor_stage_aliases: dict[int, list[int]] | None = None,
        eligible_mask=None,
    ):
        stages = np.asarray(stages, dtype=np.int64)
        episodes = np.asarray(episode_indices, dtype=np.int64)
        if stages.ndim != 1 or episodes.shape != stages.shape:
            raise ValueError("stages and episode_indices must be matching 1-D arrays")
        if eligible_mask is None:
            eligible = np.ones(stages.shape, dtype=bool)
        else:
            eligible = np.asarray(eligible_mask, dtype=bool)
            if eligible.shape != stages.shape:
                raise ValueError("eligible_mask must match stages")
        recovery_mask = (episodes >= recovery_episode_start) & eligible
        self.recovery_indices = np.flatnonzero(recovery_mask)
        if not len(self.recovery_indices):
            raise ValueError("paired recovery sampling found no recovery frames")
        aliases = anchor_stage_aliases or {}
        self.anchor_by_stage = {}
        for stage in np.unique(stages[self.recovery_indices]):
            stage = int(stage)
            anchor_stages = aliases.get(stage, [stage])
            self.anchor_by_stage[stage] = np.flatnonzero(
                (episodes < recovery_episode_start)
                & eligible
                & np.isin(stages, anchor_stages)
            )
        missing = [stage for stage, indices in self.anchor_by_stage.items() if not len(indices)]
        if missing:
            raise ValueError(f"recovery stages have no anchor frames: {missing}")
        self.stages = stages
        self.frame_indices = None if frame_indices is None else np.asarray(frame_indices, dtype=np.int64)
        if self.frame_indices is not None and self.frame_indices.shape != stages.shape:
            raise ValueError("frame_indices must match stages")
        self.recovery_start_count = recovery_start_count
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return 2 * len(self.recovery_indices)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        recovery = self.recovery_indices.copy()
        if self.frame_indices is not None and self.recovery_start_count > 0:
            priority_mask = self.frame_indices[recovery] < self.recovery_start_count
            priority, remainder = recovery[priority_mask], recovery[~priority_mask]
            rng.shuffle(priority)
            rng.shuffle(remainder)
            recovery = np.concatenate((priority, remainder))
        else:
            rng.shuffle(recovery)
        paired = []
        for recovery_index in recovery:
            anchors = self.anchor_by_stage[int(self.stages[recovery_index])]
            anchor_index = int(anchors[rng.randint(len(anchors))])
            paired.extend((int(recovery_index), anchor_index))
        return iter(paired)

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
import builtins
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs import parser
from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.hub import HubMixin

from .default import DatasetConfig, EvalConfig, PeftConfig, WandBConfig
from .policies import PreTrainedConfig

TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Set to True to use deterministic cuDNN algorithms for reproducibility.
    # This disables cudnn.benchmark and may reduce training speed by ~10-20 percent.
    cudnn_deterministic: bool = False
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    steps: int = 100_000
    # Optional per-frame sampling weights stored in the dataset. Zero-weight
    # continuation frames may carry an action chunk without becoming training
    # observations. This is used by episode-safe correction materializations.
    dataset_sample_weight_key: str | None = None
    # Optional weighted sampling for episode startup frames. A value above 1
    # increases the probability of frames whose frame_index is below
    # initial_frame_sampling_count without changing the stored dataset.
    initial_frame_sampling_weight: float = 1.0
    initial_frame_sampling_count: int = 0
    # Balance action-learning samples using the dataset's semantic task_index.
    # This is useful when the language prompt is intentionally held constant
    # while the recorded trajectory still contains distinct task phases.
    semantic_task_balanced_sampling: bool = False
    semantic_task_count: int | None = None
    # Balance the generalized PAP-MoE local-phase target directly. This is
    # distinct from task_index balancing: several local phases can occur
    # inside one semantic task, especially during recovery.
    skill_progress_balanced_sampling: bool = False
    # Optional boost for episodes at or above an index. With a positive
    # recovery_start_sampling_count it applies only to the first N positive-
    # weight observations after takeover; zero-weight rollout prefixes remain
    # excluded. During phase-balanced training it can instead boost all
    # appended observations before per-phase renormalization.
    semantic_episode_sampling_start_index: int | None = None
    semantic_episode_sampling_weight: float = 1.0
    semantic_paired_recovery_sampling: bool = False
    semantic_recovery_start_sampling_count: int = 0
    # Optional boost for the first N frames after every semantic phase
    # transition. This targets closed-loop decision boundaries (for example,
    # the onset of release) without exposing semantic labels to the policy.
    semantic_phase_start_sampling_count: int = 0
    semantic_phase_start_sampling_weight: float = 1.0
    # Optional model-agnostic boost around transitions in the generalized
    # local skill-progress label.  The label is used only by the sampler and
    # is never added to the policy observation.  Unlike phase balancing, this
    # also works for ordinary action-only behavior cloning.
    skill_progress_transition_sampling_window: int = 0
    skill_progress_transition_sampling_weight: float = 1.0
    # Optional directed local-phase filter.  When both are set, only the
    # configured from -> to boundary is boosted; None keeps all transitions.
    skill_progress_transition_from_phase: int | None = None
    skill_progress_transition_to_phase: int | None = None
    # Optional within-phase oversampling around binary gripper transitions.
    # This targets the narrow close/open decision boundary that semantic phase
    # balancing alone leaves extremely sparse.
    gripper_transition_sampling_window: int = 0
    gripper_transition_sampling_weight: float = 1.0
    eval_freq: int = 20_000
    log_freq: int = 200
    tolerance_s: float = 1e-4
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # RA-BC (Reward-Aligned Behavior Cloning) parameters
    use_rabc: bool = False  # Enable reward-weighted training
    rabc_progress_path: str | None = None  # Path to precomputed SARM progress parquet file
    rabc_kappa: float = 0.01  # Hard threshold for high-quality samples
    rabc_epsilon: float = 1e-6  # Small constant for numerical stability
    rabc_head_mode: str | None = "sparse"  # For dual-head models: "sparse" or "dense"

    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    def validate(self) -> None:
        if self.dataset_sample_weight_key is not None and not self.dataset_sample_weight_key.strip():
            raise ValueError("dataset_sample_weight_key cannot be empty")
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )

            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )

            policy_dir = Path(config_path).parent
            if self.policy is not None:
                self.policy.pretrained_path = policy_dir
            self.checkpoint_path = policy_dir.parent

        if self.policy is None:
            raise ValueError(
                "Policy is not configured. Please specify a pretrained policy with `--policy.path`."
            )
        if self.initial_frame_sampling_weight < 1.0:
            raise ValueError("initial_frame_sampling_weight must be at least 1.0")
        if self.initial_frame_sampling_count < 0:
            raise ValueError("initial_frame_sampling_count cannot be negative")
        if self.semantic_task_count is not None and self.semantic_task_count < 1:
            raise ValueError("semantic_task_count must be positive when provided")
        if self.semantic_paired_recovery_sampling:
            if self.semantic_episode_sampling_start_index is None:
                raise ValueError(
                    "semantic_paired_recovery_sampling requires semantic_episode_sampling_start_index"
                )
            if self.batch_size % 2:
                raise ValueError("paired recovery sampling requires an even batch_size")
        if self.semantic_recovery_start_sampling_count < 0:
            raise ValueError("semantic_recovery_start_sampling_count cannot be negative")
        if self.semantic_episode_sampling_weight < 1.0:
            raise ValueError("semantic_episode_sampling_weight must be at least 1.0")
        if (
            self.semantic_recovery_start_sampling_count > 0
            and self.semantic_episode_sampling_start_index is None
        ):
            raise ValueError(
                "semantic_recovery_start_sampling_count requires "
                "semantic_episode_sampling_start_index"
            )
        if self.semantic_phase_start_sampling_count < 0:
            raise ValueError(
                "semantic_phase_start_sampling_count cannot be negative"
            )
        if self.semantic_phase_start_sampling_weight < 1.0:
            raise ValueError(
                "semantic_phase_start_sampling_weight must be at least 1.0"
            )
        if self.skill_progress_transition_sampling_window < 0:
            raise ValueError(
                "skill_progress_transition_sampling_window cannot be negative"
            )
        if self.skill_progress_transition_sampling_weight < 1.0:
            raise ValueError(
                "skill_progress_transition_sampling_weight must be at least 1.0"
            )
        if (self.skill_progress_transition_from_phase is None) != (
            self.skill_progress_transition_to_phase is None
        ):
            raise ValueError(
                "skill-progress transition from/to phases must be set together"
            )
        if (
            self.skill_progress_transition_from_phase is not None
            and (
                self.skill_progress_transition_from_phase < 0
                or self.skill_progress_transition_to_phase < 0
            )
        ):
            raise ValueError("skill-progress transition phases cannot be negative")
        if self.gripper_transition_sampling_window < 0:
            raise ValueError("gripper_transition_sampling_window cannot be negative")
        if self.gripper_transition_sampling_weight < 1.0:
            raise ValueError("gripper_transition_sampling_weight must be at least 1.0")

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "'policy.repo_id' argument missing. Please specify it to push the model to the hub."
            )

        if self.use_rabc and not self.rabc_progress_path:
            # Auto-detect from dataset path
            repo_id = self.dataset.repo_id
            if self.dataset.root:
                self.rabc_progress_path = str(Path(self.dataset.root) / "sarm_progress.parquet")
            else:
                self.rabc_progress_path = f"hf://datasets/{repo_id}/sarm_progress.parquet"

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # because of the third-party library draccus uses Any as the return type

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional

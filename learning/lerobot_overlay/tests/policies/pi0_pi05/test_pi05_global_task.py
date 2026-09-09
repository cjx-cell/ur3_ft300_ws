import torch

from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


def _transition(task: str) -> dict:
    return {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros((1, 7))},
        TransitionKey.COMPLEMENTARY_DATA: {"task": [task]},
    }


def test_pi05_prompt_can_force_global_task_while_dataset_keeps_phase_label() -> None:
    step = Pi05PrepareStateTokenizerProcessorStep(
        global_task="pick up the peg and insert it into the hole"
    )

    result = step(_transition("recover contact and relocate the hole"))
    prompt = result[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert prompt.startswith("Task: pick up the peg and insert it into the hole,")
    assert "recover contact" not in prompt


def test_pi05_prompt_preserves_dataset_task_by_default() -> None:
    step = Pi05PrepareStateTokenizerProcessorStep()

    result = step(_transition("grasp_the_peg"))
    prompt = result[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert prompt.startswith("Task: grasp the peg,")

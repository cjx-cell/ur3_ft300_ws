#!/usr/bin/env python3
"""Serve an official LeRobot Diffusion Policy to the Gazebo ROS bridge.

Every fresh Gazebo observation is passed through ``select_action``. The policy
therefore owns its two-frame observation history and eight-action queue. The
ROS side must execute one returned action at a time (``--action-chunk-size 1``)
so the two observations remain adjacent 10 Hz samples as in training.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/lerobot/src")

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


JOINT_STATE_FILE = "/tmp/ur3_joint_state.txt"
ACTION_FILE = "/tmp/ur3_action.txt"
ACTION_CHUNK_FILE = "/tmp/ur3_action_chunk.npy"
ACTION_CHUNK_TMP_FILE = "/tmp/ur3_action_chunk_tmp.npy"
CAMERA0_FILE = "/tmp/ur3_camera0.npy"
CAMERA1_FILE = "/tmp/ur3_camera1.npy"
READY_FILE = "/tmp/ur3_inference_ready.txt"
EXPECTED_ACTION_DIM = 7
GRIPPER_OPEN_RAD = 0.0
GRIPPER_CLOSED_COMMAND_RAD = 0.8


def _load_observation(device: torch.device) -> dict[str, torch.Tensor]:
    with open(JOINT_STATE_FILE) as stream:
        state = np.fromstring(stream.read().strip(), sep=" ", dtype=np.float32)
    if state.shape != (EXPECTED_ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"invalid joint state shape/value: {state}")

    camera0 = np.load(CAMERA0_FILE)
    camera1 = np.load(CAMERA1_FILE)
    for name, image in (("camera0", camera0), ("camera1", camera1)):
        if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
            raise ValueError(f"invalid {name} image shape/value: {image.shape}")

    def image_tensor(image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32, copy=False)
        if float(image.max()) > 1.0:
            image = image / 255.0
        return torch.from_numpy(np.transpose(image, (2, 0, 1))).to(device)

    return {
        "observation.state": torch.from_numpy(state).to(device),
        "observation.images.camera0": image_tensor(camera0),
        "observation.images.camera1": image_tensor(camera1),
    }


def _publish_action(action: np.ndarray) -> None:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (EXPECTED_ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError(f"invalid policy action: {action}")
    action[6] = np.clip(action[6], GRIPPER_OPEN_RAD, GRIPPER_CLOSED_COMMAND_RAD)

    # Keep the learned official-style continuous gripper trajectory. Thresholding
    # here would silently change both the training contract and closed loop.
    np.save(ACTION_CHUNK_TMP_FILE, action[None, :])
    os.replace(ACTION_CHUNK_TMP_FILE, ACTION_CHUNK_FILE)
    action_tmp = ACTION_FILE + ".tmp"
    with open(action_tmp, "w") as stream:
        stream.write(" ".join(f"{value:.6f}" for value in action))
    os.replace(action_tmp, ACTION_FILE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading official DiffusionPolicy from {checkpoint}", flush=True)
    policy = DiffusionPolicy.from_pretrained(checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=checkpoint
    )
    policy.reset()

    for path in (READY_FILE, ACTION_FILE, ACTION_CHUNK_FILE, ACTION_CHUNK_TMP_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    Path(READY_FILE).write_text("READY\n")
    print(
        "Inference ready: native DP history/queue enabled; ROS action chunk must be 1.",
        flush=True,
    )
    while not os.path.exists(JOINT_STATE_FILE):
        time.sleep(0.05)

    step = 0
    last_mtime_ns = os.stat(JOINT_STATE_FILE).st_mtime_ns
    while True:
        try:
            while os.stat(JOINT_STATE_FILE).st_mtime_ns == last_mtime_ns:
                time.sleep(0.001)
            last_mtime_ns = os.stat(JOINT_STATE_FILE).st_mtime_ns
            start = time.perf_counter()

            observation = _load_observation(device)
            processed = preprocessor(observation)
            normalized_action = policy.select_action(processed)
            action = postprocessor(normalized_action).squeeze(0).detach().cpu().numpy()
            _publish_action(action)

            if step % 10 == 0:
                latency_ms = 1000.0 * (time.perf_counter() - start)
                print(
                    f"[DP {step:04d}] action={np.round(action, 4)} "
                    f"latency={latency_ms:.1f} ms",
                    flush=True,
                )
            step += 1
        except KeyboardInterrupt:
            break
        except Exception as error:
            print(f"Inference error at step {step}: {error}", flush=True)
            time.sleep(0.05)


if __name__ == "__main__":
    main()

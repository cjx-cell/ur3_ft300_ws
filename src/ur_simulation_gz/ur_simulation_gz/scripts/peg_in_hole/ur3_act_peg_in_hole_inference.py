#!/usr/bin/env python3
"""Serve an official LeRobot ACT checkpoint to the Gazebo ROS bridge."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/lerobot/src")

from lerobot.policies.act.modeling_act import ACTPolicy
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


def _image_tensor(path: str, device: torch.device) -> torch.Tensor:
    image = np.load(path)
    if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
        raise ValueError(f"invalid image {path}: {image.shape}")
    image = image.astype(np.float32, copy=False)
    if float(image.max()) > 1.0:
        image = image / 255.0
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).to(device)


def _load_observation(device: torch.device) -> dict[str, torch.Tensor]:
    with open(JOINT_STATE_FILE) as stream:
        state = np.fromstring(stream.read().strip(), sep=" ", dtype=np.float32)
    if state.shape != (EXPECTED_ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"invalid joint state: {state}")
    return {
        "observation.state": torch.from_numpy(state).to(device),
        "observation.images.camera0": _image_tensor(CAMERA0_FILE, device),
        "observation.images.camera1": _image_tensor(CAMERA1_FILE, device),
    }


def _publish_chunk(chunk: np.ndarray) -> None:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != EXPECTED_ACTION_DIM:
        raise ValueError(f"invalid ACT action chunk: {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("ACT action chunk contains non-finite values")
    chunk[:, 6] = np.clip(
        chunk[:, 6], GRIPPER_OPEN_RAD, GRIPPER_CLOSED_COMMAND_RAD
    )
    np.save(ACTION_CHUNK_TMP_FILE, chunk)
    os.replace(ACTION_CHUNK_TMP_FILE, ACTION_CHUNK_FILE)
    action_tmp = ACTION_FILE + ".tmp"
    with open(action_tmp, "w") as stream:
        stream.write(" ".join(f"{value:.6f}" for value in chunk[0]))
    os.replace(action_tmp, ACTION_FILE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--execute-steps", type=int, default=10)
    args = parser.parse_args()
    if args.execute_steps < 1:
        raise ValueError("execute-steps must be positive")

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    print(f"Loading official ACTPolicy from {checkpoint}", flush=True)
    policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=checkpoint
    )
    if args.execute_steps > policy.config.chunk_size:
        raise ValueError(
            f"execute-steps={args.execute_steps} exceeds chunk_size={policy.config.chunk_size}"
        )

    for path in (READY_FILE, ACTION_FILE, ACTION_CHUNK_FILE, ACTION_CHUNK_TMP_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    Path(READY_FILE).write_text("READY\n")
    print(
        f"Inference ready: ACT predicts {policy.config.chunk_size}, "
        f"executes {args.execute_steps}.",
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

            processed = preprocessor(_load_observation(device))
            normalized_chunk = policy.predict_action_chunk(processed)
            chunk = postprocessor(normalized_chunk)[0, : args.execute_steps]
            chunk = chunk.detach().cpu().numpy()
            _publish_chunk(chunk)

            if step % 10 == 0:
                latency_ms = 1000.0 * (time.perf_counter() - start)
                print(
                    f"[ACT {step:04d}] first={np.round(chunk[0], 4)} "
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

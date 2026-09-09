#!/usr/bin/env python3
"""Collect a gap-free batch of successful keyboard/mouse demonstrations."""

import argparse
import os
from pathlib import Path
import subprocess
import zipfile


SCRIPT_DIR = Path(__file__).resolve().parent
RECORDER = SCRIPT_DIR / "pap_moe_keyboard_teleop_record.py"
RETURN_HOME = SCRIPT_DIR / "pap_moe_teleop_return_home.py"
SYSTEM_PYTHON = "/usr/bin/python3"


def episode_dir(output, episode):
    return output / (
        "pick_up_the_peg_and_insert_it_into_the_hole_"
        f"episode_{episode:04d}_success"
    )


def is_complete_episode(path):
    """Accept only a structurally valid recorder archive as an episode."""
    archive = path / "data.npz"
    if not path.is_dir() or not archive.is_file() or archive.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(archive) as content:
            members = set(content.namelist())
            return {
                "state.npy",
                "action.npy",
                "camera0.npy",
                "camera1.npy",
                "force.npy",
                "schema_version.npy",
            }.issubset(members) and content.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def run_checked(command):
    return subprocess.run(command, check=False).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-episode", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--output",
        default=os.path.expanduser(
            "~/ur3_ft300_ws/pap_moe_framework/datasets/teleop_workspace_50"
        ),
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.start_episode < 1:
        parser.error("episodes and start-episode must be >= 1")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    first = args.start_episode
    final = first + args.episodes - 1
    print(f"COLLECTION: episodes {first:04d}..{final:04d} -> {output}")
    print("A saved success advances the number; X/rejection retries the same seed.\n")

    # The one-command collection launch creates Gazebo with the arm already at
    # HOME and the gripper open.  Do not issue a redundant stop/start reset
    # here: it can race the freshly started Servo/Tk client.  Every completed
    # or discarded attempt is still reset below before the next episode.

    episode = first
    while episode <= final:
        expected = episode_dir(output, episode)
        if expected.exists():
            if is_complete_episode(expected):
                print(f"REFUSE: valid episode already exists: {expected}")
                raise SystemExit(2)
            if expected.is_dir() and not any(expected.iterdir()):
                expected.rmdir()
                print(f"REMOVED stale empty episode shell: {expected}")
            else:
                print(f"REFUSE: incomplete non-empty episode needs inspection: {expected}")
                raise SystemExit(2)
        seed = args.seed_start + (episode - first)
        print("\n" + "=" * 72)
        print(f"EPISODE {episode:04d}/{final:04d}  seed={seed}")
        print("Operate in GUI: B=start, P=payload reference after lift, V=success, X=retry")
        rc = run_checked(
            [
                SYSTEM_PYTHON,
                str(RECORDER),
                "--episode",
                str(episode),
                "--seed",
                str(seed),
                "--output",
                str(output),
            ]
        )
        home_rc = run_checked([SYSTEM_PYTHON, str(RETURN_HOME)])
        if home_rc != 0:
            raise SystemExit("automatic home failed; collection stopped safely")
        if is_complete_episode(expected):
            print(f"ACCEPTED {episode:04d}; HOME ready; next episode")
            episode += 1
        elif expected.exists():
            raise SystemExit(
                f"recorder left an incomplete episode at {expected}; "
                "collection stopped without advancing the episode number"
            )
        else:
            print(
                f"RETRY {episode:04d}: no accepted dataset was written "
                f"(recorder rc={rc}); seed remains {seed}"
            )
    print(f"\nCOMPLETE: {args.episodes} accepted demonstrations in {output}")


if __name__ == "__main__":
    main()

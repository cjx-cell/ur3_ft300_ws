#!/usr/bin/env python3
"""Record one independent policy-rollout to expert-recovery episode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.executors import ExternalShutdownException

from pap_moe_framework.rollout_recovery.recorder import (
    RolloutRecoveryRecorder,
    save_validated_episode,
)
from pap_moe_framework.rollout_recovery.schema import RECOVERY_PHASES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--outcome-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-policy-checkpoint", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--recovery-phase", choices=RECOVERY_PHASES, required=True)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-sensor-skew-s", type=float, default=0.15)
    parser.add_argument("--max-action-skew-s", type=float, default=0.18)
    parser.add_argument("--max-pose-age-s", type=float, default=1.0)
    parser.add_argument(
        "--pre-takeover-context-s",
        type=float,
        default=None,
        help=(
            "Keep only this many seconds of policy context before takeover, plus "
            "all expert frames. Use for failure-local recovery episodes."
        ),
    )
    args = parser.parse_args()
    if args.hz <= 0.0:
        parser.error("--hz must be positive")
    if args.pre_takeover_context_s is not None and args.pre_takeover_context_s <= 0.0:
        parser.error("--pre-takeover-context-s must be positive")
    if args.output.name != "data.npz":
        parser.error("--output must end in an episode directory's data.npz")
    if not (args.session_dir / "session.json").is_file():
        parser.error("session.json does not exist; start the recovery-enabled ROS side first")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    rclpy.init()
    recorder = RolloutRecoveryRecorder(
        session_dir=args.session_dir,
        outcome_file=args.outcome_file,
        hz=args.hz,
        max_sensor_skew_s=args.max_sensor_skew_s,
        max_action_skew_s=args.max_action_skew_s,
        max_pose_age_s=args.max_pose_age_s,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(recorder)
    try:
        while rclpy.ok() and not recorder.finished.is_set():
            executor.spin_once(timeout_sec=0.1)
            recorder.poll_outcome_wall_clock()
        if recorder.error is not None:
            raise recorder.error
        values = recorder.build_episode(
            source_policy_checkpoint=args.source_policy_checkpoint,
            episode_id=args.episode_id,
            recovery_phase=args.recovery_phase,
            pre_takeover_context_s=args.pre_takeover_context_s,
        )
        save_validated_episode(args.output, values)
        print(f"Saved validated rollout recovery: {args.output} ({len(values['timestamp'])} frames)")
        return 0
    except (KeyboardInterrupt, ExternalShutdownException):
        print("Interrupted before an explicit outcome; no training episode was saved.")
        return 130
    finally:
        recorder.close()
        executor.remove_node(recorder)
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only Gazebo shadow scorer. JSON lines on stdout, never robot commands."""
import argparse
import json
import signal
from pathlib import Path
import subprocess
import sys
import time

ACTIVE = None


def stop(signum, frame):
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.kill()
        ACTIVE.wait()
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/ur_simulation_gz/ur_simulation_gz/scripts/peg_in_hole'))
from fixture_geometry_audit import parse_model_poses, score_seating


def main():
    global ACTIVE
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--period', type=float, default=1.)
    args = parser.parse_args()
    if args.samples < 1 or args.period < 0:
        parser.error('samples must be positive and period nonnegative')
    for index in range(args.samples):
        started = time.monotonic()
        row = dict(index=index, wall_time_s=time.time())
        try:
            ACTIVE = subprocess.Popen(['ign', 'topic', '-t', '/world/simulation_world/pose/info',
                                       '-e', '-n', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                output, error = ACTIVE.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                ACTIVE.kill()
                ACTIVE.communicate()
                raise
            if ACTIVE.returncode:
                raise ValueError('Pose subscription failed: ' + error[:200])
            poses = parse_model_poses(output)
            if not all(name in poses for name in ('peg', 'hole_plate')):
                raise ValueError('Required model poses absent')
            row.update(valid=True, geometry=score_seating(poses['peg'], poses['hole_plate']),
                       poses={name: [v.tolist() for v in pose] for name, pose in poses.items()})
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            row.update(valid=False, reason=str(exc))
        print(json.dumps(row, allow_nan=False), flush=True)
        if index + 1 < args.samples:
            time.sleep(max(0., args.period - (time.monotonic() - started)))


if __name__ == '__main__':
    main()

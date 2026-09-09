"""Versioned, atomically paired action reply for the file-based policy bridge.

The bridge permits one outstanding observation, and aborts on timeout in this
mode. A reply cannot authorize a different observation's action. This checks
request identity, not sensor freshness (which requires source timestamps).
"""
import os
from pathlib import Path
import tempfile

import numpy as np


def enabled():
    return bool(os.environ.get("POLICY_PAIRED_ACTION_FILE"))


def observation_id(path):
    stat = os.stat(path)
    return f"{stat.st_dev}:{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"


def validate_source_stamps(stamps, sim_now, wall_now, *, max_sim_age=.5, max_skew=.25, max_wall_age=5.):
    """Sensor liveness is not image quality: black frames may still be fresh."""
    required = ('joint', 'camera0', 'camera1', 'force')
    for name in required:
        if name not in stamps:
            raise ValueError(f"Missing observation source: {name}")
        sim, wall = stamps[name]
        if not np.isfinite([sim, wall, sim_now, wall_now]).all():
            raise ValueError(f"Nonfinite observation timestamp: {name}")
        if not -.05 <= sim_now - sim <= max_sim_age or not 0 <= wall_now - wall <= max_wall_age:
            raise ValueError(f"Stale/future observation source: {name}")
    times = [stamps[name][0] for name in required]
    if max(times) - min(times) > max_skew:
        raise ValueError("Cross-modal timestamp skew exceeds contract")


def publish_reply(request_id, action, path=None):
    path = path or os.environ.get("POLICY_PAIRED_ACTION_FILE")
    if not path:
        return
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != 7 or not np.isfinite(action).all():
        raise ValueError("Invalid paired action: expected finite [K,7]")
    destination = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, schema=np.array(1), request_id=np.array(request_id), action=action)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_reply(request_id, path=None):
    path = path or os.environ["POLICY_PAIRED_ACTION_FILE"]
    try:
        with np.load(path, allow_pickle=False) as reply:
            if reply["schema"].item() != 1:
                raise ValueError("Unknown paired-action schema")
            if reply["request_id"].item() != request_id:
                return None
            action = reply["action"].copy()
    except FileNotFoundError:
        return None
    if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != 7 or not np.isfinite(action).all():
        raise ValueError("Invalid paired action: expected finite [K,7]")
    return action

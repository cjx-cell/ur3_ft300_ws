"""Read a stable published observation before mutating model/RTC state.

The ROS publisher replaces joint_state atomically AFTER writing all modalities.
The full paired-v1 identity is retained for replies. ctime alone is not a new
content version: replacing/unlinking a file can change that metadata.
"""
import time


def content_version(identity):
    parts = identity.split(':')
    if len(parts) != 4:
        raise ValueError(f'Invalid paired-v1 identity: {identity!r}')
    return tuple(parts[:3])  # device, inode, nanosecond mtime


def read_stable(load, identify, previous_id, *, report=lambda event: None,
                timeout=.25, pause=.001):
    deadline = time.monotonic() + timeout
    while True:
        before = None
        try:
            before = identify()
            if previous_id is not None and content_version(before) == content_version(previous_id):
                return None
            observation = load()
            after = identify()
            if before == after:
                return before, observation
            report(dict(event='observation_read_retry', before=before, after=after))
        except (FileNotFoundError, ValueError, OSError) as error:
            if before is None and isinstance(error, FileNotFoundError):
                return None
            # A stable malformed observation is a real error, not a reason to
            # wait forever or feed a partial sample to the policy.
            try:
                after = identify()
            except FileNotFoundError:
                after = None
            if before is not None and before == after:
                raise
            report(dict(event='observation_read_retry', before=before, after=after,
                        error=str(error)))
        if time.monotonic() >= deadline:
            raise RuntimeError('Observation failed to stabilize before inference')
        time.sleep(pause)

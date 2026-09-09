"""Serialized, fail-closed episode lifecycle; independent of ROS and torch.

Candidate only: the frozen evaluation launchers do not import this module.
"""
from threading import RLock
from uuid import uuid4


class EpisodeSession:
    def __init__(self, backend):
        self.backend = backend
        self.lock = RLock()
        self.token = None
        self.sequence = 0

    def begin(self, seed):
        with self.lock:
            if self.token is not None:
                raise RuntimeError("End the active episode before starting another")
            # Do not advertise readiness until reset AND cold-compatible warmup finish.
            self.backend.reset_episode(seed)
            self.sequence = 0
            self.token = uuid4().hex
            return self.token

    def _check(self, token):
        if self.token is None or token != self.token:
            raise RuntimeError("Inactive or stale episode token")

    def infer(self, token, sequence, observation):
        with self.lock:
            self._check(token)
            if type(sequence) is not int or sequence != self.sequence:
                raise RuntimeError("Duplicate, skipped or out-of-order observation")
            try:
                result = self.backend.predict(observation)
            except BaseException:
                # A partially executed Flow may have mutated memory/RNG/RTC.
                # Never retry that observation in a contaminated episode.
                self.token = None
                raise
            self.sequence += 1
            return result

    def end(self, token):
        with self.lock:
            self._check(token)
            self.token = None


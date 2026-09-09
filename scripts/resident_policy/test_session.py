import unittest
from session import EpisodeSession


class FakeBackend:
    def reset_episode(self, seed):
        self.history = [seed]  # Simulate warmup memory that MUST be reproduced.

    def predict(self, observation):
        self.history.append(observation)
        if observation == 'fault':
            raise RuntimeError('Partial inference failed')
        return tuple(self.history)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.session = EpisodeSession(FakeBackend())

    def test_a_b_a_isolation(self):
        outputs = []
        for seed in [0, 1, 0]:
            token = self.session.begin(seed)
            outputs.append([self.session.infer(token, i, i) for i in range(2)])
            self.session.end(token)
        self.assertEqual(outputs[0], outputs[2])
        self.assertNotEqual(outputs[0], outputs[1])

    def test_stale_session(self):
        old = self.session.begin(0)
        self.session.end(old)
        new = self.session.begin(0)
        self.assertNotEqual(old, new)
        with self.assertRaises(RuntimeError):
            self.session.infer(old, 0, 0)
        self.assertEqual(self.session.infer(new, 0, 0), (0, 0))

    def test_duplicate_and_skipped(self):
        token = self.session.begin(0)
        self.session.infer(token, 0, 0)
        for index in [0, 2, True]:
            with self.assertRaises(RuntimeError):
                self.session.infer(token, index, 99)
        self.assertEqual(self.session.infer(token, 1, 1), (0, 0, 1))

    def test_active_episode_cannot_be_reset(self):
        self.session.begin(0)
        with self.assertRaises(RuntimeError):
            self.session.begin(1)

    def test_partial_failure_invalidates_episode(self):
        token = self.session.begin(0)
        with self.assertRaises(RuntimeError):
            self.session.infer(token, 0, 'fault')
        with self.assertRaises(RuntimeError):
            self.session.infer(token, 0, 0)
        new = self.session.begin(1)
        self.assertEqual(self.session.infer(new, 0, 0), (1, 0))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from action_contract import clip_physical_gripper


class ActionContractTests(unittest.TestCase):
    def setUp(self):
        self.action = np.arange(350, dtype=np.float32).reshape(50, 7) / 100
        self.action[:, 6] = np.linspace(-.2, 1., 50)

    def test_pi05_all_steps_and_arms_unchanged(self):
        got = clip_physical_gripper(self.action.copy(), 'pi05')
        np.testing.assert_array_equal(got[:, :6], self.action[:, :6])
        np.testing.assert_array_equal(got[:, 6], np.clip(self.action[:, 6], 0., .8))

    def test_pap_original_prefix_only(self):
        got = clip_physical_gripper(self.action.copy(), 'pap_moe')
        np.testing.assert_array_equal(got[:, :6], self.action[:, :6])
        np.testing.assert_array_equal(got[10:], self.action[10:])
        np.testing.assert_array_equal(got[:10, 6], np.clip(self.action[:10, 6], 0., .8))

    def test_executed_prefix_identical_between_modes(self):
        a = clip_physical_gripper(self.action.copy(), 'pi05')
        b = clip_physical_gripper(self.action.copy(), 'pap_moe')
        np.testing.assert_array_equal(a[:10], b[:10])


if __name__ == '__main__':
    unittest.main()

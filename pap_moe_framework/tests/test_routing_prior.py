import unittest

import numpy as np

from pap_moe_routing_prior import PhysicsRoutingPrior


class PhysicsRoutingPriorTest(unittest.TestCase):
    def _weights(
        self,
        force,
        *,
        degraded=False,
        degradation_type="normal",
        joint_vel_norm=0.0,
        tool0_z=None,
        current_stage=0,
        semantic_subtask=None,
    ):
        prior = PhysicsRoutingPrior()
        force = np.asarray(force, dtype=np.float32)
        fast = np.tile(force, (64, 1))
        return prior.compute(
            force,
            fast,
            tool0_z,
            degraded,
            degradation_type,
            6.0,
            0.5,
            joint_vel_norm,
            current_stage,
            semantic_subtask,
        )

    def test_normal_vision_free_space_routes_to_e1(self):
        weights = self._weights(np.zeros(6))
        self.assertGreater(float(weights[0]), 0.999)
        self.assertLess(float(weights[2]), 1e-6)
        self.assertLess(float(weights[3]), 1e-6)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_sub_deadband_numerical_wrench_is_not_rigid_contact(self):
        weights = self._weights([0.00012, -0.00008, 0.00003, 0.00002, 0, 0])
        self.assertGreater(float(weights[0]), 0.999)
        self.assertLess(float(weights[2]), 1e-6)
        self.assertLess(float(weights[3]), 1e-6)

    def test_visual_failure_without_contact_routes_to_e2(self):
        weights = self._weights(
            np.zeros(6), degraded=True, degradation_type="dropout"
        )
        self.assertGreater(float(weights[1]), 0.95)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_rigid_and_movable_contacts_separate_by_motion(self):
        rigid = self._weights([0, 0, 5, 0, 0, 0], joint_vel_norm=0.0)
        movable = self._weights([0, 0, 5, 0, 0, 0], joint_vel_norm=1.0)
        self.assertGreater(float(rigid[2]), float(rigid[3]))
        self.assertGreater(float(movable[3]), float(movable[2]))

    def test_blind_contact_coactivates_e2_and_contact_expert(self):
        weights = self._weights(
            [0, 0, 5, 0, 0, 0],
            degraded=True,
            degradation_type="dropout",
            joint_vel_norm=0.0,
        )
        self.assertGreater(float(weights[1]), 0.45)
        self.assertGreater(float(weights[2]), 0.40)

    def test_task_geometry_and_semantics_do_not_change_prior(self):
        first = self._weights(
            [1, 0, 3, 0, 0, 0],
            tool0_z=0.1,
            current_stage=0,
            semantic_subtask="transport to the hole",
        )
        second = self._weights(
            [1, 0, 3, 0, 0, 0],
            tool0_z=9.0,
            current_stage=3,
            semantic_subtask="insert the peg into the hole",
        )
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()

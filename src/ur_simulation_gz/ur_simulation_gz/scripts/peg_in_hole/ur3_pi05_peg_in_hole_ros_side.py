#!/usr/bin/env python3
"""Pi0.5 baseline ROS-side entry point for UR3 peg-in-hole evaluation.

The baseline and PAP-MoE entries intentionally share the same online time
base so model comparisons do not mix policy quality with controller timing.
"""

import os

from ur3_peg_in_hole_ros_side_base import PegInHoleROSSide, run_ros_side


class PI05ROSSide(PegInHoleROSSide):
    """Pi0.5 baseline timing aligned with the 10 Hz training dataset."""

    CONTROL_HZ = 10
    ACTION_DT_S = 0.1
    REPLAN_INTERVAL_S = 5.0
    ACTION_CHUNK_SIZE = 50
    ACTION_CHUNK_MAX_STEP_RAD = float(
        os.environ.get("POLICY_ACTION_CHUNK_MAX_STEP_RAD", "0.025")
    )
    MAX_EPISODE_DURATION_S = 95.0
    # Distance from the measured finger-tip midpoint to the peg center.
    ATTACH_MAX_XY_M = 0.04
    ATTACH_MAX_Z_M = 0.04
    RESET_AFTER_EPISODE = False
    ENABLE_GAZEBO_SUCCESS_CHECK = True
    # The five-episode training state extrema with a 0.05-rad guard margin.
    # This is an evaluation OOD guard, not a UR3 hardware joint limit.
    TRAINING_STATE_MIN = (
        -0.0501, -2.0897, 0.5952, -1.7077, -1.6297, -0.0501, -0.05,
    )
    TRAINING_STATE_MAX = (
        1.9021, -1.0979, 1.9709, -1.0194, -1.5069, 1.9021, 1.05,
    )


if __name__ == "__main__":
    run_ros_side(
        controller_cls=PI05ROSSide,
        node_name="ur3_pi05_peg_in_hole_ros_side",
    )

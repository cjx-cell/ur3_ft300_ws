#!/usr/bin/env python3
"""ROS bridge for policies trained on the canonical Workspace50 v10 data."""

import os

from ur3_pap_moe_peg_in_hole_ros_side import PAPMoEROSSide
from ur3_peg_in_hole_ros_side_base import run_ros_side


class Workspace50ROSSide(PAPMoEROSSide):
    """Shared deployment contract for Pi0.5, ACT, DP and PAP-MoE."""

    GRIPPER_ACTION_MODE = "continuous_radians"
    CONTROL_HZ = 10
    ACTION_DT_S = 0.1
    REPLAN_INTERVAL_S = 1.0
    ACTION_CHUNK_SIZE = 10
    # Canonical 50 episodes reach 0.126958 rad per 100 ms. The old
    # inherited 0.025 cap deformed coordinated demonstration trajectories.
    # Seven physical replay checks cover five anchors and both >0.12 cases.
    ACTION_CHUNK_MAX_STEP_RAD = float(os.environ.get("POLICY_ACTION_CHUNK_MAX_STEP_RAD", "0.13"))
    # Canonical v10 uses x in [0.285, 0.315] m and y around +/-0.18 m;
    # its furthest radial pose is below 0.37 m and is reachable by UR3.
    SPAWN_MAX_XY_SQ = 0.37**2
    MAX_EPISODE_DURATION_S = float(
        os.environ.get("WORKSPACE50_MAX_EPISODE_DURATION_S", "120.0")
    )
    ENABLE_DETACHABLE_JOINT = False
    RESET_AFTER_EPISODE = False
    ENABLE_GAZEBO_SUCCESS_CHECK = True
    SUCCESS_MAX_XY_M = float(
        os.environ.get("WORKSPACE50_SUCCESS_MAX_XY_M", "0.008")
    )
    SUCCESS_MAX_PEG_Z_M = float(
        os.environ.get("WORKSPACE50_SUCCESS_MAX_PEG_Z_M", "0.890")
    )
    SUCCESS_REQUIRED_CHECKS = int(
        os.environ.get("WORKSPACE50_SUCCESS_REQUIRED_CHECKS", "5")
    )

    # Canonical v10 episode start, measured from the formal 50-episode set.
    START_POSE = (
        -1.2539711,
        -1.5706971,
        1.5708373,
        -1.5708448,
        -1.5706999,
        0.0000286,
        0.0,
    )
    START_POSE_TOLERANCE_RAD = 0.002

    # Formal dataset extrema plus a diagnostic guard band. These are
    # dataset-distribution diagnostics, not hardware joint limits.  Wrist 2
    # stays nearly constant in the demonstrations, so its former 0.05-rad
    # band caused contact-induced controller transients (about 0.06 rad) to
    # abort otherwise valid transport rollouts.  Keep a wider but still
    # conservative diagnostic envelope for that joint.
    TRAINING_STATE_MIN = (
        -1.3041,
        -1.6626,
        -1.1655,
        -1.6209,
        -1.7500,
        -0.0501,
        -0.0501,
    )
    TRAINING_STATE_MAX = (
        0.2741,
        -0.2872,
        1.6209,
        -0.0677,
        -1.3900,
        1.5281,
        0.8500,
    )


if __name__ == "__main__":
    run_ros_side(
        controller_cls=Workspace50ROSSide,
        node_name="ur3_workspace50_peg_in_hole_ros_side",
    )

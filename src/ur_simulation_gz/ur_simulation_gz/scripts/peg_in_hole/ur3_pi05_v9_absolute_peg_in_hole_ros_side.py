#!/usr/bin/env python3
"""ROS-side controller for the v9 absolute-action Pi0.5 baseline.

This entry point is intentionally separate from the legacy relative-action
baseline.  Its online horizon matches the checkpoint contract: predict 50
samples at 10 Hz, execute a 10-sample prefix, then replan.
"""

import os

from ur3_pap_moe_peg_in_hole_ros_side import PAPMoEROSSide
from ur3_peg_in_hole_ros_side_base import run_ros_side


class PI05V9AbsoluteROSSide(PAPMoEROSSide):
    GRIPPER_ACTION_MODE = "continuous_radians"
    # The auxiliary release/insertion feedback modules were trained on raw v6
    # physical joint histories, unlike current PAP-MoE's binary D2 history.
    STATE_HISTORY_GRIPPER_MODE = "v6_analog_0.100_open_0.629_closed"
    CONTROL_HZ = 10
    ACTION_DT_S = 0.1
    REPLAN_INTERVAL_S = 1.0
    ACTION_CHUNK_SIZE = 10
    # Inherited from PAPMoEROSSide so baseline and PAP-MoE use the identical
    # task-agnostic actuator rate limit during ablations.
    MAX_EPISODE_DURATION_S = float(
        os.environ.get("PI05_MAX_EPISODE_DURATION_S", "100.0")
    )
    ATTACH_MAX_XY_M = 0.04
    # In this model the midpoint of the two finger-tip TF frames stays about
    # 10.4 cm above the peg centre at the demonstrated grasp pose.  Treat that
    # fixed tool/object offset as a window; the old 4 cm absolute-distance
    # check rejected even the canonical successful grasp configuration.
    ATTACH_MIN_Z_M = 0.08
    ATTACH_MAX_Z_M = 0.14
    # Formal evaluation uses the same physical fingertip grasp as the current
    # admittance dataset; no DetachableJoint or object-pose snap is permitted.
    ENABLE_DETACHABLE_JOINT = False
    RESET_AFTER_EPISODE = False
    ENABLE_GAZEBO_SUCCESS_CHECK = True
    # The real tapered pair seats concentrically.  Five centimetres, used by
    # the legacy evaluator, can classify a peg dropped beside the socket as a
    # success; use a sub-centimetre geometric check for this reproduction.
    SUCCESS_MAX_XY_M = float(os.environ.get("PI05_SUCCESS_MAX_XY_M", "0.008"))
    # The rigid floor places a fully seated peg centre at z=0.885 m.  Do not
    # require an impossible 10 mm penetration below that floor.  This online
    # gate intentionally checks the geometric task result only; the generic
    # controller still rejects sustained force overloads independently.
    # Successful scripted seating is centred near 0.885 m. Keep 2.5 mm of
    # numerical/contact settling tolerance so a nearly-zero residual is not
    # sent back through IK and mapped to a different kinematic branch.
    SUCCESS_MAX_PEG_Z_M = float(
        os.environ.get("PI05_SUCCESS_MAX_PEG_Z_M", "0.890")
    )
    SUCCESS_REQUIRED_CHECKS = int(
        os.environ.get("PI05_SUCCESS_REQUIRED_CHECKS", "5")
    )
    # The converted v9 dataset represents the open gripper as exactly zero.
    # Match the canonical v9 demonstration reset.  The launch-file pose differs
    # by less than 1 mrad, but the repaired visual action head is sensitive to
    # that camera-pose shift, so the generic 0.03-rad reset tolerance is unsafe.
    START_POSE = (
        -0.00001,
        -1.57015,
        1.56995,
        -1.56993,
        -1.56995,
        -0.00005,
        0.0,
    )
    # Gazebo's trajectory controller declares the reset reached within its
    # own numerical tolerance.  One milliradian remains far below the v9 data
    # envelope/camera-pose variation while avoiding false aborts at the open
    # gripper stop.
    START_POSE_TOLERANCE_RAD = 0.001
    # v9 dataset extrema plus a 0.05 rad guard band.  These are evaluation OOD
    # guards rather than the UR3 hardware joint limits.
    TRAINING_STATE_MIN = (
        -0.0501,
        -2.1506,
        0.5133,
        -1.6210,
        -1.6208,
        -0.0501,
        -0.05,
    )
    TRAINING_STATE_MAX = (
        1.8903,
        -1.3317,
        1.7425,
        -0.7025,
        -1.5194,
        1.8903,
        1.05,
    )


if __name__ == "__main__":
    run_ros_side(
        controller_cls=PI05V9AbsoluteROSSide,
        node_name="ur3_pi05_v9_absolute_peg_in_hole_ros_side",
    )

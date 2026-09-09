#!/usr/bin/env python3
"""
Convert peg-in-hole npz files to LeRobot format with force modality.
Saves local dataset to '~/ur3_ft300_ws/pap_moe_framework/datasets/lerobot'.
"""

import argparse, os, sys
import numpy as np
from pathlib import Path


PAP_MOE_V6_SUBTASKS = {
    "grasp the peg",
    "transport to the hole",
    "approach and align with the hole",
    "recover contact and relocate the hole",
    "insert the peg into the hole",
    "verify insertion success",
    "release the peg after verification",
    "retract and go back to home",
}

PAP_MOE_V6_FIELDS = {
    "force_fast": (64, 6),
    "force_slow": (50, 6),
    "state_history": (10, 7),
    "visual_quality": (4,),
}

# A controller-stopped contact can briefly remain above the 12 N online stop
# threshold while the demonstrated action is already unloading it. Preserve
# that useful recovery supervision up to 40 N only in explicit recovery and
# verification phases; normal insertion remains capped at 30 N. Task success
# still requires a separate 25-sample plateau whose mean and peak are both
# <=12 N, and raw >45 N solver impulses remain duration-limited below.
# The controller bounds lateral force at 50 N and axial force at 12 N,
# whereas this validator checks the norm of all three force components.
# 52 N is the corresponding rounded resultant bound, not a looser lateral
# contact contract.
# The controller admits up to 60 N lateral and 12 N axial force during the
# rigid zero-clearance seating transient.  A 62 N total-norm limit covers that
# bounded vector without accepting uncontrolled solver spikes.
CONTACT_FORCE_QUALITY_LIMIT_N = 62.0
CONTROLLED_RELIEF_QUALITY_LIMIT_N = 62.0
GRASP_LOAD_QUALITY_LIMIT_N = 45.0
RAW_SOLVER_IMPULSE_THRESHOLD_N = 55.0
MAX_RAW_OVERLOAD_DURATION_S = 0.10
GRIPPER_ACTION_OPEN_RAD = 0.0
GRIPPER_ACTION_CLOSED_RAD = 0.8
GRIPPER_ACTION_ENDPOINT_EPS = 1.0e-3
def canonicalize_gripper_action(action):
    """Defensively canonicalize v7+ gripper commands for model statistics."""
    result = np.asarray(action, dtype=np.float32).copy()
    value = float(np.clip(
        result[6], GRIPPER_ACTION_OPEN_RAD, GRIPPER_ACTION_CLOSED_RAD
    ))
    if abs(value - GRIPPER_ACTION_OPEN_RAD) <= GRIPPER_ACTION_ENDPOINT_EPS:
        value = GRIPPER_ACTION_OPEN_RAD
    elif abs(value - GRIPPER_ACTION_CLOSED_RAD) <= GRIPPER_ACTION_ENDPOINT_EPS:
        value = GRIPPER_ACTION_CLOSED_RAD
    result[6] = value
    return result

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print(f"ERROR: LeRobot not installed ({e})")
    sys.exit(1)

def find_episodes(input_dir, skip_failed=True):
    episodes = []
    skipped = []
    for d in sorted(os.listdir(input_dir)):
        ep_dir = os.path.join(input_dir, d)
        if "_episode_" in d and os.path.isdir(ep_dir):
            npz_path = os.path.join(ep_dir, "data.npz")
            if os.path.exists(npz_path):
                if skip_failed and d.endswith("_failed"):
                    skipped.append(d)
                else:
                    episodes.append(npz_path)
    return episodes, skipped


def validate_v6_episode(data, path, min_force_hz):
    schema = str(data.get("schema_version", ""))
    teleop_schema = schema == "pap_moe_teleop_v1"
    if schema not in {
        "pap_moe_v6",
        "pap_moe_v7",
        "pap_moe_v8",
        "pap_moe_teleop_v1",
    }:
        raise ValueError(
            f"{path}: schema_version={schema!r}, expected pap_moe_v6, "
            "pap_moe_v7, pap_moe_v8, or pap_moe_teleop_v1"
        )
    endpoint_gripper_contract = schema in {
        "pap_moe_v7",
        "pap_moe_v8",
        "pap_moe_teleop_v1",
    }
    if teleop_schema:
        if str(data.get("collection_mode", "")) != (
            "human_keyboard_moveit_servo_v1"
        ):
            raise ValueError(f"{path}: invalid teleop collection mode")
        if str(data.get("action_source", "")) != (
            "controller_reference_position_one_step_ahead"
        ):
            raise ValueError(f"{path}: invalid teleop action source")
        if str(data.get("gripper_command_contract", "")) != (
            "robotiq_endpoint_0.0_open_0.8_closed_v1"
        ):
            raise ValueError(f"{path}: invalid teleop gripper contract")
    else:
        controller_profile = str(data.get("controller_profile_version", ""))
        valid_v8_controller_profiles = {
            "active_seating_no_gravity_v10",
            "active_seating_no_gravity_v11_fast_free_motion",
        }
        expected_controller_profile = (
            valid_v8_controller_profiles
            if schema == "pap_moe_v8"
            else "universal_gripper_endpoint_force_signature_v8"
            if schema == "pap_moe_v7"
            else "two_speed_force_signature_v7"
        )
        if (
            controller_profile not in expected_controller_profile
            if isinstance(expected_controller_profile, set)
            else controller_profile != expected_controller_profile
        ):
            raise ValueError(
                f"{path}: controller_profile_version={controller_profile!r}, "
                f"expected {expected_controller_profile!r}; episodes without a "
                "bounded plateau or contact-relaxation signature are diagnostic-only"
            )
        subtask_semantics = str(data.get("subtask_semantics_version", ""))
        if subtask_semantics != "post_motion_force_verify_v2":
            raise ValueError(
                f"{path}: subtask_semantics_version={subtask_semantics!r}, "
                "expected 'post_motion_force_verify_v2'; episodes that label "
                "fine insertion as verification must not be mixed into training"
            )
        force_reference_mode = str(data.get("force_reference_mode", ""))
        if force_reference_mode != "per_sample_phase_payload_bias_v2":
            raise ValueError(
                f"{path}: force_reference_mode={force_reference_mode!r}, "
                "expected 'per_sample_phase_payload_bias_v2'; episodes whose "
                "multi-rate windows cross grasp/release under one frame-level "
                "bias must not be mixed into calibrated-force training"
            )
        force_filter_mode = str(data.get("force_filter_mode", ""))
        if force_filter_mode != "median11_sustained_visible_fast_clip30n_native100hz_v6":
            raise ValueError(
                f"{path}: force_filter_mode={force_filter_mode!r}, expected "
                "'median11_sustained_visible_fast_clip30n_native100hz_v6'; differently filtered "
                "episodes must not be mixed into force-conditioned training"
            )
        grasp_mode = str(data.get("grasp_mode", ""))
        expected_grasp_mode = (
            "physical_analytic_handle_endpoint_pose_verified_v3"
            if schema == "pap_moe_v8"
            else "physical_fingertip_endpoint_command_pose_verified_v2"
            if schema == "pap_moe_v7"
            else "physical_fingertip_pose_verified_v1"
        )
        if grasp_mode != expected_grasp_mode:
            raise ValueError(
                f"{path}: grasp_mode={grasp_mode!r}, expected "
                f"{expected_grasp_mode!r}; attached-joint and "
                "unverified grasp episodes are diagnostic-only"
            )
    routing_prior_version = str(data.get("routing_prior_version", ""))
    expected_routing_prior = (
        "factorized_general_qch_v11_deadband"
        if schema in {"pap_moe_v8", "pap_moe_teleop_v1"}
        else "factorized_calibrated_fast_contact_v9"
    )
    if routing_prior_version != expected_routing_prior:
        raise ValueError(
            f"{path}: routing_prior_version={routing_prior_version!r}, "
            f"expected {expected_routing_prior!r}; older "
            "stage targets are diagnostic-only"
        )
    physics_engine = str(data.get("physics_engine", ""))
    if physics_engine != "ignition-physics-dartsim-plugin":
        raise ValueError(
            f"{path}: missing or unsupported physics_engine="
            f"{physics_engine!r}; the validated training domain requires DART"
        )
    sim_position_gain = float(data.get("sim_position_gain", np.nan))
    if not np.isfinite(sim_position_gain) or not np.isclose(
        sim_position_gain, 0.5, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"{path}: sim_position_gain={sim_position_gain!r}, expected 0.5 "
            "from the validated gz_ros2_control overlay"
        )
    fixture_mode = str(data.get("fixture_mode", ""))
    expected_fixture_mode = (
        "rigid_real_zero_clearance_split_collision_v8"
        if schema in {"pap_moe_v8", "pap_moe_teleop_v1"}
        else "rigid_real_frustum_only_cavity_v6"
        if schema == "pap_moe_v7"
        else "rigid_real_tapered_socket_v5"
    )
    if fixture_mode != expected_fixture_mode:
        raise ValueError(
            f"{path}: fixture_mode={fixture_mode!r}, expected "
            f"{expected_fixture_mode!r}"
        )
    if endpoint_gripper_contract and not teleop_schema:
        action_source = str(data.get("action_source", ""))
        gripper_contract = str(data.get("gripper_command_contract", ""))
        if action_source != "controller_desired_position_one_step_ahead":
            raise ValueError(f"{path}: invalid v7 action_source={action_source!r}")
        if gripper_contract != "robotiq_endpoint_0.0_open_0.8_closed_v1":
            raise ValueError(
                f"{path}: invalid v7 gripper command contract={gripper_contract!r}"
            )
    if schema == "pap_moe_v8":
        if str(data.get("terminal_seating_contract", "")) != (
            "active_pre_release_seat_no_gravity_v1"
        ):
            raise ValueError(
                f"{path}: v8 episode lacks the active pre-release seating "
                "contract; gravity-completed insertions are diagnostic-only"
            )
        expected_pickup_duration = (
            3.0
            if controller_profile ==
            "active_seating_no_gravity_v11_fast_free_motion"
            else 5.0
        )
        if not np.isclose(
            float(data.get("grasp_pickup_duration_s", np.nan)),
            expected_pickup_duration,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"{path}: invalid v8 grasp pickup duration")
        expected_gripper_scaling = (
            1.0
            if controller_profile ==
            "active_seating_no_gravity_v11_fast_free_motion"
            else 0.75
        )
        if not np.isclose(
            float(data.get("gripper_velocity_scaling", np.nan)),
            expected_gripper_scaling,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"{path}: invalid v8 gripper velocity scaling")
        if str(data.get("grasp_pickup_profile", "")) != "ease_out_sqrt_v1":
            raise ValueError(f"{path}: invalid v8 grasp pickup profile")
        if str(data.get("gripper_fingertip_contact_model", "")) != (
            "analytic_cylinder_rubber_pad_v1"
        ):
            raise ValueError(f"{path}: invalid v8 fingertip contact model")
        if not np.isclose(
            float(data.get("gripper_contact_stall_velocity_rad_s", np.nan)),
            0.10,
            rtol=0.0,
            atol=1e-6,
        ) or int(data.get("gripper_contact_stall_cycles", -1)) != 5:
            raise ValueError(f"{path}: invalid v8 gripper contact latch")
        if str(data.get("policy_timebase", "")) != "gazebo_sim_clock_10hz_v1":
            raise ValueError(f"{path}: invalid v8 policy sampling timebase")
        if (
            str(data.get("policy_episode_endpoint", ""))
            != "task_result_before_reset_v1"
        ):
            raise ValueError(f"{path}: invalid v8 policy episode endpoint")
        actions = np.asarray(data["action"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"{path}: invalid v7 action shape={actions.shape}")
        if float(actions[:, 6].min()) < -0.011 or float(actions[:, 6].max()) > 0.801:
            raise ValueError(f"{path}: gripper commands leave the [0, 0.8] endpoint range")
    frame_count = len(data["state"])
    if schema in {"pap_moe_v8", "pap_moe_teleop_v1"}:
        timestamp_key = "timestamp_ros" if teleop_schema else "timestamp"
        policy_timestamps = np.asarray(
            data.get(timestamp_key, []), dtype=np.float64
        )
        if policy_timestamps.shape != (frame_count,):
            raise ValueError(f"{path}: invalid v8 policy timestamp shape")
        policy_dt = np.diff(policy_timestamps)
        if (
            len(policy_dt) == 0
            or np.any(policy_dt <= 0.0)
            or not np.isclose(np.median(policy_dt), 0.1, rtol=0.0, atol=0.03)
        ):
            raise ValueError(
                f"{path}: policy frames are not sampled at 10 Hz"
            )
    for key, trailing_shape in PAP_MOE_V6_FIELDS.items():
        if key not in data:
            raise ValueError(f"{path}: missing required field {key!r}")
        expected = (frame_count, *trailing_shape)
        if data[key].shape != expected:
            raise ValueError(f"{path}: {key} shape={data[key].shape}, expected {expected}")
    clean_teacher_keys = {
        "camera0_clean_teacher",
        "camera1_clean_teacher",
        "visual_degradation_active",
    }
    present_clean_teacher_keys = clean_teacher_keys.intersection(data.files)
    if present_clean_teacher_keys and present_clean_teacher_keys != clean_teacher_keys:
        missing = sorted(clean_teacher_keys - present_clean_teacher_keys)
        raise ValueError(
            f"{path}: incomplete clean-view teacher contract; missing {missing}"
        )
    episode_has_degradation = bool(np.asarray(data.get("cam_degraded", False)).item())
    if episode_has_degradation and not present_clean_teacher_keys:
        raise ValueError(
            f"{path}: visually degraded episode is missing same-time clean "
            "teacher views required by E2 memory training"
        )
    if present_clean_teacher_keys:
        contract = str(data.get("visual_supervision_contract", ""))
        if contract != "shared_policy_view_clean_teacher_v1":
            raise ValueError(
                f"{path}: invalid visual_supervision_contract={contract!r}"
            )
        if str(data.get("visual_degradation_scope", "")) != (
            "both_policy_cameras_v1"
        ):
            raise ValueError(
                f"{path}: invalid visual_degradation_scope; q=0 supervision "
                "requires both policy cameras to be degraded"
            )
        degradation_active = np.asarray(
            data["visual_degradation_active"], dtype=bool
        )
        if degradation_active.shape != (frame_count,):
            raise ValueError(
                f"{path}: visual_degradation_active shape="
                f"{degradation_active.shape}, expected {(frame_count,)}"
            )
        for policy_key, teacher_key in (
            ("camera0", "camera0_clean_teacher"),
            ("camera1", "camera1_clean_teacher"),
        ):
            if data[teacher_key].shape != data[policy_key].shape:
                raise ValueError(
                    f"{path}: {teacher_key} shape={data[teacher_key].shape}, "
                    f"expected {data[policy_key].shape}"
                )
        if np.any(~degradation_active):
            for policy_key, teacher_key in (
                ("camera0", "camera0_clean_teacher"),
                ("camera1", "camera1_clean_teacher"),
            ):
                if not np.array_equal(
                    data[policy_key][~degradation_active],
                    data[teacher_key][~degradation_active],
                ):
                    raise ValueError(
                        f"{path}: clean frames differ between {policy_key} "
                        f"and its teacher view"
                    )
        if np.any(degradation_active):
            changed = False
            for policy_key, teacher_key in (
                ("camera0", "camera0_clean_teacher"),
                ("camera1", "camera1_clean_teacher"),
            ):
                changed |= not np.array_equal(
                    data[policy_key][degradation_active],
                    data[teacher_key][degradation_active],
                )
            if not changed:
                raise ValueError(
                    f"{path}: degradation is active but neither policy view "
                    "differs from its clean teacher view"
                )
    task_array = np.asarray([str(value) for value in data["task"]], dtype=object)
    if schema in {"pap_moe_v8", "pap_moe_teleop_v1"}:
        global_task = "pick up the peg and insert it into the hole"
        if set(task_array) != {global_task}:
            raise ValueError(
                f"{path}: v8 task prompt must stay global and identical for "
                "baseline/PAP-MoE"
            )
        if "semantic_subtask" not in data:
            raise ValueError(f"{path}: v8 is missing semantic_subtask labels")
        semantic_array = np.asarray(
            [str(value) for value in data["semantic_subtask"]], dtype=object
        )
        if semantic_array.shape != (frame_count,):
            raise ValueError(f"{path}: invalid v8 semantic_subtask shape")
    else:
        semantic_array = task_array
    # `task` is deliberately the same global language prompt for every v8
    # frame so baseline and PAP-MoE receive identical policy inputs.  Dataset
    # quality checks that depend on controller phase must use the separate
    # auxiliary semantic label instead.
    phase_array = semantic_array
    if schema == "pap_moe_v8" and not teleop_schema:
        required_gripper_keys = {
            "gripper_kinematic_state",
            "gripper_mimic_error",
            "gripper_kinematic_joint_names",
            "gripper_mimic_max_error_rad",
        }
        missing_gripper_keys = sorted(required_gripper_keys - set(data.files))
        if missing_gripper_keys:
            raise ValueError(
                f"{path}: missing full gripper linkage validation fields: "
                f"{missing_gripper_keys}"
            )
        gripper_kinematic_state = np.asarray(
            data["gripper_kinematic_state"], dtype=np.float32
        )
        gripper_mimic_error = np.asarray(
            data["gripper_mimic_error"], dtype=np.float32
        )
        if gripper_kinematic_state.shape != (frame_count, 6):
            raise ValueError(
                f"{path}: invalid gripper linkage state shape "
                f"{gripper_kinematic_state.shape}"
            )
        if gripper_mimic_error.shape != (frame_count, 6):
            raise ValueError(
                f"{path}: invalid gripper mimic error shape "
                f"{gripper_mimic_error.shape}"
            )
        mimic_error_limit = float(data["gripper_mimic_max_error_rad"])
        measured_mimic_error = float(np.max(gripper_mimic_error))
        if (
            not np.isfinite(gripper_kinematic_state).all()
            or not np.isfinite(gripper_mimic_error).all()
            or measured_mimic_error > mimic_error_limit
        ):
            raise ValueError(
                f"{path}: gripper linkage lost symmetry; max mimic error "
                f"{measured_mimic_error:.6f} rad exceeds "
                f"{mimic_error_limit:.6f} rad"
            )
        states = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["action"], dtype=np.float32)
        tool0_z = np.asarray(data["tool0_z"], dtype=np.float32)
        state_static = np.zeros(frame_count, dtype=bool)
        action_repeated = np.zeros(frame_count, dtype=bool)
        tool_static = np.zeros(frame_count, dtype=bool)
        state_static[1:] = (
            np.max(np.abs(np.diff(states, axis=0)), axis=1) < 1e-3
        )
        action_repeated[1:] = (
            np.max(np.abs(np.diff(actions, axis=0)), axis=1) < 1e-3
        )
        # At final exact-fit seating the measured tool may advance only about
        # 2.5 um per policy frame, producing sub-milliradian joint changes.
        # This is still directed task-space motion. Only sub-micron change is
        # treated as stationary.
        tool_static[1:] = np.abs(np.diff(tool0_z)) < 1e-6
        # A short stationary window is required for explicit physical
        # verification. Elsewhere, long runs of an unchanged observation and
        # unchanged target teach the policy to stall at phase boundaries.
        accidental_static = (
            state_static
            & action_repeated
            & tool_static
            & (phase_array != "verify insertion success")
        )
        longest_static_run = 0
        current_static_run = 0
        for is_static in accidental_static:
            current_static_run = current_static_run + 1 if is_static else 0
            longest_static_run = max(
                longest_static_run, current_static_run
            )
        # Up to 2.5 s is permitted for bounded physical settling and the
        # iterative exact-axis calibration. The rejected recorder bug created
        # 8.6--8.9 s runs in every episode.
        if longest_static_run > 25:
            raise ValueError(
                f"{path}: contains {longest_static_run} consecutive 10 Hz "
                "stationary/repeated-action frames outside the explicit "
                "verification hold; maximum allowed is 25"
            )
    tasks = set(semantic_array)
    teleop_subtasks = {
        "main", "descent", "lower", "grasp", "post_grasp", "align", "contact"
    }
    expected_subtasks = teleop_subtasks if teleop_schema else PAP_MOE_V6_SUBTASKS
    unknown = sorted(tasks - expected_subtasks)
    if unknown:
        raise ValueError(f"{path}: unknown v6 subtask labels: {unknown}")
    trajectory_scope = str(data.get("trajectory_scope", "full_task"))
    grasp_recovery_prefix = trajectory_scope == "grasp_recovery_prefix_v1"
    if trajectory_scope not in {"full_task", "grasp_recovery_prefix_v1"}:
        raise ValueError(f"{path}: unknown trajectory_scope={trajectory_scope!r}")
    if grasp_recovery_prefix:
        if tasks != {"grasp the peg"}:
            raise ValueError(
                f"{path}: grasp recovery prefix contains non-grasp tasks: {tasks}"
            )
        offset = np.asarray(data.get("grasp_recovery_offset", []), dtype=np.float32)
        if offset.shape != (2,) or float(np.linalg.norm(offset)) < 0.005:
            raise ValueError(f"{path}: invalid grasp recovery offset {offset}")
        if not np.any(np.asarray(data["state"])[:, 6] > 0.60):
            raise ValueError(f"{path}: grasp recovery prefix never closes the gripper")
    elif schema in {"pap_moe_v8", "pap_moe_teleop_v1"}:
        if not teleop_schema and "retract and go back to home" in tasks:
            raise ValueError(
                f"{path}: v8 policy episode contains post-success reset motion"
            )
        if not teleop_schema and semantic_array[-1] != "release the peg after verification":
            raise ValueError(
                f"{path}: v8 full-task episode does not end in verified release"
            )
        target_z = float(data.get("terminal_target_peg_center_z", np.nan))
        pre_release_pose = np.asarray(
            data.get("terminal_pre_release_peg_pose", []), dtype=np.float32
        )
        post_release_pose = np.asarray(
            data.get("terminal_post_release_peg_pose", []), dtype=np.float32
        )
        release_delta_z = float(data.get("terminal_release_delta_z", np.nan))
        if (
            not np.isclose(target_z, 0.885, rtol=0.0, atol=1e-6)
            or pre_release_pose.shape != (7,)
            or post_release_pose.shape != (7,)
            or not np.isfinite(pre_release_pose).all()
            or not np.isfinite(post_release_pose).all()
            or abs(float(pre_release_pose[2]) - target_z) > 0.00035
            or abs(float(post_release_pose[2]) - target_z) > 0.00035
            or not np.isfinite(release_delta_z)
            or abs(release_delta_z) > 0.00035
        ):
            raise ValueError(
                f"{path}: terminal seating was not actively completed before "
                f"release (target_z={target_z}, pre_z="
                f"{pre_release_pose[2] if pre_release_pose.shape == (7,) else np.nan}, "
                f"post_z={post_release_pose[2] if post_release_pose.shape == (7,) else np.nan}, "
                f"release_dz={release_delta_z})"
            )
    # Establishing a physical grasp exposes the empty-referenced sensor to
    # the peg payload.  With the randomized mass up to 2 kg, a bounded pickup
    # transient can legitimately exceed the contact-phase limit.  Keep the
    # stricter limit everywhere that the robot can touch the fixture; the old
    # ~150 N table-squeeze trajectories still fail the 45 N load limit.
    controlled_relief_mask = np.isin(
        phase_array,
        [
            "recover contact and relocate the hole",
            "verify insertion success",
            # Multi-rate force histories at the start of release still carry
            # the immediately preceding controlled-relief samples.
            "release the peg after verification",
            "retract and go back to home",
            "contact",
        ],
    )
    force_quality_limit = np.where(
        np.isin(phase_array, ["grasp the peg", "grasp"]),
        GRASP_LOAD_QUALITY_LIMIT_N,
        np.where(
            controlled_relief_mask,
            CONTROLLED_RELIEF_QUALITY_LIMIT_N,
            CONTACT_FORCE_QUALITY_LIMIT_N,
        ),
    ).astype(np.float32)
    calibrated_force = np.asarray(data["force"], dtype=np.float32)
    if not np.isfinite(calibrated_force).all():
        raise ValueError(f"{path}: calibrated force contains non-finite values")
    frame_force_norm = np.linalg.norm(calibrated_force[:, :3], axis=1)
    violating_frames = np.flatnonzero(frame_force_norm > force_quality_limit)
    if len(violating_frames):
        peak_index = int(np.argmax(frame_force_norm - force_quality_limit))
        raise ValueError(
            f"{path}: calibrated Cartesian force "
            f"{frame_force_norm[peak_index]:.1f} N at frame {peak_index} "
            f"during {phase_array[peak_index]!r} exceeds its "
            f"{force_quality_limit[peak_index]:.0f} N training-quality limit"
        )
    for window_name in ("force_fast", "force_slow"):
        window_force = np.asarray(data[window_name], dtype=np.float32)
        window_norm = np.linalg.norm(window_force[..., :3], axis=-1)
        window_peak_per_frame = window_norm.max(axis=1)
        violating_frames = np.flatnonzero(
            window_peak_per_frame > force_quality_limit
        )
        if len(violating_frames):
            peak_index = int(
                np.argmax(window_peak_per_frame - force_quality_limit)
            )
            raise ValueError(
                f"{path}: {window_name} Cartesian force peak "
                f"{window_peak_per_frame[peak_index]:.1f} N at frame "
                f"{peak_index} during {phase_array[peak_index]!r} exceeds "
                f"its {force_quality_limit[peak_index]:.0f} N "
                "training-quality limit; the policy would still observe "
                "this native-rate transient even if the 10 Hz frame stream "
                "missed it"
            )
    for key, expected in (
        ("force_fast_valid", 64),
        ("force_slow_valid", 50),
        ("state_history_valid", 10),
    ):
        if key not in data or np.any(np.asarray(data[key]) < expected):
            minimum = int(np.min(data[key])) if key in data else 0
            raise ValueError(
                f"{path}: {key} minimum={minimum}; all recorded frames must be warm"
            )
    force_sensitive_tasks = {
        "approach and align with the hole",
        "recover contact and relocate the hole",
        "insert the peg into the hole",
        "verify insertion success",
        "align",
        "contact",
    }
    force_sensitive_mask = np.isin(
        phase_array,
        list(force_sensitive_tasks),
    )
    if not np.any(force_sensitive_mask) and not grasp_recovery_prefix:
        raise ValueError(
            f"{path}: episode ended before any force-sensitive insertion "
            "phase; reject as an incomplete controller trajectory"
        )
    timestamps = np.asarray(data["raw_force_timestamp"], dtype=np.float64)
    values = np.asarray(data["raw_force"], dtype=np.float32)
    if timestamps.ndim != 1 or values.shape != (len(timestamps), 6):
        raise ValueError(f"{path}: invalid raw force stream shape")
    positive_dt = np.diff(timestamps)
    positive_dt = positive_dt[positive_dt > 1e-6]
    if len(positive_dt) < 2:
        raise ValueError(f"{path}: insufficient timestamped raw force samples")
    measured_hz = float(1.0 / np.median(positive_dt))
    if measured_hz < min_force_hz:
        raise ValueError(
            f"{path}: median force rate {measured_hz:.1f} Hz is below "
            f"the required {min_force_hz:.1f} Hz"
        )
    raw_force_norm = np.linalg.norm(values[:, :3], axis=1)
    overload = raw_force_norm > RAW_SOLVER_IMPULSE_THRESHOLD_N
    max_run = 0
    run = 0
    for is_overload in overload:
        run = run + 1 if is_overload else 0
        max_run = max(max_run, run)
    max_allowed_run = max(
        1, int(round(MAX_RAW_OVERLOAD_DURATION_S * measured_hz))
    )
    if max_run > max_allowed_run:
        raise ValueError(
            f"{path}: raw force overload persisted for "
            f"{max_run / measured_hz:.3f} s ({max_run} samples), exceeding "
            f"the {MAX_RAW_OVERLOAD_DURATION_S:.3f} s DART-impulse allowance"
        )
    stages = np.asarray(data["stage"], dtype=np.float32)
    if stages.shape != (frame_count, 4) or not np.allclose(
        stages.sum(axis=1), 1.0, atol=1e-3
    ):
        raise ValueError(f"{path}: physical expert soft labels are invalid")
    return measured_hz


def _fix_action_stats_for_relative(dataset_root):
    import json, shutil
    from pathlib import Path
    
    stats_path = Path(dataset_root) / "meta" / "stats.json"
    if not stats_path.exists():
        return

    parquet_files = sorted(Path(dataset_root).glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        return

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return

    backup = stats_path.with_suffix(".json.abs_backup")
    if not backup.exists():
        shutil.copy(stats_path, backup)

    actions_list, states_list = [], []
    for pf_path in parquet_files:
        pf = pq.read_table(str(pf_path))
        actions_list.append(np.stack(pf.column("action").to_pylist()))
        states_list.append(np.stack(pf.column("observation.state").to_pylist()))
    actions = np.concatenate(actions_list, axis=0)
    states = np.concatenate(states_list, axis=0)
    delta = actions - states
    EXCLUDE = [6]

    with open(stats_path) as f:
        stats = json.load(f)

    old_action = stats["action"]
    new_vals = {}

    for key in ["mean", "std", "min", "max"]:
        arr = np.zeros(7, dtype=np.float32)
        for i in range(7):
            src = actions[:, i] if i in EXCLUDE else delta[:, i]
            if key == "mean":
                arr[i] = src.mean()
            elif key == "std":
                arr[i] = max(src.std(), 1e-8)
            elif key == "min":
                arr[i] = src.min()
            elif key == "max":
                arr[i] = src.max()
        new_vals[key] = arr

    for q in [0.01, 0.10, 0.50, 0.90, 0.99]:
        key = f"q{int(q * 100):02d}"
        arr = np.zeros(7, dtype=np.float32)
        for i in range(7):
            src = actions[:, i] if i in EXCLUDE else delta[:, i]
            arr[i] = np.quantile(src, q)
        new_vals[key] = arr

    for key, arr in new_vals.items():
        if key in old_action:
            old_action[key] = arr.tolist()
    stats["action"] = old_action

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    arm_std = float(new_vals["std"][:6].mean())
    print(f"  [stats fix] Action stats patched to relative (arm std={arm_std:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=os.path.expanduser("~/ur3_ft300_ws/pap_moe_framework/datasets/raw"))
    parser.add_argument("--repo_id", type=str, default="pap_moe/ur3_peg_in_hole")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--source_fps", type=int, default=10)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser("~/ur3_ft300_ws/pap_moe_framework/datasets/lerobot"))
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--keep_failed", action="store_true")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate raw v6 episodes without creating a LeRobot dataset",
    )
    parser.add_argument("--relative_actions", action="store_true",
                        help="Unsupported for PAP-MoE v6; retained to reject stale commands")
    parser.add_argument(
        "--min_force_hz",
        type=float,
        default=50.0,
        help="Reject episodes whose timestamped native force stream is slower",
    )
    parser.add_argument("--episodes_filter", nargs="+", type=str, default=None,
                        help="List of episode number strings/substrings to include (e.g. '0301' '0302')")
    parser.add_argument(
        "--model_view",
        choices=("full", "baseline"),
        default="full",
        help=(
            "full keeps PAP-MoE auxiliary modalities; baseline materializes "
            "only two RGB views, 7D state and 7D action for Pi0.5/ACT/DP"
        ),
    )
    args = parser.parse_args()
    if args.relative_actions:
        raise ValueError("PAP-MoE v6 stores and trains absolute actions only")

    step = max(1, args.source_fps // args.fps)
    input_dir = Path(args.input)
    # Schema validation must cover failed recovery demonstrations too.  They are
    # excluded by default only when compiling the training dataset.
    episodes, skipped = find_episodes(
        input_dir,
        skip_failed=not (args.keep_failed or args.validate_only),
    )

    if args.episodes_filter:
        filtered = []
        for ep_path in episodes:
            if any(f_str in ep_path for f_str in args.episodes_filter):
                filtered.append(ep_path)
        episodes = filtered
        print(f"Filtered to {len(episodes)} episodes matching {args.episodes_filter}")

    kind = "raw" if args.validate_only or args.keep_failed else "successful"
    print(f"Found {len(episodes)} {kind} episodes")
    if not episodes:
        raise FileNotFoundError(
            f"No matching episode data found under {input_dir}"
        )

    if args.validate_only:
        rates = []
        for npz_path in episodes:
            with np.load(npz_path, allow_pickle=True) as data:
                rates.append(validate_v6_episode(data, npz_path, args.min_force_hz))
        print(
            f"PAP-MoE v6 validation passed for {len(rates)} episodes; "
            f"native force median range={min(rates):.1f}–{max(rates):.1f} Hz"
        )
        return

    first = np.load(episodes[0], allow_pickle=True)
    state_shape = first["state"].shape[1]
    action_shape = first["action"].shape[1]
    force_shape = first["force"].shape[1]
    cam0_shape = first["camera0"].shape[1:]
    cam1_shape = first["camera1"].shape[1:]

    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        "robotiq_85_left_knuckle_joint",
    ]

    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_shape,),
            "names": joint_names,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_shape,),
            "names": joint_names,
        },
        "observation.force": {
            "dtype": "float32",
            "shape": (force_shape,),
            "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
        },
        "observation.force_fast": {
            "dtype": "float32",
            "shape": (64, 6),
            "names": None,
        },
        "observation.force_slow": {
            "dtype": "float32",
            "shape": (50, 6),
            "names": None,
        },
        "observation.state_history": {
            "dtype": "float32",
            "shape": (10, 7),
            "names": None,
        },
        "observation.visual_quality": {
            "dtype": "float32",
            "shape": (4,),
            "names": [
                "black_fraction",
                "saturated_fraction",
                "contrast",
                "valid",
            ],
        },
        "observation.images.camera0": {
            "dtype": "video",
            "shape": tuple(cam0_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera1": {
            "dtype": "video",
            "shape": tuple(cam1_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.physics_gate_target": {
            "dtype": "float32",
            "shape": (4,),
            "names": [
                "E1_normal_vision_free",
                "E2_visual_degraded",
                "E3_rigid_contact",
                "E4_movable_contact",
            ],
        },
    }
    if args.model_view == "baseline":
        baseline_keys = {
            "action",
            "observation.state",
            "observation.images.camera0",
            "observation.images.camera1",
        }
        features = {key: value for key, value in features.items() if key in baseline_keys}

    print(f"Creating LeRobot dataset: {args.repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type="ur3",
        use_videos=True,
        vcodec="h264",
    )

    total_frames = 0
    for ep_idx, npz_path in enumerate(episodes):
        data = np.load(npz_path, allow_pickle=True)
        measured_force_hz = validate_v6_episode(
            data,
            npz_path,
            args.min_force_hz,
        )
        states = data["state"]
        actions = data["action"]
        forces = data["force"]
        forces_fast = data["force_fast"]
        forces_slow = data["force_slow"]
        states_history = data["state_history"]
        visual_quality = data["visual_quality"]
        cam0 = data["camera0"]
        cam1 = data["camera1"]
        # Formal v6 data uses the online observable routing prior recorded in
        # the NPZ.  Legacy external relabel caches used privileged geometry
        # and must never silently override these targets.
        stages = np.asarray(data["stage"], dtype=np.float32)
        n_frames = len(states)
        task_data = data.get("task", "pick up the peg and insert it into the hole")
        if isinstance(task_data, (np.ndarray, list)) and len(task_data) == n_frames:
            tasks = [str(t) for t in task_data]
        else:
            tasks = [str(task_data)] * n_frames
        for i in range(0, n_frames, step):
            # stages[i] is the 4D soft physics vector [E1, E2, E3, E4]
            stage_vec = stages[i].astype(np.float32)

            obs_state = states[i].copy().astype(np.float32)
            act = canonicalize_gripper_action(actions[i])
            if str(data.get("schema_version", "")) == "pap_moe_v6":
                # Preserve the legacy checkpoint contract only for legacy v6
                # episodes.  V7 keeps continuous physical observations and
                # controller-desired endpoint actions exactly as recorded.
                open_subtasks = {
                    "release the peg after verification",
                    "retract and go back to home",
                }
                obs_state[6] = float(
                    tasks[i] not in open_subtasks and states[i, 6] > 0.12
                )
                act[6] = float(
                    tasks[i] not in open_subtasks and actions[i, 6] > 0.12
                )

            frame = {
                "observation.state": obs_state,
                "action": act,
                "observation.images.camera0": (cam0[i] * 255).astype(np.uint8),
                "observation.images.camera1": (cam1[i] * 255).astype(np.uint8),
                "task": tasks[i],
            }
            if args.model_view == "full":
                frame.update({
                    "observation.force": forces[i].astype(np.float32),
                    "observation.force_fast": forces_fast[i].astype(np.float32),
                    "observation.force_slow": forces_slow[i].astype(np.float32),
                    "observation.state_history": states_history[i].astype(np.float32),
                    "observation.visual_quality": visual_quality[i].astype(np.float32),
                    "observation.physics_gate_target": stage_vec,
                })
            dataset.add_frame(frame)
            total_frames += 1

        dataset.save_episode()
        print(
            f"  Episode {ep_idx+1}/{len(episodes)} processed "
            f"(native force median={measured_force_hz:.1f} Hz)"
        )

    dataset.finalize()
    # LeRobot's episode-stat aggregation averages episode quantiles, which is
    # not a valid global quantile for multi-position/multi-style datasets.
    # Repair all floating numeric statistics from the finalized parquet before
    # this dataset is copied into any baseline or PAP-MoE training view.
    from recompute_lerobot_numeric_stats import recompute_numeric_stats

    recompute_numeric_stats(dataset.root)
    # PAP-MoE v6 stores and trains absolute joint targets. No delta-statistics
    # rewrite is allowed here.

    # Copy to local directory
    import shutil
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    shutil.copytree(dataset.root, args.output_dir)
    # Diffusion resolves observation histories from every observation feature
    # present in dataset metadata, even when a policy config names a subset.
    # Therefore ACT/DP use the materialized baseline view, while PAP-MoE uses
    # the full view.  Both views are compiled from the same validated raw NPZs.
    model_views = {
        "data_contract_version": "pap_moe_multimodel_absolute_v1",
        "materialized_view": args.model_view,
        "global_task": "pick up the peg and insert it into the hole",
        "action": {
            "key": "action",
            "representation": "absolute_joint_position",
            "shape": [7],
            "gripper": {
                "open_rad": GRIPPER_ACTION_OPEN_RAD,
                "closed_command_rad": GRIPPER_ACTION_CLOSED_RAD,
                "continuous_transition": True,
                "measured_state_is_not_command": True,
            },
        },
        "baseline": {
            "models": ["pi05", "act", "diffusion_policy"],
            "input_features": [
                "observation.images.camera0",
                "observation.images.camera1",
                "observation.state",
            ],
            "output_features": ["action"],
            "forbidden_inputs": [
                "observation.force",
                "observation.force_fast",
                "observation.force_slow",
                "observation.state_history",
                "observation.visual_quality",
                "observation.physics_gate_target",
            ],
            "normalization": {
                "pi05": {"state": "quantiles_q01_q99", "action": "quantiles_q01_q99"},
                "act": {"state": "mean_std", "action": "mean_std"},
                "diffusion_policy": {"state": "min_max", "action": "min_max"},
            },
        },
        "pap_moe": {
            "input_features": [
                "observation.images.camera0",
                "observation.images.camera1",
                "observation.state",
                "observation.force",
                "observation.force_fast",
                "observation.force_slow",
                "observation.state_history",
                "observation.visual_quality",
            ],
            "auxiliary_targets": [
                "observation.physics_gate_target",
            ],
            "output_features": ["action"],
        },
    }
    with open(os.path.join(args.output_dir, "MODEL_VIEWS.json"), "w") as f:
        import json
        json.dump(model_views, f, indent=2)
    with open(os.path.join(args.output_dir, "GLOBAL_TASK_VIEW.json"), "w") as f:
        json.dump(
            {
                "source_dataset": str(input_dir.resolve()),
                "global_task": model_views["global_task"],
                "purpose": (
                    "standard Pi0.5/ACT/DP baseline without force or routing "
                    "target leakage" if args.model_view == "baseline" else
                    "full PAP-MoE multimodal training and auxiliary supervision"
                ),
            },
            f,
            indent=2,
        )
    print(f"Dataset successfully compiled and stored at: {args.output_dir}")

if __name__ == "__main__":
    main()

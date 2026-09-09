#!/usr/bin/env python3
"""Explicit simulation-only Cartesian expert for recovery data collection.

This process is never part of baseline deployment.  It reads Gazebo peg/hole
poses only after entering the recovery-data workflow, publishes every expert
joint target through the auditable recovery IPC, and permanently takes over
from the policy.  The resulting expert actions may be used as DAgger labels;
formal baseline evaluation must run without this process or recovery IPC.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import time
import threading

import numpy as np
import rclpy
from moveit_msgs.srv import GetPositionIK
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pap_moe_framework.rollout_recovery.protocol import (
    EXPERT_MODE,
    POLICY_MODE,
    PROTOCOL_VERSION,
    publish_expert_chunk,
    read_control_mode,
    read_takeover,
    release_to_policy,
    request_takeover,
    wait_for_expert_execution_ack,
)
from pap_moe_framework.rollout_recovery.recorder import read_gazebo_fixture_poses


JOINT_STATE_FILE = Path("/tmp/ur3_joint_state.txt")
FORCE_FILE = Path("/tmp/ur3_force.npy")
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_MIN = np.asarray([-0.0501, -2.1506, 0.5133, -1.6210, -1.6208, -0.0501])
ARM_MAX = np.asarray([1.8903, -1.3317, 1.7425, -0.7025, -1.5194, 1.8903])


class LocalPhaseProgress:
    """Convert an observable completion estimate to phase-local supervision."""

    def __init__(self) -> None:
        self.phase: int | None = None
        self.start_completion = 0.0
        self.max_progress = 0.0

    def update(self, phase: int, completion: float) -> tuple[float, float]:
        completion = float(np.clip(completion, 0.0, 1.0))
        if self.phase != int(phase):
            self.phase = int(phase)
            self.start_completion = completion
            self.max_progress = 0.0
            return 0.0, 0.0
        denominator = max(1.0 - self.start_completion, 1e-6)
        progress = float(np.clip((completion - self.start_completion) / denominator, 0.0, 1.0))
        self.max_progress = max(self.max_progress, progress)
        readiness = float(np.clip((self.max_progress - 0.75) / 0.25, 0.0, 1.0))
        return self.max_progress, readiness


class MoveItRecoveryIK:
    """Use live TF and MoveIt's kinematics plugin without executing MoveIt plans."""

    def __init__(self, *, group_name: str, link_name: str):
        rclpy.init()
        self.node = Node("pap_moe_rollout_recovery_moveit_ik")
        self.group_name = group_name
        self.link_name = link_name
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.client = self.node.create_client(GetPositionIK, "/compute_ik")
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        if not self.client.wait_for_service(timeout_sec=10.0):
            self.close()
            raise RuntimeError("MoveIt /compute_ik service is unavailable")

    def close(self) -> None:
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(timeout_sec=5.0)
            self.executor = None
        if getattr(self, "spin_thread", None) is not None:
            self.spin_thread.join(timeout=5.0)
            self.spin_thread = None
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None
        if rclpy.ok():
            rclpy.shutdown()

    def current_pose(self):
        try:
            return self.tf_buffer.lookup_transform("world", self.link_name, rclpy.time.Time())
        except Exception:
            return None

    def frame_midpoint(self, frames: tuple[str, ...]) -> np.ndarray | None:
        positions = []
        for frame in frames:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "world", frame, rclpy.time.Time()
                )
            except Exception:
                return None
            translation = transform.transform.translation
            positions.append([translation.x, translation.y, translation.z])
        return np.mean(np.asarray(positions, dtype=np.float64), axis=0)

    def solve_translation(
        self,
        current_arm: np.ndarray,
        world_delta: np.ndarray,
        *,
        timeout_s: float = 2.0,
    ) -> np.ndarray | None:
        transform = self.current_pose()
        if transform is None:
            return None
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self.group_name
        ik.ik_link_name = self.link_name
        ik.pose_stamped.header.frame_id = "world"
        ik.pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        ik.pose_stamped.pose.position.x = float(translation.x + world_delta[0])
        ik.pose_stamped.pose.position.y = float(translation.y + world_delta[1])
        ik.pose_stamped.pose.position.z = float(translation.z + world_delta[2])
        ik.pose_stamped.pose.orientation.x = rotation.x
        ik.pose_stamped.pose.orientation.y = rotation.y
        ik.pose_stamped.pose.orientation.z = rotation.z
        ik.pose_stamped.pose.orientation.w = rotation.w
        ik.robot_state.joint_state = JointState(
            name=list(ARM_JOINTS),
            position=[float(value) for value in current_arm],
        )
        ik.avoid_collisions = True
        ik.timeout.sec = 1
        future = self.client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        if not future.done():
            return None
        response = future.result()
        if response is None or response.error_code.val != 1:
            return None
        names = response.solution.joint_state.name
        positions = response.solution.joint_state.position
        if not all(name in names for name in ARM_JOINTS):
            return None
        solution = np.asarray(
            [positions[names.index(name)] for name in ARM_JOINTS], dtype=np.float64
        )
        return solution if np.isfinite(solution).all() else None


def ur3_fk(q: np.ndarray) -> np.ndarray:
    d1, a2, a3, d4, d5, d6 = 0.1519, -0.24365, -0.21325, 0.11235, 0.08535, 0.2619
    q1, q2, q3, q4, q5, q6 = q[:6]
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)
    s234, c234 = np.sin(q2 + q3 + q4), np.cos(q2 + q3 + q4)
    return np.asarray(
        [
            c1 * (a2 * c2 + a3 * c23 - d4 * s234 + d6 * c234) - d5 * s1,
            s1 * (a2 * c2 + a3 * c23 - d4 * s234 + d6 * c234) + d5 * c1,
            d1 + a2 * s2 + a3 * s23 + d4 * c234 + d6 * s234,
        ],
        dtype=np.float64,
    )


def translation_jacobian(q: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    origin = ur3_fk(q)
    jacobian = np.empty((3, 6), dtype=np.float64)
    for index in range(6):
        shifted = q.copy()
        shifted[index] += epsilon
        jacobian[:, index] = (ur3_fk(shifted) - origin) / epsilon
    return jacobian


def constrained_cartesian_step(
    q: np.ndarray,
    cartesian_delta: np.ndarray,
    *,
    max_joint_step_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = translation_jacobian(q)
    orientation_constraints = np.asarray(
        [
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix = np.concatenate([jacobian, 0.35 * orientation_constraints], axis=0)
    target = np.concatenate([cartesian_delta, np.zeros(3, dtype=np.float64)])
    damping = 2e-4
    delta_q = matrix.T @ np.linalg.solve(
        matrix @ matrix.T + damping * np.eye(matrix.shape[0]), target
    )
    delta_q = np.clip(delta_q, -max_joint_step_rad, max_joint_step_rad)
    candidate = np.clip(q + delta_q, ARM_MIN, ARM_MAX)
    actual_delta = ur3_fk(candidate) - ur3_fk(q)
    return candidate, actual_delta


def build_expert_chunk(
    state: np.ndarray,
    peg_position: np.ndarray,
    hole_position: np.ndarray,
    *,
    chunk_size: int,
    xy_step_m: float,
    descent_step_m: float,
    descent_xy_gate_m: float,
    max_joint_step_rad: float,
    descend: bool,
) -> np.ndarray:
    q = np.asarray(state[:6], dtype=np.float64).copy()
    estimated_peg = np.asarray(peg_position, dtype=np.float64).copy()
    hole = np.asarray(hole_position, dtype=np.float64)
    chunk = np.empty((chunk_size, 7), dtype=np.float32)
    for index in range(chunk_size):
        xy_error = hole[:2] - estimated_peg[:2]
        xy_norm = float(np.linalg.norm(xy_error))
        if xy_norm > xy_step_m:
            xy_delta = xy_error * (xy_step_m / xy_norm)
        else:
            xy_delta = xy_error
        z_delta = -descent_step_m if descend and xy_norm <= descent_xy_gate_m else 0.0
        q, achieved_delta = constrained_cartesian_step(
            q,
            np.asarray([xy_delta[0], xy_delta[1], z_delta], dtype=np.float64),
            max_joint_step_rad=max_joint_step_rad,
        )
        estimated_peg += achieved_delta
        chunk[index, :6] = q.astype(np.float32)
        chunk[index, 6] = 1.0
    return chunk


def build_moveit_expert_chunk(
    ik_solver: MoveItRecoveryIK,
    state: np.ndarray,
    peg_position: np.ndarray,
    hole_position: np.ndarray,
    *,
    chunk_size: int,
    chunk_xy_step_m: float,
    chunk_descent_m: float,
    descent_xy_gate_m: float,
    max_joint_step_rad: float,
    descend: bool,
    gripper_command: float,
) -> np.ndarray | None:
    xy_error = np.asarray(hole_position[:2] - peg_position[:2], dtype=np.float64)
    xy_norm = float(np.linalg.norm(xy_error))
    if xy_norm > chunk_xy_step_m:
        xy_error *= chunk_xy_step_m / xy_norm
    z_delta = -chunk_descent_m if descend and xy_norm <= descent_xy_gate_m else 0.0
    world_delta = np.asarray([xy_error[0], xy_error[1], z_delta], dtype=np.float64)
    solution = ik_solver.solve_translation(np.asarray(state[:6], dtype=np.float64), world_delta)
    if solution is None:
        # Retry half a step near reach/collision boundaries.
        solution = ik_solver.solve_translation(
            np.asarray(state[:6], dtype=np.float64), 0.5 * world_delta
        )
    if solution is None:
        return None
    current = np.asarray(state[:6], dtype=np.float64)
    delta = solution - current
    if np.max(np.abs(delta)) > max_joint_step_rad * chunk_size:
        return None
    if np.any(solution < ARM_MIN) or np.any(solution > ARM_MAX):
        return None
    fractions = np.linspace(1.0 / chunk_size, 1.0, chunk_size, dtype=np.float64)
    arm_chunk = current[None, :] + fractions[:, None] * delta[None, :]
    if np.max(np.abs(np.diff(np.vstack([current, arm_chunk]), axis=0))) > max_joint_step_rad + 1e-6:
        return None
    chunk = np.full((chunk_size, 7), float(gripper_command), dtype=np.float32)
    chunk[:, :6] = arm_chunk.astype(np.float32)
    return chunk


def build_moveit_translation_chunk(
    ik_solver: MoveItRecoveryIK,
    state: np.ndarray,
    world_delta: np.ndarray,
    *,
    chunk_size: int,
    max_joint_step_rad: float,
    gripper_command: float,
) -> np.ndarray | None:
    """Interpolate one bounded IK translation with an explicit gripper command."""
    current = np.asarray(state[:6], dtype=np.float64)
    solution = ik_solver.solve_translation(current, np.asarray(world_delta, dtype=np.float64))
    if solution is None:
        solution = ik_solver.solve_translation(current, 0.5 * np.asarray(world_delta))
    if solution is None:
        return None
    delta = solution - current
    if np.max(np.abs(delta)) > max_joint_step_rad * chunk_size:
        return None
    if np.any(solution < ARM_MIN) or np.any(solution > ARM_MAX):
        return None
    fractions = np.linspace(1.0 / chunk_size, 1.0, chunk_size, dtype=np.float64)
    arm_chunk = current[None, :] + fractions[:, None] * delta[None, :]
    if np.max(np.abs(np.diff(np.vstack([current, arm_chunk]), axis=0))) > max_joint_step_rad + 1e-6:
        return None
    chunk = np.full((chunk_size, 7), float(gripper_command), dtype=np.float32)
    chunk[:, :6] = arm_chunk.astype(np.float32)
    return chunk


def build_joint_target_chunk(
    state: np.ndarray,
    target_arm: np.ndarray,
    *,
    chunk_size: int,
    max_joint_step_rad: float,
    gripper_command: float,
) -> np.ndarray | None:
    """Interpolate an explicit bounded recovery posture inside the data envelope."""
    current = np.asarray(state[:6], dtype=np.float64)
    target = np.asarray(target_arm, dtype=np.float64)
    if target.shape != (6,) or not np.isfinite(target).all():
        return None
    if np.any(target < ARM_MIN) or np.any(target > ARM_MAX):
        return None
    delta = target - current
    if np.max(np.abs(delta)) > max_joint_step_rad * chunk_size:
        return None
    fractions = np.linspace(1.0 / chunk_size, 1.0, chunk_size, dtype=np.float64)
    arm_chunk = current[None, :] + fractions[:, None] * delta[None, :]
    chunk = np.full((chunk_size, 7), float(gripper_command), dtype=np.float32)
    chunk[:, :6] = arm_chunk.astype(np.float32)
    return chunk


def _read_state() -> np.ndarray | None:
    try:
        values = np.asarray(
            [float(value) for value in JOINT_STATE_FILE.read_text().split()], dtype=np.float32
        )
    except (OSError, ValueError):
        return None
    return values if values.shape == (7,) and np.isfinite(values).all() else None


def _read_force() -> np.ndarray | None:
    try:
        force = np.asarray(np.load(FORCE_FILE, allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError):
        return None
    return force if force.shape == (6,) and np.isfinite(force).all() else None


def _atomic_outcome(path: Path, outcome: str, note: str) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": time.time(),
        "outcome": outcome,
        "operator": "privileged_recovery_expert_v2",
        "note": note,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _policy_rollin_started(session_dir: Path) -> bool:
    """Return true only after the controller dispatched a real policy chunk."""
    return any((Path(session_dir) / "action_events").glob("event_*.npz"))


class DemonstrationManifoldRecovery:
    """Bridge an off-policy state onto an exact-scene successful demonstration."""

    def __init__(
        self,
        episode_paths: list[Path],
        *,
        session_dir: Path,
        scene_tolerance_m: float,
        bridge_steps: int,
        safe_open_steps: int,
        max_anchor_l2_rad: float,
        gripper_distance_weight: float,
        physical_distance_weight_rad_per_m: float = 5.0,
    ) -> None:
        with (Path(session_dir) / "session.json").open(encoding="utf-8") as stream:
            session = json.load(stream)
        self.peg_spawn_xy = np.asarray(session["peg_spawn_xy"], dtype=np.float64)
        self.hole_spawn_xy = np.asarray(session["hole_spawn_xy"], dtype=np.float64)
        self.bridge_steps = int(bridge_steps)
        self.safe_open_steps = int(safe_open_steps)
        self.max_anchor_l2_rad = float(max_anchor_l2_rad)
        self.gripper_distance_weight = float(gripper_distance_weight)
        self.physical_distance_weight_rad_per_m = float(
            physical_distance_weight_rad_per_m
        )
        self.references: list[dict[str, object]] = []
        self.selected: dict[str, object] | None = None
        self.cursor: int | None = None
        self.dense_actions: np.ndarray | None = None
        self.dense_source_frames: np.ndarray | None = None
        self.dense_cursor = 0
        self.first_chunk = True
        self.open_before_bridge = False
        self.domain_first: int | None = None
        self.domain_last: int | None = None

        for episode_path in episode_paths:
            path = Path(episode_path)
            if not path.is_file():
                raise FileNotFoundError(f"missing grasp reference episode: {path}")
            with np.load(path, allow_pickle=True) as episode:
                required = {
                    "state", "action", "semantic_subtask", "tool0_z",
                    "peg_x", "peg_y", "hole_x", "hole_y",
                }
                missing = sorted(required.difference(episode.files))
                if missing:
                    raise ValueError(f"reference {path} is missing keys: {missing}")
                state = np.asarray(episode["state"], dtype=np.float32)
                action = np.asarray(episode["action"], dtype=np.float32)
                subtask = np.asarray(episode["semantic_subtask"]).astype(str)
                tool0_z = np.asarray(episode["tool0_z"], dtype=np.float32)
                reference_peg = np.asarray(
                    [float(episode["peg_x"]), float(episode["peg_y"])], dtype=np.float64
                )
                reference_hole = np.asarray(
                    [float(episode["hole_x"]), float(episode["hole_y"])], dtype=np.float64
                )
            if state.ndim != 2 or state.shape[1] != 7 or action.shape != state.shape:
                raise ValueError(
                    f"reference {path} must have matching state/action [T,7], "
                    f"got {state.shape}/{action.shape}"
                )
            if len(subtask) != len(state) or not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"reference {path} has invalid semantic or numeric arrays")
            if np.any(action[:, 6] < -1e-4) or np.any(action[:, 6] > 0.8001):
                raise ValueError(f"reference {path} violates continuous 0..0.8 gripper contract")
            # Gazebo feedback around the open stop contains harmless numerical
            # noise near -2e-5 rad. Canonicalize only the physical endpoints;
            # all interior continuous close values remain unchanged.
            action[:, 6] = np.clip(action[:, 6], 0.0, 0.8)
            scene_error = max(
                float(np.linalg.norm(reference_peg - self.peg_spawn_xy)),
                float(np.linalg.norm(reference_hole - self.hole_spawn_xy)),
            )
            if scene_error > scene_tolerance_m:
                raise ValueError(
                    f"reference {path} is not the same scene: error={scene_error:.6f}m "
                    f"> tolerance={scene_tolerance_m:.6f}m"
                )
            grasp_indices = np.flatnonzero(subtask == "grasp the peg")
            if grasp_indices.size < 2 or not np.array_equal(
                grasp_indices, np.arange(grasp_indices[0], grasp_indices[-1] + 1)
            ):
                raise ValueError(f"reference {path} has no contiguous grasp subtask")
            closing = np.flatnonzero(action[grasp_indices, 6] > 0.02)
            if closing.size == 0:
                raise ValueError(f"reference {path} has no continuous close trajectory")
            first_close = int(grasp_indices[int(closing[0])])
            preclose = max(int(grasp_indices[0]), first_close - 1)
            approach_domain = grasp_indices[grasp_indices < first_close]
            approach_entry = int(approach_domain[np.argmax(tool0_z[approach_domain])])
            transport_indices = np.flatnonzero(subtask == "transport to the hole")
            if transport_indices.size < 2 or not np.array_equal(
                transport_indices,
                np.arange(transport_indices[0], transport_indices[-1] + 1),
            ):
                raise ValueError(f"reference {path} has no contiguous transport subtask")
            align_indices = np.flatnonzero(
                subtask == "approach and align with the hole"
            )
            if align_indices.size < 2:
                raise ValueError(f"reference {path} has no alignment subtask")
            # Derive a high-clearance lateral-alignment manifold from the
            # demonstration itself.  Do not drag a held peg sideways after
            # the demonstrated descent toward the zero-clearance socket has
            # already started.
            align_max_tool_z = float(np.max(tool0_z[align_indices]))
            safe_align_candidates = align_indices[
                tool0_z[align_indices] >= align_max_tool_z - 0.001
            ]
            if safe_align_candidates.size == 0:
                raise ValueError(f"reference {path} has no safe alignment plateau")
            insert_indices = np.flatnonzero(
                subtask == "insert the peg into the hole"
            )
            if insert_indices.size < 2:
                raise ValueError(f"reference {path} has no insertion subtask")
            # Object poses were not recorded per policy frame in v8.  Build the
            # task-manifold quantity needed by the privileged collector from
            # the exact-scene fixture and the demonstrated semantic progress:
            # the peg stays at its spawn during grasp, moves continuously to
            # the socket during transport, and is concentric during
            # align/insert.  This is never exposed to the learned policy.
            expected_peg_hole_xy = np.zeros((len(state), 2), dtype=np.float32)
            initial_peg_hole_xy = (reference_peg - reference_hole).astype(np.float32)
            expected_peg_hole_xy[: int(transport_indices[0])] = initial_peg_hole_xy
            transport_alpha = np.linspace(
                0.0, 1.0, len(transport_indices), dtype=np.float32
            )
            expected_peg_hole_xy[transport_indices] = (
                (1.0 - transport_alpha)[:, None] * initial_peg_hole_xy[None, :]
            )
            self.references.append(
                {
                    "path": path,
                    "state": state,
                    "action": action,
                    "subtask": subtask,
                    "first": int(grasp_indices[0]),
                    "last": int(grasp_indices[-1]),
                    "candidates": grasp_indices[:-1],
                    "preclose": preclose,
                    "approach_entry": approach_entry,
                    "transport_candidates": transport_indices[:-1],
                    "transport_first": int(transport_indices[0]),
                    "transport_last": int(transport_indices[-1]),
                    "safe_align_candidates": safe_align_candidates,
                    "safe_align_first": int(safe_align_candidates[0]),
                    "safe_align_last": int(safe_align_candidates[-1]),
                    "insert_first": int(insert_indices[0]),
                    "expected_peg_hole_xy": expected_peg_hole_xy,
                    # Exclude release: recovery must teach task execution, not
                    # let the expert finish by dropping an already seated peg.
                    "full_candidates": np.arange(
                        0,
                        int(np.flatnonzero(subtask == "verify insertion success")[-1]) + 1,
                        dtype=np.int64,
                    ),
                }
            )
        if not self.references:
            raise ValueError("at least one exact-scene grasp reference episode is required")

    def select_anchor(
        self,
        measured_state: np.ndarray,
        *,
        mode: str = "nearest",
        min_anchor_frame: int | None = None,
        measured_peg_hole_xy: np.ndarray | None = None,
    ) -> tuple[Path, int, float]:
        measured = np.asarray(measured_state, dtype=np.float32)
        if measured.shape != (7,) or not np.isfinite(measured).all():
            raise ValueError("measured recovery state must be finite [7]")
        best: tuple[float, float, dict[str, object], int] | None = None
        for reference in self.references:
            state = reference["state"]
            candidates = (
                reference["full_candidates"]
                if mode == "full"
                else
                reference["safe_align_candidates"]
                if mode == "safe_align"
                else
                np.asarray([reference["insert_first"]], dtype=np.int64)
                if mode == "insert_entry"
                else
                reference["transport_candidates"]
                if mode == "transport"
                else
                np.asarray([reference["approach_entry"]], dtype=np.int64)
                if mode == "approach_entry"
                else np.asarray([reference["preclose"]], dtype=np.int64)
                if mode == "preclose"
                else reference["candidates"]
            )
            candidates = np.asarray(candidates, dtype=np.int64)
            if min_anchor_frame is not None:
                candidates = candidates[candidates >= int(min_anchor_frame)]
                if candidates.size == 0:
                    continue
            assert isinstance(state, np.ndarray) and isinstance(candidates, np.ndarray)
            arm_delta = state[candidates, :6] - measured[None, :6]
            arm_l2 = np.linalg.norm(arm_delta, axis=1)
            gripper_delta = state[candidates, 6] - measured[6]
            score_squared = (
                np.square(arm_l2) + self.gripper_distance_weight * np.square(gripper_delta)
            )
            if measured_peg_hole_xy is not None:
                measured_relative = np.asarray(measured_peg_hole_xy, dtype=np.float32)
                if measured_relative.shape != (2,) or not np.isfinite(measured_relative).all():
                    raise ValueError("measured peg-hole relative XY must be finite [2]")
                expected_relative = np.asarray(
                    reference["expected_peg_hole_xy"], dtype=np.float32
                )[candidates]
                physical_l2_m = np.linalg.norm(
                    expected_relative - measured_relative[None, :], axis=1
                )
                score_squared += np.square(
                    self.physical_distance_weight_rad_per_m * physical_l2_m
                )
            score = np.sqrt(score_squared)
            local = int(np.argmin(score))
            candidate = (
                float(score[local]), float(arm_l2[local]), reference, int(candidates[local])
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            raise ValueError(
                f"no demonstration anchor remains at/after frame {min_anchor_frame}"
            )
        total_distance, arm_l2, self.selected, self.cursor = best
        self.dense_actions = None
        self.dense_source_frames = None
        self.dense_cursor = 0
        self.first_chunk = True
        self.open_before_bridge = mode == "approach_entry"
        if mode == "full":
            self.domain_first = int(np.asarray(self.selected["full_candidates"])[0])
            self.domain_last = int(np.asarray(self.selected["full_candidates"])[-1])
        elif mode == "safe_align":
            self.domain_first = int(self.selected["safe_align_first"])
            self.domain_last = int(self.selected["safe_align_last"])
        elif mode == "insert_entry":
            self.domain_first = int(self.selected["insert_first"])
            self.domain_last = int(np.asarray(self.selected["full_candidates"])[-1])
        elif mode == "transport":
            self.domain_first = int(self.selected["transport_first"])
            self.domain_last = int(self.selected["transport_last"])
        else:
            self.domain_first = int(self.selected["first"])
            self.domain_last = int(self.selected["last"])
        if arm_l2 > self.max_anchor_l2_rad:
            raise ValueError(
                f"nearest successful grasp anchor is too far: arm_l2={arm_l2:.6f}rad "
                f"> limit={self.max_anchor_l2_rad:.6f}rad"
            )
        path = self.selected["path"]
        assert isinstance(path, Path) and self.cursor is not None
        return path, self.cursor, total_distance

    def _densify_suffix(self, max_joint_step_rad: float) -> None:
        assert self.selected is not None and self.cursor is not None
        actions = self.selected["action"]
        last = self.domain_last
        assert isinstance(actions, np.ndarray) and isinstance(last, int)
        start = self.cursor
        raw = actions[start : last + 1].astype(np.float64)
        dense = [raw[0].copy()]
        source_frames = [float(start)]
        for index in range(1, len(raw)):
            previous = raw[index - 1]
            target = raw[index]
            subdivisions = max(
                1,
                int(np.ceil(np.max(np.abs(target[:6] - previous[:6])) / max_joint_step_rad)),
            )
            for subdivision in range(1, subdivisions + 1):
                alpha = subdivision / subdivisions
                dense.append((1.0 - alpha) * previous + alpha * target)
                source_frames.append(start + index - 1 + alpha)
        self.dense_actions = np.asarray(dense, dtype=np.float64)
        self.dense_source_frames = np.asarray(source_frames, dtype=np.float64)
        self.dense_cursor = 0

    def build_chunk(
        self,
        measured_state: np.ndarray,
        *,
        chunk_size: int,
        max_joint_step_rad: float,
    ) -> tuple[np.ndarray, int, int]:
        if self.selected is None or self.cursor is None:
            self.select_anchor(measured_state)
        assert self.selected is not None and self.cursor is not None
        last = self.domain_last
        assert isinstance(last, int)
        if self.dense_actions is None or self.dense_source_frames is None:
            self._densify_suffix(max_joint_step_rad)
        assert self.dense_actions is not None and self.dense_source_frames is not None
        if self.dense_cursor >= len(self.dense_actions):
            raise ValueError("successful grasp demonstration suffix is exhausted")
        measured = np.asarray(measured_state, dtype=np.float64)
        reference_start = int(np.floor(self.dense_source_frames[self.dense_cursor]))
        target = self.dense_actions[self.dense_cursor]
        required_bridge = max(
            1,
            int(np.ceil(np.max(np.abs(target[:6] - measured[:6])) / max_joint_step_rad)),
        )
        bridge = max(self.bridge_steps if self.first_chunk else 1, required_bridge)
        opening_steps = self.safe_open_steps if self.first_chunk and self.open_before_bridge else 0
        if opening_steps + bridge > chunk_size:
            raise ValueError(
                "demonstration anchor cannot be reached within one bounded expert chunk"
            )
        opened = measured.copy()
        opened[6] = 0.0
        if opening_steps:
            open_alpha = np.arange(1, opening_steps + 1, dtype=np.float64) / opening_steps
            opening = np.repeat(measured[None, :], opening_steps, axis=0)
            opening[:, 6] = (1.0 - open_alpha) * measured[6]
        else:
            opening = np.empty((0, 7), dtype=np.float64)
            opened = measured
        alpha = np.arange(1, bridge + 1, dtype=np.float64) / bridge
        bridge_chunk = (
            (1.0 - alpha[:, None]) * opened[None, :]
            + alpha[:, None] * target[None, :]
        )
        correction = np.concatenate([opening, bridge_chunk], axis=0)
        available = chunk_size - len(correction)
        suffix_start = self.dense_cursor + 1
        suffix_end = min(suffix_start + available, len(self.dense_actions))
        suffix = self.dense_actions[suffix_start:suffix_end]
        chunk = np.concatenate([correction, suffix], axis=0)
        self.dense_cursor = suffix_end
        if len(chunk) < chunk_size:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1][None, :], chunk_size - len(chunk), axis=0)], axis=0
            )
        arm_steps = np.diff(np.vstack([measured[None, :6], chunk[:, :6]]), axis=0)
        if float(np.max(np.abs(arm_steps))) > max_joint_step_rad + 1e-6:
            raise ValueError("internal error: densified demonstration violates joint-step limit")
        self.cursor = int(np.floor(self.dense_source_frames[self.dense_cursor - 1]))
        self.first_chunk = False
        return chunk.astype(np.float32), reference_start, len(correction)

    @property
    def progress(self) -> float:
        if self.selected is None or self.cursor is None:
            return 0.0
        assert self.domain_first is not None and self.domain_last is not None
        first = self.domain_first
        last = self.domain_last
        return float(np.clip((self.cursor - first) / max(last - first, 1), 0.0, 1.0))

    def expected_peg_hole_xy(self, frame: int) -> np.ndarray:
        """Return privileged physical-manifold XY for the selected frame."""
        if self.selected is None:
            raise ValueError("select a demonstration anchor before querying physical state")
        expected = np.asarray(self.selected["expected_peg_hole_xy"], dtype=np.float32)
        if not 0 <= int(frame) < len(expected):
            raise ValueError(f"reference frame {frame} is outside the selected episode")
        return expected[int(frame)].copy()

    def build_rejoin_chunk(
        self,
        measured_state: np.ndarray,
        *,
        chunk_size: int,
        max_joint_step_rad: float,
        lookahead_frames: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Move only to a nearby demonstration anchor, never replay its suffix."""
        if self.selected is None or self.cursor is None or self.domain_last is None:
            raise ValueError("select a demonstration anchor before building a rejoin chunk")
        states = self.selected["state"]
        assert isinstance(states, np.ndarray)
        target_frame = min(self.cursor + int(lookahead_frames), self.domain_last)
        # Rejoin a demonstrated *measured state*.  The action gripper endpoint
        # is the universal 0.8-rad close command, while a healthy physical
        # grasp settles near 0.627 rad.  Using action[target_frame] here makes
        # a valid held-object state mathematically unreachable and prevents
        # handoff back to the policy.
        target = np.asarray(states[target_frame], dtype=np.float64)
        measured = np.asarray(measured_state, dtype=np.float64)
        delta = target - measured
        arm_scale = min(
            1.0,
            max_joint_step_rad * chunk_size / max(float(np.max(np.abs(delta[:6]))), 1e-9),
        )
        bounded_target = target.copy()
        bounded_target[:6] = measured[:6] + arm_scale * delta[:6]
        alpha = np.arange(1, chunk_size + 1, dtype=np.float64) / chunk_size
        chunk = measured[None, :] + alpha[:, None] * (bounded_target - measured)[None, :]
        # ``target`` is a measured demonstration state.  With an object in
        # the fingers that state settles below the controller endpoint (about
        # 0.627 rad for the current peg), but the object-independent action
        # contract is still a full-close 0.8-rad command.  Never leak the
        # contact-limited measurement into an expert action label.
        if float(measured[6]) >= 0.35 and float(target[6]) >= 0.35:
            chunk[:, 6] = 0.8
        if float(np.max(np.abs(np.diff(np.vstack([measured, chunk])[:, :6], axis=0)))) > (
            max_joint_step_rad + 1e-6
        ):
            raise ValueError("rejoin chunk violates the joint-step limit")
        return chunk.astype(np.float32), target_frame, target.astype(np.float32)

    def project_local_expected_state(
        self,
        measured_state: np.ndarray,
        *,
        expected_frame: int,
        radius_frames: int,
    ) -> tuple[int, float]:
        """Align only inside a small causal window around expected progress."""
        if self.selected is None or self.domain_first is None or self.domain_last is None:
            raise ValueError("a demonstration trajectory must be locked before local alignment")
        state = self.selected["state"]
        assert isinstance(state, np.ndarray)
        start = max(self.domain_first, int(expected_frame) - int(radius_frames))
        stop = min(self.domain_last, int(expected_frame) + int(radius_frames))
        candidates = np.arange(start, stop + 1, dtype=np.int64)
        measured = np.asarray(measured_state, dtype=np.float32)
        arm_l2 = np.linalg.norm(state[candidates, :6] - measured[None, :6], axis=1)
        gripper_delta = state[candidates, 6] - measured[6]
        score = np.sqrt(
            np.square(arm_l2) + self.gripper_distance_weight * np.square(gripper_delta)
        )
        local = int(np.argmin(score))
        self.cursor = int(candidates[local])
        return self.cursor, float(score[local])


def _completed_policy_event_count(session_dir: Path) -> int:
    count = 0
    for path in (Path(session_dir) / "completion_events").glob("completion_*.json"):
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            count += int(payload.get("mode", EXPERT_MODE)) == POLICY_MODE
        except (OSError, ValueError, TypeError):
            continue
    return count


def _completed_policy_step_count(session_dir: Path) -> int:
    """Count action points that the controller actually completed in policy mode."""
    total = 0
    session_dir = Path(session_dir)
    for completion_path in (session_dir / "completion_events").glob("completion_*.json"):
        try:
            with completion_path.open(encoding="utf-8") as stream:
                completion = json.load(stream)
            if int(completion.get("mode", EXPERT_MODE)) != POLICY_MODE:
                continue
            event_sequence = int(completion["action_event_sequence"])
            event_path = session_dir / "action_events" / f"event_{event_sequence:08d}.npz"
            with np.load(event_path, allow_pickle=False) as event:
                total += len(np.asarray(event["executed_action"]))
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return total


def _latest_completed_policy_chunk(session_dir: Path) -> np.ndarray | None:
    """Return the newest controller-completed policy trajectory segment."""
    session_dir = Path(session_dir)
    newest: tuple[int, np.ndarray] | None = None
    for completion_path in (session_dir / "completion_events").glob("completion_*.json"):
        try:
            with completion_path.open(encoding="utf-8") as stream:
                completion = json.load(stream)
            if int(completion.get("mode", EXPERT_MODE)) != POLICY_MODE:
                continue
            sequence = int(completion["action_event_sequence"])
            event_path = session_dir / "action_events" / f"event_{sequence:08d}.npz"
            with np.load(event_path, allow_pickle=False) as event:
                chunk = np.asarray(event["executed_action"], dtype=np.float32).copy()
            if newest is None or sequence > newest[0]:
                newest = (sequence, chunk)
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return None if newest is None else newest[1]


def run_demonstration_multi_handoff_recovery(
    args, ik_solver: MoveItRecoveryIK
) -> int:
    """Alternate rollout and short demonstration-manifold corrections to task end."""
    tracker = DemonstrationManifoldRecovery(
        args.grasp_reference_episode,
        session_dir=args.session_dir,
        scene_tolerance_m=args.grasp_reference_scene_tolerance_m,
        bridge_steps=args.grasp_reference_bridge_steps,
        safe_open_steps=args.grasp_reference_safe_open_steps,
        max_anchor_l2_rad=args.grasp_reference_max_anchor_l2_rad,
        gripper_distance_weight=args.grasp_reference_gripper_weight,
        physical_distance_weight_rad_per_m=args.rejoin_physical_weight_rad_per_m,
    )
    deadline = time.monotonic() + args.timeout_s
    last_state_mtime = None
    expert_sequence = 0
    control_sequence = 0
    takeover_active = False
    target_state = None
    target_frame = None
    target_mode = "full"
    physical_correction_active = False
    progress_floor = 0
    policy_count_at_release = 0
    interventions = 0
    initial_peg = None
    insertion_best_peg_z = None
    insertion_last_progress_time = None
    # Optional late-takeover contract for contact-recovery collection.  The
    # policy must first demonstrate that it can reach the near-concentric
    # region by itself.  Only a subsequent loss of alignment is corrected by
    # the expert.  This avoids recording another long, generic align suffix
    # when the actual learned-policy failure is rim contact followed by drift.
    policy_alignment_reached = False
    policy_alignment_best_peg_z = None
    policy_alignment_last_descent_time = None
    # Track transport progress after every policy handoff.  Previously the
    # full-task collector deliberately waited for the policy to reach the
    # near-hole band before allowing another intervention.  That made a
    # policy which recovered grasp but then hovered over the peg wait until
    # the whole episode timed out, so the resulting trajectory was not the
    # requested policy/expert alternating recovery episode.
    policy_transport_best_xy = None
    policy_transport_last_progress_time = None
    print(
        "Full-episode multi-handoff collection active: policy rollout -> short "
        "demonstration rejoin -> policy resume.",
        flush=True,
    )
    while time.monotonic() < deadline:
        if not JOINT_STATE_FILE.exists() or not (args.session_dir / "session.json").exists():
            time.sleep(0.02)
            continue
        state_mtime = JOINT_STATE_FILE.stat().st_mtime_ns
        if state_mtime == last_state_mtime:
            time.sleep(0.005)
            continue
        last_state_mtime = state_mtime
        state = _read_state()
        force = _read_force()
        poses = read_gazebo_fixture_poses()
        if state is None or poses is None:
            continue
        peg, hole = poses
        if initial_peg is None:
            initial_peg = np.asarray(peg, dtype=np.float64).copy()
        peg_hole_xy = float(np.linalg.norm(peg[:2] - hole[:2]))
        policy_count = _completed_policy_event_count(args.session_dir)

        # A peg below its spawned support height by more than the configured
        # recovery margin has slipped out of the gripper / through the table.
        # Joint-space demonstration replay cannot recover that object state;
        # continuing would only record a long, invalid expert suffix.
        peg_drop_m = float(initial_peg[2] - peg[2])
        if peg_drop_m > args.grasp_max_recoverable_drop_m:
            note = (
                "unrecoverable peg drop during multi-handoff recovery: "
                f"drop={peg_drop_m:.6f}m peg_z={peg[2]:.6f}m"
            )
            _atomic_outcome(args.outcome_file, "failure", note)
            print(note, flush=True)
            return 1

        # A non-terminal correction must be followed by a real policy suffix.
        # A terminal insertion correction is handled below while expert mode
        # is active, avoiding unsafe motion after the peg is already seated.
        if (
            not takeover_active
            and interventions > 0
            and policy_count - policy_count_at_release >= args.rejoin_policy_suffix_events
            and peg_hole_xy <= args.success_xy_m
            and args.success_peg_z_m is not None
            and peg[2] <= args.success_peg_z_m
        ):
            note = (
                f"multi-handoff full task success: interventions={interventions}, "
                f"xy={peg_hole_xy:.6f}m peg_z={peg[2]:.6f}m"
            )
            _atomic_outcome(args.outcome_file, "success", note)
            print(note, flush=True)
            return 0

        if takeover_active:
            assert target_state is not None and target_frame is not None
            arm_l2 = float(np.linalg.norm(state[:6] - target_state[:6]))
            gripper_error = float(abs(state[6] - target_state[6]))
            target_subtask = str(np.asarray(tracker.selected["subtask"])[target_frame])
            target_requires_insertion = target_subtask in {
                "insert the peg into the hole",
                "verify insertion success",
            }
            target_requires_concentric_alignment = target_subtask in {
                "approach and align with the hole",
                "insert the peg into the hole",
                "verify insertion success",
            }
            # The demonstration supplies a feasible arm posture and height,
            # but its residual fixture-pose error is not a valid contact
            # target for the exact-fit socket. Alignment/insertion must be
            # concentric in measured Gazebo coordinates.
            expected_relative_xy = (
                np.zeros(2, dtype=np.float64)
                if target_requires_concentric_alignment
                else tracker.expected_peg_hole_xy(target_frame)
            )
            physical_error_m = float(
                np.linalg.norm((peg[:2] - hole[:2]) - expected_relative_xy)
            )
            physical_release_m = (
                min(args.rejoin_release_physical_m, args.descent_xy_gate_m)
                if target_requires_concentric_alignment
                else args.rejoin_release_physical_m
            )
            terminal_peg_z_m = (
                None
                if args.success_peg_z_m is None
                # Leave an acquisition margin below the validation boundary.
                # The asynchronous recorder can observe roughly 0.6 mm of
                # settling rebound after the expert's terminal sample.  Keep
                # the schema threshold strict and command a deeper terminal
                # target instead of accepting a geometrically shallow sample.
                # The terminal block is still capped to the remaining distance,
                # so this does not reintroduce the old fixed 4 mm over-drive.
                else args.success_peg_z_m - 0.0013
            )
            insertion_geometry_valid = (
                peg_hole_xy <= args.success_xy_m
                and args.success_peg_z_m is not None
                and peg[2] <= args.success_peg_z_m
            )
            if target_mode == "insert_entry" and not insertion_geometry_valid:
                now = time.monotonic()
                if (
                    insertion_best_peg_z is None
                    or float(peg[2])
                    <= insertion_best_peg_z - args.insertion_min_progress_m
                ):
                    insertion_best_peg_z = float(peg[2])
                    insertion_last_progress_time = now
                elif (
                    insertion_last_progress_time is not None
                    and now - insertion_last_progress_time
                    >= args.insertion_stall_s
                ):
                    note = (
                        "insertion recovery stalled without measured peg descent: "
                        f"best_z={insertion_best_peg_z:.6f}m "
                        f"current_z={peg[2]:.6f}m "
                        f"for={now - insertion_last_progress_time:.1f}s"
                    )
                    _atomic_outcome(args.outcome_file, "failure", note)
                    print(note, flush=True)
                    return 1
            target_requires_grasp = float(target_state[6]) >= 0.35
            target_physics_valid = (
                not target_requires_grasp
                or (
                    float(state[6]) >= 0.35
                    and (
                        peg[2] - initial_peg[2] >= 0.03
                        or insertion_geometry_valid
                    )
                )
            )
            # Cartesian correction can legitimately differ from a scripted
            # absolute-joint anchor during hole alignment, where the same
            # intervention continues through terminal insertion.  A grasp
            # recovery, however, hands control back to the learned policy for
            # transport.  It must therefore re-enter the demonstrated arm
            # manifold; checking joint limits alone caused premature handoff
            # at arm_l2=0.15 rad and an immediate policy release/drop.
            arm_manifold_valid = (
                arm_l2 <= args.rejoin_release_l2_rad
                if target_mode == "grasp_recovery"
                else bool(np.all(state[:6] >= ARM_MIN) and np.all(state[:6] <= ARM_MAX))
                if physical_correction_active
                else arm_l2 <= args.rejoin_release_l2_rad
            )
            rejoin_ready = (
                arm_manifold_valid
                and gripper_error <= args.rejoin_release_gripper_rad
                and physical_error_m <= physical_release_m
                and (not target_requires_insertion or insertion_geometry_valid)
                and target_physics_valid
                and (
                    target_mode != "grasp_recovery"
                    or (
                        float(state[6]) >= 0.35
                        and float(peg[2] - initial_peg[2]) >= 0.03
                    )
                )
            )
            if rejoin_ready and target_mode == "safe_align":
                # Alignment alone is not a stable handoff for the present
                # policy: it immediately drifts before initiating descent.
                # Continue the same bounded expert intervention only through
                # the missing align-to-insert capability. Seating is a
                # verified terminal state, so no policy motion follows it.
                _, target_frame, _ = tracker.select_anchor(
                    state,
                    mode="insert_entry",
                    measured_peg_hole_xy=np.asarray(peg[:2] - hole[:2]),
                )
                assert tracker.selected is not None
                # Alignment is already physically correct in the current
                # grasp frame. Replaying the demonstration's absolute joint
                # state here can convert grasp/calibration residuals into a
                # large XY jump before the IK correction starts. Preserve the
                # measured state and use the selected frame only for insertion
                # semantics / terminal progress; the next branch performs a
                # concentric Cartesian descent from the current pose.
                target_state = np.asarray(state, dtype=np.float64).copy()
                target_mode = "insert_entry"
                physical_correction_active = False
                insertion_best_peg_z = float(peg[2])
                insertion_last_progress_time = time.monotonic()
                print(
                    "Safe alignment restored; extending the same intervention "
                    f"to demonstrated insertion entry frame {target_frame}.",
                    flush=True,
                )
                continue
            if rejoin_ready:
                if target_requires_insertion and insertion_geometry_valid:
                    note = (
                        "terminal expert recovery success: "
                        f"interventions={interventions}, reference_frame={target_frame}, "
                        f"xy={peg_hole_xy:.6f}m peg_z={peg[2]:.6f}m; "
                        "task ended without unsafe post-seating policy motion"
                    )
                    _atomic_outcome(args.outcome_file, "success", note)
                    print(note, flush=True)
                    return 0
                release_to_policy(
                    args.session_dir,
                    reason=(
                        f"demonstration manifold restored at frame {target_frame}: "
                        f"arm_l2={arm_l2:.5f} gripper_error={gripper_error:.5f} "
                        f"physical_xy={physical_error_m:.5f}m"
                    ),
                    requester="demonstration_rejoin_expert_v1",
                    sequence=control_sequence,
                )
                control_sequence += 1
                takeover_active = False
                physical_correction_active = False
                if target_mode == "safe_align":
                    # Reusing the same high-clearance alignment state is
                    # monotonic (never earlier) and remains physically valid
                    # if the resumed policy drifts again before insertion.
                    progress_floor = max(progress_floor, int(target_frame))
                else:
                    progress_floor = max(
                        progress_floor,
                        min(
                            int(target_frame) + args.rejoin_min_progress_frames,
                            int(tracker.domain_last),
                        ),
                    )
                policy_count_at_release = policy_count
                policy_transport_best_xy = None
                policy_transport_last_progress_time = None
                print(
                    f"Released to policy after intervention {interventions}; "
                    f"reference_frame={target_frame} arm_l2={arm_l2:.4f}",
                    flush=True,
                )
                continue
        else:
            # Contact is an urgent physical event, not an ordinary manifold
            # mismatch.  Arm and evaluate it before the generic minimum-policy-
            # events debounce; otherwise a zero-clearance rim contact can remain
            # under policy control long enough to create an unsafe overload.
            likely_carried_peg = (
                float(state[6]) >= 0.35
                and float(peg[2] - initial_peg[2]) >= 0.03
            )
            now = time.monotonic()
            if (
                likely_carried_peg
                and args.rejoin_policy_alignment_entry_m is not None
                and peg_hole_xy > args.rejoin_policy_alignment_entry_m
            ):
                if policy_transport_best_xy is None:
                    policy_transport_best_xy = peg_hole_xy
                    policy_transport_last_progress_time = now
                elif (
                    peg_hole_xy
                    <= policy_transport_best_xy - args.transport_trigger_min_progress_m
                ):
                    policy_transport_best_xy = peg_hole_xy
                    policy_transport_last_progress_time = now
            policy_transport_stall_trigger = (
                likely_carried_peg
                and args.rejoin_policy_alignment_entry_m is not None
                and peg_hole_xy > args.rejoin_policy_alignment_entry_m
                and policy_transport_last_progress_time is not None
                and now - policy_transport_last_progress_time
                >= args.transport_trigger_stall_s
            )
            if (
                args.rejoin_policy_alignment_entry_m is not None
                and likely_carried_peg
                and peg_hole_xy <= args.rejoin_policy_alignment_entry_m
                and not policy_alignment_reached
            ):
                policy_alignment_reached = True
                policy_alignment_best_peg_z = float(peg[2])
                policy_alignment_last_descent_time = time.monotonic()
                print(
                    "Policy reached the near-concentric region; late contact-"
                    f"recovery trigger armed at xy={peg_hole_xy:.6f}m.",
                    flush=True,
                )
            policy_contact_force_n = (
                None
                if force is None
                else float(np.linalg.norm(np.asarray(force, dtype=np.float64)[:3]))
            )
            if (
                policy_alignment_reached
                and policy_alignment_best_peg_z is not None
                and float(peg[2])
                <= policy_alignment_best_peg_z
                - args.rejoin_policy_alignment_min_descent_m
            ):
                policy_alignment_best_peg_z = float(peg[2])
                policy_alignment_last_descent_time = time.monotonic()
            policy_alignment_stall_trigger = (
                policy_alignment_reached
                and args.rejoin_policy_alignment_stall_s is not None
                and policy_alignment_last_descent_time is not None
                and time.monotonic() - policy_alignment_last_descent_time
                >= args.rejoin_policy_alignment_stall_s
            )
            urgent_policy_contact = (
                policy_alignment_reached
                and args.rejoin_policy_contact_force_n is not None
                and policy_contact_force_n is not None
                and policy_contact_force_n >= args.rejoin_policy_contact_force_n
            )
            if (
                not urgent_policy_contact
                and not policy_alignment_stall_trigger
                and not policy_transport_stall_trigger
                and policy_count
                < max(2, policy_count_at_release + args.rejoin_min_policy_events)
            ):
                continue
            peg_lift_m = float(peg[2] - initial_peg[2])
            unfinished_grasp = interventions == 0 and (
                float(state[6]) < 0.35
                or peg_lift_m < 0.03
            )
            # Intervene while a failed close is still recoverable.  Waiting
            # only for a long no-lift timeout lets a correctly predicted close
            # ramp hit the peg from an offset/low wrist pose and knock it out
            # of the exact-scene demonstration domain.  This guard uses only
            # privileged collector geometry and the already-configured generic
            # grasp corridor; it is never exposed to the learned policy.
            pre_drop_grasp_risk = False
            pre_drop_grasp_error = None
            if (
                interventions == 0
                and float(state[6]) >= args.grasp_trigger_closed_rad
                and peg_lift_m <= args.grasp_trigger_max_lift_m
            ):
                gripper = ik_solver.frame_midpoint(
                    (
                        "robotiq_85_left_finger_tip_link",
                        "robotiq_85_right_finger_tip_link",
                    )
                )
                if gripper is not None:
                    gripper_peg_xy = float(
                        np.linalg.norm(np.asarray(gripper[:2]) - peg[:2])
                    )
                    gripper_peg_z = float(gripper[2] - peg[2])
                    z_error = abs(
                        gripper_peg_z - args.grasp_center_z_offset_m
                    )
                    pre_drop_grasp_risk = (
                        gripper_peg_xy > args.grasp_xy_gate_m
                        or z_error > args.grasp_z_gate_m
                    )
                    pre_drop_grasp_error = (
                        gripper_peg_xy,
                        gripper_peg_z,
                        z_error,
                    )
            # A policy can undo a valid expert grasp immediately after control
            # is returned: the fingers remain closed and centered over the peg,
            # but the peg is lowered back onto the table before transport.  The
            # old detector only handled the very first grasp attempt
            # (``interventions == 0``), so this state could run until the episode
            # timeout because it satisfies neither the carried-peg transport
            # trigger nor the open-gripper failure trigger.  Treat it as another
            # grasp recovery once the policy has received a real post-handoff
            # action window.  Keep the far-from-hole guard so a normally seated
            # peg at the end of insertion is never interpreted as a lost grasp.
            post_handoff_grasp_lost = False
            if (
                interventions > 0
                and float(state[6]) >= 0.35
                and peg_lift_m < 0.03
                and peg_hole_xy >= args.transport_trigger_min_hole_distance_m
                and policy_count
                >= policy_count_at_release + args.rejoin_min_policy_events
            ):
                gripper = ik_solver.frame_midpoint(
                    (
                        "robotiq_85_left_finger_tip_link",
                        "robotiq_85_right_finger_tip_link",
                    )
                )
                if gripper is not None:
                    gripper_peg_xy = float(
                        np.linalg.norm(np.asarray(gripper[:2]) - peg[:2])
                    )
                    post_handoff_grasp_lost = (
                        gripper_peg_xy <= args.grasp_trigger_xy_m
                    )
            grasp_stalled = (
                (
                    unfinished_grasp
                    and policy_count >= args.rejoin_grasp_stall_policy_events
                )
                or pre_drop_grasp_risk
                or post_handoff_grasp_lost
            )
            # Give the policy a real opportunity to finish its learned grasp.
            # If it is still open/unlifted after a long policy roll-in, recover
            # through the demonstrated grasp suffix instead of timing out.
            if unfinished_grasp and not grasp_stalled:
                continue
            try:
                if grasp_stalled:
                    _, anchor, nearest_distance = tracker.select_anchor(
                        state,
                        mode="approach_entry",
                        measured_peg_hole_xy=np.asarray(peg[:2] - hole[:2]),
                    )
                    target_mode = "grasp_recovery"
                elif (
                    interventions >= 3
                    and likely_carried_peg
                    and peg_hole_xy > args.transport_success_xy_m
                ):
                    # The policy has already been given several genuine
                    # re-entry opportunities and repeatedly returned to a
                    # far-from-hole transport state.  This includes both an
                    # explicit no-progress timeout and fast deviations that
                    # cross the joint-space recovery threshold before the
                    # timeout.  Escalate this intervention
                    # to the safe alignment entry so the recorded episode has
                    # a successful terminal suffix instead of dozens of
                    # nearly identical oscillations.  The earlier short
                    # handoffs remain in the episode as local corrections.
                    _, anchor, nearest_distance = tracker.select_anchor(
                        state,
                        mode="safe_align",
                        min_anchor_frame=progress_floor,
                        measured_peg_hole_xy=np.asarray(peg[:2] - hole[:2]),
                    )
                    target_mode = "safe_align"
                else:
                    _, anchor, nearest_distance = tracker.select_anchor(
                        state,
                        mode="full",
                        min_anchor_frame=progress_floor,
                        measured_peg_hole_xy=np.asarray(peg[:2] - hole[:2]),
                    )
                    assert tracker.selected is not None
                    selected_subtask = str(
                        np.asarray(tracker.selected["subtask"])[anchor]
                    )
                    target_mode = "full"
                    if (
                        selected_subtask
                        in {
                            "approach and align with the hole",
                            "insert the peg into the hole",
                            "verify insertion success",
                        }
                        and peg_hole_xy > args.descent_xy_gate_m
                    ):
                        _, anchor, nearest_distance = tracker.select_anchor(
                            state,
                            mode="safe_align",
                            min_anchor_frame=progress_floor,
                            measured_peg_hole_xy=np.asarray(peg[:2] - hole[:2]),
                        )
                        target_mode = "safe_align"
            except ValueError:
                continue
            assert tracker.selected is not None
            subtask = np.asarray(tracker.selected["subtask"]).astype(str)
            expected_subtask = str(subtask[anchor])
            expects_carried_peg = expected_subtask in {
                "transport to the hole",
                "approach and align with the hole",
                "insert the peg into the hole",
                "verify insertion success",
            }
            physical_mismatch = expects_carried_peg and (
                float(state[6]) < 0.35 or float(peg[2] - initial_peg[2]) < 0.03
            )
            alignment_mismatch = (
                expected_subtask
                in {
                    "approach and align with the hole",
                    "insert the peg into the hole",
                    "verify insertion success",
                }
                and peg_hole_xy > args.rejoin_release_physical_m
            )
            if (
                args.rejoin_policy_alignment_entry_m is not None
                and expects_carried_peg
                and peg_hole_xy <= args.rejoin_policy_alignment_entry_m
                and not policy_alignment_reached
            ):
                policy_alignment_reached = True
                policy_alignment_best_peg_z = float(peg[2])
                policy_alignment_last_descent_time = time.monotonic()
                print(
                    "Policy reached the near-concentric region; late contact-"
                    f"recovery trigger armed at xy={peg_hole_xy:.6f}m.",
                    flush=True,
                )
            waiting_for_late_alignment_failure = (
                args.rejoin_policy_alignment_entry_m is not None
                and expects_carried_peg
                and not policy_alignment_reached
                and not policy_transport_stall_trigger
            )
            if waiting_for_late_alignment_failure:
                # Do not let the joint-space manifold distance trigger an
                # early generic correction.  Failed runs that never reach the
                # entry band time out and are rejected rather than polluting
                # the contact-recovery bucket with pre-contact supervision.
                continue
            policy_contact_trigger = (
                policy_alignment_reached
                and args.rejoin_policy_contact_force_n is not None
                and policy_contact_force_n is not None
                and policy_contact_force_n >= args.rejoin_policy_contact_force_n
            )
            # Once the policy has entered the configured near-hole band, do
            # not let the generic joint/manifold mismatch path pre-empt the
            # late-contact experiment.  At this point an XY mismatch is the
            # behavior we need to observe and repair *after* a real contact or
            # a measured descent stall; taking over immediately would merely
            # record another scripted alignment suffix.  A lost peg still
            # falls through to the normal physical-mismatch recovery/failure
            # handling below.
            waiting_for_late_contact_failure = (
                policy_alignment_reached
                and expects_carried_peg
                and not physical_mismatch
                and not grasp_stalled
                and not policy_contact_trigger
                and not policy_alignment_stall_trigger
            )
            if waiting_for_late_contact_failure:
                continue
            if (
                not grasp_stalled
                and
                nearest_distance < args.rejoin_trigger_l2_rad
                and not physical_mismatch
                and not alignment_mismatch
                and not policy_contact_trigger
                and not policy_alignment_stall_trigger
                and not policy_transport_stall_trigger
            ):
                continue
            if interventions >= args.rejoin_max_interventions:
                _atomic_outcome(
                    args.outcome_file,
                    "failure",
                    f"exceeded {args.rejoin_max_interventions} bounded interventions",
                )
                return 1
            interventions += 1
            # Bind this takeover to the anchor selected in the current policy
            # state.  Leaving target_state/target_frame from the previous
            # intervention alive for one loop iteration can falsely satisfy a
            # rejoin check and advance safe_align directly to insertion while
            # the peg is still far from the hole.
            target_frame = int(anchor)
            target_state = np.asarray(
                tracker.selected["state"], dtype=np.float32
            )[target_frame].copy()
            reason = (
                "current-state distance to nearest feasible demonstration point: "
                f"nearest_frame={anchor}, distance={nearest_distance:.5f}, "
                f"physical_mismatch={physical_mismatch}, "
                f"alignment_mismatch={alignment_mismatch}, recovery_mode={target_mode}"
            )
            if policy_contact_trigger:
                reason += f", contact_force={policy_contact_force_n:.2f}N"
            if policy_alignment_stall_trigger:
                reason += (
                    ", aligned_descent_stall="
                    f"{time.monotonic() - policy_alignment_last_descent_time:.2f}s"
                )
            if policy_transport_stall_trigger:
                reason += (
                    ", transport_stall="
                    f"{time.monotonic() - policy_transport_last_progress_time:.2f}s"
                )
            if post_handoff_grasp_lost:
                reason += f", post_handoff_grasp_lost: peg_lift={peg_lift_m:.4f}m"
            if pre_drop_grasp_risk and pre_drop_grasp_error is not None:
                risk_xy, risk_z, risk_z_error = pre_drop_grasp_error
                reason += (
                    ", pre_drop_grasp_risk: "
                    f"gripper_peg_xy={risk_xy:.4f}m, "
                    f"gripper_peg_z={risk_z:.4f}m, z_error={risk_z_error:.4f}m"
                )
            request_takeover(
                args.session_dir,
                trigger=reason,
                requester="demonstration_rejoin_expert_v1",
                sequence=control_sequence,
            )
            control_sequence += 1
            takeover_active = True
            physical_correction_active = False
            insertion_best_peg_z = None
            insertion_last_progress_time = None
            print(f"Intervention {interventions}: {reason}", flush=True)

        try:
            if target_state is not None and target_frame is not None:
                arm_l2 = float(np.linalg.norm(state[:6] - target_state[:6]))
                gripper_error = float(abs(state[6] - target_state[6]))
                target_subtask = str(np.asarray(tracker.selected["subtask"])[target_frame])
                target_requires_insertion = target_subtask in {
                    "insert the peg into the hole",
                    "verify insertion success",
                }
                target_requires_concentric_alignment = target_subtask in {
                    "approach and align with the hole",
                    "insert the peg into the hole",
                    "verify insertion success",
                }
                expected_relative_xy = (
                    np.zeros(2, dtype=np.float64)
                    if target_requires_concentric_alignment
                    else tracker.expected_peg_hole_xy(target_frame)
                )
                physical_error_m = float(
                    np.linalg.norm((peg[:2] - hole[:2]) - expected_relative_xy)
                )
                physical_release_m = (
                    min(args.rejoin_release_physical_m, args.descent_xy_gate_m)
                    if target_requires_concentric_alignment
                    else args.rejoin_release_physical_m
                )
                insertion_geometry_valid = (
                    peg_hole_xy <= args.success_xy_m
                    and args.success_peg_z_m is not None
                    and peg[2] <= args.success_peg_z_m
                )
            else:
                arm_l2 = float("inf")
                gripper_error = float("inf")
                physical_error_m = float("inf")
                target_requires_insertion = False
                physical_release_m = args.rejoin_release_physical_m
                insertion_geometry_valid = False
            if (
                not physical_correction_active
                and arm_l2 <= args.rejoin_physical_entry_l2_rad
                and gripper_error <= args.rejoin_release_gripper_rad
            ):
                physical_correction_active = True
                print(
                    f"Physical XY correction latched at frame {target_frame}: "
                    f"arm_l2={arm_l2:.5f}, physical_xy={physical_error_m:.5f}m",
                    flush=True,
                )
            if (
                target_mode != "grasp_recovery"
                and
                physical_correction_active
                and (
                    physical_error_m > physical_release_m
                    or (target_requires_insertion and not insertion_geometry_valid)
                )
                and float(state[6]) >= 0.35
            ):
                physical_target = np.asarray(hole, dtype=np.float64).copy()
                physical_target[:2] += expected_relative_xy
                descent_m = args.chunk_descent_m
                if target_requires_insertion and terminal_peg_z_m is not None:
                    # Do not issue one more full descent block when the peg is
                    # already close to the accepted seated height.  The old
                    # fixed 4 mm endpoint over-drove the zero-clearance socket
                    # by several millimetres and produced sustained 100+ N
                    # contact after the geometry was already successful.
                    descent_m = min(
                        descent_m,
                        max(0.0, float(peg[2]) - terminal_peg_z_m),
                    )
                if target_requires_insertion:
                    physical_chunk_size = args.insertion_chunk_size
                elif target_mode == "safe_align":
                    physical_chunk_size = args.alignment_chunk_size
                else:
                    physical_chunk_size = args.chunk_size
                if target_requires_insertion and float(peg[2]) <= 0.905:
                    # Retain twice as many controller samples in the exact-fit
                    # lower section. This halves the nominal insertion speed
                    # without changing the Cartesian endpoint solve.
                    physical_chunk_size = 2 * args.insertion_chunk_size
                if target_requires_insertion and descent_m < args.chunk_descent_m:
                    # Preserve the nominal Cartesian speed for the shorter
                    # terminal block instead of filling it with near-static
                    # samples.
                    physical_chunk_size = max(
                        2,
                        int(
                            np.ceil(
                                physical_chunk_size
                                * descent_m
                                / args.chunk_descent_m
                            )
                        ),
                    )
                chunk = build_moveit_expert_chunk(
                    ik_solver,
                    state,
                    peg,
                    physical_target,
                    chunk_size=physical_chunk_size,
                    chunk_xy_step_m=args.chunk_xy_step_m,
                    chunk_descent_m=descent_m,
                    descent_xy_gate_m=args.descent_xy_gate_m,
                    max_joint_step_rad=args.max_joint_step_rad,
                    descend=target_requires_insertion,
                    gripper_command=0.8,
                )
                if chunk is None:
                    # A millimetre-scale Cartesian request can occasionally
                    # cross a local IK branch boundary during exact-fit
                    # insertion even though a smaller motion from the same
                    # measured state is feasible. Retry once at half scale;
                    # this keeps the expert monotonic and avoids discarding a
                    # physically healthy rollout because of one numerical IK
                    # miss. A second rejection remains a hard failure.
                    retry_xy_step_m = max(0.001, 0.5 * args.chunk_xy_step_m)
                    retry_descent_m = max(0.001, 0.5 * descent_m)
                    print(
                        "MoveIt IK rejected the primary physical correction; "
                        f"retrying at xy_step={retry_xy_step_m:.4f}m "
                        f"descent={retry_descent_m:.4f}m",
                        flush=True,
                    )
                    chunk = build_moveit_expert_chunk(
                        ik_solver,
                        state,
                        peg,
                        physical_target,
                        chunk_size=physical_chunk_size,
                        chunk_xy_step_m=retry_xy_step_m,
                        chunk_descent_m=retry_descent_m,
                        descent_xy_gate_m=args.descent_xy_gate_m,
                        max_joint_step_rad=args.max_joint_step_rad,
                        descend=target_requires_insertion,
                        gripper_command=0.8,
                    )
                if chunk is None:
                    raise ValueError("MoveIt IK rejected physical-manifold XY correction")
            elif target_mode == "grasp_recovery":
                grasp_chunk_size = max(
                    args.chunk_size, args.grasp_pickup_chunk_size
                )
                chunk, _, _ = tracker.build_chunk(
                    state,
                    chunk_size=grasp_chunk_size,
                    max_joint_step_rad=args.max_joint_step_rad,
                )
                # The replay action closes to the universal 0.8-rad command,
                # while contact with this peg settles near 0.627 rad. Rejoin
                # against the demonstrated measured state at the replay
                # cursor, not against the action command endpoint.
                assert tracker.selected is not None and tracker.cursor is not None
                target_frame = int(tracker.cursor)
                target_state = np.asarray(
                    tracker.selected["state"], dtype=np.float32
                )[target_frame].copy()
            else:
                rejoin_chunk_size = (
                    args.alignment_chunk_size
                    if target_mode == "safe_align"
                    else args.chunk_size
                )
                chunk, target_frame, target_state = tracker.build_rejoin_chunk(
                    state,
                    chunk_size=rejoin_chunk_size,
                    max_joint_step_rad=args.max_joint_step_rad,
                    lookahead_frames=0,
                )
        except ValueError as error:
            _atomic_outcome(args.outcome_file, "failure", str(error))
            return 1
        publish_expert_chunk(
            args.session_dir,
            chunk,
            sequence=expert_sequence,
            publisher="demonstration_rejoin_expert_v1",
            skill_progress_phase=7,
            skill_progress=tracker.progress,
            transition_readiness=1.0,
            label_confidence=1.0,
        )
        ack = wait_for_expert_execution_ack(
            args.session_dir,
            expert_sequence=expert_sequence,
            timeout_s=args.ack_timeout_s,
        )
        if ack is None:
            _atomic_outcome(
                args.outcome_file, "failure", f"expert chunk {expert_sequence} ACK timeout"
            )
            return 1
        expert_sequence += 1

    _atomic_outcome(args.outcome_file, "failure", "multi-handoff full episode timeout")
    return 1


def _unused_stage_aware_multi_handoff_recovery(
    args, ik_solver: MoveItRecoveryIK
) -> int:
    """Deprecated diagnostic prototype; never dispatched by ``main``.

    Retained temporarily only to make the failed dry-run auditable. New full
    episode collection uses causal local alignment in
    ``run_demonstration_multi_handoff_recovery``.
    """
    tracker = DemonstrationManifoldRecovery(
        args.grasp_reference_episode,
        session_dir=args.session_dir,
        scene_tolerance_m=args.grasp_reference_scene_tolerance_m,
        bridge_steps=args.grasp_reference_bridge_steps,
        safe_open_steps=args.grasp_reference_safe_open_steps,
        max_anchor_l2_rad=args.grasp_reference_max_anchor_l2_rad,
        gripper_distance_weight=args.grasp_reference_gripper_weight,
    )
    finger_frames = (
        "robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"
    )
    deadline = time.monotonic() + args.timeout_s
    last_state_mtime = None
    initial_peg = None
    expert_sequence = 0
    intervention_count = 0
    recovery_kind: str | None = None
    target_state = None
    target_frame = None
    grasp_ever = False
    best_xy = float("inf")
    progress_time = time.monotonic()
    policy_start_time = None
    policy_count_at_release = 0
    close_failure_since = None

    def next_control_sequence() -> int:
        current = read_control_mode(args.session_dir)
        return 0 if current is None else int(current["sequence"]) + 1

    def release(reason: str, policy_count: int) -> None:
        nonlocal recovery_kind, policy_count_at_release, progress_time
        release_to_policy(
            args.session_dir,
            reason=reason,
            requester="stage_aware_demonstration_rejoin_expert_v2",
            sequence=next_control_sequence(),
        )
        print(f"Released to policy: {reason}", flush=True)
        recovery_kind = None
        policy_count_at_release = policy_count
        progress_time = time.monotonic()

    print(
        "Stage-aware full episode recovery active (grasp / transport / align).",
        flush=True,
    )
    while time.monotonic() < deadline:
        if not JOINT_STATE_FILE.exists() or not (args.session_dir / "session.json").exists():
            time.sleep(0.02)
            continue
        state_mtime = JOINT_STATE_FILE.stat().st_mtime_ns
        if state_mtime == last_state_mtime:
            time.sleep(0.005)
            continue
        last_state_mtime = state_mtime
        state = _read_state()
        poses = read_gazebo_fixture_poses()
        gripper = ik_solver.frame_midpoint(finger_frames)
        if state is None or poses is None or gripper is None:
            continue
        peg, hole = poses
        if initial_peg is None:
            initial_peg = np.asarray(peg, dtype=np.float64).copy()
        peg_lift = float(peg[2] - initial_peg[2])
        gripper_peg_xy = float(np.linalg.norm(gripper[:2] - peg[:2]))
        peg_hole_xy = float(np.linalg.norm(peg[:2] - hole[:2]))
        closed = float(state[6]) >= 0.35
        physical_grasp = closed and peg_lift >= 0.03 and gripper_peg_xy <= 0.025
        grasp_ever = grasp_ever or physical_grasp
        policy_count = _completed_policy_event_count(args.session_dir)
        if policy_count and policy_start_time is None:
            policy_start_time = time.monotonic()

        if (
            recovery_kind is None
            and intervention_count > 0
            and policy_count - policy_count_at_release >= args.rejoin_policy_suffix_events
            and peg_hole_xy <= args.success_xy_m
            and args.success_peg_z_m is not None
            and peg[2] <= args.success_peg_z_m
        ):
            _atomic_outcome(
                args.outcome_file,
                "success",
                f"stage-aware full task: interventions={intervention_count}, "
                f"xy={peg_hole_xy:.6f}, peg_z={peg[2]:.6f}",
            )
            return 0

        if recovery_kind == "grasp":
            if physical_grasp and peg_lift >= 0.05:
                release(
                    f"physical grasp/lift restored (lift={peg_lift:.4f}m)", policy_count
                )
                best_xy = peg_hole_xy
                continue
        elif recovery_kind == "transport":
            assert target_state is not None and target_frame is not None
            arm_l2 = float(np.linalg.norm(state[:6] - target_state[:6]))
            if arm_l2 <= args.rejoin_release_l2_rad and closed and peg_lift >= 0.03:
                release(
                    f"transport manifold frame {target_frame} restored "
                    f"(arm_l2={arm_l2:.4f})",
                    policy_count,
                )
                best_xy = peg_hole_xy
                continue
        elif recovery_kind == "align":
            if peg_hole_xy <= 0.005 and closed and peg_lift >= 0.0:
                release(
                    f"safe hole alignment restored (xy={peg_hole_xy:.4f}m)", policy_count
                )
                best_xy = peg_hole_xy
                continue

        if recovery_kind is None:
            if policy_count < max(2, policy_count_at_release + args.rejoin_min_policy_events):
                continue
            now = time.monotonic()
            kind = None
            reason = None
            if not grasp_ever:
                bad_close = float(state[6]) >= 0.25 and (
                    gripper_peg_xy > 0.010 or peg_lift < 0.005
                )
                close_failure_since = (
                    now if bad_close and close_failure_since is None
                    else close_failure_since if bad_close
                    else None
                )
                approach_stall = (
                    policy_start_time is not None
                    and now - policy_start_time >= 25.0
                    and peg_lift < 0.005
                )
                if approach_stall or (
                    close_failure_since is not None and now - close_failure_since >= 1.5
                ):
                    kind = "grasp"
                    reason = (
                        f"grasp physical failure: gripper_peg_xy={gripper_peg_xy:.4f}, "
                        f"gripper={state[6]:.4f}, lift={peg_lift:.4f}"
                    )
                    try:
                        tracker.select_anchor(state, mode="approach_entry")
                    except ValueError as error:
                        _atomic_outcome(args.outcome_file, "failure", str(error))
                        return 1
            else:
                if peg_hole_xy <= best_xy - 0.005:
                    best_xy = peg_hole_xy
                    progress_time = now
                stalled = now - progress_time >= args.rejoin_stall_s
                if peg_hole_xy > 0.060 and (stalled or not closed):
                    kind = "transport"
                    reason = (
                        f"transport stalled/released: xy={peg_hole_xy:.4f}, "
                        f"closed={closed}, lift={peg_lift:.4f}"
                    )
                    try:
                        tracker.select_anchor(state, mode="transport")
                    except ValueError as error:
                        _atomic_outcome(args.outcome_file, "failure", str(error))
                        return 1
                elif peg_hole_xy > 0.005 and (
                    stalled or (peg[2] < 0.94 and peg_hole_xy > args.success_xy_m)
                ):
                    kind = "align"
                    reason = (
                        f"unsafe descent/alignment drift: xy={peg_hole_xy:.4f}, "
                        f"peg_z={peg[2]:.4f}"
                    )
            if kind is None:
                continue
            if intervention_count >= args.rejoin_max_interventions:
                _atomic_outcome(args.outcome_file, "failure", "intervention budget exhausted")
                return 1
            recovery_kind = kind
            intervention_count += 1
            request_takeover(
                args.session_dir,
                trigger=reason,
                requester="stage_aware_demonstration_rejoin_expert_v2",
                sequence=next_control_sequence(),
            )
            print(
                f"Intervention {intervention_count} ({recovery_kind}): {reason}", flush=True
            )

        if recovery_kind == "grasp":
            try:
                chunk, target_frame, _ = tracker.build_chunk(
                    state,
                    chunk_size=args.chunk_size,
                    max_joint_step_rad=args.max_joint_step_rad,
                )
                target_state = np.asarray(chunk[-1], dtype=np.float32)
            except ValueError as error:
                _atomic_outcome(args.outcome_file, "failure", str(error))
                return 1
        elif recovery_kind == "transport":
            try:
                chunk, target_frame, target_state = tracker.build_rejoin_chunk(
                    state,
                    chunk_size=args.chunk_size,
                    max_joint_step_rad=args.max_joint_step_rad,
                    lookahead_frames=args.rejoin_lookahead_frames,
                )
            except ValueError as error:
                _atomic_outcome(args.outcome_file, "failure", str(error))
                return 1
        else:
            chunk = build_moveit_expert_chunk(
                ik_solver,
                state,
                peg,
                hole,
                chunk_size=args.chunk_size,
                chunk_xy_step_m=min(args.chunk_xy_step_m, 0.010),
                chunk_descent_m=0.0,
                descent_xy_gate_m=args.descent_xy_gate_m,
                max_joint_step_rad=args.max_joint_step_rad,
                descend=False,
                gripper_command=0.8,
            )
            if chunk is None:
                continue
            target_frame = -1
            target_state = np.asarray(chunk[-1], dtype=np.float32)

        publish_expert_chunk(
            args.session_dir,
            chunk,
            sequence=expert_sequence,
            publisher="stage_aware_demonstration_rejoin_expert_v2",
            skill_progress_phase=7,
            skill_progress=tracker.progress if recovery_kind != "align" else 0.0,
            transition_readiness=0.0,
            label_confidence=1.0,
        )
        ack = wait_for_expert_execution_ack(
            args.session_dir, expert_sequence=expert_sequence, timeout_s=args.ack_timeout_s
        )
        if ack is None:
            _atomic_outcome(args.outcome_file, "failure", "expert execution ACK timeout")
            return 1
        expert_sequence += 1

    _atomic_outcome(args.outcome_file, "failure", "stage-aware full episode timeout")
    return 1


def run_demonstration_grasp_lift_recovery(args, ik_solver: MoveItRecoveryIK) -> int:
    """Collect recovery labels that return to an exact successful trajectory."""
    tracker = DemonstrationManifoldRecovery(
        args.grasp_reference_episode,
        session_dir=args.session_dir,
        scene_tolerance_m=args.grasp_reference_scene_tolerance_m,
        bridge_steps=args.grasp_reference_bridge_steps,
        safe_open_steps=args.grasp_reference_safe_open_steps,
        max_anchor_l2_rad=args.grasp_reference_max_anchor_l2_rad,
        gripper_distance_weight=args.grasp_reference_gripper_weight,
    )
    finger_frames = (
        "robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"
    )
    deadline = time.monotonic() + args.timeout_s
    trigger_deadline = None
    closed_no_lift_since = None
    initial_peg = None
    last_state_mtime = None
    sequence = 0
    takeover_active = False
    print(
        f"Waiting for grasp failure; recovery will use {len(tracker.references)} "
        "exact-scene successful demonstrations.", flush=True
    )
    while time.monotonic() < deadline:
        if not JOINT_STATE_FILE.exists():
            time.sleep(0.02)
            continue
        state_mtime = JOINT_STATE_FILE.stat().st_mtime_ns
        if state_mtime == last_state_mtime:
            time.sleep(0.005)
            continue
        last_state_mtime = state_mtime
        state = _read_state()
        poses = read_gazebo_fixture_poses()
        gripper = ik_solver.frame_midpoint(finger_frames)
        if state is None or poses is None or gripper is None:
            continue
        peg, _ = poses
        if initial_peg is None:
            initial_peg = np.asarray(peg, dtype=np.float64).copy()
        peg_lift = float(peg[2] - initial_peg[2])
        peg_xy_shift = float(np.linalg.norm(peg[:2] - initial_peg[:2]))
        xy_norm = float(np.linalg.norm(peg[:2] - gripper[:2]))
        gripper_height = float(gripper[2] - peg[2])

        if peg_lift < -args.grasp_max_recoverable_drop_m or (
            not takeover_active and peg_xy_shift > args.grasp_max_recoverable_xy_shift_m
        ):
            note = (
                "unrecoverable peg displacement before demonstration bridge: "
                f"drop={max(0.0, -peg_lift):.6f}m xy_shift={peg_xy_shift:.6f}m"
            )
            _atomic_outcome(args.outcome_file, "failure", note)
            print(f"Refusing recovery sample: {note}", flush=True)
            return 1

        if takeover_active and state[6] >= 0.35 and peg_lift >= args.success_lift_m:
            _atomic_outcome(
                args.outcome_file, "success",
                f"demonstration-manifold grasp/lift: peg_lift={peg_lift:.6f}",
            )
            print(
                f"Demonstration-manifold recovery success: peg_lift={peg_lift:.4f}m",
                flush=True,
            )
            return 0

        if not takeover_active:
            if args.grasp_trigger_mode == "time":
                if trigger_deadline is None:
                    if not _policy_rollin_started(args.session_dir):
                        continue
                    trigger_deadline = time.monotonic() + args.trigger_after_s
                if time.monotonic() < trigger_deadline:
                    continue
            elif args.grasp_trigger_mode == "aligned_high":
                if not (
                    xy_norm <= args.grasp_trigger_xy_m
                    and gripper_height >= args.grasp_trigger_min_height_m
                ):
                    continue
            elif args.grasp_trigger_mode == "gripper_misaligned":
                # The earliest observable grasp failure in the current model:
                # it starts closing while the fingertips are still outside the
                # demonstrated grasp corridor. Intervene before contact can
                # knock the still-recoverable peg over.
                if not (
                    float(state[6]) >= args.grasp_trigger_closed_rad
                    and xy_norm > args.grasp_trigger_xy_m
                ):
                    continue
            else:
                failed = (
                    float(state[6]) >= args.grasp_trigger_closed_rad
                    and xy_norm <= args.grasp_trigger_xy_m
                    and peg_lift <= args.grasp_trigger_max_lift_m
                )
                if not failed:
                    closed_no_lift_since = None
                    continue
                if closed_no_lift_since is None:
                    closed_no_lift_since = time.monotonic()
                    continue
                if time.monotonic() - closed_no_lift_since < args.grasp_trigger_sustain_s:
                    continue
            try:
                reference_path, anchor, anchor_l2 = tracker.select_anchor(
                    state,
                    mode=(
                        "approach_entry"
                        if args.grasp_trigger_mode == "gripper_misaligned"
                        else "nearest"
                    ),
                )
            except ValueError as error:
                _atomic_outcome(args.outcome_file, "failure", str(error))
                print(f"Refusing recovery sample: {error}", flush=True)
                return 1
            print(
                f"Selected demonstration anchor: {reference_path} frame={anchor} "
                f"arm_l2={anchor_l2:.4f}rad measured_gripper={state[6]:.4f}", flush=True
            )

        try:
            chunk, reference_start, bridge = tracker.build_chunk(
                state, chunk_size=args.chunk_size, max_joint_step_rad=args.max_joint_step_rad
            )
        except ValueError as error:
            _atomic_outcome(args.outcome_file, "failure", str(error))
            print(f"Stopping invalid recovery: {error}", flush=True)
            return 1
        publish_expert_chunk(
            args.session_dir, chunk, sequence=sequence,
            publisher="exact_scene_demonstration_manifold_expert_v1",
            skill_progress_phase=7, skill_progress=tracker.progress,
            transition_readiness=tracker.progress, label_confidence=1.0,
        )
        if not takeover_active:
            request_takeover(
                args.session_dir,
                trigger=(
                    "grasp rollout failure; bridge to exact-scene successful "
                    f"demonstration frame {reference_start}"
                ),
                requester="exact_scene_demonstration_manifold_expert_v1",
            )
            takeover_active = True
        print(
            f"Demonstration recovery chunk {sequence}: reference_start={reference_start} "
            f"bridge_steps={bridge} progress={tracker.progress:.3f}", flush=True
        )
        ack = wait_for_expert_execution_ack(
            args.session_dir, expert_sequence=sequence, timeout_s=args.ack_timeout_s
        )
        if ack is None:
            _atomic_outcome(args.outcome_file, "failure", f"expert chunk {sequence} ACK timeout")
            return 1
        sequence += 1

    if takeover_active:
        _atomic_outcome(args.outcome_file, "failure", "demonstration recovery timeout")
    print("Demonstration-manifold recovery expert timed out.", flush=True)
    return 1


def run_demonstration_transport_recovery(args) -> int:
    """Recover release or stalled motion back to demonstrated transport."""
    tracker = DemonstrationManifoldRecovery(
        args.grasp_reference_episode,
        session_dir=args.session_dir,
        scene_tolerance_m=args.grasp_reference_scene_tolerance_m,
        bridge_steps=args.grasp_reference_bridge_steps,
        safe_open_steps=args.grasp_reference_safe_open_steps,
        max_anchor_l2_rad=args.grasp_reference_max_anchor_l2_rad,
        gripper_distance_weight=args.grasp_reference_gripper_weight,
    )
    deadline = time.monotonic() + args.timeout_s
    last_state_mtime = None
    initial_peg = None
    grasp_seen = False
    transport_best_xy = None
    transport_progress_time = None
    takeover_active = False
    sequence = 0
    print(
        "Waiting for a premature release or stalled physical transport; "
        f"loaded {len(tracker.references)} exact-scene demonstrations.",
        flush=True,
    )
    while time.monotonic() < deadline:
        if not JOINT_STATE_FILE.exists():
            time.sleep(0.02)
            continue
        state_mtime = JOINT_STATE_FILE.stat().st_mtime_ns
        if state_mtime == last_state_mtime:
            time.sleep(0.005)
            continue
        last_state_mtime = state_mtime
        state = _read_state()
        poses = read_gazebo_fixture_poses()
        if state is None or poses is None:
            continue
        peg, hole = poses
        if initial_peg is None:
            initial_peg = np.asarray(peg, dtype=np.float64).copy()
        peg_lift = float(peg[2] - initial_peg[2])
        peg_hole_xy = float(np.linalg.norm(peg[:2] - hole[:2]))
        if state[6] >= args.grasp_trigger_closed_rad and peg_lift >= args.transport_trigger_min_lift_m:
            grasp_seen = True
            if transport_best_xy is None:
                transport_best_xy = peg_hole_xy
                transport_progress_time = time.monotonic()
            elif peg_hole_xy <= transport_best_xy - args.transport_trigger_min_progress_m:
                transport_best_xy = peg_hole_xy
                transport_progress_time = time.monotonic()

        if takeover_active:
            if peg_lift < args.transport_trigger_min_lift_m * 0.5:
                note = f"peg dropped during transport recovery: lift={peg_lift:.6f}m"
                _atomic_outcome(args.outcome_file, "failure", note)
                print(f"Stopping invalid recovery: {note}", flush=True)
                return 1
            if (
                state[6] >= args.grasp_trigger_closed_rad
                and peg_hole_xy <= args.transport_success_xy_m
            ):
                note = (
                    "demonstration-manifold transport: "
                    f"peg_hole_xy={peg_hole_xy:.6f}m lift={peg_lift:.6f}m"
                )
                _atomic_outcome(args.outcome_file, "success", note)
                print(f"Demonstration transport recovery success: {note}", flush=True)
                return 0
        else:
            if not _policy_rollin_started(args.session_dir) or not grasp_seen:
                continue
            guard_request = read_takeover(args.session_dir)
            guard_triggered = (
                guard_request is not None
                and guard_request.get("requester")
                in {
                    "ros_predispatch_premature_release_guard_v1",
                    "ros_predispatch_transport_deviation_guard_v1",
                }
            )
            premature_release = guard_triggered or (
                state[6] <= args.transport_trigger_open_rad
                and peg_lift >= args.transport_trigger_min_lift_m
                and peg_hole_xy >= args.transport_trigger_min_hole_distance_m
            )
            stalled_transport = (
                not guard_triggered
                and state[6] >= args.grasp_trigger_closed_rad
                and peg_lift >= args.transport_trigger_min_lift_m
                and peg_hole_xy >= args.transport_trigger_min_hole_distance_m
                and transport_progress_time is not None
                and time.monotonic() - transport_progress_time
                >= args.transport_trigger_stall_s
            )
            if not premature_release and not stalled_transport:
                continue
            try:
                reference_path, anchor, anchor_l2 = tracker.select_anchor(
                    state, mode="transport"
                )
            except ValueError as error:
                _atomic_outcome(args.outcome_file, "failure", str(error))
                print(f"Refusing recovery sample: {error}", flush=True)
                return 1
            print(
                f"Transport failure intercepted: premature_release={premature_release}, "
                f"stalled={stalled_transport}, gripper={state[6]:.4f}, "
                f"peg_hole_xy={peg_hole_xy:.4f}m; selected {reference_path} "
                f"frame={anchor} arm_l2={anchor_l2:.4f}rad",
                flush=True,
            )

        try:
            chunk, reference_start, bridge = tracker.build_chunk(
                state,
                chunk_size=args.chunk_size,
                max_joint_step_rad=args.max_joint_step_rad,
            )
        except ValueError as error:
            _atomic_outcome(args.outcome_file, "failure", str(error))
            print(f"Stopping invalid recovery: {error}", flush=True)
            return 1
        publish_expert_chunk(
            args.session_dir,
            chunk,
            sequence=sequence,
            publisher="exact_scene_demonstration_transport_expert_v1",
            skill_progress_phase=7,
            skill_progress=tracker.progress,
            transition_readiness=tracker.progress,
            label_confidence=1.0,
        )
        if not takeover_active:
            request_takeover(
                args.session_dir,
                trigger=(
                    "transport release/stall; bridge to exact-scene "
                    f"successful demonstration frame {reference_start}"
                ),
                requester="exact_scene_demonstration_transport_expert_v1",
            )
            takeover_active = True
        print(
            f"Transport recovery chunk {sequence}: reference_start={reference_start} "
            f"bridge_steps={bridge} progress={tracker.progress:.3f}",
            flush=True,
        )
        ack = wait_for_expert_execution_ack(
            args.session_dir, expert_sequence=sequence, timeout_s=args.ack_timeout_s
        )
        if ack is None:
            _atomic_outcome(args.outcome_file, "failure", f"expert chunk {sequence} ACK timeout")
            return 1
        sequence += 1

    _atomic_outcome(
        args.outcome_file,
        "failure",
        (
            "transport recovery timeout after takeover"
            if takeover_active
            else "policy never reached a recoverable physical transport state"
        ),
    )
    print("Demonstration transport recovery expert timed out.", flush=True)
    return 1


def run_grasp_lift_recovery(args, ik_solver: MoveItRecoveryIK) -> int:
    """Recover an arbitrary free-space rollout state to a verified physical grasp/lift."""
    finger_frames = (
        "robotiq_85_left_finger_tip_link",
        "robotiq_85_right_finger_tip_link",
    )
    deadline = time.monotonic() + args.timeout_s
    # The IPC session is initialized before the starting posture and observation
    # history are ready.  Starting this timer at process launch therefore turns
    # startup latency (or an initial-posture abort) into fake policy roll-in.
    # Arm the timer only after the controller has dispatched its first action.
    trigger_deadline = None
    initial_peg_z = None
    last_state_mtime = None
    sequence = 0
    takeover_active = False
    close_sent = False
    closed_no_lift_since = None
    probe_close_pending = False
    probe_initialized = False
    probe_chunks_remaining = 0
    probe_complete = False
    probe_failed_retreat = False
    reuse_existing_grasp = False
    # Hysteresis for the XY gate.  Once the fingers have entered the grasp
    # corridor, millimetre-scale IK drift during descent must not switch the
    # vertical target back to traverse height.  Re-arm the high-clearance path
    # only after leaving the wider trigger corridor.
    grasp_xy_acquired = False
    progress_tracker = LocalPhaseProgress()
    if args.grasp_trigger_mode == "aligned_high":
        print(
            "Waiting for aligned-high grasp failure before takeover: "
            f"xy<={args.grasp_trigger_xy_m:.3f}m, "
            f"height>={args.grasp_trigger_min_height_m:.3f}m",
            flush=True,
        )
    elif args.grasp_trigger_mode == "closed_no_lift":
        print(
            "Waiting for closed-without-lift grasp failure before takeover: "
            f"gripper>={args.grasp_trigger_closed_rad:.3f}rad, "
            f"xy<={args.grasp_trigger_xy_m:.3f}m, "
            f"peg_lift<={args.grasp_trigger_max_lift_m:.3f}m sustained for "
            f"{args.grasp_trigger_sustain_s:.1f}s",
            flush=True,
        )
    else:
        print(
            f"Waiting {args.trigger_after_s:.1f}s for policy roll-in before grasp/lift takeover...",
            flush=True,
        )
    while time.monotonic() < deadline:
        if not JOINT_STATE_FILE.exists() or not (args.session_dir / "session.json").exists():
            time.sleep(0.02)
            continue
        state_mtime = JOINT_STATE_FILE.stat().st_mtime_ns
        if state_mtime == last_state_mtime:
            time.sleep(0.005)
            continue
        last_state_mtime = state_mtime
        state = _read_state()
        poses = read_gazebo_fixture_poses()
        gripper = ik_solver.frame_midpoint(finger_frames)
        if state is None or poses is None or gripper is None:
            continue
        peg, _ = poses
        if initial_peg_z is None:
            initial_peg_z = float(peg[2])
        xy_error = np.asarray(peg[:2] - gripper[:2], dtype=np.float64)
        xy_norm = float(np.linalg.norm(xy_error))
        if xy_norm <= args.grasp_xy_gate_m:
            grasp_xy_acquired = True
        elif xy_norm > args.grasp_trigger_xy_m:
            grasp_xy_acquired = False
        gripper_height = float(gripper[2] - peg[2])
        peg_lift = float(peg[2] - initial_peg_z)
        if not takeover_active and peg_lift < -args.grasp_max_recoverable_drop_m:
            _atomic_outcome(
                args.outcome_file,
                "failure",
                "peg dropped before grasp recovery takeover: "
                f"peg_drop={-peg_lift:.6f}m",
            )
            print(
                "Refusing unrecoverable grasp sample: peg already dropped "
                f"{-peg_lift:.4f}m.",
                flush=True,
            )
            return 1
        if not takeover_active:
            if args.grasp_trigger_mode == "time":
                if trigger_deadline is None:
                    if not _policy_rollin_started(args.session_dir):
                        continue
                    trigger_deadline = time.monotonic() + args.trigger_after_s
                    print(
                        "First policy action observed; "
                        f"starting {args.trigger_after_s:.1f}s rollout timer.",
                        flush=True,
                    )
                if time.monotonic() < trigger_deadline:
                    continue
            elif args.grasp_trigger_mode == "aligned_high":
                if not (
                    xy_norm <= args.grasp_trigger_xy_m
                    and gripper_height >= args.grasp_trigger_min_height_m
                ):
                    continue
            else:
                closed_without_lift = (
                    float(state[6]) >= args.grasp_trigger_closed_rad
                    and xy_norm <= args.grasp_trigger_xy_m
                    and peg_lift <= args.grasp_trigger_max_lift_m
                )
                if not closed_without_lift:
                    closed_no_lift_since = None
                    continue
                if closed_no_lift_since is None:
                    closed_no_lift_since = time.monotonic()
                    continue
                if time.monotonic() - closed_no_lift_since < args.grasp_trigger_sustain_s:
                    continue
            if not probe_initialized:
                probe_chunks_remaining = args.grasp_probe_chunks
                probe_close_pending = True
                probe_initialized = True
        elif peg_lift < -args.grasp_max_recoverable_drop_m:
            _atomic_outcome(
                args.outcome_file,
                "failure",
                "peg dropped during grasp recovery: "
                f"peg_drop={-peg_lift:.6f}m",
            )
            print(
                "Stopping unrecoverable grasp sample: peg dropped "
                f"{-peg_lift:.4f}m after takeover.",
                flush=True,
            )
            return 1
        if (
            takeover_active
            and probe_initialized
            and not probe_complete
            and peg_lift >= args.grasp_probe_success_lift_m
            and float(state[6]) >= 0.35
        ):
            # Do not continue lifting a payload with the policy's weak partial
            # aperture. Strengthen the grasp as soon as measured peg motion
            # proves that the object is following the fingers.
            probe_chunks_remaining = 0
            probe_complete = True
            reuse_existing_grasp = True
        elif takeover_active and probe_initialized and not probe_complete and probe_chunks_remaining == 0:
            probe_complete = True
            reuse_existing_grasp = False
            probe_failed_retreat = True
        target_offset = (
            args.traverse_z_offset_m
            if not grasp_xy_acquired
            else args.grasp_center_z_offset_m
        )
        target_gripper_z = float(peg[2] + target_offset)
        z_error = target_gripper_z - float(gripper[2])

        if close_sent:
            if state[6] >= 0.35 and peg_lift >= args.success_lift_m:
                _atomic_outcome(
                    args.outcome_file,
                    "success",
                    f"physical grasp/lift: xy={xy_norm:.6f}, peg_lift={peg_lift:.6f}",
                )
                print(
                    f"Grasp/lift recovery success: xy={xy_norm:.4f}m "
                    f"peg_lift={peg_lift:.4f}m",
                    flush=True,
                )
                return 0
            world_delta = np.asarray([0.0, 0.0, args.lift_step_m], dtype=np.float64)
            gripper_command = 0.8
            phase = "lift"
            readiness = min(1.0, max(0.0, peg_lift) / args.success_lift_m)
            progress_phase, progress = 6, readiness
        elif reuse_existing_grasp:
            # The policy made a weak but physically valid grasp. Strengthen it
            # while adding the same synchronized pickup used by demonstrations
            # instead of reopening the fingers around a suspended peg.
            world_delta = np.asarray(
                [0.0, 0.0, args.grasp_pickup_step_m], dtype=np.float64
            )
            gripper_command = 0.8
            phase = "close"
            progress_phase, progress, readiness = 3, 0.0, 0.0
        elif probe_close_pending:
            # A partial measured aperture is not a grasp command. Strengthen it
            # to the universal full-close endpoint while holding the arm still
            # before deciding whether the peg is actually captured.
            world_delta = np.zeros(3, dtype=np.float64)
            gripper_command = 0.8
            phase = "probe_full_close"
            progress_phase, progress, readiness = 3, 0.0, 0.0
        elif probe_chunks_remaining > 0:
            # Test the strengthened grasp with a small lift. If the peg does
            # not follow, retreat with closed fingers before reopening for a
            # clean regrasp so the object is not swept sideways.
            world_delta = np.asarray(
                [0.0, 0.0, args.grasp_probe_step_m], dtype=np.float64
            )
            gripper_command = 0.8
            phase = "probe_existing_grasp"
            progress_phase, progress, readiness = 7, 0.0, 0.0
        elif probe_failed_retreat:
            if gripper_height < args.traverse_z_offset_m:
                world_delta = np.asarray(
                    [0.0, 0.0, args.lift_step_m], dtype=np.float64
                )
                gripper_command = 0.8
                phase = "retreat_failed_grasp"
                progress_phase, progress, readiness = 7, 0.0, 0.0
            else:
                # The next ordinary descent chunk opens at safe clearance and
                # returns to the peg for a complete regrasp.
                probe_failed_retreat = False
                continue
        elif not grasp_xy_acquired and abs(z_error) > args.grasp_z_gate_m:
            # From the home configuration the fingertips sit below the
            # desired peg-centre offset. Establish vertical clearance before
            # the long XY traverse; solving the XY motion first drives UR3
            # toward a near-singular elbow limit around 35--40 cm away.
            z_step = float(
                np.clip(z_error, -args.descent_step_m, args.descent_step_m)
            )
            world_delta = np.asarray([0.0, 0.0, z_step], dtype=np.float64)
            gripper_command = 0.0
            phase = "set_traverse_height"
            # This motion starts after an explicit expert takeover from a
            # failed rollout.  It is a recovery manoeuvre, not the ordinary
            # entry into a skill; labelling it as ``enter`` conflicts with
            # successful demonstrations that begin from the same home pose.
            readiness = float(np.clip(1.0 - z_error / max(args.traverse_z_offset_m, 1e-6), 0.0, 1.0))
            progress_phase, progress = 7, readiness
        elif (
            xy_norm > args.grasp_xy_gate_m
            and float(state[4]) > args.grasp_wrist_2_reset_trigger_rad
        ):
            target_arm = np.asarray(state[:6], dtype=np.float64).copy()
            target_arm[4] = args.grasp_wrist_2_reset_target_rad
            world_delta = None
            gripper_command = 0.0
            phase = "reset_wrist_orientation"
            progress_phase, progress, readiness = 7, 0.0, 0.0
        elif xy_norm > args.grasp_xy_gate_m:
            step_xy = xy_error
            if xy_norm > args.chunk_xy_step_m:
                step_xy *= args.chunk_xy_step_m / xy_norm
            world_delta = np.asarray([step_xy[0], step_xy[1], 0.0], dtype=np.float64)
            gripper_command = 0.0
            phase = "approach_xy"
            progress_phase = 1
            readiness = float(np.clip(1.0 - xy_norm / max(args.trigger_xy_m, args.grasp_xy_gate_m), 0.0, 1.0))
            progress = readiness
        elif abs(z_error) > args.grasp_z_gate_m:
            z_step = float(np.clip(z_error, -args.descent_step_m, args.descent_step_m))
            world_delta = np.asarray([0.0, 0.0, z_step], dtype=np.float64)
            gripper_command = 0.0
            phase = "descend"
            progress_phase = 2
            readiness = float(np.clip(1.0 - abs(z_error) / 0.05, 0.0, 1.0))
            progress = readiness
        else:
            # Match the physically validated scripted demonstrations: closing
            # against a peg resting on the table must be synchronized with a
            # small upward pickup.  A stationary full-close trajectory pins
            # the contact against the table and the knuckle stalls almost at
            # zero.  This remains an ordinary learned arm+gripper action (no
            # attachment and no object-specific aperture); contact determines
            # the achieved gripper angle.
            world_delta = np.asarray(
                [0.0, 0.0, args.grasp_pickup_step_m], dtype=np.float64
            )
            gripper_command = 0.8
            phase = "close"
            # A new current phase starts at zero.  ``readiness`` always means
            # readiness to leave the current phase, not readiness to enter it.
            progress_phase, progress, readiness = 3, 0.0, 0.0

        close_chunk_size = (
            max(args.chunk_size, args.grasp_pickup_chunk_size)
            if phase == "close"
            else args.chunk_size
        )
        if phase == "reset_wrist_orientation":
            chunk = build_joint_target_chunk(
                state,
                target_arm,
                chunk_size=close_chunk_size,
                max_joint_step_rad=args.max_joint_step_rad,
                gripper_command=gripper_command,
            )
        else:
            chunk = build_moveit_translation_chunk(
                ik_solver,
                state,
                world_delta,
                chunk_size=close_chunk_size,
                max_joint_step_rad=args.max_joint_step_rad,
                gripper_command=gripper_command,
            )
        if chunk is None:
            print(
                f"MoveIt IK rejected grasp phase={phase} xy={xy_norm:.4f}m "
                f"z_error={z_error:.4f}m",
                flush=True,
            )
            continue
        if phase in {"close", "probe_full_close"}:
            # Three-second eased close, matching the validated success
            # collector.  The arm simultaneously performs the 15 mm pickup.
            fractions = np.sqrt(
                np.arange(1, len(chunk) + 1, dtype=np.float32) / len(chunk)
            )
            chunk[:, 6] = (
                float(state[6]) + fractions * (0.8 - float(state[6]))
            )
        progress, readiness = progress_tracker.update(progress_phase, progress)
        publish_expert_chunk(
            args.session_dir,
            chunk,
            sequence=sequence,
            publisher="privileged_grasp_lift_expert_v1",
            skill_progress_phase=progress_phase,
            skill_progress=progress,
            transition_readiness=readiness,
            label_confidence=1.0,
        )
        if not takeover_active:
            request_takeover(
                args.session_dir,
                trigger=(
                    f"PAP-MoE grasp roll-in failure mode={args.grasp_trigger_mode}; "
                    f"gripper-peg xy={xy_norm:.6f}m height={gripper_height:.6f}m "
                    f"z_error={z_error:.6f}m"
                ),
                requester="privileged_grasp_lift_expert_v1",
            )
            takeover_active = True
        print(
            f"Grasp expert chunk {sequence}: phase={phase} xy={xy_norm:.4f}m "
            f"z_error={z_error:.4f}m",
            flush=True,
        )
        ack = wait_for_expert_execution_ack(
            args.session_dir,
            expert_sequence=sequence,
            timeout_s=args.ack_timeout_s,
        )
        if ack is None:
            _atomic_outcome(args.outcome_file, "failure", f"expert chunk {sequence} ACK timeout")
            return 1
        if phase == "close":
            close_sent = True
        elif phase == "probe_full_close":
            probe_close_pending = False
        elif phase == "probe_existing_grasp":
            probe_chunks_remaining -= 1
        sequence += 1

    if takeover_active:
        _atomic_outcome(args.outcome_file, "failure", "grasp/lift expert timeout")
    print("Grasp/lift recovery expert timed out.", flush=True)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--outcome-file", type=Path, required=True)
    parser.add_argument(
        "--recovery-phase",
        choices=("grasp_lift", "transport", "align_precontact", "contact", "insertion", "full_task"),
        default="contact",
    )
    parser.add_argument("--trigger-after-s", type=float, default=20.0)
    parser.add_argument(
        "--grasp-trigger-mode",
        choices=("time", "aligned_high", "gripper_misaligned", "closed_no_lift"),
        default="time",
    )
    parser.add_argument("--grasp-trigger-xy-m", type=float, default=0.020)
    parser.add_argument("--grasp-trigger-min-height-m", type=float, default=0.160)
    # 0.40 rad catches the common low/offset partial-close failure while the
    # peg is still upright. Waiting for the nominal 0.60 rad contact aperture
    # lets the same failed action knock the peg down before takeover.
    parser.add_argument("--grasp-trigger-closed-rad", type=float, default=0.400)
    parser.add_argument("--grasp-trigger-max-lift-m", type=float, default=0.005)
    parser.add_argument("--grasp-trigger-sustain-s", type=float, default=3.0)
    parser.add_argument("--grasp-max-recoverable-drop-m", type=float, default=0.020)
    parser.add_argument("--grasp-max-recoverable-xy-shift-m", type=float, default=0.012)
    parser.add_argument(
        "--grasp-reference-episode",
        type=Path,
        action="append",
        default=[],
        help="Exact-scene successful data.npz; repeat for trajectory diversity.",
    )
    parser.add_argument("--grasp-reference-scene-tolerance-m", type=float, default=0.001)
    parser.add_argument("--grasp-reference-bridge-steps", type=int, default=20)
    parser.add_argument("--grasp-reference-safe-open-steps", type=int, default=10)
    parser.add_argument("--grasp-reference-max-anchor-l2-rad", type=float, default=0.50)
    parser.add_argument("--grasp-reference-gripper-weight", type=float, default=0.50)
    parser.add_argument("--transport-trigger-open-rad", type=float, default=0.55)
    parser.add_argument("--transport-trigger-min-lift-m", type=float, default=0.050)
    parser.add_argument("--transport-trigger-min-hole-distance-m", type=float, default=0.080)
    parser.add_argument("--transport-trigger-stall-s", type=float, default=5.0)
    parser.add_argument("--transport-trigger-min-progress-m", type=float, default=0.010)
    parser.add_argument("--transport-success-xy-m", type=float, default=0.060)
    parser.add_argument("--grasp-probe-chunks", type=int, default=2)
    parser.add_argument("--grasp-probe-step-m", type=float, default=0.006)
    parser.add_argument("--grasp-probe-success-lift-m", type=float, default=0.008)
    parser.add_argument("--trigger-xy-m", type=float, default=0.050)
    parser.add_argument("--trigger-min-peg-z-m", type=float, default=0.94)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument(
        "--alignment-chunk-size",
        type=int,
        default=None,
        help=(
            "Controller samples for safe-align/rejoin corrections. Defaults "
            "to --chunk-size; separate from grasp recovery bridge length."
        ),
    )
    parser.add_argument(
        "--insertion-chunk-size",
        type=int,
        default=None,
        help=(
            "Controller samples for each insertion correction. Defaults to "
            "--chunk-size, but can be smaller without shortening grasp/rejoin bridges."
        ),
    )
    parser.add_argument("--chunk-xy-step-m", type=float, default=0.015)
    parser.add_argument("--chunk-descent-m", type=float, default=0.004)
    # The current peg/hole geometry has zero nominal radial clearance.  A
    # millimetre-scale gate lets the peg touch the rim before the XY loop has
    # converged, so insertion recovery must use a sub-millimetre default.
    parser.add_argument("--descent-xy-gate-m", type=float, default=0.0005)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.025)
    parser.add_argument("--contact-force-rise-n", type=float, default=0.5)
    parser.add_argument("--success-xy-m", type=float, default=0.008)
    parser.add_argument("--success-peg-z-m", type=float, default=None)
    parser.add_argument("--rejoin-trigger-l2-rad", type=float, default=0.20)
    parser.add_argument(
        "--rejoin-physical-weight-rad-per-m",
        type=float,
        default=5.0,
        help="Weight for privileged peg-hole relative-XY mismatch in nearest-manifold search.",
    )
    parser.add_argument("--rejoin-release-l2-rad", type=float, default=0.06)
    parser.add_argument("--rejoin-physical-entry-l2-rad", type=float, default=0.12)
    parser.add_argument("--rejoin-release-gripper-rad", type=float, default=0.08)
    parser.add_argument("--rejoin-release-physical-m", type=float, default=0.008)
    parser.add_argument(
        "--rejoin-policy-alignment-entry-m",
        type=float,
        default=None,
        help=(
            "If set, suppress post-grasp expert takeover until the policy has "
            "first reached this peg-hole XY distance. A later manifold or "
            "alignment failure then records contact-local recovery instead of "
            "another generic approach/alignment suffix."
        ),
    )
    parser.add_argument(
        "--rejoin-policy-contact-force-n",
        type=float,
        default=None,
        help=(
            "After the optional policy-alignment entry has been reached, "
            "request expert takeover when measured translational force reaches "
            "this value, before it becomes a sustained overload."
        ),
    )
    parser.add_argument(
        "--rejoin-policy-alignment-stall-s",
        type=float,
        default=None,
        help=(
            "After entering the optional alignment band, request takeover if "
            "the policy makes no minimum downward progress for this duration."
        ),
    )
    parser.add_argument(
        "--rejoin-policy-alignment-min-descent-m",
        type=float,
        default=0.003,
        help="Downward peg progress that resets the aligned-hover stall timer.",
    )
    parser.add_argument("--rejoin-stall-s", type=float, default=6.0)
    parser.add_argument("--insertion-stall-s", type=float, default=30.0)
    parser.add_argument("--insertion-min-progress-m", type=float, default=0.001)
    parser.add_argument("--rejoin-lookahead-frames", type=int, default=4)
    parser.add_argument("--rejoin-local-window-frames", type=int, default=2)
    parser.add_argument("--rejoin-min-progress-frames", type=int, default=2)
    parser.add_argument("--rejoin-min-policy-events", type=int, default=2)
    parser.add_argument("--rejoin-grasp-stall-policy-events", type=int, default=30)
    parser.add_argument("--rejoin-policy-suffix-events", type=int, default=3)
    parser.add_argument("--rejoin-max-interventions", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--ack-timeout-s", type=float, default=30.0)
    parser.add_argument("--align-only", action="store_true")
    parser.add_argument("--grasp-center-z-offset-m", type=float, default=0.105)
    parser.add_argument("--grasp-pickup-step-m", type=float, default=0.015)
    parser.add_argument("--grasp-pickup-chunk-size", type=int, default=30)
    parser.add_argument("--traverse-z-offset-m", type=float, default=0.200)
    parser.add_argument("--grasp-xy-gate-m", type=float, default=0.008)
    parser.add_argument("--grasp-z-gate-m", type=float, default=0.006)
    parser.add_argument("--grasp-wrist-2-reset-trigger-rad", type=float, default=-1.545)
    parser.add_argument("--grasp-wrist-2-reset-target-rad", type=float, default=-1.570)
    parser.add_argument("--descent-step-m", type=float, default=0.012)
    parser.add_argument("--lift-step-m", type=float, default=0.012)
    # The schema requires 50 mm measured after takeover. Keep 15 mm margin for
    # asynchronous Gazebo pose sampling and the recorder's final grace window.
    parser.add_argument("--success-lift-m", type=float, default=0.065)
    parser.add_argument("--moveit-group", default="ur_manipulator")
    parser.add_argument("--ik-link", default="tool0")
    args = parser.parse_args()
    if args.alignment_chunk_size is None:
        args.alignment_chunk_size = args.chunk_size
    if args.insertion_chunk_size is None:
        args.insertion_chunk_size = args.chunk_size
    if args.outcome_file.exists():
        parser.error(f"refusing stale outcome file: {args.outcome_file}")
    if (
        args.chunk_size < 1
        or args.alignment_chunk_size < 1
        or args.insertion_chunk_size < 1
        or args.chunk_xy_step_m <= 0.0
        or args.chunk_descent_m <= 0.0
        or args.descent_xy_gate_m <= 0.0
        or args.max_joint_step_rad <= 0.0
        or args.ack_timeout_s <= 0.0
        or args.trigger_after_s < 0.0
        or args.grasp_trigger_xy_m <= 0.0
        or args.grasp_trigger_min_height_m <= args.grasp_center_z_offset_m
        or args.grasp_trigger_closed_rad <= 0.0
        or args.grasp_trigger_max_lift_m < 0.0
        or args.grasp_trigger_sustain_s <= 0.0
        or args.grasp_max_recoverable_drop_m <= 0.0
        or args.grasp_max_recoverable_xy_shift_m <= 0.0
        or args.grasp_reference_scene_tolerance_m <= 0.0
        or args.grasp_reference_bridge_steps < 1
        or args.grasp_reference_safe_open_steps < 1
        or args.grasp_reference_max_anchor_l2_rad <= 0.0
        or args.grasp_reference_gripper_weight < 0.0
        or not 0.0 < args.transport_trigger_open_rad < 0.8
        or args.transport_trigger_min_lift_m <= 0.0
        or args.transport_trigger_min_hole_distance_m <= args.transport_success_xy_m
        or args.transport_trigger_stall_s <= 0.0
        or args.transport_trigger_min_progress_m <= 0.0
        or args.transport_success_xy_m <= 0.0
        or args.grasp_probe_chunks < 1
        or args.grasp_probe_step_m <= 0.0
        or args.grasp_probe_success_lift_m <= 0.0
        or args.grasp_center_z_offset_m <= 0.0
        or args.traverse_z_offset_m <= args.grasp_center_z_offset_m
        or args.grasp_xy_gate_m <= 0.0
        or args.grasp_z_gate_m <= 0.0
        or not ARM_MIN[4] <= args.grasp_wrist_2_reset_target_rad <= ARM_MAX[4]
        or args.grasp_wrist_2_reset_target_rad >= args.grasp_wrist_2_reset_trigger_rad
        or args.descent_step_m <= 0.0
        or args.lift_step_m <= 0.0
        or args.success_lift_m <= 0.0
        or args.rejoin_trigger_l2_rad <= args.rejoin_release_l2_rad
        or args.rejoin_physical_weight_rad_per_m <= 0.0
        or args.rejoin_release_l2_rad <= 0.0
        or not args.rejoin_release_l2_rad < args.rejoin_physical_entry_l2_rad < args.rejoin_trigger_l2_rad
        or args.rejoin_release_gripper_rad <= 0.0
        or args.rejoin_release_physical_m <= 0.0
        or (
            args.rejoin_policy_alignment_stall_s is not None
            and args.rejoin_policy_alignment_stall_s <= 0.0
        )
        or args.rejoin_policy_alignment_min_descent_m <= 0.0
        or args.rejoin_stall_s <= 0.0
        or args.insertion_stall_s <= 0.0
        or args.insertion_min_progress_m <= 0.0
        or args.rejoin_lookahead_frames < 0
        or args.rejoin_local_window_frames < 0
        or args.rejoin_min_progress_frames < 1
        or args.rejoin_min_policy_events < 1
        or args.rejoin_grasp_stall_policy_events < args.rejoin_min_policy_events
        or args.rejoin_policy_suffix_events < 1
        or args.rejoin_max_interventions < 1
    ):
        parser.error("chunk size and step limits must be positive")

    if args.recovery_phase == "full_task":
        if not args.grasp_reference_episode:
            parser.error("full_task recovery requires --grasp-reference-episode")
        if args.success_peg_z_m is None:
            parser.error("full_task recovery requires --success-peg-z-m")
        ik_solver = MoveItRecoveryIK(group_name=args.moveit_group, link_name=args.ik_link)
        atexit.register(ik_solver.close)
        status = run_demonstration_multi_handoff_recovery(args, ik_solver)
        ik_solver.close()
        try:
            atexit.unregister(ik_solver.close)
        except Exception:
            pass
        return status

    ik_solver = MoveItRecoveryIK(group_name=args.moveit_group, link_name=args.ik_link)
    atexit.register(ik_solver.close)

    def close_ik() -> None:
        ik_solver.close()
        try:
            atexit.unregister(ik_solver.close)
        except Exception:
            pass

    if args.recovery_phase == "grasp_lift":
        status = (
            run_demonstration_grasp_lift_recovery(args, ik_solver)
            if args.grasp_reference_episode
            else run_grasp_lift_recovery(args, ik_solver)
        )
        close_ik()
        return status
    if args.recovery_phase == "transport":
        if not args.grasp_reference_episode:
            parser.error("transport recovery requires --grasp-reference-episode")
        status = run_demonstration_transport_recovery(args)
        close_ik()
        return status

    print("Waiting for policy to reach the privileged recovery trigger...", flush=True)
    deadline = time.monotonic() + args.timeout_s
    last_state_mtime = None
    sequence = 0
    takeover_active = False
    trigger_deadline = None
    baseline_force_norm = None
    progress_tracker = LocalPhaseProgress()
    while time.monotonic() < deadline:
        if not JOINT_STATE_FILE.exists() or not (args.session_dir / "session.json").exists():
            time.sleep(0.02)
            continue
        state_mtime = JOINT_STATE_FILE.stat().st_mtime_ns
        if state_mtime == last_state_mtime:
            time.sleep(0.005)
            continue
        last_state_mtime = state_mtime
        state = _read_state()
        force = _read_force()
        poses = read_gazebo_fixture_poses()
        if state is None or force is None or poses is None:
            continue
        peg, hole = poses
        xy_error = float(np.linalg.norm(peg[:2] - hole[:2]))
        if not takeover_active:
            if state[6] < 0.35 or peg[2] < args.trigger_min_peg_z_m or xy_error > args.trigger_xy_m:
                trigger_deadline = None
                continue
            if not _policy_rollin_started(args.session_dir):
                continue
            if trigger_deadline is None:
                trigger_deadline = time.monotonic() + args.trigger_after_s
                print(
                    f"Recovery corridor reached at xy={xy_error:.4f}m; "
                    f"waiting {args.trigger_after_s:.1f}s before takeover...",
                    flush=True,
                )
            if time.monotonic() < trigger_deadline:
                continue
            baseline_force_norm = float(np.linalg.norm(force[:3]))
        force_rise = 0.0 if baseline_force_norm is None else (
            float(np.linalg.norm(force[:3])) - baseline_force_norm
        )
        if takeover_active:
            geometry_success = (
                args.success_peg_z_m is not None
                and xy_error <= args.success_xy_m
                and peg[2] <= args.success_peg_z_m
            )
            contact_success = (
                xy_error <= args.success_xy_m
                and (args.align_only or force_rise >= args.contact_force_rise_n)
            )
            if contact_success or geometry_success:
                _atomic_outcome(
                    args.outcome_file,
                    "success",
                    f"xy={xy_error:.6f}, peg_z={peg[2]:.6f}, force_rise={force_rise:.3f}",
                )
                print(
                    f"Recovery success: xy={xy_error:.4f}m peg_z={peg[2]:.4f}m "
                    f"force_rise={force_rise:.2f}N"
                )
                close_ik()
                return 0

        chunk = build_moveit_expert_chunk(
            ik_solver,
            state,
            peg,
            hole,
            chunk_size=args.chunk_size,
            chunk_xy_step_m=args.chunk_xy_step_m,
            chunk_descent_m=args.chunk_descent_m,
            descent_xy_gate_m=args.descent_xy_gate_m,
            max_joint_step_rad=args.max_joint_step_rad,
            descend=not args.align_only,
            gripper_command=0.8,
        )
        if chunk is None:
            print(
                f"MoveIt IK rejected recovery target at xy={xy_error:.4f}m; waiting for next state",
                flush=True,
            )
            continue
        readiness = float(
            np.clip(1.0 - xy_error / max(args.trigger_xy_m, args.descent_xy_gate_m), 0.0, 1.0)
        )
        if args.align_only or xy_error > args.descent_xy_gate_m:
            progress_phase = 2  # shared "align" phase
            progress = readiness
        else:
            progress_phase = 3  # shared "interact" phase
            contact_readiness = float(
                np.clip(force_rise / max(args.contact_force_rise_n, 1e-6), 0.0, 1.0)
            )
            readiness = contact_readiness
            progress = contact_readiness
        progress, readiness = progress_tracker.update(progress_phase, progress)
        publish_expert_chunk(
            args.session_dir,
            chunk,
            sequence=sequence,
            publisher="privileged_cartesian_expert_v1",
            skill_progress_phase=progress_phase,
            skill_progress=progress,
            transition_readiness=readiness,
            label_confidence=1.0,
        )
        if not takeover_active:
            request_takeover(
                args.session_dir,
                trigger=f"policy peg-hole drift at xy={xy_error:.6f}m",
                requester="privileged_cartesian_expert_v1",
            )
            takeover_active = True
            print(
                f"Expert takeover requested at xy={xy_error:.4f}m peg_z={peg[2]:.4f}m",
                flush=True,
            )
        else:
            print(
                f"Expert chunk {sequence}: xy={xy_error:.4f}m peg_z={peg[2]:.4f}m "
                f"force_rise={force_rise:.2f}N",
                flush=True,
            )
        ack = wait_for_expert_execution_ack(
            args.session_dir,
            expert_sequence=sequence,
            timeout_s=args.ack_timeout_s,
        )
        if ack is None:
            _atomic_outcome(
                args.outcome_file,
                "failure",
                f"expert chunk {sequence} execution ACK timeout",
            )
            print(
                f"Recovery failed: no execution-complete ACK for expert chunk {sequence}.",
                flush=True,
            )
            close_ik()
            return 1
        sequence += 1

    if takeover_active:
        _atomic_outcome(args.outcome_file, "failure", "expert recovery timeout")
    print("Recovery expert timed out.", flush=True)
    close_ik()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

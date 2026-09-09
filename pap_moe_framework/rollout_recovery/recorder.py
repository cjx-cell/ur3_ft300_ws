"""Independent ROS recorder for rollout-failure expert recoveries."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformListener

from .protocol import (
    EXPERT_MODE,
    POLICY_MODE,
    PROTOCOL_VERSION,
    SKILL_PROGRESS_PHASES,
    load_expert_annotation,
    load_action_completion,
    load_action_event,
)
from .schema import (
    FAST_FORCE_SAMPLES,
    SCHEMA_VERSION,
    SLOW_FORCE_SAMPLES,
    STATE_HISTORY_SAMPLES,
    TRAJECTORY_SCOPE,
    validate_episode,
)


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
ALL_JOINTS = ARM_JOINTS + (GRIPPER_JOINT,)
IMAGE_SIZE = (224, 224)
POSE_TOPIC = "/world/simulation_world/pose/info"
PHASE_SUBTASK = {
    "grasp_lift": "grasp the peg",
    "transport": "transport to the hole",
    "align_precontact": "approach and align with the hole",
    "contact": "recover contact and relocate the hole",
    "insertion": "insert the peg into the hole",
    "full_task": "full peg-in-hole task",
}
PHASE_STAGE = {
    "grasp_lift": (1.0, 0.0, 0.0, 0.0),
    "transport": (1.0, 0.0, 0.0, 0.0),
    "align_precontact": (1.0, 0.0, 0.0, 0.0),
    "contact": (0.0, 0.0, 1.0, 0.0),
    "insertion": (0.0, 0.0, 0.0, 1.0),
    "full_task": (1.0, 0.0, 0.0, 0.0),
}

# Full recovery-assisted episodes currently take roughly 190 wall-clock
# seconds.  The old 12k callback ring could evict the causal state/force
# samples belonging to the policy prefix before build_episode() ran.  Keep a
# generous 20-minute buffer at 100 Hz; numeric history is small compared with
# the synchronized RGB frames already retained for the episode.
NATIVE_HISTORY_MAX_SAMPLES = 120_000


def _visual_quality(camera0: np.ndarray, camera1: np.ndarray) -> np.ndarray:
    cameras = np.stack([camera0, camera1]).astype(np.float32) / 255.0
    gray = cameras.mean(axis=-1)
    finite = bool(np.isfinite(cameras).all())
    black = float(np.mean(gray <= 0.02)) if finite else 1.0
    saturated = float(np.mean(gray >= 0.98)) if finite else 1.0
    contrast = float(np.mean(np.std(gray, axis=(1, 2)))) if finite else 0.0
    return np.asarray(
        [black, saturated, contrast, float(finite and contrast >= 0.01)],
        dtype=np.float32,
    )


def _causal_windows(
    values: np.ndarray,
    sample_timestamps: np.ndarray,
    anchor_timestamps: np.ndarray,
    *,
    count: int,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample raw callback data without ever selecting a future sample."""
    offsets = np.linspace(-window_s, 0.0, count, dtype=np.float64)
    windows = np.empty((len(anchor_timestamps), count, values.shape[1]), dtype=np.float32)
    source_timestamps = np.empty((len(anchor_timestamps), count), dtype=np.float64)
    for frame, anchor in enumerate(anchor_timestamps):
        latest = int(np.searchsorted(sample_timestamps, anchor, side="right") - 1)
        if latest < 0:
            raise ValueError("history has no causal sample at an episode frame")
        indices = np.searchsorted(
            sample_timestamps[: latest + 1], anchor + offsets, side="right"
        ) - 1
        indices = np.clip(indices, 0, latest)
        windows[frame] = values[indices]
        source_timestamps[frame] = sample_timestamps[indices]
    return windows, source_timestamps


@dataclass(frozen=True)
class TimelinePoint:
    timestamp: float
    mode: int
    policy_action: np.ndarray
    expert_action: np.ndarray
    executed_action: np.ndarray
    requested_action: np.ndarray
    controller_action: np.ndarray
    skill_progress_phase: int
    skill_progress: float
    transition_readiness: float
    skill_progress_valid: bool
    skill_progress_label_source: str
    skill_progress_confidence: float


class ActionTimeline:
    def __init__(self, session_dir: Path):
        self.event_dir = Path(session_dir) / "action_events"
        self._loaded_sequences: set[int] = set()
        self._points: list[TimelinePoint] = []

    def refresh(self) -> None:
        for path in sorted(self.event_dir.glob("event_*.npz")):
            event = load_action_event(path)
            if event.sequence in self._loaded_sequences:
                continue
            completion_path = (
                self.event_dir.parent
                / "completion_events"
                / f"completion_{event.sequence:08d}.json"
            )
            # Do not commit predicted wall timestamps for an in-flight event.
            # Gazebo may run slower than real time; completion makes the wall
            # timeline authoritative and immutable.
            if not completion_path.exists():
                break
            completion = load_action_completion(completion_path)
            if (
                completion.action_event_sequence != event.sequence
                or completion.mode != event.mode
                or completion.source_sequence != event.source_sequence
                or completion.completed_timestamp <= event.dispatch_timestamp
            ):
                raise ValueError(f"action completion does not match event {event.sequence}")
            if self._loaded_sequences and event.sequence != max(self._loaded_sequences) + 1:
                raise ValueError(
                    f"non-contiguous action-event sequence at {event.sequence}"
                )
            actual_timestamps = event.dispatch_timestamp + (
                completion.completed_timestamp - event.dispatch_timestamp
            ) * np.arange(1, len(event.action_timestamp) + 1, dtype=np.float64) / len(
                event.action_timestamp
            )
            annotation = None
            if event.mode == EXPERT_MODE:
                annotation = load_expert_annotation(self.event_dir.parent, event.source_sequence)
            for index, timestamp in enumerate(actual_timestamps):
                self._points.append(
                    TimelinePoint(
                        timestamp=float(timestamp),
                        mode=event.mode,
                        policy_action=event.policy_action[index].copy(),
                        expert_action=event.expert_action[index].copy(),
                        executed_action=event.executed_action[index].copy(),
                        requested_action=event.requested_action[index].copy(),
                        controller_action=event.controller_action[index].copy(),
                        skill_progress_phase=(
                            annotation.skill_progress_phase if annotation is not None else 7
                        ),
                        skill_progress=(annotation.skill_progress if annotation is not None else 0.0),
                        transition_readiness=(
                            annotation.transition_readiness if annotation is not None else 0.0
                        ),
                        skill_progress_valid=bool(
                            annotation is not None and annotation.label_confidence > 0.0
                        ),
                        skill_progress_label_source=(
                            annotation.publisher if annotation is not None else "unlabelled_policy_rollin"
                        ),
                        skill_progress_confidence=(
                            annotation.label_confidence if annotation is not None else 0.0
                        ),
                    )
                )
            self._loaded_sequences.add(event.sequence)
        self._points.sort(key=lambda point: point.timestamp)

    def closest(self, timestamp: float, *, max_skew_s: float) -> TimelinePoint | None:
        self.refresh()
        if not self._points:
            return None
        times = np.fromiter((point.timestamp for point in self._points), dtype=np.float64)
        index = int(np.argmin(np.abs(times - timestamp)))
        point = self._points[index]
        return point if abs(point.timestamp - timestamp) <= max_skew_s else None


def _parse_model_position(text: str, name: str) -> np.ndarray | None:
    index = text.find(f'name: "{name}"')
    if index < 0:
        return None
    match = re.search(
        r"position\s*\{\s*x:\s*([-+\d.eE]+)\s*"
        r"y:\s*([-+\d.eE]+)\s*z:\s*([-+\d.eE]+)",
        text[index : index + 350],
    )
    if match is None:
        return None
    return np.asarray([float(value) for value in match.groups()], dtype=np.float32)


def read_gazebo_fixture_poses() -> tuple[np.ndarray, np.ndarray] | None:
    try:
        result = subprocess.run(
            ["ign", "topic", "-t", POSE_TOPIC, "-e", "-n", "1"],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    text = result.stdout.decode(errors="replace")
    peg = _parse_model_position(text, "peg")
    hole = _parse_model_position(text, "hole_plate")
    return None if peg is None or hole is None else (peg, hole)


class RolloutRecoveryRecorder(Node):
    def __init__(
        self,
        *,
        session_dir: Path,
        outcome_file: Path,
        hz: float,
        max_sensor_skew_s: float,
        max_action_skew_s: float,
        max_pose_age_s: float,
    ):
        super().__init__("pap_moe_rollout_recovery_recorder")
        self.session_dir = Path(session_dir)
        self.outcome_file = Path(outcome_file)
        self.max_sensor_skew_s = max_sensor_skew_s
        self.max_action_skew_s = max_action_skew_s
        self.max_pose_age_s = max_pose_age_s
        self.timeline = ActionTimeline(self.session_dir)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.stop_pose_thread = threading.Event()
        self.finished = threading.Event()
        self.error: Exception | None = None
        self.outcome: dict[str, Any] | None = None
        self.outcome_seen_monotonic: float | None = None
        self.outcome_grace_s = 1.0

        self.state: np.ndarray | None = None
        self.state_timestamp = np.nan
        self.camera0: np.ndarray | None = None
        self.camera0_timestamp = np.nan
        self.camera1: np.ndarray | None = None
        self.camera1_timestamp = np.nan
        self.force: np.ndarray | None = None
        self.force_timestamp = np.nan
        self.peg_position: np.ndarray | None = None
        self.hole_position: np.ndarray | None = None
        self.pose_timestamp = np.nan
        self.last_recorded_state_timestamp = -np.inf
        self.frames: list[dict[str, Any]] = []
        # Preserve native callback streams. Histories are built from these raw,
        # causal samples, not reconstructed from the lower-rate recorded frames.
        self.state_samples: deque[tuple[float, np.ndarray]] = deque(
            maxlen=NATIVE_HISTORY_MAX_SAMPLES
        )
        self.force_samples: deque[tuple[float, np.ndarray]] = deque(
            maxlen=NATIVE_HISTORY_MAX_SAMPLES
        )

        callback_group = ReentrantCallbackGroup()
        self.create_subscription(
            JointState, "/joint_states", self._joint_state, 20, callback_group=callback_group
        )
        self.create_subscription(
            Image,
            "/wrist_camera/color/image_raw",
            self._camera0,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            Image,
            "/global_camera/color/image_raw",
            self._camera1,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            WrenchStamped,
            "/force_torque_sensor_broadcaster/wrench",
            self._wrench,
            50,
            callback_group=callback_group,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(1.0 / hz, self._sample, callback_group=callback_group)
        self.pose_thread = threading.Thread(target=self._pose_loop, daemon=True)
        self.pose_thread.start()

    def close(self) -> None:
        self.stop_pose_thread.set()
        self.pose_thread.join(timeout=2.5)

    def _joint_state(self, message: JointState) -> None:
        if not all(name in message.name for name in ALL_JOINTS):
            return
        state = np.asarray(
            [message.position[message.name.index(name)] for name in ALL_JOINTS],
            dtype=np.float32,
        )
        with self.lock:
            self.state = state
            self.state_timestamp = time.time()
            self.state_samples.append((self.state_timestamp, state.copy()))

    def _decode_image(self, message: Image) -> np.ndarray:
        bgr = self.bridge.imgmsg_to_cv2(message, "bgr8")
        return cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8)

    def _camera0(self, message: Image) -> None:
        image = self._decode_image(message)
        with self.lock:
            self.camera0 = image
            self.camera0_timestamp = time.time()

    def _camera1(self, message: Image) -> None:
        image = self._decode_image(message)
        with self.lock:
            self.camera1 = image
            self.camera1_timestamp = time.time()

    def _wrench(self, message: WrenchStamped) -> None:
        wrench = message.wrench
        value = np.asarray(
            [
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                wrench.torque.x,
                wrench.torque.y,
                wrench.torque.z,
            ],
            dtype=np.float32,
        )
        with self.lock:
            self.force = value
            self.force_timestamp = time.time()
            self.force_samples.append((self.force_timestamp, value.copy()))

    def _pose_loop(self) -> None:
        while not self.stop_pose_thread.is_set():
            poses = read_gazebo_fixture_poses()
            if poses is not None:
                with self.lock:
                    self.peg_position, self.hole_position = poses
                    self.pose_timestamp = time.time()
            self.stop_pose_thread.wait(0.2)

    def _gripper_position(self) -> np.ndarray | None:
        positions = []
        for frame in (
            "robotiq_85_left_finger_tip_link",
            "robotiq_85_right_finger_tip_link",
        ):
            try:
                transform = self.tf_buffer.lookup_transform("world", frame, rclpy.time.Time())
            except Exception:
                return None
            translation = transform.transform.translation
            positions.append([translation.x, translation.y, translation.z])
        return np.mean(np.asarray(positions, dtype=np.float32), axis=0)

    @staticmethod
    def _pose_confirms_grasp(
        state: np.ndarray, peg_position: np.ndarray, gripper_position: np.ndarray
    ) -> bool:
        distance_xy = float(np.linalg.norm(peg_position[:2] - gripper_position[:2]))
        distance_z = float(gripper_position[2] - peg_position[2])
        # Match the controller's measured fingertip-midpoint window. The long
        # tapered peg can be stably held off-axis by about 38 mm; the old
        # 20 mm recorder-only threshold rejected visible, load-bearing grasps.
        return bool(state[6] >= 0.35 and distance_xy <= 0.040 and 0.070 <= distance_z <= 0.140)

    def _read_outcome(self) -> dict[str, Any] | None:
        if not self.outcome_file.exists():
            return None
        with self.outcome_file.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("outcome protocol version mismatch")
        if payload.get("outcome") not in ("success", "failure"):
            raise ValueError("outcome must be success or failure")
        return payload

    def poll_outcome_wall_clock(self) -> None:
        """Finish after an explicit outcome even if Gazebo's ROS clock stops."""
        if self.finished.is_set():
            return
        try:
            pending_outcome = self._read_outcome()
            if pending_outcome is not None and self.outcome is None:
                self.outcome = pending_outcome
                self.outcome_seen_monotonic = time.monotonic()
            if (
                self.outcome is not None
                and self.outcome_seen_monotonic is not None
                and time.monotonic() - self.outcome_seen_monotonic >= self.outcome_grace_s
            ):
                self.finished.set()
        except Exception as error:
            self.error = error
            self.finished.set()

    def _sample(self) -> None:
        if self.finished.is_set():
            return
        try:
            self.poll_outcome_wall_clock()
            with self.lock:
                snapshot = {
                    "state": None if self.state is None else self.state.copy(),
                    "timestamp": float(self.state_timestamp),
                    "camera0": None if self.camera0 is None else self.camera0.copy(),
                    "camera0_timestamp": float(self.camera0_timestamp),
                    "camera1": None if self.camera1 is None else self.camera1.copy(),
                    "camera1_timestamp": float(self.camera1_timestamp),
                    "force": None if self.force is None else self.force.copy(),
                    "force_timestamp": float(self.force_timestamp),
                    "peg_position": (
                        None if self.peg_position is None else self.peg_position.copy()
                    ),
                    "hole_position": (
                        None if self.hole_position is None else self.hole_position.copy()
                    ),
                    "pose_timestamp": float(self.pose_timestamp),
                }
                histories_warm = (
                    len(self.state_samples) >= STATE_HISTORY_SAMPLES
                    and self.state_samples[-1][0] - self.state_samples[0][0] >= 1.0
                    and len(self.force_samples) >= max(FAST_FORCE_SAMPLES, SLOW_FORCE_SAMPLES)
                    and self.force_samples[-1][0] - self.force_samples[0][0] >= 5.0
                )
            timestamp = snapshot["timestamp"]
            if not np.isfinite(timestamp) or timestamp <= self.last_recorded_state_timestamp:
                return
            if any(
                snapshot[key] is None
                for key in ("state", "camera0", "camera1", "force", "peg_position", "hole_position")
            ):
                return
            if not histories_warm:
                return
            sensor_skews = (
                abs(snapshot["camera0_timestamp"] - timestamp),
                abs(snapshot["camera1_timestamp"] - timestamp),
                abs(snapshot["force_timestamp"] - timestamp),
            )
            if max(sensor_skews) > self.max_sensor_skew_s:
                return
            if time.time() - snapshot["pose_timestamp"] > self.max_pose_age_s:
                return
            gripper_position = self._gripper_position()
            if gripper_position is None:
                return
            pose_grasped = self._pose_confirms_grasp(
                snapshot["state"], snapshot["peg_position"], gripper_position
            )
            self.frames.append(
                {
                    **snapshot,
                    "gripper_position": gripper_position,
                    "peg_attached": pose_grasped,
                }
            )
            self.last_recorded_state_timestamp = timestamp
        except Exception as error:
            self.error = error
            self.finished.set()
        finally:
            self.poll_outcome_wall_clock()

    def build_episode(
        self,
        *,
        source_policy_checkpoint: str,
        episode_id: str,
        recovery_phase: str,
        pre_takeover_context_s: float | None = None,
    ) -> dict[str, np.ndarray]:
        if self.outcome is None:
            raise ValueError("no explicit recovery outcome was received")
        if not self.frames:
            raise ValueError("no synchronized recovery frames were recorded")
        control_events = sorted((self.session_dir / "control_events").glob("control_*.json"))
        if not control_events:
            raise ValueError("no expert intervention control event was recorded")
        takeover = None
        for control_path in control_events:
            with control_path.open(encoding="utf-8") as stream:
                candidate = json.load(stream)
            if int(candidate.get("mode", POLICY_MODE)) == EXPERT_MODE:
                takeover = candidate
                break
        if takeover is None:
            raise ValueError("no expert takeover was recorded")

        # Sensor snapshots are buffered independently while a trajectory is in
        # flight.  Completion events arrive only after execution; align here so
        # slow-than-real-time Gazebo does not discard every intermediate frame.
        self.timeline.refresh()
        aligned_frames: list[dict[str, Any]] = []
        for sensor_frame in self.frames:
            action = self.timeline.closest(
                float(sensor_frame["timestamp"]), max_skew_s=self.max_action_skew_s
            )
            if action is None:
                continue
            aligned_frames.append(
                {
                    **sensor_frame,
                    "control_mode": action.mode,
                    "policy_action": action.policy_action,
                    "expert_action": action.expert_action,
                    "executed_action": action.executed_action,
                    "requested_action": action.requested_action,
                    "controller_action": action.controller_action,
                    "action_timestamp": action.timestamp,
                    "skill_progress_phase": action.skill_progress_phase,
                    "skill_progress": action.skill_progress,
                    "transition_readiness": action.transition_readiness,
                    "skill_progress_valid": action.skill_progress_valid,
                    "skill_progress_label_source": action.skill_progress_label_source,
                    "skill_progress_confidence": action.skill_progress_confidence,
                }
            )
        if not aligned_frames:
            raise ValueError("no buffered sensor frames align to completed action events")
        if pre_takeover_context_s is not None:
            if pre_takeover_context_s <= 0.0:
                raise ValueError("pre_takeover_context_s must be positive")
            takeover_timestamp = float(takeover["timestamp"])
            context_start = takeover_timestamp - pre_takeover_context_s
            time_start_index = next(
                (
                    index
                    for index, frame in enumerate(aligned_frames)
                    if float(frame["timestamp"]) >= context_start
                ),
                len(aligned_frames),
            )
            expert_start_index = next(
                (
                    index
                    for index, frame in enumerate(aligned_frames)
                    if int(frame["control_mode"]) == EXPERT_MODE
                ),
                len(aligned_frames),
            )
            if expert_start_index == len(aligned_frames):
                raise ValueError("no expert-aligned frames were recorded after takeover")
            policy_indices = [
                index
                for index, frame in enumerate(aligned_frames[:expert_start_index])
                if int(frame["control_mode"]) == POLICY_MODE
            ]
            # A short failure-local time window can contain fewer than the
            # schema's three policy frames because action completions and
            # sensor callbacks are asynchronous.  Preserve the last three
            # genuine policy-aligned frames without extending the episode all
            # the way back through the successful prefix.
            required_policy_start = (
                policy_indices[-3] if len(policy_indices) >= 3 else 0
            )
            start_index = min(time_start_index, required_policy_start)
            aligned_frames = aligned_frames[start_index:]
            if not aligned_frames:
                raise ValueError("no action-aligned frames remain in takeover context window")

        modes = np.asarray([frame["control_mode"] for frame in aligned_frames], dtype=np.int8)
        timestamps = np.asarray(
            [frame["timestamp"] for frame in aligned_frames], dtype=np.float64
        )
        with self.lock:
            state_samples = list(self.state_samples)
            force_samples = list(self.force_samples)
        if not state_samples or not force_samples:
            raise ValueError("native state/FT300 callback history is unavailable")
        state_sample_ts = np.asarray([item[0] for item in state_samples], dtype=np.float64)
        state_sample_values = np.stack([item[1] for item in state_samples]).astype(np.float32)
        force_sample_ts = np.asarray([item[0] for item in force_samples], dtype=np.float64)
        force_sample_values = np.stack([item[1] for item in force_samples]).astype(np.float32)
        state_history, state_history_timestamp = _causal_windows(
            state_sample_values,
            state_sample_ts,
            timestamps,
            count=STATE_HISTORY_SAMPLES,
            window_s=1.0,
        )
        force_fast, force_fast_timestamp = _causal_windows(
            force_sample_values,
            force_sample_ts,
            timestamps,
            count=FAST_FORCE_SAMPLES,
            window_s=0.64,
        )
        force_slow, force_slow_timestamp = _causal_windows(
            force_sample_values,
            force_sample_ts,
            timestamps,
            count=SLOW_FORCE_SAMPLES,
            window_s=5.0,
        )
        # Gazebo can briefly stall and then deliver native sensor callbacks in
        # a wall-clock burst. Such a frame is synchronized to a fresh wrench,
        # but its causal resampling window can still contain repeated padding.
        # Never fabricate timestamps or weaken the schema: discard only the
        # affected synchronized frames before materializing every modality.
        history_complete = (
            (state_history_timestamp[:, -1] - state_history_timestamp[:, 0] >= 0.90)
            & (force_fast_timestamp[:, -1] - force_fast_timestamp[:, 0] >= 0.55)
            & (force_slow_timestamp[:, -1] - force_slow_timestamp[:, 0] >= 4.50)
        )
        if not np.all(history_complete):
            aligned_frames = [
                frame for frame, keep in zip(aligned_frames, history_complete, strict=True) if keep
            ]
            state_history = state_history[history_complete]
            state_history_timestamp = state_history_timestamp[history_complete]
            force_fast = force_fast[history_complete]
            force_fast_timestamp = force_fast_timestamp[history_complete]
            force_slow = force_slow[history_complete]
            force_slow_timestamp = force_slow_timestamp[history_complete]
            if not aligned_frames:
                raise ValueError("all synchronized frames have incomplete native sensor history")
            modes = np.asarray(
                [frame["control_mode"] for frame in aligned_frames], dtype=np.int8
            )
            timestamps = np.asarray(
                [frame["timestamp"] for frame in aligned_frames], dtype=np.float64
            )
        visual_quality = np.stack(
            [_visual_quality(frame["camera0"], frame["camera1"]) for frame in aligned_frames]
        )
        # Physics supervision is observation-derived, not a fixed synonym for
        # the semantic task.  This keeps the Physics Gate distinct from the
        # Subtask/Skill-Progress heads and avoids Gazebo pose truth as input.
        force_values = np.stack([frame["force"] for frame in aligned_frames]).astype(np.float32)
        baseline_force = np.median(force_values[: min(10, len(force_values)), :3], axis=0)
        force_delta = np.linalg.norm(force_values[:, :3] - baseline_force[None, :], axis=1)
        contact = force_delta >= 0.5
        stage = np.zeros((len(aligned_frames), 4), dtype=np.float32)
        stage[:, 0] = 1.0
        # During a complete episode, contact type follows observed motion:
        # sustained contact with joint motion is movable/compliant (E4), while
        # contact without motion is rigid/stuck (E3). No object truth is used.
        state_values = np.stack([frame["state"] for frame in aligned_frames]).astype(np.float32)
        state_motion = np.linalg.norm(
            np.diff(state_values[:, :6], axis=0, prepend=state_values[:1, :6]), axis=1
        )
        if recovery_phase == "full_task":
            compliant = contact & (state_motion >= 2e-3)
            rigid = contact & ~compliant
        else:
            rigid = contact & (recovery_phase != "insertion")
            compliant = contact & (recovery_phase == "insertion")
        stage[rigid] = np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float32)
        stage[compliant] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        if recovery_phase == "full_task":
            peg_values = np.stack([frame["peg_position"] for frame in aligned_frames])
            hole_values = np.stack([frame["hole_position"] for frame in aligned_frames])
            peg_hole_xy = np.linalg.norm(peg_values[:, :2] - hole_values[:, :2], axis=1)
            peg_lift = peg_values[:, 2] - peg_values[0, 2]
            closed = state_values[:, 6] >= 0.35
            semantic_subtask = np.full(
                len(aligned_frames), "grasp the peg", dtype="<U40"
            )
            semantic_subtask[closed & (peg_lift >= 0.03) & (peg_hole_xy > 0.06)] = (
                "transport to the hole"
            )
            semantic_subtask[closed & (peg_hole_xy <= 0.06) & (peg_values[:, 2] > 0.89)] = (
                "approach and align with the hole"
            )
            semantic_subtask[closed & (peg_hole_xy <= 0.012) & (peg_values[:, 2] <= 0.89)] = (
                "insert the peg into the hole"
            )
        else:
            semantic_subtask = np.asarray(
                [PHASE_SUBTASK[recovery_phase]] * len(aligned_frames), dtype=np.str_
            )
        skill_progress_phase = np.asarray(
            [frame["skill_progress_phase"] for frame in aligned_frames], dtype=np.int64
        )
        action_timestamp = np.asarray(
            [frame["action_timestamp"] for frame in aligned_frames], dtype=np.float64
        )
        stack = lambda key: np.stack([frame[key] for frame in aligned_frames], axis=0)
        values = {
            "schema_version": np.asarray(SCHEMA_VERSION),
            "trajectory_scope": np.asarray(TRAJECTORY_SCOPE),
            "source_policy_checkpoint": np.asarray(source_policy_checkpoint),
            "recovery_trigger": np.asarray(str(takeover["trigger"])),
            "episode_id": np.asarray(episode_id),
            "recovery_phase": np.asarray(recovery_phase),
            "recovery_outcome": np.asarray(str(self.outcome["outcome"])),
            "state": stack("state").astype(np.float32),
            "state_history": state_history,
            "state_history_timestamp": state_history_timestamp,
            "policy_action": stack("policy_action").astype(np.float32),
            "expert_action": stack("expert_action").astype(np.float32),
            "executed_action": stack("executed_action").astype(np.float32),
            "requested_action": stack("requested_action").astype(np.float32),
            "controller_action": stack("controller_action").astype(np.float32),
            "control_mode": modes,
            "intervention_mask": modes == EXPERT_MODE,
            "timestamp": timestamps,
            "policy_action_timestamp": np.where(modes == POLICY_MODE, action_timestamp, np.nan),
            "expert_action_timestamp": np.where(modes == EXPERT_MODE, action_timestamp, np.nan),
            "executed_action_timestamp": action_timestamp,
            "camera0_timestamp": np.asarray(
                [frame["camera0_timestamp"] for frame in aligned_frames], dtype=np.float64
            ),
            "camera1_timestamp": np.asarray(
                [frame["camera1_timestamp"] for frame in aligned_frames], dtype=np.float64
            ),
            "force_timestamp": np.asarray(
                [frame["force_timestamp"] for frame in aligned_frames], dtype=np.float64
            ),
            "pose_timestamp": np.asarray(
                [frame["pose_timestamp"] for frame in aligned_frames], dtype=np.float64
            ),
            "camera0": stack("camera0").astype(np.uint8),
            "camera1": stack("camera1").astype(np.uint8),
            "force": stack("force").astype(np.float32),
            "force_fast": force_fast,
            "force_fast_timestamp": force_fast_timestamp,
            "force_slow": force_slow,
            "force_slow_timestamp": force_slow_timestamp,
            "visual_quality": visual_quality,
            "stage": stage,
            "modality_validity": np.ones((len(aligned_frames), 7), dtype=np.float32),
            "semantic_subtask": semantic_subtask,
            "skill_progress_phase": skill_progress_phase,
            "skill_progress_phase_name": np.asarray(
                [SKILL_PROGRESS_PHASES[index] for index in skill_progress_phase], dtype=np.str_
            ),
            "skill_progress": np.asarray(
                [frame["skill_progress"] for frame in aligned_frames], dtype=np.float32
            ),
            "transition_readiness": np.asarray(
                [frame["transition_readiness"] for frame in aligned_frames], dtype=np.float32
            ),
            "skill_progress_valid": np.asarray(
                [frame["skill_progress_valid"] for frame in aligned_frames], dtype=bool
            ),
            "skill_progress_label_source": np.asarray(
                [frame["skill_progress_label_source"] for frame in aligned_frames], dtype=np.str_
            ),
            "skill_progress_confidence": np.asarray(
                [frame["skill_progress_confidence"] for frame in aligned_frames], dtype=np.float32
            ),
            "peg_attached": np.asarray(
                [frame["peg_attached"] for frame in aligned_frames], dtype=bool
            ),
            "peg_position": stack("peg_position").astype(np.float32),
            "hole_position": stack("hole_position").astype(np.float32),
            "gripper_position": stack("gripper_position").astype(np.float32),
        }
        return values


def save_validated_episode(path: Path, values: dict[str, np.ndarray]) -> None:
    require_success = str(np.asarray(values["recovery_outcome"]).item()) == "success"
    recovery_phase = str(np.asarray(values["recovery_phase"]).item())
    intervention = np.asarray(values["intervention_mask"], dtype=bool)
    force_norm = np.linalg.norm(np.asarray(values["force"])[:, :3], axis=1)
    # Persist the audit mask explicitly.  It is context metadata, not an
    # action-training weight: policy actions remain excluded by intervention_mask.
    values["policy_failure_force_overload_mask"] = (
        (~intervention) & (force_norm > 80.0)
    )
    validate_episode(
        values,
        require_success=require_success,
        require_full_modalities=True,
        allow_policy_failure_force_context=(recovery_phase == "full_task"),
    )
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite recovery episode: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)

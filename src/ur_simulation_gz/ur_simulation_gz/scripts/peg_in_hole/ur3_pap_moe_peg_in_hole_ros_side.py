#!/usr/bin/env python3
"""PAP-MoE v6 ROS-side entry point for UR3 peg-in-hole evaluation.

The policy replans once per second but consumes a fixed multi-rate snapshot:

- action / policy frame rate: 10 Hz;
- latest 64 native force samples (about one second in the current Gazebo);
- latest 50 force samples resampled at 10 Hz (five seconds);
- latest 10 joint states resampled at 10 Hz (one second);
- four deterministic image-quality values.
"""

import json
import os
import time
from collections import deque

import numpy as np
from ros_gz_interfaces.msg import Contacts
from pap_moe_online_observation import (
    calibrate_wrench_window,
    estimate_wrench_bias,
    is_settled_gripper_close,
    map_gripper_history_to_training_units,
)
from ur3_peg_in_hole_ros_side_base import PegInHoleROSSide, run_ros_side

FORCE_FAST_FILE = "/tmp/ur3_force_fast.npy"
FORCE_SLOW_FILE = "/tmp/ur3_force_slow.npy"
STATE_HISTORY_FILE = "/tmp/ur3_state_history.npy"
VISUAL_QUALITY_FILE = "/tmp/ur3_visual_quality.npy"
OBSERVATION_META_FILE = "/tmp/ur3_pap_moe_observation_meta.json"

FAST_FORCE_SAMPLES = 64
SLOW_FORCE_SAMPLES = 50
STATE_HISTORY_SAMPLES = 10
RESAMPLE_PERIOD_S = 0.1
FORCE_FILTER_SAMPLES = 11
EMPTY_BIAS_SAMPLES = 64
PAYLOAD_BIAS_SAMPLES = 20
# Detect a settled close from gripper motion rather than an object-specific
# finger angle.  This keeps the force-reference contract usable for objects
# that stop the 2F-85 at different widths.
PAYLOAD_GRIPPER_MIN_CLOSED_RAD = 0.20
PAYLOAD_GRIPPER_RELEASE_RAD = 0.15
PAYLOAD_GRIPPER_STABLE_SAMPLES = 5
PAYLOAD_GRIPPER_STABLE_RANGE_RAD = 0.01
DART_IMPULSE_REJECT_N = 30.0
CONTACT_TRUTH_TIMEOUT_S = 0.25
CONTACT_TRUTH_TOPICS = (
    "/pap_moe/peg_body_contacts",
    "/pap_moe/peg_handle_contacts",
    "/pap_moe/hole_side_contacts",
    "/pap_moe/hole_floor_contacts",
)


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as file:
        np.save(file, array)
    os.replace(temporary, path)


def _atomic_write_json(path: str, value: dict) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def _left_pad_window(values: list[np.ndarray], size: int, width: int) -> np.ndarray:
    """Return a fixed window, padding startup history with the oldest sample."""
    if not values:
        return np.zeros((size, width), dtype=np.float32)
    selected = [np.asarray(value, dtype=np.float32) for value in values[-size:]]
    if len(selected) < size:
        selected = [selected[0]] * (size - len(selected)) + selected
    return np.stack(selected, axis=0)


def _left_pad_flags(values: list[bool], size: int) -> np.ndarray:
    if not values:
        return np.zeros(size, dtype=bool)
    selected = [bool(value) for value in values[-size:]]
    if len(selected) < size:
        selected = [selected[0]] * (size - len(selected)) + selected
    return np.asarray(selected, dtype=bool)


def compute_visual_quality(camera0: np.ndarray, camera1: np.ndarray) -> np.ndarray:
    """Return [black_fraction, saturated_fraction, contrast, valid]."""
    cameras = np.stack([camera0, camera1], axis=0).astype(np.float32, copy=False)
    gray = cameras.mean(axis=-1)
    finite = bool(np.isfinite(cameras).all())
    black_fraction = float(np.mean(gray <= 0.02)) if finite else 1.0
    saturated_fraction = float(np.mean(gray >= 0.98)) if finite else 1.0
    contrast = float(np.mean(np.std(gray, axis=(1, 2)))) if finite else 0.0
    valid = float(
        finite and camera0.shape == (224, 224, 3) and camera1.shape == (224, 224, 3) and contrast >= 0.01
    )
    return np.asarray(
        [black_fraction, saturated_fraction, contrast, valid],
        dtype=np.float32,
    )


class PAPMoEROSSide(PegInHoleROSSide):
    GRIPPER_ACTION_MODE = "continuous_radians"
    STATE_HISTORY_GRIPPER_MODE = "continuous_radians_0_0.8"
    """PAP-MoE controller and asynchronous multi-rate observation producer."""

    CONTROL_HZ = 10
    ACTION_DT_S = 0.1
    REPLAN_INTERVAL_S = 1.0
    ACTION_CHUNK_SIZE = 10
    # Match the causal rate limit used by the privileged recovery expert.  It
    # is a task-agnostic actuator safety contract and can be overridden for a
    # controlled experiment without changing policy weights.
    ACTION_CHUNK_MAX_STEP_RAD = float(
        os.environ.get("POLICY_ACTION_CHUNK_MAX_STEP_RAD", "0.025")
    )
    MAX_EPISODE_DURATION_S = float(
        os.environ.get("PAP_MOE_MAX_EPISODE_DURATION_S", "100.0")
    )
    ENABLE_DETACHABLE_JOINT = False
    RESET_AFTER_EPISODE = False
    ENABLE_GAZEBO_SUCCESS_CHECK = True
    SUCCESS_MAX_XY_M = float(os.environ.get("PAP_MOE_SUCCESS_MAX_XY_M", "0.008"))
    SUCCESS_MAX_PEG_Z_M = float(
        os.environ.get("PAP_MOE_SUCCESS_MAX_PEG_Z_M", "0.905")
    )
    SUCCESS_REQUIRED_CHECKS = int(
        os.environ.get("PAP_MOE_SUCCESS_REQUIRED_CHECKS", "5")
    )
    FIRST_ACTION_TIMEOUT_S = 30.0
    ATTACH_MAX_XY_M = 0.04
    ATTACH_MAX_Z_M = 0.04
    # Match the successful scene15001 baseline camera/kinematic reset exactly.
    START_POSE = (-0.00001, -1.57015, 1.56995, -1.56993, -1.56995, -0.00005, 0.0)
    START_POSE_TOLERANCE_RAD = 0.001
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

    def __init__(self, *args, **kwargs):
        self._force_native = deque(maxlen=512)
        self._force_slow = deque(maxlen=SLOW_FORCE_SAMPLES)
        self._force_filter = deque(maxlen=FORCE_FILTER_SAMPLES)
        self._empty_bias_samples = deque(maxlen=EMPTY_BIAS_SAMPLES)
        self._payload_bias_samples = deque(maxlen=PAYLOAD_BIAS_SAMPLES)
        self._empty_force_bias = None
        self._payload_force_bias = None
        self._current_force_payload = False
        self._payload_phase_latched = False
        self._state_history = deque(maxlen=STATE_HISTORY_SAMPLES)
        self._last_force_timestamp = None
        self._last_force_slow_timestamp = None
        self._last_state_timestamp = None
        self._last_calibrated_force = np.zeros(6, dtype=np.float32)
        self._contact_truth_by_topic = {}
        super().__init__(*args, **kwargs)
        self._contact_truth_subscriptions = [
            self.create_subscription(
                Contacts,
                topic,
                lambda msg, contact_topic=topic: self._contacts(msg, contact_topic),
                10,
            )
            for topic in CONTACT_TRUTH_TOPICS
        ]

    def _contacts(self, msg, topic):
        """Cache privileged Gazebo contact truth for oracle diagnostics only."""
        now = self.get_clock().now().nanoseconds * 1e-9
        names = []
        for contact in msg.contacts:
            names.extend(
                [contact.collision1.name.lower(), contact.collision2.name.lower()]
            )
        joined = " ".join(names)
        state = {
            "timestamp": now,
            "any": bool(msg.contacts),
            "gripper": any(
                token in joined for token in ("robotiq", "finger", "knuckle")
            ),
            "hole": any(
                token in joined
                for token in ("hole_plate", "socket_col", "rigid_floor_col")
            ),
        }
        with self.lock:
            self._contact_truth_by_topic[topic] = state

    def _observation_ready(self):
        """Require real (not left-padded) PAP-MoE temporal context."""
        with self.lock:
            return bool(
                self._empty_force_bias is not None
                and len(self._force_native) >= FAST_FORCE_SAMPLES
                and len(self._force_slow) >= SLOW_FORCE_SAMPLES
                and len(self._state_history) >= STATE_HISTORY_SAMPLES
            )

    def _reset_policy_observation_context(self):
        """Rebuild all temporal windows from a stationary policy start pose."""
        with self.lock:
            self._force_native.clear()
            self._force_slow.clear()
            self._force_filter.clear()
            self._state_history.clear()
            self._last_force_timestamp = None
            self._last_force_slow_timestamp = None
            self._last_state_timestamp = None
        self.get_logger().info("Cleared policy temporal context collected during start-pose motion.")

    def _status_force_norm(self):
        with self.lock:
            force = self._last_calibrated_force[:3].copy()
        return float(np.linalg.norm(force))

    def _message_timestamp(self, message) -> float:
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        if stamp is not None:
            # Stay in one clock domain even for the first zero-stamped Gazebo
            # sample. Falling back to wall monotonic time for stamp==0 and then
            # switching to simulation time clears every temporal window on the
            # next message because simulation time is numerically smaller.
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _should_resample(timestamp: float, previous: float | None) -> bool:
        return previous is None or timestamp < previous or timestamp - previous >= 0.095

    def _wrench(self, msg):
        wrench = msg.wrench
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
        timestamp = self._message_timestamp(msg)
        # Reject before updating either freshness or median/bias history.
        # Sustained invalid samples must expire the last valid force source.
        if not np.isfinite(value).all() or not np.isfinite(timestamp):
            return
        with self.lock:
            self._source_stamps["force"] = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, time.monotonic())
            if self._last_force_timestamp is not None and timestamp < self._last_force_timestamp:
                self._force_native.clear()
                self._force_slow.clear()
                self._force_filter.clear()
                self._empty_bias_samples.clear()
                self._payload_bias_samples.clear()
                self._empty_force_bias = None
                self._payload_force_bias = None
                self._last_force_slow_timestamp = None

            # Match collection v6: advance even during sustained contact.
            # The median rejects isolated impulses, not every sample >30 N.
            self._force_filter.append(value.copy())
            filtered_value = np.median(np.stack(self._force_filter), axis=0).astype(np.float32)

            # ``is_attached`` tracks only the optional Gazebo detachable
            # joint.  Formal Workspace50 rollouts deliberately disable that
            # safeguard, so using it as a payload flag leaves the measured peg
            # weight in every PAP force channel and makes the Gate interpret
            # ordinary transport as rigid contact.  A settled measured close
            # provides an online, object-width-independent approximation to
            # the training recorder's post-grasp stable payload reference,
            # without requiring attachment or privileged object pose.
            measured_gripper = None if self.latest_pos is None else float(self.latest_pos[6])
            if measured_gripper is not None and measured_gripper <= PAYLOAD_GRIPPER_RELEASE_RAD:
                if self._payload_phase_latched:
                    self._payload_bias_samples.clear()
                    self._payload_force_bias = None
                self._payload_phase_latched = False
            recent_gripper = np.asarray(
                [state[6] for _, state in list(self._state_history)[-PAYLOAD_GRIPPER_STABLE_SAMPLES:]],
                dtype=np.float32,
            )
            settled_close = bool(
                measured_gripper is not None
                and is_settled_gripper_close(
                    recent_gripper,
                    measured_gripper,
                    min_closed_position=PAYLOAD_GRIPPER_MIN_CLOSED_RAD,
                    stable_range=PAYLOAD_GRIPPER_STABLE_RANGE_RAD,
                    min_samples=PAYLOAD_GRIPPER_STABLE_SAMPLES,
                )
            )
            if self.is_attached or settled_close:
                self._payload_phase_latched = True
            payload_phase = bool(self._payload_phase_latched)
            self._current_force_payload = payload_phase
            self.wrench = filtered_value
            self._force_native.append((timestamp, value.copy(), payload_phase))
            self._last_force_timestamp = timestamp
            if self._should_resample(timestamp, self._last_force_slow_timestamp):
                self._force_slow.append((timestamp, filtered_value.copy(), payload_phase))
                self._last_force_slow_timestamp = timestamp

            if not payload_phase and self._empty_force_bias is None:
                self._empty_bias_samples.append(filtered_value.copy())
                if len(self._empty_bias_samples) >= EMPTY_BIAS_SAMPLES:
                    self._empty_force_bias = estimate_wrench_bias(list(self._empty_bias_samples))
            elif payload_phase and self._payload_force_bias is None:
                self._payload_bias_samples.append(filtered_value.copy())
                if len(self._payload_bias_samples) >= PAYLOAD_BIAS_SAMPLES:
                    self._payload_force_bias = estimate_wrench_bias(list(self._payload_bias_samples))

    def _record_joint_history(self, timestamp, position):
        """Called inside the base joint-state lock, with this message's values.

        Source timestamp, current state and history now form one transaction.
        Real timestamp rollback still resets history; it is not hidden by
        relabeling old samples or padding them as valid observations.
        """
        if self._last_state_timestamp is not None and timestamp < self._last_state_timestamp:
            self.get_logger().warning(
                f"JOINT HISTORY RESET: stamp={timestamp:.9f}, previous={self._last_state_timestamp:.9f}, "
                f"sim_now={self.get_clock().now().nanoseconds * 1e-9:.9f}"
            )
            self._state_history.clear()
            self._last_state_timestamp = None
        if self._should_resample(timestamp, self._last_state_timestamp):
            self._state_history.append((timestamp, np.asarray(position, dtype=np.float32).copy()))
            self._last_state_timestamp = timestamp

    def _save_images(self):
        with self.lock:
            self._snapshot_joint_pos = None if self.latest_pos is None else list(self.latest_pos)
            snapshot_source_stamps = dict(self._source_stamps)
            camera0 = self.wrist_img.copy()
            camera1 = self.global_img.copy()
            current_filtered_force = self.wrench.copy()
            current_force_payload = bool(self._current_force_payload)
            fast_pairs = list(self._force_native)[-FAST_FORCE_SAMPLES:]
            slow_pairs = list(self._force_slow)
            state_pairs = list(self._state_history)
            empty_force_bias = None if self._empty_force_bias is None else self._empty_force_bias.copy()
            payload_force_bias = None if self._payload_force_bias is None else self._payload_force_bias.copy()
            contact_now = self.get_clock().now().nanoseconds * 1e-9
            fresh_contacts = [
                state
                for state in self._contact_truth_by_topic.values()
                if contact_now - state["timestamp"] <= CONTACT_TRUTH_TIMEOUT_S
            ]
            contact_truth = {
                "valid": True,
                "any": any(state["any"] for state in fresh_contacts),
                "gripper": any(state["gripper"] for state in fresh_contacts),
                "hole": any(state["hole"] for state in fresh_contacts),
            }

        # Do not let inference consume raw gravity/payload wrench while the
        # empty-tool reference is still warming. Metadata below keeps the
        # inference process blocked until this reference is valid.
        active_empty_bias = np.zeros(6, dtype=np.float32) if empty_force_bias is None else empty_force_bias

        raw_force_fast = _left_pad_window(
            [value for _, value, _ in fast_pairs],
            FAST_FORCE_SAMPLES,
            6,
        )
        raw_force_slow = _left_pad_window(
            [value for _, value, _ in slow_pairs],
            SLOW_FORCE_SAMPLES,
            6,
        )
        fast_payload_flags = _left_pad_flags([payload for _, _, payload in fast_pairs], FAST_FORCE_SAMPLES)
        slow_payload_flags = _left_pad_flags([payload for _, _, payload in slow_pairs], SLOW_FORCE_SAMPLES)
        current_force = calibrate_wrench_window(
            current_filtered_force[None, :],
            np.asarray([current_force_payload], dtype=bool),
            active_empty_bias,
            payload_force_bias,
        )[0]
        with self.lock:
            self._last_calibrated_force = current_force.copy()
        force_fast = calibrate_wrench_window(
            raw_force_fast,
            fast_payload_flags,
            active_empty_bias,
            payload_force_bias,
            clip_fast=True,
        )
        force_slow = calibrate_wrench_window(
            raw_force_slow,
            slow_payload_flags,
            active_empty_bias,
            payload_force_bias,
        )
        state_history = map_gripper_history_to_training_units(
            _left_pad_window(
                [value for _, value in state_pairs],
                STATE_HISTORY_SAMPLES,
                7,
            ),
            mode=self.STATE_HISTORY_GRIPPER_MODE,
        )
        visual_quality = compute_visual_quality(camera0, camera1)

        _atomic_save_npy("/tmp/ur3_camera0.npy", camera0)
        _atomic_save_npy("/tmp/ur3_camera1.npy", camera1)
        _atomic_save_npy("/tmp/ur3_force.npy", current_force)
        _atomic_save_npy(FORCE_FAST_FILE, force_fast)
        _atomic_save_npy(FORCE_SLOW_FILE, force_slow)
        _atomic_save_npy(STATE_HISTORY_FILE, state_history)
        _atomic_save_npy(VISUAL_QUALITY_FILE, visual_quality)

        fast_span = float(fast_pairs[-1][0] - fast_pairs[0][0]) if len(fast_pairs) >= 2 else 0.0
        slow_span = float(slow_pairs[-1][0] - slow_pairs[0][0]) if len(slow_pairs) >= 2 else 0.0
        state_span = float(state_pairs[-1][0] - state_pairs[0][0]) if len(state_pairs) >= 2 else 0.0
        metadata = {
            "schema": "pap_moe_v6",
            "wall_time_s": time.time(),
            "source_stamps_sim_and_monotonic_s": snapshot_source_stamps,
            "simulation_time_s": self.get_clock().now().nanoseconds * 1e-9,
            "force_reference_mode": "per_sample_phase_payload_bias_v2",
            "force_filter_mode": "median11_sustained_visible_fast_clip30n_native100hz_v6",
            "empty_force_bias_valid": empty_force_bias is not None,
            "empty_force_bias": (active_empty_bias.tolist() if empty_force_bias is not None else None),
            "payload_force_bias_valid": payload_force_bias is not None,
            "payload_force_bias": (payload_force_bias.tolist() if payload_force_bias is not None else None),
            "current_force_reference_payload": current_force_payload,
            "state_history_gripper_units": self.STATE_HISTORY_GRIPPER_MODE,
            "force_fast_valid": len(fast_pairs),
            "force_fast_span_s": fast_span,
            "force_fast_estimated_hz": ((len(fast_pairs) - 1) / fast_span if fast_span > 0.0 else 0.0),
            "force_slow_valid": len(slow_pairs),
            "force_slow_span_s": slow_span,
            "state_history_valid": len(state_pairs),
            "state_history_span_s": state_span,
            "visual_quality": visual_quality.tolist(),
            # Never part of the policy observation. Read only by the explicit
            # contact_oracle routing ablation in the separate inference process.
            "contact_truth": contact_truth,
        }
        _atomic_write_json(OBSERVATION_META_FILE, metadata)


def main(args=None):
    run_ros_side(
        PAPMoEROSSide,
        node_name="ur3_pap_moe_peg_in_hole_ros_side",
        args=args,
    )


if __name__ == "__main__":
    main()

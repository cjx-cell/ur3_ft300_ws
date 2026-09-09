#!/usr/bin/env python3
"""Smooth keyboard and ordinary-mouse teleoperation for UR3 MoveIt Servo.

The original keyboard collectors remain unchanged.  This v2 frontend adds a
proportional mouse joystick, acceleration-limited commands and useful runtime
diagnostics while preserving the existing recording event contract.
"""

from __future__ import annotations

import json
import math
import time
import tkinter as tk

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from std_msgs.msg import Bool, Float64, String

from pap_moe_keyboard_teleop import ARM_JOINTS, KeyboardTeleop
from pap_moe_teleop_motion import (
    KEY_MOTIONS,
    SPEED_MODES,
    MotionCommandShaper,
    compose_unit_command,
)


MOTION_KEYS = set(KEY_MOTIONS)
JOYSTICK_RADIUS = 105.0
JOYSTICK_DEADZONE = 7.0
# X11 physical keycodes make ordinary letter controls independent of the
# active Chinese/Latin input method.  Unknown platforms still fall back to
# Tk's keysym below.
X11_KEYCODE_NAMES = {
    9: "\x1b", 10: "1", 11: "2", 12: "3", 13: "4", 14: "5", 15: "6", 16: "7",
    24: "q", 25: "w", 26: "e", 27: "r", 31: "i", 33: "p",
    38: "a", 39: "s", 40: "d", 41: "f", 42: "g",
    44: "j", 45: "k", 46: "l", 53: "x", 55: "v", 56: "b", 58: "m",
    65: " ",
}


class SmoothKeyboardMouseTeleop(KeyboardTeleop):
    def __init__(self):
        super().__init__()
        self.speed_modes = tuple((mode.name, mode.scale, 0.0) for mode in SPEED_MODES)
        self.speed_mode_index = 0
        self.shaper = MotionCommandShaper()
        self.mouse_translation = [0.0, 0.0, 0.0]
        self.mouse_rotation = [0.0, 0.0, 0.0]
        self.wheel_deadline_ros = 0.0
        self.last_shape_time_ros = None
        self.last_shape_time_wall = time.monotonic()
        self.last_target = [0.0] * 6
        self.last_output = [0.0] * 6
        self.arm_velocity_norm = 0.0
        self.realtime_factor = 0.0
        self.collision_velocity_scale = 1.0
        self.servo_status_since = time.monotonic()
        self._last_servo_status = self.servo_status
        self.last_gui_key_event = "none"
        self.auto_reset_active = False
        self.diagnostic_pub = self.create_publisher(
            String, "/pap_moe/teleop_diagnostics", 10
        )
        self.create_subscription(
            Float64,
            "/servo_node/collision_velocity_scale",
            self.collision_scale_cb,
            10,
        )
        self.create_subscription(
            Bool,
            "/pap_moe/teleop_reset_active",
            self.reset_active_cb,
            10,
        )
        self.create_timer(0.2, self.publish_diagnostics)

    def reset_active_cb(self, msg):
        active = bool(msg.data)
        if active and not self.auto_reset_active:
            self.force_stop()
            self.get_logger().info("automatic post-episode reset has control")
        elif not active and self.auto_reset_active:
            self.force_stop()
            self.get_logger().info("manual teleoperation control restored")
        self.auto_reset_active = active

    def publish_command(self):
        # The batch collector owns Servo and the arm position controller while
        # returning home.  Publishing GUI zeros at the same time would race the
        # reset command and can make the robot appear stuck between episodes.
        if self.auto_reset_active:
            return
        super().publish_command()

    def collision_scale_cb(self, msg):
        self.collision_velocity_scale = float(msg.data)

    def servo_status_cb(self, msg):
        previous = self.servo_status
        super().servo_status_cb(msg)
        if self.servo_status != previous:
            self.servo_status_since = time.monotonic()

    def joint_state_cb(self, msg):
        super().joint_state_cb(msg)
        try:
            velocities = [
                float(msg.velocity[msg.name.index(name)]) for name in ARM_JOINTS
            ]
        except (AttributeError, ValueError, IndexError):
            return
        self.arm_velocity_norm = math.sqrt(sum(value * value for value in velocities))

    def set_mouse_translation(self, x, y):
        self.mouse_translation[0] = float(x)
        self.mouse_translation[1] = float(y)

    def set_mouse_rotation(self, roll, pitch):
        self.mouse_rotation[0] = float(roll)
        self.mouse_rotation[1] = float(pitch)

    def set_mouse_wheel(self, direction):
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        self.mouse_translation[2] = 1.0 if direction > 0 else -1.0
        self.wheel_deadline_ros = now_ros + 0.16

    def clear_mouse_wheel(self):
        self.mouse_translation[2] = 0.0
        self.wheel_deadline_ros = 0.0

    def clear_mouse(self):
        self.mouse_translation = [0.0, 0.0, 0.0]
        self.mouse_rotation = [0.0, 0.0, 0.0]
        self.wheel_deadline_ros = 0.0

    def force_stop(self):
        with self.lock:
            self.held_motion_keys.clear()
            self.command = [0.0] * 6
            self.command_deadline_ros = 0.0
            self.clear_mouse()
            self.last_target = [0.0] * 6
            self.last_output = self.shaper.reset()

    def continuous_command(self):
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        now_wall = time.monotonic()
        if now_ros > self.wheel_deadline_ros:
            self.mouse_translation[2] = 0.0
        unit = compose_unit_command(
            self.held_motion_keys,
            self.mouse_translation,
            self.mouse_rotation,
        )
        mode = SPEED_MODES[self.speed_mode_index]
        target = [mode.scale * value for value in unit]
        if self.last_shape_time_ros is None:
            dt = 0.02
        else:
            dt = now_ros - self.last_shape_time_ros
            wall_dt = now_wall - self.last_shape_time_wall
            if wall_dt > 1e-4 and dt >= 0.0:
                self.realtime_factor = dt / wall_dt
        self.last_shape_time_ros = now_ros
        self.last_shape_time_wall = now_wall
        self.last_target = target
        self.last_output = self.shaper.step(
            target, dt, mode.acceleration, mode.deceleration
        )
        return self.last_output.copy()

    def handle_key(self, key):
        if key == "m":
            self.speed_mode_index = (self.speed_mode_index + 1) % len(SPEED_MODES)
            mode = SPEED_MODES[self.speed_mode_index]
            self.get_logger().info(
                f"speed mode={mode.name}, scale={mode.scale:.2f}, "
                f"accel={mode.acceleration:.2f}, decel={mode.deceleration:.2f}"
            )
            return
        super().handle_key(key)

    def diagnostic_payload(self):
        linear_unitless = math.sqrt(sum(value * value for value in self.last_output[:3]))
        linear_sim_mps = 0.8 * linear_unitless
        return {
            "version": "keyboard_mouse_teleop_v2",
            "control_mode": self.control_mode,
            "speed_mode": SPEED_MODES[self.speed_mode_index].name,
            "held_keys": sorted(self.held_motion_keys),
            "last_gui_key_event": self.last_gui_key_event,
            "mouse_translation": [round(value, 5) for value in self.mouse_translation],
            "mouse_rotation": [round(value, 5) for value in self.mouse_rotation],
            "target_command": [round(value, 5) for value in self.last_target],
            "shaped_command": [round(value, 5) for value in self.last_output],
            "servo_status": int(self.servo_status),
            "collision_velocity_scale": round(self.collision_velocity_scale, 5),
            "servo_status_duration_wall_s": round(
                time.monotonic() - self.servo_status_since, 3
            ),
            "arm_joint_velocity_norm": round(self.arm_velocity_norm, 5),
            "gripper_interlock_active": bool(
                time.monotonic() <= self.gripper_deadline
            ),
            "servo_resume_pending": bool(self.servo_restart_pending),
            "realtime_factor_estimate": round(self.realtime_factor, 3),
            "linear_speed_sim_mps": round(linear_sim_mps, 4),
            "linear_speed_wall_estimate_mps": round(
                linear_sim_mps * self.realtime_factor, 4
            ),
            "auto_reset_active": self.auto_reset_active,
        }

    def publish_diagnostics(self):
        self.diagnostic_pub.publish(String(data=json.dumps(self.diagnostic_payload())))


class MouseJoystickWindow:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("PAP-MoE keyboard + mouse teleop v2")
        self.root.geometry("820x680")
        self.root.configure(bg="#17202a")
        self.held = set()
        self.one_shot_down = set()
        self.pending_one_shot_releases = {}
        self.window_active = True
        self.pending_focus_check = None
        self.left_origin = None
        self.right_origin = None
        self.closed = False

        tk.Label(
            self.root,
            text="UR3 顺滑键鼠摇操作 v2（窗口必须保持焦点）",
            font=("Sans", 18, "bold"),
            fg="#ecf0f1",
            bg="#17202a",
        ).pack(pady=(14, 6))
        self.status = tk.Label(
            self.root,
            text="",
            font=("Monospace", 12, "bold"),
            justify="left",
            fg="#2ecc71",
            bg="#17202a",
        )
        self.status.pack(pady=6)
        tk.Label(
            self.root,
            text=(
                "全局画面对齐：W/S=画面上/下  A/D=画面左/右  R/F=升/降\n"
                "姿态：I/K=Roll  J/L=Pitch  Q/E=Yaw（base_link）\n"
                "控制窗口获得焦点后直接按键；松开运动键立即停止\n"
                "鼠标：按住左键拖动=XY；右键拖动=Roll/Pitch；滚轮=Z\n"
                "M=速度  G=脱困  SPACE=夹爪\n"
                "P=FT基准  B=记录  V=成功  X=丢弃  Esc=急停"
            ),
            font=("Sans", 12),
            justify="left",
            fg="#d5dbdb",
            bg="#17202a",
        ).pack(padx=20, pady=5)

        self.canvas = tk.Canvas(
            self.root,
            width=700,
            height=330,
            bg="#101820",
            highlightthickness=1,
            highlightbackground="#566573",
        )
        self.canvas.pack(pady=10)
        self.canvas.create_text(
            175, 24, text="左键拖动：XY平移", fill="#ecf0f1", font=("Sans", 12, "bold")
        )
        self.canvas.create_text(
            525, 24, text="右键拖动：Roll/Pitch", fill="#ecf0f1", font=("Sans", 12, "bold")
        )
        self.centers = {"left": (175.0, 180.0), "right": (525.0, 180.0)}
        for center in self.centers.values():
            x, y = center
            self.canvas.create_oval(
                x - JOYSTICK_RADIUS,
                y - JOYSTICK_RADIUS,
                x + JOYSTICK_RADIUS,
                y + JOYSTICK_RADIUS,
                outline="#5dade2",
                width=2,
            )
            self.canvas.create_line(x - 115, y, x + 115, y, fill="#34495e")
            self.canvas.create_line(x, y - 115, x, y + 115, fill="#34495e")
        self.left_vector = self.canvas.create_line(175, 180, 175, 180, fill="#2ecc71", width=5)
        self.right_vector = self.canvas.create_line(525, 180, 525, 180, fill="#f1c40f", width=5)
        self.focus_label = tk.Label(
            self.root,
            text="点击窗口后开始；焦点离开会立即停止",
            font=("Sans", 12, "bold"),
            fg="#f1c40f",
            bg="#17202a",
        )
        self.focus_label.pack(pady=5)

        self.root.bind_all("<KeyPress>", self.on_press)
        self.root.bind_all("<KeyRelease>", self.on_release)
        self.root.bind("<FocusOut>", self.on_focus_out)
        self.root.bind("<FocusIn>", self.on_focus_in)
        self.root.bind("<ButtonPress-1>", self.on_window_click, add="+")
        self.root.bind("<ButtonPress-3>", self.on_window_click, add="+")
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.root.bind_all("<MouseWheel>", self.on_wheel)
        self.root.bind_all("<Button-4>", lambda event: self.on_linux_wheel(event, 1))
        self.root.bind_all("<Button-5>", lambda event: self.on_linux_wheel(event, -1))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(10, self.pump)

    @staticmethod
    def _key_name(event):
        physical = X11_KEYCODE_NAMES.get(int(event.keycode))
        if physical is not None:
            return physical
        if event.keysym == "space":
            return " "
        if event.keysym == "Escape":
            return "\x1b"
        return event.keysym.lower()

    @staticmethod
    def _joystick_value(origin, x, y):
        dx = float(x) - origin[0]
        dy = float(y) - origin[1]
        norm = math.hypot(dx, dy)
        if norm <= JOYSTICK_DEADZONE:
            return 0.0, 0.0
        if norm > JOYSTICK_RADIUS:
            scale = JOYSTICK_RADIUS / norm
            dx *= scale
            dy *= scale
        gain = (min(math.hypot(dx, dy), JOYSTICK_RADIUS) - JOYSTICK_DEADZONE) / (
            JOYSTICK_RADIUS - JOYSTICK_DEADZONE
        )
        direction_norm = max(math.hypot(dx, dy), 1e-9)
        return gain * dx / direction_norm, gain * dy / direction_norm

    def on_press(self, event):
        key = self._key_name(event)
        self.node.last_gui_key_event = (
            f"press keysym={event.keysym!s} keycode={event.keycode} mapped={key!r}"
        )
        pending_release = self.pending_one_shot_releases.pop(key, None)
        if pending_release is not None:
            # X11 represents keyboard auto-repeat as release/press pairs.  Keep
            # one-shot controls edge-triggered while a key is physically held.
            self.root.after_cancel(pending_release)
        if key in {"shift_l", "shift_r"}:
            return
        if key in MOTION_KEYS:
            self.held.add(key)
            self.node.set_held_motion_keys(self.held)
            return
        if key not in self.one_shot_down:
            self.one_shot_down.add(key)
            self.node.handle_key(key)
            if key == "g":
                self.held.clear()
                self.node.set_held_motion_keys(set())
            if key == "\x1b":
                self.close()

    def on_release(self, event):
        key = self._key_name(event)
        self.node.last_gui_key_event = (
            f"release keysym={event.keysym!s} keycode={event.keycode} mapped={key!r}"
        )
        if key in {"shift_l", "shift_r"}:
            return
        if key in self.one_shot_down:
            previous = self.pending_one_shot_releases.pop(key, None)
            if previous is not None:
                self.root.after_cancel(previous)
            self.pending_one_shot_releases[key] = self.root.after(
                80, lambda released_key=key: self._finish_one_shot_release(released_key)
            )
        if key in MOTION_KEYS:
            self.held.discard(key)
            self.node.set_held_motion_keys(self.held)

    def _finish_one_shot_release(self, key):
        self.pending_one_shot_releases.pop(key, None)
        self.one_shot_down.discard(key)

    def on_left_press(self, event):
        self.left_origin = (float(event.x), float(event.y))

    def on_left_drag(self, event):
        if self.left_origin is None:
            return
        dx, dy = self._joystick_value(self.left_origin, event.x, event.y)
        # Upright far-side camera: image right is base +Y and image down is
        # base +X.
        self.node.set_mouse_translation(dy, dx)
        cx, cy = self.centers["left"]
        self.canvas.coords(self.left_vector, cx, cy, cx + dx * JOYSTICK_RADIUS, cy + dy * JOYSTICK_RADIUS)

    def on_left_release(self, _event):
        self.left_origin = None
        self.node.set_mouse_translation(0.0, 0.0)
        cx, cy = self.centers["left"]
        self.canvas.coords(self.left_vector, cx, cy, cx, cy)

    def on_right_press(self, event):
        self.right_origin = (float(event.x), float(event.y))

    def on_right_drag(self, event):
        if self.right_origin is None:
            return
        dx, dy = self._joystick_value(self.right_origin, event.x, event.y)
        self.node.set_mouse_rotation(-dy, dx)
        cx, cy = self.centers["right"]
        self.canvas.coords(self.right_vector, cx, cy, cx + dx * JOYSTICK_RADIUS, cy + dy * JOYSTICK_RADIUS)

    def on_right_release(self, _event):
        self.right_origin = None
        self.node.set_mouse_rotation(0.0, 0.0)
        cx, cy = self.centers["right"]
        self.canvas.coords(self.right_vector, cx, cy, cx, cy)

    def on_wheel(self, event):
        self.node.set_mouse_wheel(1 if event.delta > 0 else -1)

    def on_linux_wheel(self, _event, direction):
        self.node.set_mouse_wheel(direction)

    def on_window_click(self, _event):
        # This is user-initiated focus acquisition (unlike focus_force at
        # startup), so it cannot steal keys while the user types elsewhere.
        self.root.focus_set()

    def on_focus_out(self, _event):
        # FocusOut also fires when focus moves between the toplevel and its
        # canvas children.  Defer the decision and stop only if focus really
        # left the complete teleop window.
        if self.pending_focus_check is not None:
            self.root.after_cancel(self.pending_focus_check)
        self.pending_focus_check = self.root.after(30, self._check_focus_left)

    def _check_focus_left(self):
        self.pending_focus_check = None
        if self.root.focus_displayof() is not None:
            self.window_active = True
            return
        self.held.clear()
        self.one_shot_down.clear()
        for callback in self.pending_one_shot_releases.values():
            self.root.after_cancel(callback)
        self.pending_one_shot_releases.clear()
        self.window_active = False
        self.left_origin = None
        self.right_origin = None
        self.node.force_stop()
        self.focus_label.configure(text="焦点已离开：运动已立即停止", fg="#e74c3c")

    def on_focus_in(self, _event):
        if self.pending_focus_check is not None:
            self.root.after_cancel(self.pending_focus_check)
            self.pending_focus_check = None
        was_active = self.window_active
        # A newly focused control window must always start from a known stop;
        # motion begins only after a fresh key press or mouse-button action.
        self.held.clear()
        self.one_shot_down.clear()
        for callback in self.pending_one_shot_releases.values():
            self.root.after_cancel(callback)
        self.pending_one_shot_releases.clear()
        self.window_active = True
        if not was_active:
            self.node.force_stop()
        self.focus_label.configure(text="控制窗口已激活", fg="#2ecc71")

    def pump(self):
        if not rclpy.ok() or not self.node.running:
            self.close()
            return
        rclpy.spin_once(self.node, timeout_sec=0.0)
        states = {
            -1: "WAITING",
            0: "OK",
            1: "SINGULARITY SLOW",
            2: "SINGULARITY STOP",
            3: "COLLISION SLOW",
            4: "COLLISION STOP",
            5: "JOINT LIMIT",
            6: "LEAVING SINGULARITY",
        }
        mode = SPEED_MODES[self.node.speed_mode_index]
        target_norm = math.sqrt(sum(v * v for v in self.node.last_target))
        output_norm = math.sqrt(sum(v * v for v in self.node.last_output))
        linear_sim_mps = 0.8 * math.sqrt(
            sum(v * v for v in self.node.last_output[:3])
        )
        linear_wall_mps = linear_sim_mps * self.node.realtime_factor
        status = states.get(self.node.servo_status, f"STATUS {self.node.servo_status}")
        self.status.configure(
            text=(
                f"控制={self.node.control_mode.upper()}  速度={mode.name.upper()}({mode.scale:.2f})\n"
                f"窗口焦点={'ON' if self.window_active else 'OFF'}  "
                f"按键={'+'.join(sorted(self.held)).upper() or 'STOP'}\n"
                f"自动复位={'ON' if self.node.auto_reset_active else 'OFF'}\n"
                f"末次键事件={self.node.last_gui_key_event}\n"
                f"目标={target_norm:.3f}  成形输出={output_norm:.3f}\n"
                f"夹爪状态={'运动中（机械臂可并行）' if time.monotonic() <= self.node.gripper_deadline else '稳定'}\n"
                f"Servo={status}  状态持续={time.monotonic() - self.node.servo_status_since:.1f}s  "
                f"碰撞缩放={self.node.collision_velocity_scale:.3f}\n"
                f"关节速度范数={self.node.arm_velocity_norm:.3f}  "
                f"Gazebo实时率估计={self.node.realtime_factor:.2f}\n"
                f"线速度：仿真={linear_sim_mps:.3f} m/s  "
                f"墙钟估计={linear_wall_mps:.3f} m/s"
            ),
            fg="#2ecc71" if self.node.servo_status == 0 else "#f39c12",
        )
        self.root.after(10, self.pump)

    def close(self):
        # Launch shutdown can invalidate the ROS context before Tk processes
        # WM_DELETE_WINDOW (and more than one pending ``after`` callback may
        # enter here).  Make teardown one-shot and never publish through an
        # invalid rclpy context.
        if self.closed:
            return
        self.closed = True
        self.node.running = False
        self.node.force_stop()
        if rclpy.ok(context=self.node.context):
            try:
                self.node.publish_command()
            except RCLError:
                pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = SmoothKeyboardMouseTeleop()
    try:
        node.start_servo()
        MouseJoystickWindow(node).run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

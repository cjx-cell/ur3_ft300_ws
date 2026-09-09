#!/usr/bin/env python3
"""Focused GUI keyboard controller with true simultaneous held-key input."""

import tkinter as tk

import rclpy

from pap_moe_keyboard_teleop import KeyboardTeleop


MOTION_KEYS = set("wsadrfikjlqe")


class TeleopWindow:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("PAP-MoE UR3 continuous keyboard teleop")
        self.root.geometry("720x430")
        self.root.configure(bg="#17202a")
        self.held = set()
        self.one_shot_down = set()

        title = tk.Label(
            self.root,
            text="UR3 连续键盘摇操作（窗口必须保持焦点）",
            font=("Sans", 18, "bold"),
            fg="#ecf0f1",
            bg="#17202a",
        )
        title.pack(pady=(18, 8))
        self.status = tk.Label(
            self.root,
            text="",
            font=("Monospace", 14, "bold"),
            fg="#2ecc71",
            bg="#17202a",
        )
        self.status.pack(pady=8)
        instructions = (
            "同时按住：W/S=X，A/D=Y，R/F=Z\n"
            "姿态：I/K=Roll，J/L=Pitch，Q/E=Yaw\n\n"
            "G：笛卡尔/关节脱困模式\n"
            "关节模式：W/S=J1 A/D=J2 R/F=J3 I/K=J4 J/L=J5 Q/E=J6\n"
            "M：coarse → precision → normal    SPACE：夹爪开/闭\n"
            "1..7：阶段    P：负载FT基准    B：开始录制\n"
            "V：成功    X：丢弃    Esc：急停退出\n\n"
            "默认 coarse；靠近物体、对孔及插入时切到 precision。"
        )
        tk.Label(
            self.root,
            text=instructions,
            justify="left",
            font=("Sans", 13),
            fg="#d5dbdb",
            bg="#17202a",
        ).pack(padx=24, pady=10)
        self.focus_label = tk.Label(
            self.root,
            text="点击此窗口后按住移动键",
            font=("Sans", 13, "bold"),
            fg="#f1c40f",
            bg="#17202a",
        )
        self.focus_label.pack(pady=10)

        self.root.bind("<KeyPress>", self.on_press)
        self.root.bind("<KeyRelease>", self.on_release)
        self.root.bind("<FocusOut>", self.on_focus_out)
        self.root.bind("<FocusIn>", self.on_focus_in)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(10, self.pump)
        self.root.focus_force()

    @staticmethod
    def key_name(event):
        if event.keysym == "space":
            return " "
        if event.keysym == "Escape":
            return "\x1b"
        return event.keysym.lower()

    def on_press(self, event):
        key = self.key_name(event)
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
        key = self.key_name(event)
        self.one_shot_down.discard(key)
        if key in MOTION_KEYS:
            self.held.discard(key)
            self.node.set_held_motion_keys(self.held)

    def on_focus_out(self, _event):
        self.held.clear()
        self.one_shot_down.clear()
        self.node.set_held_motion_keys(set())
        self.focus_label.configure(text="焦点已离开：运动已自动停止", fg="#e74c3c")

    def on_focus_in(self, _event):
        self.focus_label.configure(text="控制窗口已激活", fg="#2ecc71")

    def pump(self):
        if not rclpy.ok() or not self.node.running:
            self.close()
            return
        rclpy.spin_once(self.node, timeout_sec=0.0)
        mode, scale, _ = self.node.speed_modes[self.node.speed_mode_index]
        keys = "+".join(sorted(self.held)).upper() or "STOP"
        servo_states = {
            -1: "WAITING",
            0: "OK",
            1: "SINGULARITY SLOW",
            2: "SINGULARITY STOP",
            3: "COLLISION SLOW",
            4: "COLLISION STOP",
            5: "JOINT LIMIT",
            6: "LEAVING SINGULARITY",
        }
        servo_state = servo_states.get(
            self.node.servo_status, f"STATUS {self.node.servo_status}"
        )
        control_mode = self.node.control_mode.upper()
        self.status.configure(
            text=(
                f"控制: {control_mode}  速度: {mode.upper()}({scale:.2f})  按键: {keys}\n"
                f"Servo: {servo_state}"
            ),
            fg="#2ecc71" if self.node.servo_status == 0 else "#f39c12",
        )
        self.root.after(10, self.pump)

    def close(self):
        self.node.set_held_motion_keys(set())
        self.node.publish_command()
        self.node.running = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = KeyboardTeleop()
    try:
        node.start_servo()
        TeleopWindow(node).run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

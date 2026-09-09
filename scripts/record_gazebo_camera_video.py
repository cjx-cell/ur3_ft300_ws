#!/usr/bin/env python3
"""Record the UR3 Gazebo wrist/global camera topics as a side-by-side MP4."""

import argparse
import signal
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class GazeboCameraRecorder(Node):
    def __init__(self, output: Path, fps: float, title: str) -> None:
        super().__init__("gazebo_camera_video_recorder")
        self.output = output
        self.fps = fps
        self.title = title
        self.bridge = CvBridge()
        self.wrist = None
        self.global_camera = None
        self.writer = None
        self.started_at = time.monotonic()
        self.last_frame_at = 0.0
        self.frame_count = 0
        self.create_subscription(Image, "/wrist_camera/color/image_raw", self._on_wrist, 10)
        self.create_subscription(Image, "/global_camera/color/image_raw", self._on_global, 10)

    def _on_wrist(self, message: Image) -> None:
        self.wrist = self.bridge.imgmsg_to_cv2(message, "bgr8")

    def _on_global(self, message: Image) -> None:
        self.global_camera = self.bridge.imgmsg_to_cv2(message, "bgr8")
        self._write_frame()

    def _write_frame(self) -> None:
        now = time.monotonic()
        if self.wrist is None or self.global_camera is None:
            return
        if now - self.last_frame_at < 1.0 / self.fps:
            return

        wrist = cv2.resize(self.wrist, (448, 448), interpolation=cv2.INTER_AREA)
        global_camera = cv2.resize(self.global_camera, (448, 448), interpolation=cv2.INTER_AREA)
        frame = cv2.hconcat([wrist, global_camera])
        frame = cv2.copyMakeBorder(frame, 52, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        elapsed = now - self.started_at
        cv2.putText(frame, self.title, (18, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Wrist camera", (18, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Global camera                 elapsed {elapsed:6.1f}s",
                    (466, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (180, 220, 255), 1, cv2.LINE_AA)

        if self.writer is None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.output), fourcc, self.fps,
                                          (frame.shape[1], frame.shape[0]))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video output: {self.output}")
        self.writer.write(frame)
        self.last_frame_at = now
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.get_logger().info(f"Saved {self.frame_count} frames to {self.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--title", default="Gazebo closed-loop rollout")
    args = parser.parse_args()

    rclpy.init()
    recorder = GazeboCameraRecorder(args.output, args.fps, args.title)

    def stop(_signum, _frame) -> None:
        rclpy.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        rclpy.spin(recorder)
    finally:
        recorder.close()
        recorder.destroy_node()


if __name__ == "__main__":
    main()

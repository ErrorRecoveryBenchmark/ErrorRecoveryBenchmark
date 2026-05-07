"""
Video recording utility for MuJoCo environment visualization.

Captures offscreen renders from EnvWrapper and writes MP4 video files
with optional text overlay for diagnostic information.

Usage:
    recorder = VideoRecorder("output.mp4", fps=20)
    for step in range(100):
        env_wrapper.step(action)
        recorder.capture_frame(env_wrapper, overlay_text=f"step={step}")
    recorder.close()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Records MuJoCo environment frames to MP4 video with optional text overlay."""

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: int = 20,
        resolution: Tuple[int, int] = (512, 512),
        camera_name: str = "agentview",
        codec: str = "libx264",
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.resolution = resolution
        self.camera_name = camera_name
        self.frame_count = 0

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            str(self.output_path),
            fps=fps,
            codec=codec,
            quality=8,
        )
        self._closed = False

    def capture_frame(
        self,
        env_wrapper,
        overlay_text: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Capture one frame from the environment and write to video.

        Args:
            env_wrapper: EnvWrapper instance with get_camera_obs().
            overlay_text: Text line(s) to overlay on the frame.
                          String is treated as a single line;
                          list of strings renders multiple lines.
        """
        if self._closed:
            return

        frame = env_wrapper.get_camera_obs(
            camera_name=self.camera_name,
            resolution=self.resolution,
        )

        if overlay_text is not None:
            frame = self._add_overlay(frame, overlay_text)

        self._writer.append_data(frame)
        self.frame_count += 1

    def _add_overlay(
        self,
        frame: np.ndarray,
        text: Union[str, List[str]],
    ) -> np.ndarray:
        """Add text overlay to a frame using cv2."""
        frame = frame.copy()
        lines = [text] if isinstance(text, str) else text

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        y_offset = 18

        for i, line in enumerate(lines):
            y = y_offset + i * 18
            # Draw background rectangle for readability
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(frame, (4, y - th - 2), (8 + tw, y + 4), bg_color, -1)
            cv2.putText(frame, line, (6, y), font, font_scale, color, thickness)

        return frame

    def close(self) -> None:
        """Finalize and close the video file."""
        if not self._closed:
            self._writer.close()
            self._closed = True
            logger.debug(
                "Video saved: %s (%d frames, %.1fs)",
                self.output_path,
                self.frame_count,
                self.frame_count / max(self.fps, 1),
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def build_augmentation_overlay(
    step: int,
    total_steps: int,
    subtask_label: str = "",
    success: Optional[bool] = None,
    fail_reason: str = "",
    eef_pos: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Build overlay text lines for augmentation video frames.

    Args:
        step: Current step number.
        total_steps: Total steps so far.
        subtask_label: Current subtask label (e.g., "re_acquire").
        success: Current success status (None if unknown).
        fail_reason: Failure reason if known.
        eef_pos: End-effector position (3,).
        extra: Additional key-value pairs to display.

    Returns:
        List of text lines for overlay.
    """
    lines = [f"step: {step}/{total_steps}"]

    if subtask_label:
        lines.append(f"subtask: {subtask_label}")

    if success is not None:
        lines.append(f"success: {success}")

    if fail_reason:
        lines.append(f"FAIL: {fail_reason}")

    if eef_pos is not None:
        lines.append(f"eef: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")

    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")

    return lines

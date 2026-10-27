# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterator

import cv2
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VideoContainer:
    """
    Manages loading and accessing frames from a video file.

    Attributes:
        path (str): Path to the video.
        cap (cv2.VideoCapture): OpenCV video capture object.
        total_frames (int): Total number of frames in the video.
        fps (float): Frames per second of the video.
    """
    def __init__(self,
                 path: str) -> None:
        """
        Initializes the VideoContainer with the given video path.

        Args:
            path (str): Path to the video.

        Raises:
            IOError: If the video file cannot be opened or properties cannot be read.
        """
        self.id: str = os.path.splitext(os.path.basename(path))[0]
        self.path: str = os.path.abspath(path)
        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.path)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {path}.")

        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS)
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not np.isfinite(self.fps) or self.fps <= 0:
            self.cap.release()
            raise IOError(
                f"Invalid FPS ({self.fps}) reported for video: {self.path}. "
                "Expected a finite, positive value."
            )

    @property
    def size(self) -> tuple[int, int]:
        """Returns the size of one frame in pixels (width, height)."""
        return (self.width, self.height)

    def _get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Retrieves a specific frame by its index, converting it to RGB format.
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(f"Frame index {frame_idx} out of bounds (0..{self.total_frames-1}) for {self.path}")
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise IOError(f"Cannot reopen video file: {self.path}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise IOError(f"Could not read frame at index {frame_idx} from {self.path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """
        Retrieves a specific frame by its index, converting it to RGB format.
        Allows for direct indexing on the VideoContainer object (e.g., vc[frame_idx]).
        """
        if not isinstance(frame_idx, int):
            raise TypeError("Frame index must be an integer.")

        if frame_idx < 0:
            frame_idx += self.total_frames

        return self._get_frame(frame_idx)

    def release(self) -> None:
        """
        Releases the video capture object.
        """
        self.cap.release()

    def __del__(self) -> None:
        self.release()

    def __enter__(self) -> VideoContainer:
        """Allows the class to be used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensures release() is called when exiting the context."""
        self.release()

    def __len__(self) -> int:
        """Returns the total number of frames of the video."""
        return self.total_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        """Allows the iteration over the frames of the video."""
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise IOError(f"Failed to open video for iteration: {self.path}")

        for i in range(self.total_frames):
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != i:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            ret, frame = self.cap.read()
            if not ret:
                raise IOError(
                    f"Failed to read frame at index {i} during iteration "
                    f"from video: {self.path}. Expected {self.total_frames} frames, "
                    f"but failed at frame {i}."
                )
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __repr__(self) -> str:
        """Returns a string representation of the VideoContainer."""
        return f"<VideoContainer path='{self.path}', frames={self.total_frames}, fps={self.fps:.2f}>"

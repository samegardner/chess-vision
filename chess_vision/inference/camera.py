"""Camera capture wrapper."""

import cv2
import numpy as np

from chess_vision.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


class Camera:
    """Thin wrapper around OpenCV VideoCapture."""

    def __init__(
        self,
        device_index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
    ):
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {device_index}")

    def capture(self) -> np.ndarray:
        """Capture a single frame. Returns BGR image."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def release(self):
        """Release the camera."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

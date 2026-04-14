"""Move detection via pixel differencing on warped board images.

Instead of classifying each square with a CNN and comparing board states,
this compares raw pixel values between consecutive warped frames. Much more
robust since it doesn't depend on model accuracy.
"""

import cv2
import numpy as np


class PixelMoveDetector:
    """Detects moves by comparing pixel intensity changes per square."""

    def __init__(self, stability_frames: int = 5, change_threshold: float = 25.0):
        """
        Args:
            stability_frames: Consecutive frames with same change before accepting.
            change_threshold: Mean pixel diff per square to count as "changed".
        """
        self.stability_frames = stability_frames
        self.change_threshold = change_threshold
        self.reference_frame: np.ndarray | None = None
        self.candidate_squares: set[str] | None = None
        self.stable_count: int = 0
        self.hand_present: bool = False

    def set_reference(self, warped_board: np.ndarray) -> None:
        """Set the reference frame (stable board position)."""
        self.reference_frame = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def detect_change(self, warped_board: np.ndarray) -> list[str] | None:
        """Compare current frame to reference, return changed squares if stable.

        Returns list of changed square names, or None if no stable change detected.
        """
        if self.reference_frame is None:
            return None

        gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        sq_h = h / 8
        sq_w = w / 8

        # Compute per-square mean absolute difference
        changed = set()
        total_diff = 0.0
        for rank in range(8):
            for file in range(8):
                row = 7 - rank  # rank 1 = bottom = row 7
                y1 = int(row * sq_h)
                y2 = int((row + 1) * sq_h)
                x1 = int(file * sq_w)
                x2 = int((file + 1) * sq_w)

                sq_diff = np.mean(np.abs(gray[y1:y2, x1:x2] - self.reference_frame[y1:y2, x1:x2]))
                total_diff += sq_diff

                if sq_diff > self.change_threshold:
                    sq_name = chr(ord("a") + file) + str(rank + 1)
                    changed.add(sq_name)

        # Detect hand presence: if many squares changed, a hand is over the board
        avg_diff = total_diff / 64
        if len(changed) > 10 or avg_diff > self.change_threshold * 0.8:
            self.hand_present = True
            self.candidate_squares = None
            self.stable_count = 0
            return None

        if self.hand_present and len(changed) <= 10:
            # Hand just left, start stability counting
            self.hand_present = False

        if not changed:
            self.candidate_squares = None
            self.stable_count = 0
            return None

        # Stability check: same set of changed squares for N frames
        if changed == self.candidate_squares:
            self.stable_count += 1
        else:
            self.candidate_squares = changed
            self.stable_count = 1

        if self.stable_count >= self.stability_frames:
            result = sorted(self.candidate_squares)
            # Update reference to current frame
            self.reference_frame = gray.copy()
            self.candidate_squares = None
            self.stable_count = 0
            return result

        return None

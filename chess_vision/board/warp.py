"""Perspective warp to transform detected board to top-down view."""

import cv2
import numpy as np

from chess_vision.config import WARP_SIZE


def compute_homography(corners: np.ndarray, target_size: int = WARP_SIZE) -> np.ndarray:
    """Compute perspective transform from 4 corners to a square target.

    Args:
        corners: (4, 2) array of source corner points.
        target_size: Output image dimension (square).

    Returns:
        3x3 homography matrix.
    """
    raise NotImplementedError


def warp_board(image: np.ndarray, homography: np.ndarray, target_size: int = WARP_SIZE) -> np.ndarray:
    """Apply homography to get top-down board view.

    Returns warped image of shape (target_size, target_size, 3).
    """
    raise NotImplementedError

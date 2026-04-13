"""Perspective warp to transform detected board to top-down view."""

import cv2
import numpy as np

from chess_vision.config import WARP_SIZE


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.

    Works regardless of the input order of the corners.
    """
    corners = corners.reshape(4, 2).astype(np.float32)

    # Sort by sum (x+y): smallest = top-left, largest = bottom-right
    s = corners.sum(axis=1)
    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]

    # Sort by difference (y-x): smallest = top-right, largest = bottom-left
    d = np.diff(corners, axis=1).flatten()
    tr = corners[np.argmin(d)]
    bl = corners[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_homography(corners: np.ndarray, target_size: int = WARP_SIZE) -> np.ndarray:
    """Compute perspective transform from 4 corners to a square target.

    Args:
        corners: (4, 2) array of source corner points (any order).
        target_size: Output image dimension (square).

    Returns:
        3x3 homography matrix.
    """
    ordered = order_corners(corners)
    dst = np.array([
        [0, 0],
        [target_size, 0],
        [target_size, target_size],
        [0, target_size],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(ordered, dst)


def warp_board(image: np.ndarray, homography: np.ndarray, target_size: int = WARP_SIZE) -> np.ndarray:
    """Apply homography to get top-down board view.

    Returns warped image of shape (target_size, target_size, 3).
    """
    return cv2.warpPerspective(image, homography, (target_size, target_size))

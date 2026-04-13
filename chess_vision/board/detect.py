"""Board detection using Hough lines + RANSAC corner finding."""

import cv2
import numpy as np


def detect_lines(image: np.ndarray) -> tuple[list, list]:
    """Detect horizontal and vertical lines via Hough transform.

    Returns (horizontal_lines, vertical_lines) as lists of (rho, theta) tuples.
    """
    raise NotImplementedError


def find_corners(h_lines: list, v_lines: list) -> np.ndarray:
    """Find 4 board corner points from line intersections using RANSAC.

    Returns ndarray of shape (4, 2) with corner coordinates.
    """
    raise NotImplementedError


def detect_board(image: np.ndarray) -> np.ndarray | None:
    """Detect chessboard in image and return 4 corner points.

    Returns ndarray of shape (4, 2) or None if detection fails.
    """
    raise NotImplementedError

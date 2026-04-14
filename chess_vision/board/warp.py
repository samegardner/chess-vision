"""Corner ordering for board perspective."""

import numpy as np


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

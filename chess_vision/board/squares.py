"""Extract individual square images from a warped board."""

import numpy as np

from chess_vision.config import WARP_SIZE, CONTEXT_PAD


def extract_squares(
    warped_board: np.ndarray,
    padding: float = CONTEXT_PAD,
) -> list[np.ndarray]:
    """Extract 64 square images with contextual padding.

    Squares are indexed 0-63: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63.

    Args:
        warped_board: Top-down board image (WARP_SIZE x WARP_SIZE).
        padding: Extra padding ratio around each square (0.5 = 50%).

    Returns:
        List of 64 square images.
    """
    raise NotImplementedError


def square_index_to_name(index: int) -> str:
    """Convert 0-63 index to algebraic notation (e.g., 0 -> 'a1')."""
    file = chr(ord("a") + index % 8)
    rank = str(index // 8 + 1)
    return file + rank


def name_to_square_index(name: str) -> int:
    """Convert algebraic notation to 0-63 index (e.g., 'a1' -> 0)."""
    file = ord(name[0]) - ord("a")
    rank = int(name[1]) - 1
    return rank * 8 + file

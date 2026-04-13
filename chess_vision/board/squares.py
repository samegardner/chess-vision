"""Extract individual square images from a warped board."""

import numpy as np

from chess_vision.config import WARP_SIZE, CONTEXT_PAD


def extract_squares(
    warped_board: np.ndarray,
    padding: float = CONTEXT_PAD,
) -> list[np.ndarray]:
    """Extract 64 square images with contextual padding.

    Squares are indexed 0-63: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63.
    The warped board has rank 8 at the top (row 0) and rank 1 at the bottom.

    Args:
        warped_board: Top-down board image (WARP_SIZE x WARP_SIZE).
        padding: Extra padding ratio around each square (0.5 = 50%).

    Returns:
        List of 64 square images.
    """
    h, w = warped_board.shape[:2]
    sq_h = h / 8
    sq_w = w / 8
    pad_h = sq_h * padding
    pad_w = sq_w * padding

    squares = []
    # rank 1 (bottom of image) to rank 8 (top of image)
    for rank in range(8):
        for file in range(8):
            # Rank 1 = bottom of image = row 7, rank 8 = top = row 0
            row = 7 - rank
            col = file

            # Center of this square
            cy = row * sq_h + sq_h / 2
            cx = col * sq_w + sq_w / 2

            # Padded bounding box, clamped to image bounds
            y1 = max(0, int(cy - sq_h / 2 - pad_h))
            y2 = min(h, int(cy + sq_h / 2 + pad_h))
            x1 = max(0, int(cx - sq_w / 2 - pad_w))
            x2 = min(w, int(cx + sq_w / 2 + pad_w))

            squares.append(warped_board[y1:y2, x1:x2].copy())

    return squares


def remap_board_state(board_state: dict[str, str | None]) -> dict[str, str | None]:
    """Remap a board state from flipped orientation (black on bottom) to standard.

    When the camera sees the board from Black's side, the warped image has
    a8 at bottom-left instead of a1. This function swaps all square names
    so the board state matches standard orientation (white on bottom).
    """
    remapped = {}
    for sq_name, piece in board_state.items():
        flipped_file = chr(ord("h") - (ord(sq_name[0]) - ord("a")))
        flipped_rank = str(9 - int(sq_name[1]))
        remapped[f"{flipped_file}{flipped_rank}"] = piece
    return remapped


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

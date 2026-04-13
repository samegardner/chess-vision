"""Auto-label squares from starting position calibration photos."""

from pathlib import Path

import cv2
import numpy as np

from chess_vision.board.detect import detect_board
from chess_vision.board.warp import compute_homography, warp_board
from chess_vision.board.squares import extract_squares, square_index_to_name
from chess_vision.training.dataset import FEN_TO_PIECE_CLASS

# Known starting position: square name -> FEN piece character (None = empty)
STARTING_POSITION: dict[str, str | None] = {
    "a1": "R", "b1": "N", "c1": "B", "d1": "Q", "e1": "K", "f1": "B", "g1": "N", "h1": "R",
    "a2": "P", "b2": "P", "c2": "P", "d2": "P", "e2": "P", "f2": "P", "g2": "P", "h2": "P",
    "a3": None, "b3": None, "c3": None, "d3": None, "e3": None, "f3": None, "g3": None, "h3": None,
    "a4": None, "b4": None, "c4": None, "d4": None, "e4": None, "f4": None, "g4": None, "h4": None,
    "a5": None, "b5": None, "c5": None, "d5": None, "e5": None, "f5": None, "g5": None, "h5": None,
    "a6": None, "b6": None, "c6": None, "d6": None, "e6": None, "f6": None, "g6": None, "h6": None,
    "a7": "p", "b7": "p", "c7": "p", "d7": "p", "e7": "p", "f7": "p", "g7": "p", "h7": "p",
    "a8": "r", "b8": "n", "c8": "b", "d8": "q", "e8": "k", "f8": "b", "g8": "n", "h8": "r",
}


def label_calibration_squares(
    white_photo: np.ndarray,
    black_photo: np.ndarray,
) -> list[tuple[np.ndarray, str | None]]:
    """Extract and label all 128 squares from 2 calibration photos.

    Detects the board in each photo, warps to top-down view, extracts squares,
    and assigns labels from the known starting position.

    Args:
        white_photo: Photo with white pieces closest to camera (rank 1 nearest).
        black_photo: Photo with black pieces closest to camera (rank 8 nearest).

    Returns:
        List of (square_image, label) tuples. Label is FEN char or None (empty).
        128 total (64 from each photo).
    """
    results = []

    for photo, is_white_view in [(white_photo, True), (black_photo, False)]:
        corners = detect_board(photo)
        if corners is None:
            raise ValueError(
                f"Could not detect board in {'white' if is_white_view else 'black'}-side photo. "
                "Ensure the full board is visible with good contrast."
            )

        H = compute_homography(corners)
        warped = warp_board(photo, H)
        squares = extract_squares(warped)

        for i, sq_img in enumerate(squares):
            sq_name = square_index_to_name(i)

            if is_white_view:
                # White view: board is oriented normally (a1 bottom-left)
                label = STARTING_POSITION[sq_name]
            else:
                # Black view: board is rotated 180 degrees
                # a1 in the warped image actually shows h8, etc.
                flipped_file = chr(ord("h") - (ord(sq_name[0]) - ord("a")))
                flipped_rank = str(9 - int(sq_name[1]))
                label = STARTING_POSITION[f"{flipped_file}{flipped_rank}"]

            results.append((sq_img, label))

    return results


def save_calibration_squares(
    labeled_squares: list[tuple[np.ndarray, str | None]],
    output_dir: Path,
) -> None:
    """Save labeled squares to disk in ImageFolder format.

    Creates subdirectories for each class (empty, white_pawn, etc.)
    with individual square images as PNGs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for sq_img, label in labeled_squares:
        if label is None:
            class_name = "empty"
        else:
            class_name = FEN_TO_PIECE_CLASS[label]

        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        idx = counts.get(class_name, 0)
        counts[class_name] = idx + 1

        cv2.imwrite(str(class_dir / f"{class_name}_{idx:04d}.png"), sq_img)

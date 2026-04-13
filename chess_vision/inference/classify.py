"""Full board classification pipeline."""

import numpy as np

from chess_vision.inference.onnx_runtime import ONNXClassifier
from chess_vision.board.squares import square_index_to_name
from chess_vision.models.piece import PIECE_CLASSES, PIECE_TO_FEN


def classify_board(
    square_images: list[np.ndarray],
    occupancy_model: ONNXClassifier,
    piece_model: ONNXClassifier,
    occupancy_threshold: float = 0.5,
) -> dict[str, str | None]:
    """Classify all 64 squares using both models.

    Returns dict mapping square names to FEN piece characters (or None for empty).
    E.g., {'a1': 'R', 'a2': 'P', 'e4': None, ...}
    """
    raise NotImplementedError


def board_to_fen(board_state: dict[str, str | None]) -> str:
    """Convert board state dict to FEN position string.

    Walks ranks 8 to 1, files a to h.
    """
    ranks = []
    for rank in range(8, 0, -1):
        empty_count = 0
        rank_str = ""
        for file in "abcdefgh":
            piece = board_state.get(f"{file}{rank}")
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += piece
        if empty_count > 0:
            rank_str += str(empty_count)
        ranks.append(rank_str)
    return "/".join(ranks)

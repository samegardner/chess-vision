"""Full board classification pipeline."""

import cv2
import numpy as np
from PIL import Image

from chess_vision.inference.onnx_runtime import ONNXClassifier
from chess_vision.board.squares import square_index_to_name
from chess_vision.models.piece import PIECE_CLASSES, PIECE_TO_FEN
from chess_vision.training.augment import get_eval_transforms
from chess_vision.config import INPUT_SIZE


def _prepare_batch(square_images: list[np.ndarray], input_size: int = INPUT_SIZE) -> np.ndarray:
    """Convert list of BGR numpy images to a normalized float32 batch.

    Returns (N, 3, H, W) float32 array suitable for ONNX inference.
    """
    transform = get_eval_transforms(size=input_size)
    tensors = []
    for img in square_images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensors.append(transform(pil_img).numpy())
    return np.stack(tensors).astype(np.float32)


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
    assert len(square_images) == 64

    # Detect input size from ONNX model (handles both 100x100 and 224x224 models)
    occ_input_shape = occupancy_model.session.get_inputs()[0].shape
    input_size = occ_input_shape[2] if isinstance(occ_input_shape[2], int) else INPUT_SIZE

    # Stage 1: occupancy classification on all 64 squares
    batch = _prepare_batch(square_images, input_size=input_size)
    occ_logits = occupancy_model.predict(batch)  # (64, 1)
    occ_probs = 1 / (1 + np.exp(-occ_logits.flatten()))  # sigmoid
    occupied = occ_probs > occupancy_threshold

    # Stage 2: piece classification on occupied squares only
    occupied_indices = [i for i in range(64) if occupied[i]]
    board_state: dict[str, str | None] = {}

    if occupied_indices:
        occ_images = [square_images[i] for i in occupied_indices]
        piece_batch = _prepare_batch(occ_images, input_size=input_size)
        piece_logits = piece_model.predict(piece_batch)  # (N, 12)
        pred_classes = np.argmax(piece_logits, axis=1)

        for i, idx in enumerate(occupied_indices):
            class_name = PIECE_CLASSES[pred_classes[i]]
            board_state[square_index_to_name(idx)] = PIECE_TO_FEN[class_name]

    # Fill in empty squares
    for i in range(64):
        name = square_index_to_name(i)
        if name not in board_state:
            board_state[name] = None

    return board_state


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

"""ResNet-18 12-class classifier for chess piece identification."""

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image

from chess_vision.training.augment import get_eval_transforms

PIECE_CLASSES = [
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
]

# Mapping from class name to FEN character
PIECE_TO_FEN = {
    "white_pawn": "P",
    "white_knight": "N",
    "white_bishop": "B",
    "white_rook": "R",
    "white_queen": "Q",
    "white_king": "K",
    "black_pawn": "p",
    "black_knight": "n",
    "black_bishop": "b",
    "black_rook": "r",
    "black_queen": "q",
    "black_king": "k",
}


def create_piece_model(pretrained: bool = True) -> nn.Module:
    """Create ResNet-18 with final FC layer replaced for 12-class classification."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(PIECE_CLASSES))
    return model


def predict_pieces(
    model: nn.Module,
    square_images: list[np.ndarray],
    occupied: list[bool],
) -> list[str | None]:
    """Predict piece type for occupied squares.

    Args:
        model: Trained piece classification model.
        square_images: List of BGR numpy arrays (from OpenCV), one per square.
        occupied: List of booleans indicating which squares are occupied.

    Returns:
        List of FEN characters (e.g., 'P', 'n') or None for empty squares.
    """
    assert len(square_images) == len(occupied)

    results: list[str | None] = [None] * len(square_images)

    # Collect only occupied square images
    occupied_indices = [i for i, occ in enumerate(occupied) if occ]
    if not occupied_indices:
        return results

    transform = get_eval_transforms()
    tensors = []
    for idx in occupied_indices:
        rgb = cv2.cvtColor(square_images[idx], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensors.append(transform(pil_img))

    batch = torch.stack(tensors)
    device = next(model.parameters()).device
    batch = batch.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(batch)  # (N, 12)
        pred_indices = torch.argmax(logits, dim=1)

    for i, idx in enumerate(occupied_indices):
        class_name = PIECE_CLASSES[pred_indices[i].item()]
        results[idx] = PIECE_TO_FEN[class_name]

    return results

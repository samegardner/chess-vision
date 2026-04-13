"""ResNet-18 12-class classifier for chess piece identification."""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

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

    Returns list of FEN characters (e.g., 'P', 'n') or None for empty squares.
    """
    raise NotImplementedError

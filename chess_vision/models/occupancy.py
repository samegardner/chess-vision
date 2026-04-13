"""ResNet-18 binary classifier for square occupancy (empty vs occupied)."""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


def create_occupancy_model(pretrained: bool = True) -> nn.Module:
    """Create ResNet-18 with final FC layer replaced for binary classification.

    Output: single logit (use sigmoid for probability).
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def predict_occupancy(
    model: nn.Module,
    square_images: list[np.ndarray],
    threshold: float = 0.5,
) -> list[bool]:
    """Batch predict occupancy for square images.

    Returns list of booleans (True = occupied).
    """
    raise NotImplementedError

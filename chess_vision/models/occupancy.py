"""ResNet-18 binary classifier for square occupancy (empty vs occupied)."""

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image

from chess_vision.training.augment import get_eval_transforms


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

    Args:
        model: Trained occupancy model.
        square_images: List of BGR numpy arrays (from OpenCV).
        threshold: Sigmoid probability above which a square is "occupied".

    Returns:
        List of booleans (True = occupied).
    """
    if not square_images:
        return []

    transform = get_eval_transforms()
    tensors = []
    for img in square_images:
        # Convert BGR (OpenCV) to RGB PIL Image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensors.append(transform(pil_img))

    batch = torch.stack(tensors)
    device = next(model.parameters()).device
    batch = batch.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(batch).squeeze(-1)  # (N,)
        probs = torch.sigmoid(logits)

    return [p.item() > threshold for p in probs]

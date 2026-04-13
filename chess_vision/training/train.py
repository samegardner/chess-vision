"""Training loop for occupancy and piece classification models."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    output_dir: Path = Path("models/checkpoints"),
) -> nn.Module:
    """Train with Adam, cosine LR schedule, early stopping.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        lr: Learning rate.
        output_dir: Directory for saving checkpoints.

    Returns:
        Trained model with best validation weights loaded.
    """
    raise NotImplementedError

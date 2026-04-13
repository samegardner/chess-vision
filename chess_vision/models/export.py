"""Export PyTorch models to ONNX format."""

from pathlib import Path

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: tuple = (1, 3, 224, 224),
) -> None:
    """Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model.
        output_path: Where to save the .onnx file.
        input_size: Input tensor shape (batch, channels, height, width).
    """
    raise NotImplementedError

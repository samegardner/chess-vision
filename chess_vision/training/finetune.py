"""Fine-tune models on calibration data from a specific chess set."""

from pathlib import Path


def finetune_from_calibration(
    base_occupancy_model: Path,
    base_piece_model: Path,
    calibration_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    lr: float = 1e-4,
) -> tuple[Path, Path]:
    """Fine-tune both models on calibration squares.

    Freezes early ResNet layers, trains final blocks + FC head.
    Exports fine-tuned models to ONNX.

    Args:
        base_occupancy_model: Path to base occupancy .pt checkpoint.
        base_piece_model: Path to base piece .pt checkpoint.
        calibration_dir: Directory containing labeled square images.
        output_dir: Where to save fine-tuned ONNX models.
        epochs: Fine-tuning epochs.
        lr: Fine-tuning learning rate.

    Returns:
        (occupancy_onnx_path, piece_onnx_path)
    """
    raise NotImplementedError

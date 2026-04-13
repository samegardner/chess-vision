"""Fine-tune models on calibration data from a specific chess set."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from chess_vision.models.occupancy import create_occupancy_model
from chess_vision.models.piece import create_piece_model
from chess_vision.models.export import export_to_onnx
from chess_vision.training.dataset import CalibrationDataset
from chess_vision.training.augment import get_calibration_transforms
from chess_vision.training.train import train_model


def _freeze_early_layers(model: torch.nn.Module) -> None:
    """Freeze all layers except layer4 and fc (final block + classification head)."""
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False


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
        calibration_dir: Directory containing labeled square images
            organized as: {class_name}/*.png
        output_dir: Where to save fine-tuned ONNX models.
        epochs: Fine-tuning epochs.
        lr: Fine-tuning learning rate.

    Returns:
        (occupancy_onnx_path, piece_onnx_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = get_calibration_transforms()

    # --- Occupancy model ---
    print("Fine-tuning occupancy model...")
    occ_model = create_occupancy_model(pretrained=False)
    occ_model.load_state_dict(torch.load(base_occupancy_model, weights_only=True))
    _freeze_early_layers(occ_model)

    # Build occupancy dataset: remap classes to binary (empty=0, occupied=1)
    occ_dataset = CalibrationDataset(calibration_dir, transform=transform)
    # Remap: any non-empty class -> occupied (1), empty -> 0
    empty_idx = occ_dataset.class_to_idx.get("empty", -1)
    for i, (path, label) in enumerate(occ_dataset.samples):
        occ_dataset.samples[i] = (path, 0 if label == empty_idx else 1)

    occ_train, occ_val = _split_dataset(occ_dataset)
    occ_train_loader = DataLoader(occ_train, batch_size=16, shuffle=True)
    occ_val_loader = DataLoader(occ_val, batch_size=16)

    occ_model = train_model(
        occ_model, occ_train_loader, occ_val_loader,
        epochs=epochs, lr=lr, model_name="occupancy_ft",
        output_dir=output_dir,
    )
    occ_onnx = output_dir / "occupancy.onnx"
    export_to_onnx(occ_model, occ_onnx)

    # --- Piece model ---
    print("Fine-tuning piece model...")
    piece_model = create_piece_model(pretrained=False)
    piece_model.load_state_dict(torch.load(base_piece_model, weights_only=True))
    _freeze_early_layers(piece_model)

    # Build piece dataset: only occupied squares, remap to 0-11 class indices
    from chess_vision.models.piece import PIECE_CLASSES
    piece_dataset = CalibrationDataset(calibration_dir, transform=transform)
    # Filter to only piece classes (exclude "empty")
    piece_samples = []
    for path, label in piece_dataset.samples:
        class_name = piece_dataset.classes[label]
        if class_name in PIECE_CLASSES:
            piece_samples.append((path, PIECE_CLASSES.index(class_name)))
    piece_dataset.samples = piece_samples

    if len(piece_samples) > 0:
        piece_train, piece_val = _split_dataset(piece_dataset)
        piece_train_loader = DataLoader(piece_train, batch_size=16, shuffle=True)
        piece_val_loader = DataLoader(piece_val, batch_size=16)

        piece_model = train_model(
            piece_model, piece_train_loader, piece_val_loader,
            epochs=epochs, lr=lr, model_name="piece_ft",
            output_dir=output_dir,
        )
    piece_onnx = output_dir / "piece.onnx"
    export_to_onnx(piece_model, piece_onnx)

    return occ_onnx, piece_onnx


def _split_dataset(dataset, val_ratio: float = 0.2):
    """Split a dataset into train/val."""
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])

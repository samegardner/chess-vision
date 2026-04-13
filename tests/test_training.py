"""Tests for training infrastructure."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from chess_vision.models.occupancy import create_occupancy_model, predict_occupancy
from chess_vision.models.piece import create_piece_model, predict_pieces
from chess_vision.models.export import export_to_onnx
from chess_vision.training.train import train_model


def test_export_to_onnx():
    """Test ONNX export produces a valid file."""
    model = create_occupancy_model(pretrained=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.onnx"
        export_to_onnx(model, path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_predict_occupancy_returns_booleans():
    """Predict occupancy returns correct number of booleans."""
    model = create_occupancy_model(pretrained=False)
    model.eval()
    # Create 4 dummy BGR images
    images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
    results = predict_occupancy(model, images)
    assert len(results) == 4
    assert all(isinstance(r, bool) for r in results)


def test_predict_pieces_returns_correct_structure():
    """Predict pieces returns FEN chars for occupied, None for empty."""
    model = create_piece_model(pretrained=False)
    model.eval()
    images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
    occupied = [True, False, True, False]
    results = predict_pieces(model, images, occupied)
    assert len(results) == 4
    assert results[0] is not None  # occupied
    assert results[1] is None      # empty
    assert results[2] is not None  # occupied
    assert results[3] is None      # empty


def test_predict_pieces_all_empty():
    """All empty squares should return all None."""
    model = create_piece_model(pretrained=False)
    model.eval()
    images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
    results = predict_pieces(model, images, [False, False, False])
    assert results == [None, None, None]


def test_train_model_runs():
    """Training loop completes without errors on tiny synthetic data."""
    model = create_occupancy_model(pretrained=False)

    # Create tiny dummy dataset (20 samples, binary labels)
    x = torch.randn(20, 3, 224, 224)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        trained = train_model(
            model, train_loader, val_loader,
            epochs=2, lr=1e-3, output_dir=Path(tmpdir),
            model_name="test", patience=5,
        )
        assert trained is not None
        # Should have saved a checkpoint
        assert (Path(tmpdir) / "test_best.pt").exists()

"""Tests for piece classification model."""


def test_create_piece_model():
    """Test model creation (without pretrained weights to avoid download in CI)."""
    from chess_vision.models.piece import create_piece_model, PIECE_CLASSES
    model = create_piece_model(pretrained=False)
    assert model.fc.out_features == len(PIECE_CLASSES)

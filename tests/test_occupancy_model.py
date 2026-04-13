"""Tests for occupancy model."""


def test_create_occupancy_model():
    """Test model creation (without pretrained weights to avoid download in CI)."""
    from chess_vision.models.occupancy import create_occupancy_model
    model = create_occupancy_model(pretrained=False)
    assert model.fc.out_features == 1

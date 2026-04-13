"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_board_image():
    """A dummy 800x800 board image for testing."""
    return np.zeros((800, 800, 3), dtype=np.uint8)


@pytest.fixture
def starting_board_state():
    """The standard chess starting position as a board state dict."""
    from chess_vision.calibration.label import STARTING_POSITION
    return dict(STARTING_POSITION)

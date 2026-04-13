"""Tests for square extraction and naming."""

import numpy as np

from chess_vision.board.squares import (
    extract_squares,
    square_index_to_name,
    name_to_square_index,
)


def test_square_index_to_name():
    assert square_index_to_name(0) == "a1"
    assert square_index_to_name(7) == "h1"
    assert square_index_to_name(8) == "a2"
    assert square_index_to_name(63) == "h8"


def test_name_to_square_index():
    assert name_to_square_index("a1") == 0
    assert name_to_square_index("h1") == 7
    assert name_to_square_index("a2") == 8
    assert name_to_square_index("h8") == 63


def test_roundtrip():
    for i in range(64):
        assert name_to_square_index(square_index_to_name(i)) == i


def test_extract_squares_count():
    """Should return exactly 64 squares."""
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    squares = extract_squares(board, padding=0.0)
    assert len(squares) == 64


def test_extract_squares_no_padding_size():
    """Without padding, each square should be 100x100 from an 800x800 board."""
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    squares = extract_squares(board, padding=0.0)
    for sq in squares:
        assert sq.shape[0] == 100
        assert sq.shape[1] == 100


def test_extract_squares_with_padding_larger():
    """With padding, squares should be larger than the base 100x100."""
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    squares = extract_squares(board, padding=0.5)
    # Interior squares should be 200x200 (100 + 50 padding each side)
    # Edge squares may be smaller due to clamping
    # Check a center square (e4 = index 28)
    e4 = squares[name_to_square_index("e4")]
    assert e4.shape[0] > 100
    assert e4.shape[1] > 100


def test_extract_squares_reads_correct_region():
    """Verify that specific squares map to the correct board regions."""
    board = np.zeros((800, 800, 3), dtype=np.uint8)

    # Color a1 region (bottom-left of board = rows 700-800, cols 0-100) red
    board[700:800, 0:100] = [0, 0, 255]

    # Color h8 region (top-right = rows 0-100, cols 700-800) blue
    board[0:100, 700:800] = [255, 0, 0]

    squares = extract_squares(board, padding=0.0)

    a1 = squares[name_to_square_index("a1")]
    assert np.mean(a1[:, :, 2]) > 200  # Red channel should be high

    h8 = squares[name_to_square_index("h8")]
    assert np.mean(h8[:, :, 0]) > 200  # Blue channel should be high


def test_extract_squares_edge_clamping():
    """Corner squares with padding should be clamped to image bounds."""
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    squares = extract_squares(board, padding=0.5)

    # a1 is bottom-left corner. With 50% padding it would extend below/left
    # of the image, but should be clamped.
    a1 = squares[name_to_square_index("a1")]
    assert a1.shape[0] > 0
    assert a1.shape[1] > 0

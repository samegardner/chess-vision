"""Tests for square extraction and naming."""

from chess_vision.board.squares import square_index_to_name, name_to_square_index


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

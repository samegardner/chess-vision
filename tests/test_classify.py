"""Tests for board classification."""

from chess_vision.inference.classify import board_to_fen


def test_board_to_fen_starting_position():
    from chess_vision.calibration.label import STARTING_POSITION
    fen = board_to_fen(STARTING_POSITION)
    assert fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_board_to_fen_empty():
    empty = {f"{f}{r}": None for f in "abcdefgh" for r in range(1, 9)}
    fen = board_to_fen(empty)
    assert fen == "8/8/8/8/8/8/8/8"

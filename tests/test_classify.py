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


def test_board_to_fen_midgame():
    """Test FEN with mixed empty/occupied squares in the same rank."""
    from chess_vision.calibration.label import STARTING_POSITION

    state = dict(STARTING_POSITION)
    # 1. e4 e5 2. Nf3
    state["e2"] = None
    state["e4"] = "P"
    state["e7"] = None
    state["e5"] = "p"
    state["g1"] = None
    state["f3"] = "N"

    fen = board_to_fen(state)
    assert fen == "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"

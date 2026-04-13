"""Tests for legal move resolution."""

import chess
from chess_vision.game.rules import resolve_move, detect_orientation
from chess_vision.calibration.label import STARTING_POSITION


def _board_state_after_move(board: chess.Board) -> dict[str, str | None]:
    """Convert python-chess board to our board state dict."""
    state = {}
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        piece = board.piece_at(sq)
        state[name] = piece.symbol() if piece else None
    return state


def test_resolve_normal_move():
    board = chess.Board()
    board_after = board.copy()
    board_after.push_san("e4")
    new_state = _board_state_after_move(board_after)

    move = resolve_move(["e2", "e4"], board, new_state)
    assert move is not None
    assert move == chess.Move.from_uci("e2e4")


def test_resolve_capture():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")

    board_after = board.copy()
    board_after.push_san("exd5")
    new_state = _board_state_after_move(board_after)

    move = resolve_move(["e4", "d5"], board, new_state)
    assert move is not None
    assert move == chess.Move.from_uci("e4d5")


def test_detect_orientation_white_bottom():
    assert detect_orientation(STARTING_POSITION) == "white_bottom"

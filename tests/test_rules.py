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


def test_resolve_kingside_castling():
    board = chess.Board()
    # Clear path for white kingside castle
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Bc5")

    board_after = board.copy()
    board_after.push_san("O-O")
    new_state = _board_state_after_move(board_after)

    # Castling changes 4 squares: e1, f1, g1, h1
    move = resolve_move(["e1", "f1", "g1", "h1"], board, new_state)
    assert move is not None
    assert move == chess.Move.from_uci("e1g1")


def test_resolve_queenside_castling():
    board = chess.Board()
    board.push_san("d4")
    board.push_san("d5")
    board.push_san("Nc3")
    board.push_san("Nc6")
    board.push_san("Bf4")
    board.push_san("Bf5")
    board.push_san("Qd2")
    board.push_san("Qd7")

    board_after = board.copy()
    board_after.push_san("O-O-O")
    new_state = _board_state_after_move(board_after)

    # Queenside castling changes: a1, c1, d1, e1
    move = resolve_move(["a1", "c1", "d1", "e1"], board, new_state)
    assert move is not None
    assert move == chess.Move.from_uci("e1c1")


def test_resolve_en_passant():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("a6")
    board.push_san("e5")
    board.push_san("d5")  # Black pushes d5, enabling en passant

    board_after = board.copy()
    board_after.push_san("exd6")  # En passant capture
    new_state = _board_state_after_move(board_after)

    # En passant changes 3 squares: e5 (origin), d6 (destination), d5 (captured pawn)
    move = resolve_move(["e5", "d6", "d5"], board, new_state)
    assert move is not None
    assert move == chess.Move.from_uci("e5d6")


def test_resolve_promotion():
    # Set up a position where white can promote
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    board_after = board.copy()
    board_after.push(chess.Move.from_uci("a7a8q"))
    new_state = _board_state_after_move(board_after)

    move = resolve_move(["a7", "a8"], board, new_state)
    assert move is not None
    assert move.promotion == chess.QUEEN


def test_resolve_promotion_to_knight():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    board_after = board.copy()
    board_after.push(chess.Move.from_uci("a7a8n"))
    new_state = _board_state_after_move(board_after)

    move = resolve_move(["a7", "a8"], board, new_state)
    assert move is not None
    assert move.promotion == chess.KNIGHT


def test_detect_orientation_white_bottom():
    assert detect_orientation(STARTING_POSITION) == "white_bottom"


def test_detect_orientation_white_top():
    # Flip the board: white pieces on ranks 7-8, black on ranks 1-2
    flipped = {}
    for sq_name, piece in STARTING_POSITION.items():
        file = sq_name[0]
        rank = 9 - int(sq_name[1])  # Flip rank
        flipped[f"{file}{rank}"] = piece
    assert detect_orientation(flipped) == "white_top"


def test_detect_orientation_error():
    # Board with no king on e1 or e8
    bad_state = {f"{f}{r}": None for f in "abcdefgh" for r in range(1, 9)}
    import pytest
    with pytest.raises(ValueError, match="Cannot detect orientation"):
        detect_orientation(bad_state)

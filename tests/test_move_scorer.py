"""Tests for move scoring logic."""

import chess
import numpy as np

from chess_vision.game.move_scorer import (
    calculate_score, get_move_data, get_move_pairs, combine_data,
    MoveDetectorV2, LABELS, LABEL_MAP,
)


def _make_state_with_starting_position():
    """Create a state matrix that looks like the starting position."""
    state = np.zeros((64, 12), dtype=np.float32)
    board = chess.Board()
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            state[sq, LABEL_MAP[piece.symbol()]] = 0.8
    return state


def test_starting_position_no_move():
    """No move should score positive when the board looks like starting position."""
    state = _make_state_with_starting_position()
    board = chess.Board()

    for move in board.legal_moves:
        data = get_move_data(board, move)
        score = calculate_score(state, data)
        assert score < 0, f"{board.san(move)} scored {score} on starting position"


def test_e4_scores_positive_after_move():
    """After e2-e4 is played, the state should make e4 score positive."""
    state = _make_state_with_starting_position()
    board = chess.Board()

    # Simulate e2-e4: e2 becomes empty, e4 gets a white pawn
    state[chess.E2] = np.zeros(12)  # Empty
    state[chess.E4, LABEL_MAP["P"]] = 0.8  # White pawn

    move = board.parse_san("e4")
    data = get_move_data(board, move)
    score = calculate_score(state, data)
    assert score > 0, f"e4 should score positive, got {score}"


def test_castling_move_data():
    """Castling should include rook squares."""
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    move = board.parse_san("O-O")
    data = get_move_data(board, move)
    assert chess.E1 in data.from_squares
    assert chess.H1 in data.from_squares  # Rook leaves h1
    assert chess.G1 in data.to_squares  # King goes to g1
    assert chess.F1 in data.to_squares  # Rook goes to f1


def test_en_passant_move_data():
    """En passant should include the captured pawn's square."""
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    move = board.parse_san("exd6")
    data = get_move_data(board, move)
    assert chess.D5 in data.from_squares  # Captured pawn disappears


def test_combine_data_removes_overlaps():
    """Combined move data should not double-count overlapping squares."""
    move1 = get_move_data(chess.Board(), chess.Board().parse_san("e4"))
    board2 = chess.Board()
    board2.push_san("e4")
    move2 = get_move_data(board2, board2.parse_san("e5"))
    combined = combine_data(move1, move2)
    # e4 is in both moves; combined should use move2's version
    assert len(set(combined.from_squares)) == len(combined.from_squares)  # No duplicates


def test_greedy_delay():
    """Greedy fallback should not fire before the delay."""
    detector = MoveDetectorV2(greedy_delay=1.0)
    state = _make_state_with_starting_position()
    board = chess.Board()

    # Simulate e4 played
    state[chess.E2] = np.zeros(12)
    state[chess.E4, LABEL_MAP["P"]] = 0.8

    # First call: should not fire (timer just started)
    result = detector.detect_move(board, state)
    assert result is None


def test_pgn_generation():
    """PGN should contain the moves played."""
    from chess_vision.game.pgn import generate_pgn
    moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
    pgn = generate_pgn(moves, white_name="W", black_name="B")
    assert "e4" in pgn
    assert "e5" in pgn
    assert "W" in pgn

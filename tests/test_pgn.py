"""Tests for PGN generation."""

import chess
from chess_vision.game.pgn import generate_pgn


def test_generate_pgn_basic():
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
    ]
    pgn = generate_pgn(moves, white_name="Sam", black_name="Opponent")
    assert "Sam" in pgn
    assert "Opponent" in pgn
    assert "e4" in pgn
    assert "e5" in pgn
    assert "Nf3" in pgn


def test_generate_pgn_empty_game():
    pgn = generate_pgn([])
    assert "Result" in pgn

"""PGN file generation."""

from datetime import date
from pathlib import Path

import chess
import chess.pgn


def generate_pgn(
    moves: list[chess.Move],
    white_name: str = "White",
    black_name: str = "Black",
    game_date: str | None = None,
    result: str = "*",
) -> str:
    """Generate PGN string from a list of moves.

    Args:
        moves: List of chess.Move objects in order.
        white_name: White player name.
        black_name: Black player name.
        game_date: Date string (YYYY.MM.DD). Defaults to today.
        result: Game result ('1-0', '0-1', '1/2-1/2', '*').

    Returns:
        PGN formatted string.
    """
    game = chess.pgn.Game()
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Date"] = game_date or date.today().strftime("%Y.%m.%d")
    game.headers["Result"] = result

    node = game
    for move in moves:
        node = node.add_variation(move)

    return str(game)


def save_pgn(pgn_str: str, output_path: Path) -> None:
    """Save PGN string to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pgn_str + "\n")

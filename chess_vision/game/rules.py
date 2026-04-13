"""Legal move resolution using python-chess."""

import chess


def resolve_move(
    changed_squares: list[str],
    board: chess.Board,
    new_board_state: dict[str, str | None],
) -> chess.Move | None:
    """Find the legal move that matches the observed board change.

    Handles normal moves, captures, castling (4 squares), en passant (3 squares),
    and promotion (piece type change on back rank).

    Args:
        changed_squares: List of square names that changed.
        board: Current board state (before the move).
        new_board_state: Observed board state after the move.

    Returns:
        The matching legal move, or None if no legal move matches.
    """
    for move in board.legal_moves:
        board.push(move)
        matches = True
        for sq_name in changed_squares:
            sq = chess.parse_square(sq_name)
            piece = board.piece_at(sq)
            expected = piece.symbol() if piece else None
            observed = new_board_state.get(sq_name)
            if expected != observed:
                matches = False
                board.pop()
                break
        else:
            board.pop()
        if matches:
            return move
    return None


def detect_orientation(board_state: dict[str, str | None]) -> str:
    """Detect board orientation from starting position.

    Returns 'white_bottom' if white pieces are on ranks 1-2,
    'white_top' if white pieces are on ranks 7-8.
    """
    white_on_bottom = board_state.get("e1") == "K"
    white_on_top = board_state.get("e8") == "K"

    if white_on_bottom:
        return "white_bottom"
    elif white_on_top:
        return "white_top"
    else:
        raise ValueError("Cannot detect orientation: no king found on e1 or e8")

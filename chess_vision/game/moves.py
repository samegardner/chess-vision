"""Move detection via frame differencing with stability window."""

from chess_vision.config import STABILITY_FRAMES


class MoveDetector:
    """Detects when a move has been made by comparing board states."""

    def __init__(self, stability_frames: int = STABILITY_FRAMES):
        self.stability_frames = stability_frames
        self.previous_board: dict[str, str | None] | None = None
        self.candidate_changes: list[str] | None = None
        self.stable_count: int = 0

    def set_initial_board(self, board_state: dict[str, str | None]) -> None:
        """Set the initial board state (starting position)."""
        self.previous_board = board_state

    def detect_change(
        self, current_board: dict[str, str | None]
    ) -> list[str] | None:
        """Compare current board to previous state.

        Returns list of changed square names if change is stable
        for N consecutive frames, else None.
        """
        if self.previous_board is None:
            return None

        changed = [
            sq
            for sq in current_board
            if current_board[sq] != self.previous_board.get(sq)
        ]

        if not changed:
            self.candidate_changes = None
            self.stable_count = 0
            return None

        if changed == self.candidate_changes:
            self.stable_count += 1
        else:
            self.candidate_changes = changed
            self.stable_count = 1

        if self.stable_count >= self.stability_frames:
            self.previous_board = dict(current_board)
            self.candidate_changes = None
            self.stable_count = 0
            return changed

        return None

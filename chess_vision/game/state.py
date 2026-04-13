"""Game state tracking."""

import chess


class GameState:
    """Tracks the evolving game state across detected moves."""

    def __init__(self, starting_fen: str = chess.STARTING_FEN):
        self.board = chess.Board(starting_fen)
        self.move_history: list[chess.Move] = []
        self.fen_history: list[str] = [starting_fen]

    def apply_move(self, move: chess.Move) -> bool:
        """Apply a move if legal. Returns True if successful."""
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            self.fen_history.append(self.board.fen())
            return True
        return False

    def get_fen(self) -> str:
        """Get current FEN string."""
        return self.board.fen()

    def get_board_fen(self) -> str:
        """Get just the piece placement portion of FEN."""
        return self.board.board_fen()

    def whose_turn(self) -> str:
        """Return 'white' or 'black'."""
        return "white" if self.board.turn == chess.WHITE else "black"

    def move_number(self) -> int:
        """Current full move number."""
        return self.board.fullmove_number

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> str:
        """Game result string (e.g., '1-0', '0-1', '1/2-1/2', '*')."""
        if self.board.is_game_over():
            return self.board.result()
        return "*"

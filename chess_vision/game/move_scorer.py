"""Move scoring and detection based on ChessCam's approach.

Scores legal moves against a smoothed state matrix (64 squares x 12 classes).
Uses two-move lookahead and greedy fallback with time-based confirmation.
"""

import time
from dataclasses import dataclass, field

import chess
import numpy as np

# ChessCam class ordering
LABELS = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}


@dataclass
class MoveData:
    """Describes what a move expects on the board."""
    san: str
    from_squares: list[int]  # Squares that should become empty
    to_squares: list[int]    # Squares that should have specific pieces
    targets: list[int]       # Class index expected on each to_square


@dataclass
class MovePair:
    move1: MoveData
    move2: MoveData | None
    combined: MoveData | None


def piece_to_label_idx(piece: chess.Piece) -> int:
    """Convert a python-chess Piece to a LABELS index."""
    sym = piece.symbol()
    return LABEL_MAP[sym]


def get_move_data(board: chess.Board, move: chess.Move) -> MoveData:
    """Extract from/to/target data for a move."""
    san = board.san(move)
    from_squares = [move.from_square]
    to_squares = [move.to_square]

    # What piece ends up on the to_square
    piece = board.piece_at(move.from_square)
    if move.promotion:
        promoted = chess.Piece(move.promotion, piece.color)
        targets = [piece_to_label_idx(promoted)]
    else:
        targets = [piece_to_label_idx(piece)]

    # Castling: rook also moves
    if board.is_castling(move):
        rank = 0 if board.turn == chess.WHITE else 7
        if board.is_kingside_castling(move):
            from_squares.append(chess.square(7, rank))
            to_squares.append(chess.square(5, rank))
        else:
            from_squares.append(chess.square(0, rank))
            to_squares.append(chess.square(3, rank))
        rook = chess.Piece(chess.ROOK, board.turn)
        targets.append(piece_to_label_idx(rook))

    # En passant: captured pawn disappears
    if board.is_en_passant(move):
        cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        from_squares.append(cap_sq)

    # Capture: the to_square was occupied, now has the moving piece (already handled)

    return MoveData(san=san, from_squares=from_squares, to_squares=to_squares, targets=targets)


def combine_data(move1: MoveData, move2: MoveData) -> MoveData:
    """Combine two moves into a single scoring unit (ChessCam's combineData)."""
    bad_squares = set(move2.from_squares + move2.to_squares)

    # Keep move1's squares that don't overlap with move2
    from1 = [sq for sq in move1.from_squares if sq not in bad_squares]
    to1 = []
    targets1 = []
    for i, sq in enumerate(move1.to_squares):
        if sq not in bad_squares:
            to1.append(sq)
            targets1.append(move1.targets[i])

    return MoveData(
        san=move1.san,
        from_squares=from1 + move2.from_squares,
        to_squares=to1 + move2.to_squares,
        targets=targets1 + move2.targets,
    )


def get_move_pairs(board: chess.Board) -> list[MovePair]:
    """Enumerate all legal move pairs (move + opponent's response)."""
    pairs = []
    for move1 in board.legal_moves:
        move1_data = get_move_data(board, move1)

        board.push(move1)
        has_response = False
        for move2 in board.legal_moves:
            move2_data = get_move_data(board, move2)
            combined = combine_data(move1_data, move2_data)
            pairs.append(MovePair(move1=move1_data, move2=move2_data, combined=combined))
            has_response = True
        board.pop()

        if not has_response:
            pairs.append(MovePair(move1=move1_data, move2=None, combined=None))

    return pairs


def calculate_score(state: np.ndarray, move: MoveData, threshold: float = 0.6) -> float:
    """Score how well the state matrix matches a move (ChessCam's calculateScore).

    For 'from' squares: reward emptiness (1 - max_confidence - threshold)
    For 'to' squares: reward correct piece detection (confidence - threshold)
    """
    score = 0.0
    for sq in move.from_squares:
        score += 1.0 - float(np.max(state[sq])) - threshold
    for i, sq in enumerate(move.to_squares):
        score += float(state[sq, move.targets[i]]) - threshold
    return score


class MoveDetectorV2:
    """ChessCam-style move detection with two-move lookahead and greedy fallback."""

    def __init__(self, greedy_delay: float = 3.0, baseline_state: np.ndarray | None = None):
        self.possible_moves: set[str] = set()
        self.greedy_times: dict[str, float] = {}
        self.greedy_delay = greedy_delay
        self.last_move_san: str = ""
        self.baseline_state = baseline_state  # EMA state from warmup (starting position)

    def _state_changed_enough(self, state: np.ndarray, move: MoveData) -> bool:
        """Check that the from/to squares actually changed from the baseline.

        Rejects moves where the model had the same bias during warmup
        (e.g., consistently thinks h2 is empty even in starting position).
        """
        if self.baseline_state is None:
            return True

        min_change = 0.15  # Minimum EMA change from baseline to count
        for sq in move.from_squares:
            delta = float(np.max(self.baseline_state[sq])) - float(np.max(state[sq]))
            if delta < min_change:
                return False  # This square didn't change from baseline
        return True

    def detect_move(self, board: chess.Board, state: np.ndarray) -> str | None:
        """Check if a move should be played based on the current state matrix.

        Returns SAN string of the detected move, or None.
        """
        pairs = get_move_pairs(board)

        best_score1 = float("-inf")
        best_score2 = float("-inf")
        best_joint_score = float("-inf")
        best_move: MoveData | None = None
        best_combined: MoveData | None = None
        seen: set[str] = set()

        for pair in pairs:
            # Score move1 (deduplicated)
            if pair.move1.san not in seen:
                seen.add(pair.move1.san)
                # Skip moves where from-squares haven't changed from baseline
                if not self._state_changed_enough(state, pair.move1):
                    continue
                score1 = calculate_score(state, pair.move1)
                if score1 > 0:
                    self.possible_moves.add(pair.move1.san)
                if score1 > best_score1:
                    best_score1 = score1
                    best_move = pair.move1

            # Score move pair (only if move1 is "possible" and move2 exists)
            if pair.move2 is None or pair.combined is None:
                continue
            if pair.move1.san not in self.possible_moves:
                continue

            score2 = calculate_score(state, pair.move2)
            if score2 < 0:
                continue
            if score2 > best_score2:
                best_score2 = score2

            joint_score = calculate_score(state, pair.combined)
            if joint_score > best_joint_score:
                best_joint_score = joint_score
                best_combined = pair.combined

        # Two-move detection (strongest signal)
        if best_combined is not None and best_score2 > 0 and best_joint_score > 0:
            san = best_combined.san
            if san in self.possible_moves:
                self.possible_moves.clear()
                self.greedy_times.clear()
                self.last_move_san = san
                return san

        # Greedy fallback (single move, requires time confirmation)
        if best_move is not None and best_score1 > 0:
            san = best_move.san
            now = time.time()

            if san not in self.greedy_times:
                self.greedy_times[san] = now

            elapsed = now - self.greedy_times[san]
            is_new = san != self.last_move_san

            if elapsed > self.greedy_delay and is_new:
                self.possible_moves.clear()
                self.greedy_times.clear()
                self.last_move_san = san
                return san

        # Clean up greedy times for moves that are no longer the best
        if best_move is not None:
            self.greedy_times = {
                san: t for san, t in self.greedy_times.items()
                if san == best_move.san
            }

        return None

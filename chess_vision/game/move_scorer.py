"""Move scoring and detection based on ChessCam's approach.

Scores legal moves against a smoothed state matrix (64 squares x 12 classes).
Uses two-move lookahead and greedy fallback with time-based confirmation.
"""

import time
from dataclasses import dataclass

import chess
import numpy as np

# ChessCam class ordering
LABELS = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}


@dataclass
class MoveData:
    san: str
    from_squares: list[int]
    to_squares: list[int]
    targets: list[int]


@dataclass
class MovePair:
    move1: MoveData
    move2: MoveData | None
    combined: MoveData | None


def piece_to_label_idx(piece: chess.Piece) -> int:
    return LABEL_MAP[piece.symbol()]


def get_move_data(board: chess.Board, move: chess.Move) -> MoveData:
    san = board.san(move)
    from_squares = [move.from_square]
    to_squares = [move.to_square]

    piece = board.piece_at(move.from_square)
    if move.promotion:
        promoted = chess.Piece(move.promotion, piece.color)
        targets = [piece_to_label_idx(promoted)]
    else:
        targets = [piece_to_label_idx(piece)]

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

    if board.is_en_passant(move):
        cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        from_squares.append(cap_sq)

    return MoveData(san=san, from_squares=from_squares, to_squares=to_squares, targets=targets)


def combine_data(move1: MoveData, move2: MoveData) -> MoveData:
    bad_squares = set(move2.from_squares + move2.to_squares)
    from1 = [sq for sq in move1.from_squares if sq not in bad_squares]
    to1, targets1 = [], []
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
    """Score how well the state matrix matches a move.

    Exactly matches ChessCam's calculateScore:
    - from squares: reward emptiness (1 - max_confidence - threshold)
    - to squares: reward correct piece (confidence - threshold)
    """
    score = 0.0
    for sq in move.from_squares:
        score += 1.0 - float(np.max(state[sq])) - threshold
    for i, sq in enumerate(move.to_squares):
        score += float(state[sq, move.targets[i]]) - threshold
    return score


class MoveDetectorV2:
    """ChessCam-style move detection with two-move lookahead and greedy fallback.

    Improvements over vanilla ChessCam approach:
    - Time confirmation on BOTH single-move and two-move paths
    - Minimum score threshold (rejects barely-positive noise)
    - Score margin requirement (top move must clearly beat second-best)
    - Expiring possible_moves (stale candidates don't trigger two-move path)
    """

    MIN_SCORE = 0.15         # Minimum score to accept a move
    SCORE_MARGIN = 0.1       # Top move must beat second-best by this much
    TWO_MOVE_DELAY = 0.3     # Time confirmation for two-move detections
    POSSIBLE_MOVE_TTL = 3.0  # Expire possible_moves after this many seconds

    def __init__(self, greedy_delay: float = 1.0):
        self.possible_moves: dict[str, float] = {}  # san -> last_seen_time
        self.greedy_times: dict[str, float] = {}
        self.two_move_times: dict[str, float] = {}
        self.greedy_delay = greedy_delay
        self.last_move_san: str = ""
        self._cached_pairs: list[MovePair] | None = None
        self._cached_fen: str = ""

    def detect_move(self, board: chess.Board, state: np.ndarray) -> str | None:
        now = time.time()

        # Cache move pairs (only recompute when position changes)
        fen = board.board_fen() + str(board.turn)
        if fen != self._cached_fen:
            self._cached_pairs = get_move_pairs(board)
            self._cached_fen = fen
        pairs = self._cached_pairs

        # Expire stale possible_moves
        self.possible_moves = {
            san: t for san, t in self.possible_moves.items()
            if now - t < self.POSSIBLE_MOVE_TTL
        }

        # Score all moves
        best_score1 = float("-inf")
        second_score1 = float("-inf")
        best_joint_score = float("-inf")
        second_joint_score = float("-inf")
        best_move: MoveData | None = None
        best_combined: MoveData | None = None
        best_combined_san: str = ""
        seen: set[str] = set()

        for pair in pairs:
            if pair.move1.san not in seen:
                seen.add(pair.move1.san)
                score1 = calculate_score(state, pair.move1)
                if score1 > 0:
                    self.possible_moves[pair.move1.san] = now
                if score1 > best_score1:
                    second_score1 = best_score1
                    best_score1 = score1
                    best_move = pair.move1
                elif score1 > second_score1:
                    second_score1 = score1

            if pair.move2 is None or pair.combined is None:
                continue
            if pair.move1.san not in self.possible_moves:
                continue

            joint_score = calculate_score(state, pair.combined)
            if joint_score > best_joint_score:
                second_joint_score = best_joint_score
                best_joint_score = joint_score
                best_combined = pair.combined
                best_combined_san = pair.move1.san
            elif joint_score > second_joint_score:
                second_joint_score = joint_score

        # Two-move detection (with time confirmation + score margin)
        if (best_combined is not None
                and best_joint_score >= self.MIN_SCORE
                and best_joint_score - second_joint_score >= self.SCORE_MARGIN
                and best_combined_san in self.possible_moves
                and best_combined_san != self.last_move_san):
            san = best_combined_san
            if san not in self.two_move_times:
                self.two_move_times[san] = now
            elapsed = now - self.two_move_times[san]
            if elapsed >= self.TWO_MOVE_DELAY:
                self.possible_moves.clear()
                self.greedy_times.clear()
                self.two_move_times.clear()
                self.last_move_san = san
                return san
        else:
            self.two_move_times.clear()

        # Greedy fallback (single move, requires time confirmation + score margin)
        if (best_move is not None
                and best_score1 >= self.MIN_SCORE
                and best_score1 - second_score1 >= self.SCORE_MARGIN):
            san = best_move.san
            if san not in self.greedy_times:
                self.greedy_times[san] = now

            elapsed = now - self.greedy_times[san]
            is_new = san != self.last_move_san

            if elapsed > self.greedy_delay and is_new:
                self.possible_moves.clear()
                self.greedy_times.clear()
                self.two_move_times.clear()
                self.last_move_san = san
                return san

        # Clean up stale greedy times (only keep current top candidate)
        if best_move is not None:
            self.greedy_times = {
                san: t for san, t in self.greedy_times.items()
                if san == best_move.san
            }

        return None

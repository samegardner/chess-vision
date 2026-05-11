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
    """Build the MoveData representing the COMBINED end state of move1+move2.

    Filtering move2.from_squares against move1.to_squares: in the en-passant-
    of-move1 case (e.g. white plays c4, black plays dxc3 capturing c4), move2's
    from_squares contains move1.to_square (c4) as the en-passant captured
    square. Including it as a combined from-square gives a free "this square
    is empty" reward (+0.55 in calculate_score) that biases the scorer toward
    the en-passant interpretation. With it filtered, c4+dxc3 e.p. and
    c3+dxc3 produce the same combined data, matching the fact that their
    final positions are identical and the bot can't tell them apart from
    image data alone. Tie-breaking is handled in get_move_pairs.
    """
    move1_to_set = set(move1.to_squares)
    bad_squares = set(move2.from_squares + move2.to_squares)
    from1 = [sq for sq in move1.from_squares if sq not in bad_squares]
    to1, targets1 = [], []
    for i, sq in enumerate(move1.to_squares):
        if sq not in bad_squares:
            to1.append(sq)
            targets1.append(move1.targets[i])
    move2_from = [sq for sq in move2.from_squares if sq not in move1_to_set]

    return MoveData(
        san=move1.san,
        from_squares=from1 + move2_from,
        to_squares=to1 + move2.to_squares,
        targets=targets1 + move2.targets,
    )


def _pair_signature(pair: MovePair) -> tuple:
    """Hashable signature of a pair's combined end state.

    Two pairs with the same signature lead to identical observable board
    states, so the YOLO state matrix can't distinguish them. Used to
    deduplicate equivalent pairs (notably c3+dxc3 vs c4+dxc3 e.p.).
    """
    c = pair.combined
    if c is None:
        return ()
    return (
        frozenset(c.from_squares),
        tuple(sorted(zip(c.to_squares, c.targets))),
    )


def get_move_pairs(board: chess.Board) -> list[MovePair]:
    """Generate move-pair candidates for two-move lookahead scoring.

    Pairs whose combined end-state is observationally identical are deduped,
    keeping the FIRST emitted (which by python-chess iteration order is the
    simpler interpretation - e.g. c3+dxc3 over c4+dxc3 e.p.). Without dedup,
    equivalent pairs split the joint-score margin so neither can fire, and
    the bot stalls.
    """
    pairs = []
    for move1 in board.legal_moves:
        move1_data = get_move_data(board, move1)
        board.push(move1)
        has_response = False
        seen_signatures: set = set()
        for move2 in board.legal_moves:
            move2_data = get_move_data(board, move2)
            combined = combine_data(move1_data, move2_data)
            pair = MovePair(move1=move1_data, move2=move2_data, combined=combined)
            sig = _pair_signature(pair)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            pairs.append(pair)
            has_response = True
        board.pop()
        if not has_response:
            pairs.append(MovePair(move1=move1_data, move2=None, combined=None))
    # Cross-move1 dedup: two pairs with different move1 but identical combined
    # end-state are observationally equivalent (the c3+dxc3 / c4+dxc3 e.p.
    # case). Keep the first one emitted, which by iteration order is the
    # simpler interpretation (c3 before c4).
    deduped = []
    seen_cross: set = set()
    for pair in pairs:
        sig = _pair_signature(pair)
        if sig and sig in seen_cross:
            continue
        if sig:
            seen_cross.add(sig)
        deduped.append(pair)
    return deduped


def should_undo(state: np.ndarray, move: MoveData,
                from_threshold: float = 0.4, to_threshold: float = 0.2) -> bool:
    """Check whether the smoothed state still agrees with the last move.

    Returns True if the state suggests the move was a false positive:
    a from-square still looks occupied (piece didn't actually leave),
    OR the PRIMARY to-square fails to show the expected piece.

    For castling, move.to_squares has two entries: the king's destination
    (index 0) and the rook's destination (index 1). The rook is small and
    easily missed by YOLO at oblique camera angles - requiring it would
    spuriously undo legitimate castles. We check only the primary (first)
    to-square; the from-side check still catches the case where the rook
    never actually moved (it'd still be detected on h1/a1).

    For en passant, from_squares includes the captured pawn's square, so
    a "captured pawn still there" condition triggers via from-side. Only
    one to-square (the capturing pawn's destination), still checked.
    """
    from_occ = max(float(np.max(state[sq])) for sq in move.from_squares)
    primary_to_sq = move.to_squares[0]
    primary_target = move.targets[0]
    to_occ = float(state[primary_to_sq, primary_target])
    return from_occ > from_threshold or to_occ < to_threshold


def calculate_score(state: np.ndarray, move: MoveData, threshold: float = 0.45) -> float:
    """Score how well the state matrix matches a move.

    Matches ChessCam's calculateScore:
    - from squares: reward emptiness (1 - max_confidence - threshold)
    - to squares: reward correct piece (confidence - threshold)

    Threshold history: 0.60 -> 0.55 -> 0.45. Sam's Bf1-Be2 attempt
    failed at 0.55 because YOLO couldn't find the bishop anywhere on
    the destination (state[e2,B]=0) so both from-side and to-side
    subtracted T and the score stuck at ~ -0.10. At T=0.45 the same
    state scores +0.10 - low but positive, so a weak to-side detection
    (0.1-0.2 confidence) is enough to break ties via SCORE_MARGIN.
    Full-zero-to-side moves still tie with other bishop options and
    won't fire (correct).
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

    MIN_SCORE = 0.10         # Minimum score to accept a move
    SCORE_MARGIN = 0.1       # Top move must beat second-best by this much
    TWO_MOVE_DELAY = 0.3     # Time confirmation for two-move detections
    POSSIBLE_MOVE_TTL = 3.0  # Expire possible_moves after this many seconds
    UNDO_COOLDOWN = 3.0      # Seconds an auto-undone SAN is blocked from re-firing
    UNDO_RATE_WINDOW = 10.0  # Sliding window for the cross-SAN undo-rate guard
    UNDO_RATE_THRESHOLD = 3  # Undos within window that trigger a global freeze
    UNDO_FREEZE_DURATION = 5.0  # Length of the global firing freeze

    def __init__(self, greedy_delay: float = 1.0, event_log=None):
        self.possible_moves: dict[str, float] = {}  # san -> last_seen_time
        self.greedy_times: dict[str, float] = {}
        self.two_move_times: dict[str, float] = {}
        self.greedy_delay = greedy_delay
        self.last_move_san: str = ""
        self._cached_pairs: list[MovePair] | None = None
        self._cached_fen: str = ""
        # Top candidates from the most recent detect_move call, sorted by score.
        # Exposed for the debug HUD so the user can see what the detector is
        # considering even when nothing crosses the firing threshold.
        self.top_candidates: list[tuple[str, float]] = []
        # Per-SAN cooldowns set when auto-undo retracts a move. Prevents the
        # joint-vs-single scoring inconsistency from looping the same SAN
        # forever (e.g. d5 firing repeatedly because joint scoring sees a
        # white pawn on d5 after a hypothetical capture, while the undo
        # check looks for a black pawn on d5).
        self.undo_cooldown: dict[str, float] = {}
        # Sliding window of recent undo timestamps. When too many undos
        # happen in a short period, regardless of SAN, ALL firing is
        # frozen briefly so the user can intervene (e.g. press R to
        # reset). Catches the case where the system has diverged from
        # reality and is firing alternating wrong moves.
        self.recent_undos: list[float] = []
        self.frozen_until: float = 0.0
        # Optional structured logger; left None means no-op.
        from chess_vision.event_log import NullEventLog
        self.event_log = event_log if event_log is not None else NullEventLog()

    def mark_undone(self, san: str) -> None:
        """Record that this SAN was just auto-undone. Call from the main
        loop's undo handler. Prevents the same SAN from re-firing for
        UNDO_COOLDOWN seconds, AND triggers a global firing freeze if
        too many undos have happened recently (catches the case where
        the system has diverged from reality and keeps firing wrong
        alternating moves to bypass the per-SAN cooldown)."""
        now = time.time()
        self.undo_cooldown[san] = now + self.UNDO_COOLDOWN
        self.recent_undos.append(now)
        cutoff = now - self.UNDO_RATE_WINDOW
        self.recent_undos = [t for t in self.recent_undos if t > cutoff]
        if len(self.recent_undos) >= self.UNDO_RATE_THRESHOLD:
            self.frozen_until = now + self.UNDO_FREEZE_DURATION
            print(f"[detect] firing FROZEN for {self.UNDO_FREEZE_DURATION:.0f}s "
                  f"({len(self.recent_undos)} undos in {self.UNDO_RATE_WINDOW:.0f}s) - "
                  f"system likely diverged from reality, press R if needed")

    def detect_move(self, board: chess.Board, state: np.ndarray) -> str | None:
        now = time.time()
        # Drop expired cooldown entries
        self.undo_cooldown = {s: t for s, t in self.undo_cooldown.items() if t > now}
        # Global firing freeze: skip detection entirely. top_candidates
        # from the previous call stays as-is so the HUD doesn't go blank.
        if now < self.frozen_until:
            return None

        # Cache move pairs (only recompute when position changes).
        # Use full FEN: board_fen alone misses en passant + castling rights,
        # which can leave stale SANs in the cache that aren't legal anymore.
        fen = board.fen()
        if fen != self._cached_fen:
            self._cached_pairs = get_move_pairs(board)
            self._cached_fen = fen
            # Position changed (push or pop). Anything we tracked about the
            # previous position - timers, possible_moves, "last fired SAN" -
            # is now stale. Castling collision (white "O-O" then black "O-O")
            # would otherwise permanently block black's castle, since the
            # SAN literal is identical across colors.
            self.greedy_times.clear()
            self.possible_moves.clear()
            self.two_move_times.clear()
            self.last_move_san = ""
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
        all_scores: list[tuple[str, float]] = []  # for the debug HUD

        for pair in pairs:
            if pair.move1.san not in seen:
                seen.add(pair.move1.san)
                score1 = calculate_score(state, pair.move1)
                all_scores.append((pair.move1.san, score1))
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

        # Update greedy timers proactively for ANY positive-scoring move.
        # Decoupling timer accumulation from "is currently best?" lets a move
        # with a fluctuating score survive brief frames where another move
        # momentarily takes the top spot. Without this, oscillation between
        # candidates kept resetting the leader's timer and Bd7-style moves
        # never confirmed even when consistently dominant.
        for san, score in all_scores:
            if score > 0:
                if san not in self.greedy_times:
                    self.greedy_times[san] = now
            else:
                self.greedy_times.pop(san, None)

        # Stash top 3 candidates (with their timer age) for the HUD.
        all_scores.sort(key=lambda x: -x[1])
        self.top_candidates = [
            (san, score, max(0.0, now - self.greedy_times.get(san, now)))
            for san, score in all_scores[:3]
        ]

        # Two-move detection (with time confirmation + score margin)
        if (best_combined is not None
                and best_joint_score >= self.MIN_SCORE
                and best_joint_score - second_joint_score >= self.SCORE_MARGIN
                and best_combined_san in self.possible_moves
                and best_combined_san != self.last_move_san
                and best_combined_san not in self.undo_cooldown):
            san = best_combined_san
            if san not in self.two_move_times:
                self.two_move_times[san] = now
            elapsed = now - self.two_move_times[san]
            if elapsed >= self.TWO_MOVE_DELAY:
                print(f"[detect] FIRE two-move {san} joint={best_joint_score:.2f} elapsed={elapsed:.1f}s")
                self.event_log.log("fire", path="two-move", san=san,
                                   joint_score=round(best_joint_score, 3),
                                   elapsed=round(elapsed, 2))
                self.possible_moves.clear()
                self.greedy_times.clear()
                self.two_move_times.clear()
                self.last_move_san = san
                return san
        else:
            self.two_move_times.clear()

        # Greedy fallback: fire if the current best meets margin AND has had
        # a timer running for at least greedy_delay. Timer was set above the
        # moment the move first scored positive, so brief wobbles where
        # another move took #1 don't reset it.
        if (best_move is not None
                and best_score1 >= self.MIN_SCORE
                and best_score1 - second_score1 >= self.SCORE_MARGIN
                and best_move.san in self.greedy_times
                and best_move.san != self.last_move_san
                and best_move.san not in self.undo_cooldown):
            san = best_move.san
            elapsed = now - self.greedy_times[san]
            if elapsed > self.greedy_delay:
                print(f"[detect] FIRE greedy {san} score={best_score1:.2f} "
                      f"margin={best_score1 - second_score1:.2f} elapsed={elapsed:.1f}s")
                self.event_log.log("fire", path="greedy", san=san,
                                   score=round(best_score1, 3),
                                   margin=round(best_score1 - second_score1, 3),
                                   elapsed=round(elapsed, 2))
                self.possible_moves.clear()
                self.greedy_times.clear()
                self.two_move_times.clear()
                self.last_move_san = san
                return san

        # Diagnostic: log why the top candidate is being held back, max once
        # every 2 seconds so it doesn't spam.
        if best_move is not None and best_move.san in self.greedy_times:
            blockers = []
            san = best_move.san
            if best_score1 < self.MIN_SCORE:
                blockers.append(f"score{best_score1:.2f}<MIN({self.MIN_SCORE})")
            if best_score1 - second_score1 < self.SCORE_MARGIN:
                blockers.append(f"margin{best_score1 - second_score1:.2f}<MARGIN({self.SCORE_MARGIN})")
            if san == self.last_move_san:
                blockers.append(f"san==last_fired({self.last_move_san!r})")
            elapsed = now - self.greedy_times[san]
            if elapsed <= self.greedy_delay:
                blockers.append(f"elapsed{elapsed:.1f}<=delay({self.greedy_delay})")
            if blockers and now - getattr(self, "_last_blocker_log", 0.0) > 2.0:
                self._last_blocker_log = now
                print(f"[detect] {san} blocked: {', '.join(blockers)}")
                self.event_log.log("blocked", san=san, score=round(best_score1, 3),
                                   margin=round(best_score1 - second_score1, 3),
                                   timer=round(elapsed, 2),
                                   last_fired=self.last_move_san,
                                   reasons=blockers)

        return None

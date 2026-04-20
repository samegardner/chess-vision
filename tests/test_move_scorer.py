"""Tests for move scoring logic."""

import chess
import numpy as np

from chess_vision.game.move_scorer import (
    calculate_score, get_move_data, get_move_pairs, combine_data,
    should_undo,
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


def test_greedy_delay(monkeypatch):
    """Greedy fallback must not fire before greedy_delay elapses, and MUST
    fire once it does. Monkey-patching time.time keeps the test deterministic
    instead of relying on real wall-clock sleeps.
    """
    import time as time_module
    fake_now = [1000.0]
    monkeypatch.setattr(time_module, "time", lambda: fake_now[0])

    detector = MoveDetectorV2(greedy_delay=1.0)
    state = _make_state_with_starting_position()
    board = chess.Board()

    # Simulate e4 played
    state[chess.E2] = np.zeros(12)
    state[chess.E4, LABEL_MAP["P"]] = 0.8

    # First call starts the timer, must not fire yet.
    assert detector.detect_move(board, state) is None

    # Half the delay -> still nothing.
    fake_now[0] += 0.5
    assert detector.detect_move(board, state) is None

    # Past the delay -> the move fires.
    fake_now[0] += 1.0
    assert detector.detect_move(board, state) == "e4"


def test_pgn_generation():
    """PGN should contain the moves played."""
    from chess_vision.game.pgn import generate_pgn
    moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
    pgn = generate_pgn(moves, white_name="W", black_name="B")
    assert "e4" in pgn
    assert "e5" in pgn
    assert "W" in pgn


def test_should_undo_castling_kept_when_state_matches():
    """After O-O, state should show king on g1, rook on f1, e1 and h1 empty.
    should_undo must NOT fire (castling was real)."""
    board = chess.Board("rnbqk2r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    move = board.parse_san("O-O")
    data = get_move_data(board, move)

    state = np.zeros((64, 12), dtype=np.float32)
    state[chess.G1, LABEL_MAP["K"]] = 0.85
    state[chess.F1, LABEL_MAP["R"]] = 0.85

    assert should_undo(state, data) is False


def test_should_undo_castling_fires_when_rook_stuck():
    """If after O-O the rook detection is still on h1 (didn't actually move),
    should_undo must catch it - a naive king-only check would miss this."""
    board = chess.Board("rnbqk2r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    move = board.parse_san("O-O")
    data = get_move_data(board, move)

    state = np.zeros((64, 12), dtype=np.float32)
    state[chess.G1, LABEL_MAP["K"]] = 0.85  # king landed correctly
    state[chess.H1, LABEL_MAP["R"]] = 0.85  # rook never left -> should undo
    # f1 intentionally left empty (to-square with no piece)

    assert should_undo(state, data) is True


def test_should_undo_en_passant_kept_when_captured_pawn_gone():
    """After exd6 e.p., d5 (captured pawn) must be empty. should_undo
    should NOT fire when state reflects that."""
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    move = board.parse_san("exd6")
    data = get_move_data(board, move)

    state = np.zeros((64, 12), dtype=np.float32)
    state[chess.D6, LABEL_MAP["P"]] = 0.85  # capturing pawn landed

    assert should_undo(state, data) is False


def test_should_undo_en_passant_fires_when_captured_pawn_still_there():
    """If after exd6 e.p. the black pawn on d5 is still detected, the
    captured-pawn square entry in from_squares catches it and undo fires."""
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    move = board.parse_san("exd6")
    data = get_move_data(board, move)

    state = np.zeros((64, 12), dtype=np.float32)
    state[chess.D6, LABEL_MAP["P"]] = 0.85  # capturing pawn landed
    state[chess.D5, LABEL_MAP["p"]] = 0.85  # captured pawn still visible -> undo

    assert should_undo(state, data) is True


def test_undo_cooldown_blocks_immediate_refire(monkeypatch):
    """When auto-undo retracts a move and the same SAN scores high again
    immediately, the cooldown must block re-firing for UNDO_COOLDOWN
    seconds. Otherwise the joint-vs-single scoring mismatch (e.g. d5
    after exd5) would cycle fire-undo-fire-undo forever."""
    import time as time_module
    fake_now = [1000.0]
    monkeypatch.setattr(time_module, "time", lambda: fake_now[0])

    detector = MoveDetectorV2(greedy_delay=1.0)
    state = _make_state_with_starting_position()
    board = chess.Board()
    state[chess.E2] = np.zeros(12)
    state[chess.E4, LABEL_MAP["P"]] = 0.85

    # First call starts the timer; second call past delay fires e4.
    assert detector.detect_move(board, state) is None
    fake_now[0] += 1.5
    assert detector.detect_move(board, state) == "e4"

    # Simulate the main loop: push the move, run detect_move on the new
    # position to update the cache, then auto-undo pops back. The cache
    # rebuild on the pop clears last_move_san (matches real flow).
    board.push_san("e4")
    fake_now[0] += 0.1
    detector.detect_move(board, state)  # caches post-e4 fen
    board.pop()
    detector.mark_undone("e4")

    # State still looks like e4 was played. last_move_san is cleared
    # by the cache rebuild on pop, so the only remaining blocker is
    # the undo cooldown.
    fake_now[0] += 0.1
    detector.detect_move(board, state)  # warmup; cache rebuilds, timer set
    fake_now[0] += 1.5
    assert detector.detect_move(board, state) is None, (
        "undo cooldown should block e4 from re-firing right after undo"
    )

    # Past the cooldown window, e4 fires again.
    fake_now[0] += MoveDetectorV2.UNDO_COOLDOWN + 1.0
    assert detector.detect_move(board, state) == "e4"


def test_undo_rate_freeze_blocks_alternating_san_loop(monkeypatch):
    """Per-SAN cooldown can be bypassed by alternating between two
    different wrong SANs (e.g. g3, gxf3, g3, gxf3...). The cross-SAN
    undo-rate guard must freeze ALL firing once enough undos pile up
    in a short window, regardless of which SANs they were."""
    import time as time_module
    fake_now = [1000.0]
    monkeypatch.setattr(time_module, "time", lambda: fake_now[0])

    detector = MoveDetectorV2()
    # Manually stack 3 undos within the rate window.
    detector.mark_undone("g3")
    fake_now[0] += 1.0
    detector.mark_undone("gxf3")
    fake_now[0] += 1.0
    detector.mark_undone("g3")  # this third undo crosses the threshold

    # Freeze must now be active.
    assert detector.frozen_until > fake_now[0]

    # detect_move with any state must return None while frozen, even if
    # the score for some legal move is high.
    state = _make_state_with_starting_position()
    board = chess.Board()
    state[chess.E2] = np.zeros(12)
    state[chess.E4, LABEL_MAP["P"]] = 0.85
    fake_now[0] += 1.0
    assert detector.detect_move(board, state) is None
    fake_now[0] += 1.0
    assert detector.detect_move(board, state) is None

    # After the freeze expires, normal firing resumes.
    fake_now[0] += MoveDetectorV2.UNDO_FREEZE_DURATION + 1.0
    detector.detect_move(board, state)  # warmup
    fake_now[0] += MoveDetectorV2.UNDO_COOLDOWN + 2.0
    fake_now[0] += 1.5
    # e4 has no cooldown of its own, so it should fire eventually.
    result = detector.detect_move(board, state)
    assert result == "e4", f"expected e4 to fire post-freeze, got {result!r}"


def test_castling_collision_does_not_block_opposite_color(monkeypatch):
    """White "O-O" and black "O-O" share the same SAN literal. After white
    castles, last_move_san = "O-O"; the 'same san as last fire' guard must
    NOT block black's castle. Cache rebuild on position change should clear
    last_move_san so the new position starts fresh.
    """
    import time as time_module
    fake_now = [1000.0]
    monkeypatch.setattr(time_module, "time", lambda: fake_now[0])

    detector = MoveDetectorV2(greedy_delay=1.0)

    # Position where white can castle kingside (king e1, rook h1, path clear).
    board = chess.Board("rnbqk2r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    # Set up state so white O-O scores high
    state = np.zeros((64, 12), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            state[sq, LABEL_MAP[piece.symbol()]] = 0.85
    # Simulate white having castled: king on g1, rook on f1, e1/h1 empty.
    state[chess.E1] = np.zeros(12)
    state[chess.H1] = np.zeros(12)
    state[chess.G1, LABEL_MAP["K"]] = 0.85
    state[chess.F1, LABEL_MAP["R"]] = 0.85

    # Fire white's castle
    assert detector.detect_move(board, state) is None  # timer start
    fake_now[0] += 1.5
    assert detector.detect_move(board, state) == "O-O"

    # Main-loop equivalent: push the move. Position changes to black's turn.
    board.push_san("O-O")

    # Now set state to look like black has also castled (king g8, rook f8).
    state[chess.E8] = np.zeros(12)
    state[chess.H8] = np.zeros(12)
    state[chess.G8, LABEL_MAP["k"]] = 0.85
    state[chess.F8, LABEL_MAP["r"]] = 0.85

    # Black O-O must be able to fire despite sharing the "O-O" literal.
    fake_now[0] += 0.1
    assert detector.detect_move(board, state) is None  # timer starts fresh on new fen
    fake_now[0] += 1.5
    assert detector.detect_move(board, state) == "O-O"


def test_cache_invalidates_on_ep_rights_change():
    """Same piece placement + same turn but different EP rights must rebuild
    the move pair cache. board_fen() drops EP, so the cache key has to use
    the full fen() to avoid stale 'exd6' SANs surviving into a position
    where they're no longer legal.
    """
    detector = MoveDetectorV2()
    state = np.zeros((64, 12), dtype=np.float32)

    # White can play exd6 en passant
    ep_board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    detector.detect_move(ep_board, state)
    legal_with_ep = {p.move1.san for p in detector._cached_pairs}
    assert "exd6" in legal_with_ep

    # Same piece placement but EP no longer legal (new arrival path).
    # board_fen() is identical, but fen() differs -> cache must rebuild.
    no_ep_board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    assert ep_board.board_fen() == no_ep_board.board_fen()
    detector.detect_move(no_ep_board, state)
    legal_without_ep = {p.move1.san for p in detector._cached_pairs}
    assert "exd6" not in legal_without_ep

"""Record a chess game using ChessCam's YOLO model + move scoring.

Usage:
    python scripts/record_game.py
    python scripts/record_game.py --select-corners
    python scripts/record_game.py --output mygame.pgn
"""

import argparse
import json
import subprocess
import time
import sys
from pathlib import Path

import chess
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.board.detect import select_corners
from chess_vision.board.auto_corners import auto_detect_corners, XCornerDetector
from chess_vision.inference.yolo_detect import (
    YoloPieceDetector, compute_square_centers, compute_crop_region,
    compute_board_quad, point_in_quad,
)
from chess_vision.event_log import EventLog
from chess_vision.game.move_scorer import MoveDetectorV2, MoveData, get_move_data, should_undo
from chess_vision.game.pgn import generate_pgn, save_pgn

CORNERS_FILE = Path(__file__).parent.parent / "corners.json"
MODEL_PATH = Path(__file__).parent.parent / "models" / "chesscam_pieces.onnx"
XCORNERS_MODEL_PATH = Path(__file__).parent.parent / "models" / "chesscam_xcorners.onnx"


def align_to_existing(new_corners: np.ndarray, old_corners: np.ndarray) -> np.ndarray:
    """Find the cyclic rotation of new_corners that best matches old_corners.

    Mid-game recalibration must NOT re-derive orientation from piece positions:
    after several moves, Q/K and centroids no longer disambiguate sides
    reliably, so auto_detect_corners can return a flipped board. This locks
    the new corners to the existing a1/a8/h8/h1 assignment by picking the
    rotation with the smallest max corner displacement.
    """
    best = new_corners
    best_shift = float("inf")
    for k in range(4):
        rotated = np.roll(new_corners, -k, axis=0)
        shift = float(np.max(np.linalg.norm(rotated - old_corners, axis=1)))
        if shift < best_shift:
            best_shift = shift
            best = rotated
    return best


def read_fresh(cap, drain: int = 4):
    """Read the latest frame, dropping any stale buffered ones first.

    Internal capture buffers can hold ~5 frames; without draining, cap.read()
    returns the OLDEST one, so EMA learns from a frame ~250ms in the past.
    grab() decodes nothing, so draining is cheap.
    """
    for _ in range(drain):
        if not cap.grab():
            return False, None
    return cap.retrieve()


def load_or_select_corners(frame, force_select=False):
    if not force_select and CORNERS_FILE.exists():
        corners = np.array(json.loads(CORNERS_FILE.read_text()), dtype=np.float32)
        print(f"Using saved corners from {CORNERS_FILE}")
        return corners

    print("Click corners in order: a1, a8, h8, h1")
    corners = select_corners(frame)
    CORNERS_FILE.write_text(json.dumps(corners.tolist()))
    print(f"Corners saved to {CORNERS_FILE}")
    return corners


def _game_over_text(board: chess.Board) -> str | None:
    """Return a short label for the game-over reason, or None if still playing."""
    if not board.is_game_over():
        return None
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"CHECKMATE - {winner} wins"
    if board.is_stalemate():
        return "STALEMATE"
    if board.is_insufficient_material():
        return "DRAW - insufficient material"
    if board.is_seventyfive_moves():
        return "DRAW - 75-move rule"
    if board.is_fivefold_repetition():
        return "DRAW - fivefold repetition"
    return "GAME OVER"


def draw_debug(frame, detections, square_centers, board, san_history, corners,
               hand_on_board=False, top_candidates=None, last_fired=""):
    """Draw debug overlay. san_history is a pre-built list of SAN strings.

    top_candidates: optional list of (san, score) tuples (best first) from
    MoveDetectorV2.top_candidates, rendered as a small HUD so the user can
    see what the detector is considering when nothing crosses the firing
    threshold.
    last_fired: MoveDetectorV2.last_move_san, shown so we can see if the
    detector is stuck thinking a move was already played.
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Board outline
    pts = corners.reshape(4, 2).astype(int)
    for i in range(4):
        cv2.line(overlay, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)

    # Best detection per square
    best_per_square: dict[int, dict] = {}
    for det in detections:
        piece_x = det["cx"]
        piece_y = det["cy"] + det["h"] / 2 - det["w"] / 3
        dists = np.sqrt(
            (square_centers[:, 0] - piece_x) ** 2
            + (square_centers[:, 1] - piece_y) ** 2
        )
        sq = int(np.argmin(dists))
        if sq not in best_per_square or det["confidence"] > best_per_square[sq]["confidence"]:
            best_per_square[sq] = det

    for det in best_per_square.values():
        x1 = int(det["cx"] - det["w"] / 2)
        y1 = int(det["cy"] - det["h"] / 2)
        x2 = int(det["cx"] + det["w"] / 2)
        y2 = int(det["cy"] + det["h"] / 2)
        color = (0, 200, 0) if det["class_name"].isupper() else (200, 0, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Move list panel (right side)
    panel_w = 350
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    # Title
    turn = "White" if board.turn == chess.WHITE else "Black"
    cv2.putText(panel, "Chess Vision", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(panel, f"Move {board.fullmove_number} | {turn}", (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    over_text = _game_over_text(board)
    if over_text:
        cv2.putText(panel, over_text, (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif hand_on_board:
        cv2.putText(panel, "HAND DETECTED", (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Top-3 candidates HUD (lets the user see what the detector is "thinking"
    # even when nothing crosses the firing threshold). Each line:
    #   <san>  <score>  [<timer>s]   where timer is the move's greedy clock.
    # When timer reaches greedy_delay (default 1.0s), the move fires.
    hud_y = 160
    if top_candidates:
        cv2.putText(panel, "Considering:", (15, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        hud_y += 25
        for entry in top_candidates:
            # Backwards compatible: entry may be (san, score) or (san, score, timer)
            san, score = entry[0], entry[1]
            timer = entry[2] if len(entry) > 2 else 0.0
            color = (100, 255, 100) if score >= 0.15 else (
                (180, 180, 100) if score > 0 else (140, 140, 140))
            timer_str = f"[{timer:.1f}s]" if timer > 0 else ""
            cv2.putText(panel, f"  {san:<7} {score:+.2f}  {timer_str}", (15, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            hud_y += 22
        hud_y += 8

    # Move list
    y_start = max(160, hud_y + 5)
    line_height = 35
    max_moves_shown = (h - y_start - 60) // line_height

    # Build move text pairs from cached SAN strings
    move_lines = []
    for i, san in enumerate(san_history):
        if i % 2 == 0:
            move_lines.append(f"{i // 2 + 1}. {san}")
        else:
            move_lines[-1] += f"  {san}"

    if len(move_lines) > max_moves_shown:
        move_lines = move_lines[-max_moves_shown:]

    for i, line in enumerate(move_lines):
        y = y_start + i * line_height
        if i == len(move_lines) - 1:
            cv2.putText(panel, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            cv2.putText(panel, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 1)

    # Controls + move count at bottom
    if last_fired:
        cv2.putText(panel, f"Last fired: {last_fired}", (15, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 200), 1)
    cv2.putText(panel, f"{len(san_history)} moves | Q=Quit  R=Reset  C=Corners", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Combine frame + panel
    combined = np.hstack([overlay, panel])
    return combined


def debug_anchor_mapping(detections, square_centers, board_quad):
    """Print rook detections: anchor pixel, assigned square, and offset.

    Compares the ChessCam anchor (cx, cy + h/2 - w/3) against the assigned
    square center to spot systematic offsets.
    """
    from chess_vision.inference.yolo_detect import point_in_quad
    rooks = [d for d in detections if d["class_name"].lower() == "r"]
    if not rooks:
        print("[debug] no rooks detected this frame")
        return

    print(f"[debug] {len(rooks)} rook detection(s):")
    print(f"  {'cls':>4} {'conf':>5}  {'anchor (x,y)':>16}  {'sq':>4}  {'sq_center':>14}  {'dx,dy':>10}  in_quad")
    for d in rooks:
        ax = d["cx"]
        ay = d["cy"] + d["h"] / 2 - d["w"] / 3
        in_quad = point_in_quad(np.array([ax, ay]), board_quad) if board_quad is not None else True
        dists = np.sqrt((square_centers[:, 0] - ax) ** 2 + (square_centers[:, 1] - ay) ** 2)
        sq = int(np.argmin(dists))
        scx, scy = square_centers[sq]
        sq_name = chess.square_name(sq)
        print(f"  {d['class_name']:>4} {d['confidence']:>5.0%}  "
              f"({ax:6.0f},{ay:6.0f})  {sq_name:>4}  "
              f"({scx:6.0f},{scy:6.0f})  ({ax-scx:+4.0f},{ay-scy:+4.0f})  {in_quad}")


def main():
    parser = argparse.ArgumentParser(description="Record a chess game")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, default="game.pgn")
    parser.add_argument("--select-corners", action="store_true", help="Manually click corners")
    parser.add_argument("--auto-corners", action="store_true", help="Auto-detect corners from piece positions")
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--white", type=str, default="White")
    parser.add_argument("--black", type=str, default="Black")
    parser.add_argument("--ema", type=float, default=0.4)
    parser.add_argument("--greedy-delay", type=float, default=1.0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--debug-anchors", action="store_true",
                        help="Print rook anchor->square mappings every ~1.5s")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    # Prevent macOS sleep while recording
    caffeinate_proc = subprocess.Popen(
        ["caffeinate", "-dims"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _run_recording(args, caffeinate_proc)
    finally:
        caffeinate_proc.terminate()


def _run_recording(args, caffeinate_proc):
    print("Loading YOLO piece detector...")
    detector = YoloPieceDetector(str(MODEL_PATH), ema_decay=args.ema)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return

    # Minimize internal buffering so cap.read() returns the freshest frame.
    # On macOS AVFoundation this is often a no-op, hence the buffer-drain
    # in read_fresh() below.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Warming up camera...")
    for _ in range(30):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return
    print(f"Camera ready: {frame.shape[1]}x{frame.shape[0]}")

    if args.auto_corners:
        print("Auto-detecting board corners...")
        if XCORNERS_MODEL_PATH.exists():
            xcorner_det = XCornerDetector(str(XCORNERS_MODEL_PATH))
            dets = detector.detect_raw(frame)
            corners = auto_detect_corners(dets, xcorner_det, frame)
            if corners is not None:
                print(f"  Auto-detected corners: {corners.astype(int).tolist()}")
                CORNERS_FILE.write_text(json.dumps(corners.tolist()))
            else:
                print("  Auto-detection failed. Falling back to manual.")
                print("  Click corners in order: a1, a8, h8, h1")
                corners = select_corners(frame)
                CORNERS_FILE.write_text(json.dumps(corners.tolist()))
        else:
            print(f"  Xcorners model not found at {XCORNERS_MODEL_PATH}")
            print("  Falling back to manual selection.")
            corners = select_corners(frame)
            CORNERS_FILE.write_text(json.dumps(corners.tolist()))
    else:
        corners = load_or_select_corners(frame, force_select=args.select_corners)
    square_centers = compute_square_centers(corners, frame.shape)
    crop_region = compute_crop_region(corners)
    board_quad = compute_board_quad(corners)

    # Quick detection test
    dets = detector.detect_raw(frame, crop_region=crop_region)
    print(f"Initial detection: {len(dets)} pieces")

    # Warmup: build EMA state (no move detection yet)
    WARMUP_FRAMES = 30  # ~1.5s at 20 FPS
    print("Stabilizing (~1.5s)...")
    for _ in range(WARMUP_FRAMES):
        time.sleep(args.interval)
        ret, frame = read_fresh(cap)
        if ret:
            dets = detector.detect_raw(frame, crop_region=crop_region)
            update = detector.detections_to_board(dets, square_centers, board_quad)
            detector.update_state(update)

    # Count how many squares look occupied after warmup
    occupied = int(np.sum(np.max(detector.state, axis=1) > 0.3))
    print(f"Ready! ({occupied} squares look occupied)")

    # Structured event log: one JSONL file per game, hand to Claude after
    # play to reconstruct exactly what the detector saw and decided.
    from datetime import datetime
    session_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    games_dir = Path(__file__).parent.parent / "games"
    games_dir.mkdir(exist_ok=True)
    debug_log_path = games_dir / f"{session_timestamp}_{args.white}_vs_{args.black}_debug.jsonl"
    event_log = EventLog(debug_log_path)
    event_log.log("session_start", white=args.white, black=args.black,
                  greedy_delay=args.greedy_delay, ema=args.ema, interval=args.interval,
                  corners=corners.tolist())
    print(f"Event log: {debug_log_path}")

    board = chess.Board()
    move_history: list[chess.Move] = []
    san_history: list[str] = []  # Cached SAN strings (avoids replaying game each frame)
    move_data_history: list[MoveData] = []  # Per-move from/to squares + targets, for undo
    move_detector = MoveDetectorV2(greedy_delay=args.greedy_delay, event_log=event_log)

    print()
    print("=== RECORDING ===")
    print(f"Output: {args.output}")
    print("Press Q in window or Ctrl+C to stop.")
    print()
    print(board)
    print()

    frame_count = 0
    greedy_pending = False
    last_fire_time = 0.0          # time.monotonic() at most recent push
    UNDO_CHECK_DELAY = 0.5        # seconds after fire before undo check runs
    HAND_TRIGGER_DELAY = 0.15     # seconds of continuous low-count before freeze
    HAND_RATIO_THRESHOLD = 0.5    # detected < 50% of expected => possible hand
    hand_low_start: float | None = None

    # Timing buckets for the debug HUD snapshot. Accumulated between
    # snapshots, averaged and reset each snapshot. Useful to find the
    # bottleneck when FPS is lower than expected.
    timings: dict[str, list[float]] = {
        "read_fresh": [], "detect_raw": [], "detect_move": [],
        "draw_debug": [], "cv2_show": [], "loop": [],
    }

    # Load xcorner detector (used only by the C-key auto-recalibration now;
    # the mid-game periodic recalibration was removed - it always rejected
    # in practice and cost CPU on every 200th frame).
    xcorner_det = None
    if XCORNERS_MODEL_PATH.exists():
        xcorner_det = XCornerDetector(str(XCORNERS_MODEL_PATH))

    try:
        # Loop runs until user hits Q. After is_game_over() we keep rendering
        # the final position with the GAME OVER overlay so the user can read
        # the result, and we stop pushing moves.
        # Helper to record loop time exactly once per iteration regardless
        # of which continue path we take. Call before each continue and at
        # the natural end. Sets loop_recorded so we don't double-count.
        def _record_loop():
            timings["loop"].append(time.monotonic() - loop_start)

        while True:
            loop_start = time.monotonic()

            _t0 = time.perf_counter()
            ret, frame = read_fresh(cap)
            timings["read_fresh"].append(time.perf_counter() - _t0)
            if not ret:
                _record_loop()
                continue

            game_over = board.is_game_over()

            _t0 = time.perf_counter()
            dets = detector.detect_raw(frame, crop_region=crop_region)
            timings["detect_raw"].append(time.perf_counter() - _t0)
            update = detector.detections_to_board(dets, square_centers, board_quad)

            if args.debug_anchors and frame_count % 30 == 0:
                debug_anchor_mapping(dets, square_centers, board_quad)

            # Hand detection: freeze state when piece count drops sustainedly.
            # Time-based (not frame-count-based) so it behaves the same at
            # any FPS. Threshold softened to 50% - YOLO routinely misses a
            # few pieces in a single frame, so 30% was too tight.
            expected_pieces = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
            detected_pieces = int(np.sum(np.max(update, axis=1) > 0.3))
            now_mono = time.monotonic()
            if detected_pieces < expected_pieces * HAND_RATIO_THRESHOLD:
                if hand_low_start is None:
                    hand_low_start = now_mono
                hand_on_board = (now_mono - hand_low_start) >= HAND_TRIGGER_DELAY
            else:
                hand_low_start = None
                hand_on_board = False

            if not hand_on_board:
                detector.update_state(update)
            # else: don't update EMA, hand is blocking pieces

            frame_count += 1

            # Periodic snapshot of full state. ~1.5s cadence keeps the file
            # readable but loses no important transitions (events fill the
            # gaps). Timings are averaged over the frames since the last
            # snapshot, so the 'loop' field also tells us effective FPS.
            if frame_count % 30 == 0:
                def _avg_ms(bucket: list[float]) -> float:
                    return round(1000 * sum(bucket) / len(bucket), 1) if bucket else 0.0
                hand_low_ms = 0.0 if hand_low_start is None else round(
                    (time.monotonic() - hand_low_start) * 1000, 0
                )

                # Diagnostic: re-run YOLO with a low confidence threshold so
                # we can see weak detections that the live pipeline filters.
                # Helps answer "is YOLO seeing the bishop on e2 at 0.18 and
                # we're throwing it away?" Top 20 by confidence; filter to
                # detections whose anchor is on or near the board.
                low_dets = detector.detect_raw(frame, crop_region=crop_region, min_conf=0.05)
                low_dets.sort(key=lambda d: -d["confidence"])
                low_summary = []
                for det in low_dets[:20]:
                    ax = det["cx"]
                    ay = det["cy"] + det["h"] / 2 - det["w"] / 3
                    in_quad = bool(point_in_quad(np.array([ax, ay]), board_quad))
                    if in_quad:
                        dists = np.sqrt(
                            (square_centers[:, 0] - ax) ** 2
                            + (square_centers[:, 1] - ay) ** 2
                        )
                        sq_idx = int(np.argmin(dists))
                        sq_name = chess.square_name(sq_idx)
                        sq_dist = float(dists[sq_idx])
                    else:
                        sq_name = "off"
                        sq_dist = -1.0
                    low_summary.append({
                        "cls": det["class_name"],
                        "conf": round(det["confidence"], 3),
                        "anchor": [round(ax, 0), round(ay, 0)],
                        "sq": sq_name,
                        "sq_dist": round(sq_dist, 1),
                    })

                event_log.log(
                    "snapshot",
                    frame=frame_count,
                    fen=board.fen(),
                    turn="white" if board.turn == chess.WHITE else "black",
                    moves=len(move_history),
                    candidates=[
                        {"san": entry[0], "score": round(entry[1], 3),
                         "timer": round(entry[2] if len(entry) > 2 else 0.0, 2)}
                        for entry in move_detector.top_candidates
                    ],
                    last_fired=move_detector.last_move_san,
                    hand_on_board=hand_on_board,
                    hand_low_ms=hand_low_ms,
                    detected_pieces=detected_pieces,
                    expected_pieces=expected_pieces,
                    greedy_pending=greedy_pending,
                    timings_ms={k: _avg_ms(v) for k, v in timings.items()},
                    low_conf_dets=low_summary,
                )
                for bucket in timings.values():
                    bucket.clear()

            # Draw debug every 3rd frame
            if not args.no_display and frame_count % 3 == 0:
                _t0 = time.perf_counter()
                debug = draw_debug(frame, dets, square_centers, board, san_history, corners,
                                   hand_on_board=hand_on_board,
                                   top_candidates=move_detector.top_candidates,
                                   last_fired=move_detector.last_move_san)
                timings["draw_debug"].append(time.perf_counter() - _t0)
                _t0 = time.perf_counter()
                cv2.imshow("Chess Vision", debug)
                timings["cv2_show"].append(time.perf_counter() - _t0)
            _t0 = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF
            timings["cv2_show"].append(time.perf_counter() - _t0)
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Reset: clear all moves, restart from beginning
                board = chess.Board()
                move_history.clear()
                san_history.clear()
                move_data_history.clear()
                greedy_pending = False
                last_fire_time = 0.0
                hand_low_start = None
                move_detector = MoveDetectorV2(greedy_delay=args.greedy_delay, event_log=event_log)
                # Re-snapshot the current board as the new reference
                detector.state = np.zeros((64, 12), dtype=np.float32)
                detector.initialized = False
            elif key == ord("c"):
                # Re-detect corners mid-game. Try auto first; fall back to
                # manual click if auto fails or isn't available. Auto-corners
                # returns a guess that may be in the wrong rotation, so we
                # run align_to_existing to keep the existing a1/a8/h8/h1
                # orientation - user's mid-game intent is "board got bumped",
                # not "I flipped the board."
                ret_calib, calib_frame = read_fresh(cap)
                new_corners = None
                if ret_calib and xcorner_det is not None:
                    print("Auto-detecting corners...")
                    try:
                        full_dets = detector.detect_raw(calib_frame)
                        candidate = auto_detect_corners(full_dets, xcorner_det, calib_frame)
                        if candidate is not None:
                            new_corners = align_to_existing(candidate, corners)
                            shift = float(np.max(np.linalg.norm(new_corners - corners, axis=1)))
                            print(f"Auto-detected corners (max shift {shift:.0f}px).")
                    except Exception as e:
                        print(f"Auto-detect failed: {e}")
                if new_corners is None and ret_calib:
                    print("Falling back to manual selection. Click a1, a8, h8, h1.")
                    try:
                        new_corners = select_corners(calib_frame)
                    except KeyboardInterrupt:
                        print("Corner re-selection cancelled.")
                        new_corners = None
                if new_corners is not None:
                    corners = new_corners
                    CORNERS_FILE.write_text(json.dumps(corners.tolist()))
                    square_centers = compute_square_centers(corners, calib_frame.shape)
                    crop_region = compute_crop_region(corners)
                    board_quad = compute_board_quad(corners)
                    detector.state = np.zeros((64, 12), dtype=np.float32)
                    detector.initialized = False
                    hand_low_start = None
                    print("Corners updated. Game state preserved.")

            # No more move detection once the game has ended.
            if game_over:
                _record_loop()
                continue

            # Don't check for moves while hand is on board
            if hand_on_board:
                _record_loop()
                continue

            # Auto-undo: if last move was greedy and looks wrong, retract it.
            # Uses full MoveData so castling (rook squares) and en passant
            # (captured pawn square) are checked too, not just king/pawn travel.
            if (greedy_pending and (time.monotonic() - last_fire_time) >= UNDO_CHECK_DELAY):
                last_data = move_data_history[-1]
                from_occ = max(
                    float(np.max(detector.state[sq])) for sq in last_data.from_squares
                )
                to_occ = min(
                    float(detector.state[sq, last_data.targets[i]])
                    for i, sq in enumerate(last_data.to_squares)
                )
                if should_undo(detector.state, last_data):
                    event_log.log("undo", san=last_data.san,
                                  from_occ=round(from_occ, 3), to_occ=round(to_occ, 3))
                    board.pop()
                    move_history.pop()
                    san_history.pop()
                    move_data_history.pop()
                    greedy_pending = False
                    _record_loop()
                    continue
                else:
                    event_log.log("confirm", san=last_data.san,
                                  from_occ=round(from_occ, 3), to_occ=round(to_occ, 3))
                    greedy_pending = False

            _t0 = time.perf_counter()
            san = move_detector.detect_move(board, detector.state)
            timings["detect_move"].append(time.perf_counter() - _t0)
            if san is None:
                _record_loop()
                continue

            # Defensive: detector should only return legal SANs, but if cache
            # ever desyncs we'd rather skip the frame than crash the recording.
            try:
                move = board.parse_san(san)
            except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError) as e:
                print(f"[detect] dropping illegal SAN '{san}': {e}")
                _record_loop()
                continue

            move_data = get_move_data(board, move)  # capture before push
            board.push(move)
            move_history.append(move)
            san_history.append(san)
            move_data_history.append(move_data)
            last_fire_time = time.monotonic()
            greedy_pending = True  # All moves start as tentative

            # Moves are shown in the display window, not terminal

            # Sleep only the remaining time to hit target interval.
            # Record 'loop' BEFORE the sleep so it measures actual work,
            # not wall time (wall time is inferable from snapshot cadence).
            _record_loop()
            elapsed = time.monotonic() - loop_start
            remaining = max(0, args.interval - elapsed)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        event_log.log("session_end", moves=len(move_history),
                      final_fen=board.fen(),
                      game_over=board.is_game_over())
        event_log.close()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        if not move_history:
            print("\nNo moves recorded.")
        else:
            # Determine result
            if board.is_checkmate():
                result = "0-1" if board.turn == chess.WHITE else "1-0"
            elif board.is_game_over():
                result = "1/2-1/2"
            else:
                result = "*"

            pgn = generate_pgn(move_history, white_name=args.white,
                               black_name=args.black, result=result)

            # Save to specified output
            save_pgn(pgn, Path(args.output))

            # Save PGN with the SAME session_timestamp as the debug log so
            # the two files are easy to pair up afterwards.
            game_file = games_dir / f"{session_timestamp}_{args.white}_vs_{args.black}.pgn"
            save_pgn(pgn, game_file)

            # Copy PGN to clipboard
            try:
                subprocess.run(["pbcopy"], input=pgn.encode(), check=True)
                clipboard_ok = True
            except (FileNotFoundError, subprocess.CalledProcessError):
                clipboard_ok = False

            print(f"\nGame saved to:")
            print(f"  {args.output}")
            print(f"  {game_file}")
            if clipboard_ok:
                print(f"  (copied to clipboard)")
            print(f"Moves: {len(move_history)} | Result: {result}")
            print(f"\nFinal position:")
            print(board)
            print(f"\nFEN: {board.fen()}")


if __name__ == "__main__":
    main()

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
    compute_board_quad,
)
from chess_vision.game.move_scorer import MoveDetectorV2, MoveData, get_move_data
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
               hand_on_board=False):
    """Draw debug overlay. san_history is a pre-built list of SAN strings."""
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

    # Move list
    y_start = 160
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
    cv2.putText(panel, f"{len(san_history)} moves | Q=Quit  R=Reset", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

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

    board = chess.Board()
    move_history: list[chess.Move] = []
    san_history: list[str] = []  # Cached SAN strings (avoids replaying game each frame)
    move_data_history: list[MoveData] = []  # Per-move from/to squares + targets, for undo
    move_detector = MoveDetectorV2(greedy_delay=args.greedy_delay)

    print()
    print("=== RECORDING ===")
    print(f"Output: {args.output}")
    print("Press Q in window or Ctrl+C to stop.")
    print()
    print(board)
    print()

    frames_since_last_move = 0
    frame_count = 0
    greedy_pending = False
    UNDO_CHECK_FRAMES = 10
    RECALIBRATE_INTERVAL = 200  # Re-detect corners every ~10s at 20 FPS
    HAND_TRIGGER_FRAMES = 2     # Consecutive low-count frames before freezing
    hand_low_streak = 0

    # Load xcorner detector for periodic recalibration
    xcorner_det = None
    if XCORNERS_MODEL_PATH.exists():
        xcorner_det = XCornerDetector(str(XCORNERS_MODEL_PATH))

    try:
        # Loop runs until user hits Q. After is_game_over() we keep rendering
        # the final position with the GAME OVER overlay so the user can read
        # the result, and we stop pushing moves.
        while True:
            loop_start = time.monotonic()

            ret, frame = read_fresh(cap)
            if not ret:
                continue

            game_over = board.is_game_over()

            # Periodically re-detect corners (handles board shifting mid-game).
            # Mid-game we lock the new corners to the existing orientation
            # (a1/a8/h8/h1 mapping) and only accept SMALL shifts. Large shifts
            # almost always mean auto_detect_corners got confused (e.g. flipped
            # the board) once the position is no longer the starting one.
            if not game_over and xcorner_det and frame_count > 0 and frame_count % RECALIBRATE_INTERVAL == 0:
                try:
                    full_dets = detector.detect_raw(frame)
                    new_corners = auto_detect_corners(full_dets, xcorner_det, frame)
                    if new_corners is not None:
                        aligned = align_to_existing(new_corners, corners)
                        per_corner_shift = np.linalg.norm(aligned - corners, axis=1)
                        max_shift = float(np.max(per_corner_shift))
                        # Accept tiny adjustments only. Reject anything that
                        # smells like an orientation swap or a detection blunder.
                        if 15 < max_shift < 80:
                            corners = aligned
                            square_centers = compute_square_centers(corners, frame.shape)
                            crop_region = compute_crop_region(corners)
                            board_quad = compute_board_quad(corners)
                            # EMA was learned on the OLD square grid; reset so
                            # square assignments re-learn from fresh detections.
                            detector.state = np.zeros((64, 12), dtype=np.float32)
                            detector.initialized = False
                except Exception as e:
                    print(f"[recalibrate] skipped: {e}")

            dets = detector.detect_raw(frame, crop_region=crop_region)
            update = detector.detections_to_board(dets, square_centers, board_quad)

            if args.debug_anchors and frame_count % 30 == 0:
                debug_anchor_mapping(dets, square_centers, board_quad)

            # Hand detection: freeze state when piece count drops sustainedly.
            # Single-frame dips (one bad detection) shouldn't trigger; otherwise
            # noise freezes tracking and good moves get missed. Requires
            # HAND_TRIGGER_FRAMES consecutive low frames before declaring a hand.
            expected_pieces = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
            detected_pieces = int(np.sum(np.max(update, axis=1) > 0.3))
            if detected_pieces < expected_pieces * 0.7:
                hand_low_streak += 1
            else:
                hand_low_streak = 0
            hand_on_board = hand_low_streak >= HAND_TRIGGER_FRAMES

            if not hand_on_board:
                detector.update_state(update)
            # else: don't update EMA, hand is blocking pieces

            frames_since_last_move += 1
            frame_count += 1

            # Draw debug every 3rd frame
            if not args.no_display and frame_count % 3 == 0:
                debug = draw_debug(frame, dets, square_centers, board, san_history, corners,
                                   hand_on_board=hand_on_board)
                cv2.imshow("Chess Vision", debug)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Reset: clear all moves, restart from beginning
                board = chess.Board()
                move_history.clear()
                san_history.clear()
                move_data_history.clear()
                greedy_pending = False
                frames_since_last_move = 0
                hand_low_streak = 0
                move_detector = MoveDetectorV2(greedy_delay=args.greedy_delay)
                # Re-snapshot the current board as the new reference
                detector.state = np.zeros((64, 12), dtype=np.float32)
                detector.initialized = False

            # No more move detection once the game has ended.
            if game_over:
                continue

            # Don't check for moves while hand is on board
            if hand_on_board:
                continue

            # Auto-undo: if last move was greedy and looks wrong, retract it.
            # Uses full MoveData so castling (rook squares) and en passant
            # (captured pawn square) are checked too, not just king/pawn travel.
            if (greedy_pending and frames_since_last_move >= UNDO_CHECK_FRAMES):
                last_data = move_data_history[-1]
                # Any of the from-squares still looking occupied -> undo
                from_occ = max(
                    float(np.max(detector.state[sq])) for sq in last_data.from_squares
                )
                # Any to-square not showing the expected piece -> undo
                to_occ = min(
                    float(detector.state[sq, last_data.targets[i]])
                    for i, sq in enumerate(last_data.to_squares)
                )
                if from_occ > 0.4 or to_occ < 0.2:
                    board.pop()
                    move_history.pop()
                    san_history.pop()
                    move_data_history.pop()
                    greedy_pending = False
                    continue
                else:
                    # Move confirmed, no longer pending
                    greedy_pending = False

            san = move_detector.detect_move(board, detector.state)
            if san is None:
                continue

            # Defensive: detector should only return legal SANs, but if cache
            # ever desyncs we'd rather skip the frame than crash the recording.
            try:
                move = board.parse_san(san)
            except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError) as e:
                print(f"[detect] dropping illegal SAN '{san}': {e}")
                continue

            move_data = get_move_data(board, move)  # capture before push
            board.push(move)
            move_history.append(move)
            san_history.append(san)
            move_data_history.append(move_data)
            frames_since_last_move = 0
            greedy_pending = True  # All moves start as tentative

            # Moves are shown in the display window, not terminal

            # Sleep only the remaining time to hit target interval
            elapsed = time.monotonic() - loop_start
            remaining = max(0, args.interval - elapsed)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
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

            # Also save to games/ with timestamp
            from datetime import datetime
            games_dir = Path(__file__).parent.parent / "games"
            games_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            game_file = games_dir / f"{timestamp}_{args.white}_vs_{args.black}.pgn"
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

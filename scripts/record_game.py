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
from chess_vision.game.move_scorer import MoveDetectorV2
from chess_vision.game.pgn import generate_pgn, save_pgn

CORNERS_FILE = Path(__file__).parent.parent / "corners.json"
MODEL_PATH = Path(__file__).parent.parent / "models" / "chesscam_pieces.onnx"
XCORNERS_MODEL_PATH = Path(__file__).parent.parent / "models" / "chesscam_xcorners.onnx"


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

    if hand_on_board:
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
        ret, frame = cap.read()
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

    # Load xcorner detector for periodic recalibration
    xcorner_det = None
    if XCORNERS_MODEL_PATH.exists():
        xcorner_det = XCornerDetector(str(XCORNERS_MODEL_PATH))

    try:
        while not board.is_game_over():
            loop_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                continue

            # Periodically re-detect corners (handles board shifting mid-game)
            if xcorner_det and frame_count > 0 and frame_count % RECALIBRATE_INTERVAL == 0:
                try:
                    # Use full frame (no crop) so piece colors are detected for orientation
                    full_dets = detector.detect_raw(frame)
                    new_corners = auto_detect_corners(full_dets, xcorner_det, frame)
                    if new_corners is not None:
                        shift = np.max(np.abs(new_corners - corners))
                        if shift > 15:
                            corners = new_corners
                            square_centers = compute_square_centers(corners, frame.shape)
                            crop_region = compute_crop_region(corners)
                            board_quad = compute_board_quad(corners)
                            # Reset EMA so it re-learns with new orientation
                            detector.state = np.zeros((64, 12), dtype=np.float32)
                            detector.initialized = False
                except Exception:
                    pass

            dets = detector.detect_raw(frame, crop_region=crop_region)
            update = detector.detections_to_board(dets, square_centers, board_quad)

            if args.debug_anchors and frame_count % 30 == 0:
                debug_anchor_mapping(dets, square_centers, board_quad)

            # Hand detection: if piece count drops significantly, freeze state
            expected_pieces = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
            detected_pieces = int(np.sum(np.max(update, axis=1) > 0.3))
            hand_on_board = detected_pieces < expected_pieces * 0.7

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
                greedy_pending = False
                move_detector = MoveDetectorV2(greedy_delay=args.greedy_delay)
                # Re-snapshot the current board as the new reference
                detector.state = np.zeros((64, 12), dtype=np.float32)
                detector.initialized = False

            # Don't check for moves while hand is on board
            if hand_on_board:
                continue

            # Auto-undo: if last move was greedy and looks wrong, retract it
            if (greedy_pending and frames_since_last_move >= UNDO_CHECK_FRAMES):
                last_move = move_history[-1]
                from_sq = last_move.from_square
                to_sq = last_move.to_square
                from_occ = float(np.max(detector.state[from_sq]))
                to_occ = float(np.max(detector.state[to_sq]))
                # If from-square still looks occupied OR to-square looks empty, undo
                if from_occ > 0.4 or to_occ < 0.2:
                    board.pop()
                    move_history.pop()
                    san_history.pop()
                    greedy_pending = False
                    # Undo shown in display window via move_history update
                    continue
                else:
                    # Move confirmed, no longer pending
                    greedy_pending = False

            san = move_detector.detect_move(board, detector.state)
            if san is None:
                continue

            move = board.parse_san(san)
            board.push(move)
            move_history.append(move)
            san_history.append(san)
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

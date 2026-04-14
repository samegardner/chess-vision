"""Record a chess game using ChessCam's YOLO model + move scoring.

Usage:
    python scripts/record_game.py
    python scripts/record_game.py --select-corners
    python scripts/record_game.py --output mygame.pgn
"""

import argparse
import json
import time
import sys
from pathlib import Path

import chess
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.board.detect import select_corners
from chess_vision.board.auto_corners import auto_detect_corners
from chess_vision.inference.yolo_detect import (
    YoloPieceDetector, compute_square_centers, compute_crop_region,
    compute_board_quad,
)
from chess_vision.game.move_scorer import MoveDetectorV2
from chess_vision.game.pgn import generate_pgn, save_pgn

CORNERS_FILE = Path(__file__).parent.parent / "corners.json"
MODEL_PATH = Path(__file__).parent.parent / "models" / "chesscam_pieces.onnx"


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


def draw_debug(frame, detections, square_centers, board, last_move_san, corners):
    overlay = frame.copy()
    pts = corners.reshape(4, 2).astype(int)
    for i in range(4):
        cv2.line(overlay, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)

    # Keep only the best detection per square (avoids duplicate boxes)
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

    turn = "White" if board.turn == chess.WHITE else "Black"
    status = f"{turn} | Move {board.fullmove_number}"
    if last_move_san:
        status += f" | Last: {last_move_san}"
    h = frame.shape[0]
    cv2.putText(overlay, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Record a chess game")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, default="game.pgn")
    parser.add_argument("--select-corners", action="store_true", help="Manually click corners")
    parser.add_argument("--auto-corners", action="store_true", help="Auto-detect corners from piece positions")
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--white", type=str, default="White")
    parser.add_argument("--black", type=str, default="Black")
    parser.add_argument("--ema", type=float, default=0.2)
    parser.add_argument("--greedy-delay", type=float, default=0.4)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

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
        # Experimental: auto-detect from piece positions
        print("Auto-detecting board corners (experimental)...")
        dets = detector.detect_raw(frame)
        corners = auto_detect_corners(dets)
        if corners is None:
            print("  Auto-detection failed. Falling back to manual.")
            corners = select_corners(frame)
        else:
            print(f"  Detected corners: {corners.astype(int).tolist()}")
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
    last_move_san = ""
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
    UNDO_CHECK_FRAMES = 10  # ~0.5s at 20 FPS, check if greedy move was correct

    try:
        while not board.is_game_over():
            loop_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                continue

            dets = detector.detect_raw(frame, crop_region=crop_region)
            update = detector.detections_to_board(dets, square_centers, board_quad)

            # Hand detection: if piece count drops significantly, freeze state
            expected_pieces = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
            detected_pieces = int(np.sum(np.max(update, axis=1) > 0.3))
            hand_on_board = detected_pieces < expected_pieces * 0.6

            if not hand_on_board:
                detector.update_state(update)
            # else: don't update EMA, hand is blocking pieces

            frames_since_last_move += 1
            frame_count += 1

            # Draw debug every 3rd frame
            if not args.no_display and frame_count % 3 == 0:
                debug = draw_debug(frame, dets, square_centers, board, last_move_san, corners)
                if hand_on_board:
                    cv2.putText(debug, "HAND DETECTED", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow("Chess Vision", debug)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

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
                    undone = move_history.pop()
                    last_move_san = move_history[-1].uci() if move_history else ""
                    greedy_pending = False
                    print(f"  (undid {chess.square_name(undone.from_square)}{chess.square_name(undone.to_square)})")
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
            last_move_san = san
            frames_since_last_move = 0
            greedy_pending = True  # All moves start as tentative

            if board.turn == chess.WHITE:
                print(f"  {board.fullmove_number - 1}... {san}")
            else:
                print(f"  {board.fullmove_number}. {san}")

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
            import subprocess
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

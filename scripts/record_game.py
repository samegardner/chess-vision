"""Record a chess game using ChessCam's YOLO model + move scoring.

Uses pretrained LeYOLO for piece detection, EMA smoothing, and
ChessCam-style two-move lookahead with greedy fallback.

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
from chess_vision.board.warp import order_corners
from chess_vision.inference.yolo_detect import (
    YoloPieceDetector, compute_square_centers, YOLO_CLASSES,
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
    print("Click the 4 corners of the chessboard...")
    corners = select_corners(frame)
    CORNERS_FILE.write_text(json.dumps(corners.tolist()))
    print(f"Corners saved to {CORNERS_FILE}")
    return corners


def draw_debug(frame, detections, square_centers, board, last_move_san, corners):
    overlay = frame.copy()
    ordered = order_corners(corners)
    pts = ordered.astype(int)
    for i in range(4):
        cv2.line(overlay, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)

    for det in detections:
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
    parser.add_argument("--select-corners", action="store_true")
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--white", type=str, default="White")
    parser.add_argument("--black", type=str, default="Black")
    parser.add_argument("--ema", type=float, default=0.5)
    parser.add_argument("--greedy-delay", type=float, default=1.5,
                        help="Seconds a move must be top candidate before greedy accept")
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

    corners = load_or_select_corners(frame, force_select=args.select_corners)
    square_centers = compute_square_centers(corners, frame.shape)

    # Warmup: let EMA stabilize
    WARMUP_FRAMES = 15
    print(f"Stabilizing ({WARMUP_FRAMES * args.interval:.0f}s)...")
    for i in range(WARMUP_FRAMES):
        time.sleep(args.interval)
        ret, frame = cap.read()
        if ret:
            dets = detector.detect_raw(frame)
            update = detector.detections_to_board(dets, square_centers)
            detector.update_state(update)
            if not args.no_display:
                debug = draw_debug(frame, dets, square_centers, chess.Board(), "", corners)
                cv2.imshow("Chess Vision", debug)
                cv2.waitKey(1)

    print("Ready!")

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

    try:
        while not board.is_game_over():
            time.sleep(args.interval)
            ret, frame = cap.read()
            if not ret:
                continue

            dets = detector.detect_raw(frame)
            update = detector.detections_to_board(dets, square_centers)
            detector.update_state(update)

            if not args.no_display:
                debug = draw_debug(frame, dets, square_centers, board, last_move_san, corners)
                cv2.imshow("Chess Vision", debug)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # Check for moves using ChessCam-style scoring
            san = move_detector.detect_move(board, detector.state)
            if san is None:
                continue

            move = board.parse_san(san)
            board.push(move)
            move_history.append(move)
            last_move_san = san

            if board.turn == chess.WHITE:
                print(f"  {board.fullmove_number - 1}... {san}")
            else:
                print(f"  {board.fullmove_number}. {san}")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        pgn = generate_pgn(move_history, white_name=args.white, black_name=args.black)
        save_pgn(pgn, Path(args.output))
        print(f"\nGame saved to {args.output} ({len(move_history)} moves)")
        if move_history:
            print(f"\nFinal position:")
            print(board)
            print(f"\nFEN: {board.fen()}")


if __name__ == "__main__":
    main()

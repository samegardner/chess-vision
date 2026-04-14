"""Record a chess game using ChessCam's YOLO model + EMA smoothing.

Uses a pretrained LeYOLO model for piece detection, exponential moving
average for temporal smoothing, and python-chess legal move scoring.

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
    YoloPieceDetector, compute_square_centers, YOLO_CLASSES, YOLO_TO_FEN,
)
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


def score_move(move: chess.Move, board: chess.Board, state: np.ndarray) -> float:
    """Score how well the current state matrix matches a move being played.

    Checks: do the 'from' squares look empty? Do the 'to' squares have
    the expected piece? Higher score = better match.
    """
    score = 0.0
    threshold = 0.3

    # Simulate the move
    from_sq = move.from_square
    to_sq = move.to_square
    piece = board.piece_at(from_sq)
    if piece is None:
        return -999

    # From square should now be empty (low occupancy)
    from_max = float(np.max(state[from_sq]))
    score += (threshold - from_max)  # Positive if square looks empty

    # To square should have the moved piece
    piece_sym = piece.symbol()
    if move.promotion:
        piece_sym = chess.piece_symbol(move.promotion)
        if piece.color == chess.WHITE:
            piece_sym = piece_sym.upper()

    if piece_sym in YOLO_CLASSES:
        class_idx = YOLO_CLASSES.index(piece_sym)
        to_score = float(state[to_sq, class_idx])
        score += (to_score - threshold)

    # Castling: rook should also move
    if board.is_castling(move):
        rank = 0 if board.turn == chess.WHITE else 7
        if board.is_kingside_castling(move):
            rook_from = chess.square(7, rank)
            rook_to = chess.square(5, rank)
        else:
            rook_from = chess.square(0, rank)
            rook_to = chess.square(3, rank)

        score += (threshold - float(np.max(state[rook_from])))
        rook_sym = "R" if board.turn == chess.WHITE else "r"
        rook_class = YOLO_CLASSES.index(rook_sym)
        score += (float(state[rook_to, rook_class]) - threshold)

    # En passant: captured pawn should disappear
    if board.is_en_passant(move):
        cap_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
        score += (threshold - float(np.max(state[cap_sq])))

    return score


def find_best_move(board: chess.Board, state: np.ndarray, min_score: float = 0.25) -> chess.Move | None:
    """Find the legal move that best matches the current state matrix."""
    best_move = None
    best_score = min_score  # Minimum threshold to accept

    for move in board.legal_moves:
        s = score_move(move, board, state)
        if s > best_score:
            best_score = s
            best_move = move

    return best_move


def draw_debug(frame, detections, square_centers, board, last_move_san, corners):
    """Draw debug overlay on the camera frame."""
    overlay = frame.copy()

    # Draw board outline
    ordered = order_corners(corners)
    pts = ordered.astype(int)
    for i in range(4):
        cv2.line(overlay, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)

    # Draw detections as bounding boxes
    for det in detections:
        x1 = int(det["cx"] - det["w"] / 2)
        y1 = int(det["cy"] - det["h"] / 2)
        x2 = int(det["cx"] + det["w"] / 2)
        y2 = int(det["cy"] + det["h"] / 2)
        color = (0, 200, 0) if det["class_name"].isupper() else (200, 0, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw square centers
    for i, center in enumerate(square_centers):
        sq = chess.square_name(i)
        piece = board.piece_at(i)
        if piece:
            cv2.circle(overlay, tuple(center.astype(int)), 3, (255, 255, 0), -1)

    # Status
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
    parser.add_argument("--ema", type=float, default=0.5, help="EMA decay (0=no smoothing, 1=infinite memory)")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Download from: https://drive.google.com/file/d/1-80xp_nly9i6s3o0mF0mU9OZGEzUAlGj")
        return

    # Load YOLO model
    print("Loading YOLO piece detector...")
    detector = YoloPieceDetector(str(MODEL_PATH), ema_decay=args.ema)

    # Open camera
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

    # Get corners
    corners = load_or_select_corners(frame, force_select=args.select_corners)
    square_centers = compute_square_centers(corners, frame.shape)

    # Quick test: run detection on first frame
    dets = detector.detect_raw(frame)
    print(f"Initial detection: {len(dets)} pieces found")

    # Warmup: let EMA stabilize on starting position before accepting moves
    WARMUP_FRAMES = 20  # ~6 seconds at 0.3s interval
    print(f"Stabilizing ({WARMUP_FRAMES * args.interval:.0f}s warmup)...")
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
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{WARMUP_FRAMES} frames...")
    print("Ready!")

    # Game state
    board = chess.Board()
    move_history: list[chess.Move] = []
    last_move_san = ""
    frames_since_move = 0
    MIN_FRAMES_BETWEEN_MOVES = 8  # ~2.4s at 0.3s interval

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

            # Detect pieces
            dets = detector.detect_raw(frame)
            update = detector.detections_to_board(dets, square_centers)
            detector.update_state(update)

            frames_since_move += 1

            # Show debug
            if not args.no_display:
                debug = draw_debug(frame, dets, square_centers, board, last_move_san, corners)
                cv2.imshow("Chess Vision", debug)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # Don't check for moves too soon after the last one
            if frames_since_move < MIN_FRAMES_BETWEEN_MOVES:
                continue

            # Find best matching legal move
            move = find_best_move(board, detector.state)
            if move is None:
                continue

            san = board.san(move)
            board.push(move)
            move_history.append(move)
            last_move_san = san
            frames_since_move = 0

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

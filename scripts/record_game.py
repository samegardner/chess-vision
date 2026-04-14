"""Record a chess game from the camera using pixel-based move detection.

No CNN needed for move tracking. Uses pixel differencing on the warped
board to detect which squares changed, then python-chess to resolve
the legal move.

Usage:
    python scripts/record_game.py
    python scripts/record_game.py --select-corners
    python scripts/record_game.py --output mygame.pgn
    python scripts/record_game.py --threshold 20    # Adjust sensitivity
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
from chess_vision.board.warp import compute_homography, warp_board
from chess_vision.game.pixel_moves import PixelMoveDetector
from chess_vision.game.pgn import generate_pgn, save_pgn

CORNERS_FILE = Path(__file__).parent.parent / "corners.json"


def load_or_select_corners(frame, force_select=False):
    """Load saved corners or prompt user to select them."""
    if not force_select and CORNERS_FILE.exists():
        corners = np.array(json.loads(CORNERS_FILE.read_text()), dtype=np.float32)
        print(f"Using saved corners from {CORNERS_FILE}")
        return corners

    print("Click the 4 corners of the chessboard...")
    corners = select_corners(frame)
    CORNERS_FILE.write_text(json.dumps(corners.tolist()))
    print(f"Corners saved to {CORNERS_FILE}")
    return corners


def resolve_pixel_move(
    changed_squares: list[str],
    board: chess.Board,
) -> chess.Move | None:
    """Find the legal move that involves the changed squares.

    For each legal move, check if the squares it touches match the
    observed changed squares. This doesn't need CNN output since we
    track game state from the known starting position.
    """
    changed_set = set(changed_squares)

    best_move = None
    best_overlap = 0

    for move in board.legal_moves:
        # Squares this move touches
        move_squares = {chess.square_name(move.from_square), chess.square_name(move.to_square)}

        # Castling touches rook squares too
        if board.is_castling(move):
            if board.is_kingside_castling(move):
                rook_file = "h"
                new_rook_file = "f"
            else:
                rook_file = "a"
                new_rook_file = "d"
            rank = "1" if board.turn == chess.WHITE else "8"
            move_squares.add(f"{rook_file}{rank}")
            move_squares.add(f"{new_rook_file}{rank}")

        # En passant removes the captured pawn
        if board.is_en_passant(move):
            cap_rank = "5" if board.turn == chess.WHITE else "4"
            cap_file = chess.square_name(move.to_square)[0]
            move_squares.add(f"{cap_file}{cap_rank}")

        # Check overlap between expected and observed changes
        overlap = len(move_squares & changed_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_move = move

        # Perfect match
        if move_squares == changed_set:
            return move

    # If we found a good partial match (at least from+to squares matched)
    if best_move and best_overlap >= 2:
        return best_move

    return None


def main():
    parser = argparse.ArgumentParser(description="Record a chess game")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, default="game.pgn")
    parser.add_argument("--select-corners", action="store_true", help="Re-select board corners")
    parser.add_argument("--interval", type=float, default=0.3, help="Seconds between captures")
    parser.add_argument("--white", type=str, default="White")
    parser.add_argument("--black", type=str, default="Black")
    parser.add_argument("--threshold", type=float, default=20.0, help="Pixel change threshold per square")
    parser.add_argument("--stability", type=int, default=4, help="Stable frames before accepting move")
    args = parser.parse_args()

    # Open camera with warmup
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
    H = compute_homography(corners)

    # Get initial warped board
    warped = warp_board(frame, H)

    # Set up game and pixel detector
    board = chess.Board()
    detector = PixelMoveDetector(
        stability_frames=args.stability,
        change_threshold=args.threshold,
    )
    detector.set_reference(warped)
    move_history: list[chess.Move] = []

    print()
    print("=== RECORDING ===")
    print(f"Output: {args.output}")
    print(f"Sensitivity: threshold={args.threshold}, stability={args.stability} frames")
    print("Make moves on the board. Press Ctrl+C to stop and save.")
    print()

    try:
        while not board.is_game_over():
            time.sleep(args.interval)
            ret, frame = cap.read()
            if not ret:
                continue

            warped = warp_board(frame, H)
            changed = detector.detect_change(warped)

            if changed is None:
                continue

            # Try to resolve the move
            move = resolve_pixel_move(changed, board)
            if move is None:
                print(f"  ? Changed squares {changed} don't match any legal move")
                # Reset reference to current frame to avoid getting stuck
                detector.set_reference(warped)
                continue

            san = board.san(move)
            board.push(move)
            move_history.append(move)

            if board.turn == chess.WHITE:
                # Black just moved
                print(f"  {board.fullmove_number - 1}... {san}")
            else:
                # White just moved
                print(f"  {board.fullmove_number}. {san}")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        cap.release()
        pgn = generate_pgn(
            move_history,
            white_name=args.white,
            black_name=args.black,
        )
        save_pgn(pgn, Path(args.output))
        print(f"\nGame saved to {args.output} ({len(move_history)} moves)")
        if move_history:
            print(f"Final FEN: {board.fen()}")


if __name__ == "__main__":
    main()

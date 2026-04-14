"""Record a chess game using pixel-based move detection.

Tracks moves from the known starting position using:
1. Pixel differencing on the warped board to detect which squares changed
2. python-chess to resolve which legal move matches those changes

No CNN needed. Shows a live debug view of the board with change detection.

Usage:
    python scripts/record_game.py
    python scripts/record_game.py --select-corners
    python scripts/record_game.py --output mygame.pgn
    python scripts/record_game.py --threshold 20     # Adjust sensitivity
    python scripts/record_game.py --no-display        # No GUI, terminal only
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

# FEN chars for display
PIECE_SYMBOLS = {
    "R": "R", "N": "N", "B": "B", "Q": "Q", "K": "K", "P": "P",
    "r": "r", "n": "n", "b": "b", "q": "q", "k": "k", "p": "p",
}


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


def resolve_pixel_move(changed_squares: list[str], board: chess.Board) -> chess.Move | None:
    """Find the legal move that best matches the observed changed squares."""
    changed_set = set(changed_squares)
    best_move = None
    best_overlap = 0

    for move in board.legal_moves:
        move_squares = {chess.square_name(move.from_square), chess.square_name(move.to_square)}

        # Castling touches rook squares too
        if board.is_castling(move):
            rank = "1" if board.turn == chess.WHITE else "8"
            if board.is_kingside_castling(move):
                move_squares.update({f"h{rank}", f"f{rank}"})
            else:
                move_squares.update({f"a{rank}", f"d{rank}"})

        # En passant removes the captured pawn
        if board.is_en_passant(move):
            cap_rank = "5" if board.turn == chess.WHITE else "4"
            cap_file = chess.square_name(move.to_square)[0]
            move_squares.add(f"{cap_file}{cap_rank}")

        overlap = len(move_squares & changed_set)

        # Perfect match
        if move_squares == changed_set:
            return move

        if overlap > best_overlap:
            best_overlap = overlap
            best_move = move

    # Accept if at least the from and to squares matched
    if best_move and best_overlap >= 2:
        return best_move
    return None


def draw_debug(warped: np.ndarray, board: chess.Board, changed_squares: list[str] | None,
               detector: PixelMoveDetector) -> np.ndarray:
    """Draw debug overlay on warped board."""
    h, w = warped.shape[:2]
    sq_h = h // 8
    sq_w = w // 8
    overlay = warped.copy()

    changed_set = set(changed_squares) if changed_squares else set()
    candidate_set = detector.candidate_squares or set()

    for rank in range(8):
        for file in range(8):
            row = 7 - rank
            x1 = file * sq_w
            y1 = row * sq_h
            x2 = x1 + sq_w
            y2 = y1 + sq_h
            sq_name = chr(ord("a") + file) + str(rank + 1)

            # Get piece from python-chess board state (ground truth from game tracking)
            sq = chess.parse_square(sq_name)
            piece = board.piece_at(sq)

            # Color coding
            if sq_name in changed_set:
                # Confirmed change: bright green
                cv2.rectangle(overlay, (x1+1, y1+1), (x2-1, y2-1), (0, 255, 0), 3)
            elif sq_name in candidate_set:
                # Candidate change (not yet stable): yellow
                cv2.rectangle(overlay, (x1+1, y1+1), (x2-1, y2-1), (0, 255, 255), 2)
            elif piece:
                # Occupied square: dim green
                cv2.rectangle(overlay, (x1+1, y1+1), (x2-1, y2-1), (0, 120, 0), 1)

            # Draw piece label from game state
            if piece:
                label = piece.symbol()
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
                tx = x1 + (sq_w - text_size[0]) // 2
                ty = y1 + (sq_h + text_size[1]) // 2
                cv2.putText(overlay, label, (tx, ty), font, 0.7, (0, 0, 0), 3)
                cv2.putText(overlay, label, (tx, ty), font, 0.7, (255, 255, 255), 2)

            # Square name in corner
            cv2.putText(overlay, sq_name, (x1+2, y1+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # Status bar
    status = f"Stability: {detector.stable_count}/{detector.stability_frames}"
    if detector.hand_present:
        status += " | HAND DETECTED"
    elif candidate_set:
        status += f" | Watching: {sorted(candidate_set)}"
    cv2.putText(overlay, status, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Record a chess game")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, default="game.pgn")
    parser.add_argument("--select-corners", action="store_true")
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--white", type=str, default="White")
    parser.add_argument("--black", type=str, default="Black")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Pixel change threshold per square (lower = more sensitive)")
    parser.add_argument("--stability", type=int, default=4,
                        help="Stable frames needed before accepting a move")
    parser.add_argument("--no-display", action="store_true", help="No GUI window")
    args = parser.parse_args()

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
    H = compute_homography(corners)
    warped = warp_board(frame, H)

    # Game state (starts from known starting position)
    board = chess.Board()
    move_history: list[chess.Move] = []

    # Pixel-based move detector
    detector = PixelMoveDetector(
        stability_frames=args.stability,
        change_threshold=args.threshold,
    )
    detector.set_reference(warped)

    print()
    print("=== RECORDING ===")
    print(f"Output: {args.output}")
    print(f"Sensitivity: threshold={args.threshold}, stability={args.stability}")
    print("Make moves on the board. Press Q in the window or Ctrl+C to stop.")
    print()

    # Print starting position
    print("Starting position:")
    print(board)
    print()

    try:
        while not board.is_game_over():
            time.sleep(args.interval)
            ret, frame = cap.read()
            if not ret:
                continue

            warped = warp_board(frame, H)
            changed = detector.detect_change(warped)

            # Show debug view
            if not args.no_display:
                debug = draw_debug(warped, board, changed, detector)
                cv2.imshow("Chess Vision - Recording", debug)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nQuitting...")
                    break

            if changed is None:
                continue

            # Resolve the move
            move = resolve_pixel_move(changed, board)
            if move is None:
                print(f"  ? Changed squares {changed} don't match any legal move. Resetting.")
                detector.set_reference(warped)
                continue

            san = board.san(move)
            board.push(move)
            move_history.append(move)

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

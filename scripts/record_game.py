"""Record a chess game using pixel-based move detection.

No CNN needed. Uses pixel differencing to detect changed squares,
then python-chess to find the matching legal move. Accepts partial
matches (even a single changed square) when only one legal move
involves that square.

Usage:
    python scripts/record_game.py
    python scripts/record_game.py --select-corners
    python scripts/record_game.py --output mygame.pgn
    python scripts/record_game.py --threshold 12
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
    if not force_select and CORNERS_FILE.exists():
        corners = np.array(json.loads(CORNERS_FILE.read_text()), dtype=np.float32)
        print(f"Using saved corners from {CORNERS_FILE}")
        return corners
    print("Click the 4 corners of the chessboard...")
    corners = select_corners(frame)
    CORNERS_FILE.write_text(json.dumps(corners.tolist()))
    print(f"Corners saved to {CORNERS_FILE}")
    return corners


def get_move_squares(move: chess.Move, board: chess.Board) -> set[str]:
    """Get all squares a move touches (from, to, plus castling/en passant squares)."""
    squares = {chess.square_name(move.from_square), chess.square_name(move.to_square)}

    if board.is_castling(move):
        rank = "1" if board.turn == chess.WHITE else "8"
        if board.is_kingside_castling(move):
            squares.update({f"h{rank}", f"f{rank}"})
        else:
            squares.update({f"a{rank}", f"d{rank}"})

    if board.is_en_passant(move):
        cap_rank = "5" if board.turn == chess.WHITE else "4"
        cap_file = chess.square_name(move.to_square)[0]
        squares.add(f"{cap_file}{cap_rank}")

    return squares


def resolve_move(changed_squares: list[str], board: chess.Board) -> chess.Move | None:
    """Find the legal move that best matches the changed squares.

    Relaxed matching: accepts even a single changed square if only one
    legal move involves it. Scores candidates by overlap count.
    """
    changed_set = set(changed_squares)
    candidates: list[tuple[chess.Move, int]] = []

    for move in board.legal_moves:
        move_squares = get_move_squares(move, board)
        overlap = len(move_squares & changed_set)

        if overlap > 0:
            candidates.append((move, overlap))

    if not candidates:
        return None

    # Sort by overlap (most matching squares first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_overlap = candidates[0][1]

    # Get all candidates tied for best overlap
    best_candidates = [m for m, o in candidates if o == best_overlap]

    if len(best_candidates) == 1:
        return best_candidates[0]

    # Multiple candidates with same overlap. Try to disambiguate:
    # Prefer moves where the changed squares are exactly the move squares
    for move in best_candidates:
        if get_move_squares(move, board) == changed_set:
            return move

    # Prefer moves with higher overlap ratio (changed covers more of the move)
    best_ratio = 0
    best_move = best_candidates[0]
    for move in best_candidates:
        ms = get_move_squares(move, board)
        ratio = len(ms & changed_set) / len(ms)
        if ratio > best_ratio:
            best_ratio = ratio
            best_move = move

    return best_move


def draw_debug(warped, board, changed_squares, detector, last_move_san):
    """Draw debug overlay."""
    h, w = warped.shape[:2]
    sq_h, sq_w = h // 8, w // 8
    overlay = warped.copy()
    changed_set = set(changed_squares) if changed_squares else set()
    candidate_set = detector.candidate_squares or set()

    for rank in range(8):
        for file in range(8):
            row = 7 - rank
            x1, y1 = file * sq_w, row * sq_h
            x2, y2 = x1 + sq_w, y1 + sq_h
            sq_name = chr(ord("a") + file) + str(rank + 1)
            sq = chess.parse_square(sq_name)
            piece = board.piece_at(sq)

            if sq_name in changed_set:
                cv2.rectangle(overlay, (x1+1, y1+1), (x2-1, y2-1), (0, 255, 0), 3)
            elif sq_name in candidate_set:
                cv2.rectangle(overlay, (x1+1, y1+1), (x2-1, y2-1), (0, 255, 255), 2)
            elif piece:
                cv2.rectangle(overlay, (x1+1, y1+1), (x2-1, y2-1), (0, 120, 0), 1)

            if piece:
                label = piece.symbol()
                font = cv2.FONT_HERSHEY_SIMPLEX
                ts = cv2.getTextSize(label, font, 0.7, 2)[0]
                tx = x1 + (sq_w - ts[0]) // 2
                ty = y1 + (sq_h + ts[1]) // 2
                cv2.putText(overlay, label, (tx, ty), font, 0.7, (0, 0, 0), 3)
                cv2.putText(overlay, label, (tx, ty), font, 0.7, (255, 255, 255), 2)

    turn = "White" if board.turn == chess.WHITE else "Black"
    status = f"{turn} to move | Move {board.fullmove_number}"
    if last_move_san:
        status += f" | Last: {last_move_san}"
    if detector.hand_present:
        status += " | HAND"
    elif candidate_set:
        status += f" | Watching {sorted(candidate_set)}"
    cv2.putText(overlay, status, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Record a chess game")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, default="game.pgn")
    parser.add_argument("--select-corners", action="store_true")
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--white", type=str, default="White")
    parser.add_argument("--black", type=str, default="Black")
    parser.add_argument("--threshold", type=float, default=12.0)
    parser.add_argument("--stability", type=int, default=4)
    parser.add_argument("--no-display", action="store_true")
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

    corners = load_or_select_corners(frame, force_select=args.select_corners)
    H = compute_homography(corners)
    warped = warp_board(frame, H)

    board = chess.Board()
    detector = PixelMoveDetector(
        stability_frames=args.stability,
        change_threshold=args.threshold,
    )
    detector.set_reference(warped)
    move_history: list[chess.Move] = []
    last_move_san = ""

    print()
    print("=== RECORDING ===")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.threshold}, Stability: {args.stability} frames")
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

            warped = warp_board(frame, H)
            changed = detector.detect_change(warped)

            if not args.no_display:
                debug = draw_debug(warped, board, changed, detector, last_move_san)
                cv2.imshow("Chess Vision", debug)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if changed is None:
                continue

            move = resolve_move(changed, board)
            if move is None:
                print(f"  ? Changed: {changed}, no legal move matched. Resetting.")
                detector.set_reference(warped)
                continue

            san = board.san(move)
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

"""Collect training data from your camera by playing through a game.

How it works:
1. Set up the starting position, select corners
2. Play moves on the board. After each move, press SPACE to capture
3. The script tracks the game state (it knows what's on every square)
4. Each capture saves 64 labeled square images to disk
5. When done, press Q. Then run fine-tuning on the collected data.

You don't need to type any FEN or label anything. Just play chess
and press SPACE after each move.

Usage:
    python scripts/collect_data.py
    python scripts/collect_data.py --select-corners
    python scripts/collect_data.py --output-dir data/my_board
"""

import argparse
import json
import sys
from pathlib import Path

import chess
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.board.detect import select_corners
from chess_vision.board.warp import compute_homography, warp_board
from chess_vision.board.squares import extract_squares, square_index_to_name
from chess_vision.models.piece import PIECE_TO_FEN

CORNERS_FILE = Path(__file__).parent.parent / "corners.json"

# Reverse mapping: FEN char -> class folder name
FEN_TO_FOLDER = {
    "P": "white_pawn", "N": "white_knight", "B": "white_bishop",
    "R": "white_rook", "Q": "white_queen", "K": "white_king",
    "p": "black_pawn", "n": "black_knight", "b": "black_bishop",
    "r": "black_rook", "q": "black_queen", "k": "black_king",
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


def save_squares(warped: np.ndarray, board: chess.Board, output_dir: Path, capture_id: int):
    """Extract squares from warped board and save with labels from game state."""
    squares = extract_squares(warped, padding=0.5)
    saved = 0

    for i, sq_img in enumerate(squares):
        sq_name = square_index_to_name(i)
        sq = chess.parse_square(sq_name)
        piece = board.piece_at(sq)

        if piece:
            folder = FEN_TO_FOLDER[piece.symbol()]
        else:
            folder = "empty"

        class_dir = output_dir / folder
        class_dir.mkdir(parents=True, exist_ok=True)
        filename = f"cap{capture_id:04d}_{sq_name}.png"
        cv2.imwrite(str(class_dir / filename), sq_img)
        saved += 1

    return saved


def draw_overlay(warped: np.ndarray, board: chess.Board, capture_count: int, move_num: int) -> np.ndarray:
    """Draw current game state and instructions on the warped board."""
    h, w = warped.shape[:2]
    sq_h = h // 8
    sq_w = w // 8
    overlay = warped.copy()

    for rank in range(8):
        for file in range(8):
            row = 7 - rank
            x1 = file * sq_w
            y1 = row * sq_h
            sq_name = chr(ord("a") + file) + str(rank + 1)
            sq = chess.parse_square(sq_name)
            piece = board.piece_at(sq)

            if piece:
                label = piece.symbol()
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(label, font, 0.6, 2)[0]
                tx = x1 + (sq_w - text_size[0]) // 2
                ty = y1 + (sq_h + text_size[1]) // 2
                cv2.putText(overlay, label, (tx, ty), font, 0.6, (0, 0, 0), 3)
                cv2.putText(overlay, label, (tx, ty), font, 0.6, (255, 255, 255), 2)

    # Instructions
    turn = "White" if board.turn == chess.WHITE else "Black"
    info = f"Move {move_num} | {turn} to play | Captures: {capture_count} | SPACE=capture  U=undo  Q=quit"
    cv2.putText(overlay, info, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Collect training data by playing chess")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--select-corners", action="store_true")
    parser.add_argument("--output-dir", type=str, default="data/my_board")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Game state
    board = chess.Board()
    capture_count = 0
    total_squares = 0
    move_history: list[str] = []

    print()
    print("=== DATA COLLECTION ===")
    print(f"Output: {output_dir}")
    print()
    print("Instructions:")
    print("  1. Board should show the starting position now")
    print("  2. Press SPACE to capture the current position")
    print("  3. Make a move on the physical board")
    print("  4. Type the move (e.g., e4, Nf3, O-O) and press ENTER")
    print("  5. Press SPACE to capture the new position")
    print("  6. Repeat until done. Press Q to quit.")
    print()

    # Capture starting position first
    print("Press SPACE to capture the starting position...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        warped = warp_board(frame, H)
        overlay = draw_overlay(warped, board, capture_count, board.fullmove_number)
        cv2.imshow("Data Collection", overlay)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(" "):
            # Capture current position
            n = save_squares(warped, board, output_dir, capture_count)
            capture_count += 1
            total_squares += n
            print(f"  Captured position {capture_count} ({n} squares, {total_squares} total)")

            if capture_count == 1:
                print()
                print("Now make a move on the board, then type it below.")
                print()

        elif key == ord("q"):
            break

        elif key == 13:  # ENTER - this won't work in OpenCV window
            pass

        elif key == ord("u"):
            # Undo last move
            if move_history:
                board.pop()
                undone = move_history.pop()
                print(f"  Undid move: {undone}")

    cap.release()
    cv2.destroyAllWindows()

    # Now prompt for moves in the terminal
    print()
    print(f"Window closed. Captured {capture_count} positions so far.")
    print()
    print("Now enter moves to collect more data.")
    print("After typing each move, the script will prompt you to capture.")
    print("Type 'done' when finished.")
    print()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not reopen camera")
        _print_summary(output_dir, capture_count, total_squares)
        return

    # Drain a few frames
    for _ in range(10):
        cap.read()

    while True:
        turn = "White" if board.turn == chess.WHITE else "Black"
        move_str = input(f"  {board.fullmove_number}. {turn}'s move (or 'done'): ").strip()

        if move_str.lower() == "done":
            break
        if move_str.lower() == "undo":
            if move_history:
                board.pop()
                undone = move_history.pop()
                print(f"    Undid: {undone}")
            continue

        try:
            move = board.parse_san(move_str)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
            print(f"    Invalid move: {e}")
            print(f"    Legal moves: {', '.join(board.san(m) for m in board.legal_moves)}")
            continue

        board.push(move)
        move_history.append(move_str)
        print(f"    Played: {move_str}")
        print(f"    Position the board to match, then press ENTER to capture...")
        input()

        # Capture
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret:
            warped = warp_board(frame, H)
            n = save_squares(warped, board, output_dir, capture_count)
            capture_count += 1
            total_squares += n
            print(f"    Captured position {capture_count} ({total_squares} total squares)")
        else:
            print("    Failed to capture frame")

    cap.release()
    _print_summary(output_dir, capture_count, total_squares)


def _print_summary(output_dir: Path, capture_count: int, total_squares: int):
    print()
    print(f"=== DONE ===")
    print(f"Captured {capture_count} positions, {total_squares} labeled squares")
    print(f"Saved to {output_dir}/")
    print()

    # Count per class
    for class_dir in sorted(output_dir.iterdir()):
        if class_dir.is_dir():
            n = len(list(class_dir.glob("*.png")))
            print(f"  {class_dir.name}: {n}")

    print()
    print("To train on this data:")
    print(f"  python scripts/train_optimized.py")
    print("  (will need to add support for custom data dir)")


if __name__ == "__main__":
    main()

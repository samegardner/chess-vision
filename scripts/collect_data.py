"""Collect training data from your camera by playing through a game.

Fully terminal-based. No OpenCV windows during collection.

Flow:
1. Select corners (one-time)
2. Set up starting position, press ENTER to capture
3. Make a move on the board, type it (e.g. e4), press ENTER to capture
4. Repeat. Type 'done' when finished.

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

CORNERS_FILE = Path(__file__).parent.parent / "corners.json"

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


def capture_and_save(cap, H, board, output_dir, capture_id):
    """Capture a frame, warp it, extract squares, save with labels."""
    # Read a few frames to get a fresh one
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        print("    ERROR: Failed to capture frame!")
        return 0

    warped = warp_board(frame, H)
    squares = extract_squares(warped, padding=0.5)
    saved = 0

    for i, sq_img in enumerate(squares):
        sq_name = square_index_to_name(i)
        sq = chess.parse_square(sq_name)
        piece = board.piece_at(sq)

        folder = FEN_TO_FOLDER[piece.symbol()] if piece else "empty"
        class_dir = output_dir / folder
        class_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(class_dir / f"cap{capture_id:04d}_{sq_name}.png"), sq_img)
        saved += 1

    return saved


def print_board_compact(board):
    """Print a compact board representation."""
    print()
    print("    a b c d e f g h")
    print("    ---------------")
    for rank in range(7, -1, -1):
        row = f" {rank+1}| "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            row += (piece.symbol() if piece else ".") + " "
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Collect training data")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--select-corners", action="store_true")
    parser.add_argument("--output-dir", type=str, default="data/my_board")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Open camera and get corners
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

    # Game state
    board = chess.Board()
    capture_count = 0
    total_squares = 0
    move_list = []

    print()
    print("=" * 50)
    print("  CHESS DATA COLLECTION")
    print("=" * 50)
    print()
    print("Set up the starting position on the board.")
    print()

    # Step 1: Capture starting position
    input(">>> Press ENTER to capture the starting position...")
    n = capture_and_save(cap, H, board, output_dir, capture_count)
    capture_count += 1
    total_squares += n
    print(f"    CAPTURED! Position #{capture_count} ({total_squares} squares total)")
    print_board_compact(board)

    # Step 2: Play moves
    print("Now play moves. After each move on the board:")
    print("  1. Type the move (e.g. e4, Nf3, Bb5, O-O)")
    print("  2. Press ENTER to record it and capture the position")
    print()
    print("Commands: 'undo' to take back, 'done' to finish")
    print()

    while True:
        turn = "White" if board.turn == chess.WHITE else "Black"
        move_str = input(f">>> {board.fullmove_number}. {turn}'s move: ").strip()

        if not move_str:
            continue

        if move_str.lower() == "done":
            break

        if move_str.lower() == "undo":
            if move_list:
                board.pop()
                undone = move_list.pop()
                print(f"    Undid: {undone}")
                print_board_compact(board)
            else:
                print("    Nothing to undo.")
            continue

        # Parse the move
        try:
            move = board.parse_san(move_str)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
            print(f"    Invalid: {e}")
            legal = [board.san(m) for m in board.legal_moves]
            # Show a few examples
            if len(legal) > 10:
                print(f"    Some legal moves: {', '.join(legal[:10])}...")
            else:
                print(f"    Legal moves: {', '.join(legal)}")
            continue

        # Apply the move
        san = board.san(move)
        board.push(move)
        move_list.append(san)

        # Show what happened
        if board.turn == chess.WHITE:
            print(f"    Played: {board.fullmove_number - 1}... {san}")
        else:
            print(f"    Played: {board.fullmove_number}. {san}")

        # Capture
        print("    Make sure the board matches, then press ENTER to capture...")
        input()
        n = capture_and_save(cap, H, board, output_dir, capture_count)
        capture_count += 1
        total_squares += n
        print(f"    CAPTURED! Position #{capture_count} ({total_squares} squares total)")
        print_board_compact(board)

    cap.release()

    # Summary
    print()
    print("=" * 50)
    print("  COLLECTION COMPLETE")
    print("=" * 50)
    print(f"  Positions captured: {capture_count}")
    print(f"  Total squares: {total_squares}")
    print(f"  Saved to: {output_dir}/")
    print()

    for class_dir in sorted(output_dir.iterdir()):
        if class_dir.is_dir():
            n = len(list(class_dir.glob("*.png")))
            print(f"    {class_dir.name}: {n}")

    if move_list:
        print()
        print(f"  Moves played: {' '.join(move_list)}")

    print()
    print("  Next step: retrain the model on this data")
    print()


if __name__ == "__main__":
    main()

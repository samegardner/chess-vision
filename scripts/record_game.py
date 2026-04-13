"""Record a chess game from the camera.

Usage:
    python scripts/record_game.py
    python scripts/record_game.py --select-corners   # Re-select corners
    python scripts/record_game.py --output mygame.pgn
"""

import argparse
import json
import time
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.board.detect import select_corners
from chess_vision.board.warp import compute_homography, warp_board
from chess_vision.board.squares import extract_squares, remap_board_state
from chess_vision.inference.onnx_runtime import ONNXClassifier
from chess_vision.inference.classify import classify_board, board_to_fen
from chess_vision.game.state import GameState
from chess_vision.game.moves import MoveDetector
from chess_vision.game.rules import resolve_move, detect_orientation
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


def main():
    parser = argparse.ArgumentParser(description="Record a chess game")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, default="game.pgn")
    parser.add_argument("--select-corners", action="store_true", help="Re-select board corners")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between captures")
    parser.add_argument("--white", type=str, default="White")
    parser.add_argument("--black", type=str, default="Black")
    parser.add_argument("--profile", type=str, default=None, help="Use calibrated models from profile")
    args = parser.parse_args()

    # Load models
    project_root = Path(__file__).resolve().parent.parent
    if args.profile:
        occ_path = project_root / f"profiles/{args.profile}/occupancy.onnx"
        piece_path = project_root / f"profiles/{args.profile}/piece.onnx"
    else:
        occ_path = project_root / "models/occupancy.onnx"
        piece_path = project_root / "models/piece.onnx"

    if not occ_path.exists() or not piece_path.exists():
        print(f"Model files not found: {occ_path}, {piece_path}")
        print("Run 'chess-vision train --stage base --fast' first.")
        return

    print("Loading models...")
    occ = ONNXClassifier(occ_path)
    piece = ONNXClassifier(piece_path)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return
    print(f"Camera ready: {frame.shape[1]}x{frame.shape[0]}")

    # Get corners
    corners = load_or_select_corners(frame, force_select=args.select_corners)
    H = compute_homography(corners)

    # Initial board state
    warped = warp_board(frame, H)
    squares = extract_squares(warped)
    board_state = classify_board(squares, occ, piece)

    flipped = False
    try:
        orientation = detect_orientation(board_state)
        print(f"Orientation: {orientation}")
        if orientation == "white_top":
            flipped = True
            board_state = remap_board_state(board_state)
            print("Board is flipped, remapping squares.")
    except ValueError:
        print("Could not detect orientation, assuming white on bottom.")

    fen = board_to_fen(board_state)
    print(f"Detected FEN: {fen}")
    print()

    # Game loop
    game = GameState()
    detector = MoveDetector()
    detector.set_initial_board(board_state)

    print("=== RECORDING ===")
    print(f"Output: {args.output}")
    print("Make moves on the board. Press Ctrl+C to stop and save.")
    print()

    try:
        while not game.is_game_over():
            time.sleep(args.interval)
            ret, frame = cap.read()
            if not ret:
                continue

            warped = warp_board(frame, H)
            squares = extract_squares(warped)
            board_state = classify_board(squares, occ, piece)
            if flipped:
                board_state = remap_board_state(board_state)

            changed = detector.detect_change(board_state)
            if changed is None:
                continue

            move = resolve_move(changed, game.board, board_state)
            if move is None:
                print(f"  ? Unresolved change: {changed}")
                continue

            san = game.board.san(move)
            game.apply_move(move)

            if game.whose_turn() == "white":
                print(f"  {game.move_number() - 1}... {san}")
            else:
                print(f"  {game.move_number()}. {san}")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        cap.release()
        pgn = generate_pgn(
            game.move_history,
            white_name=args.white,
            black_name=args.black,
        )
        save_pgn(pgn, Path(args.output))
        print(f"\nGame saved to {args.output} ({len(game.move_history)} moves)")
        if game.move_history:
            print(f"Final FEN: {game.get_fen()}")


if __name__ == "__main__":
    main()

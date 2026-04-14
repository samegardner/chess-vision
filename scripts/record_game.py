"""Record a chess game.

Uses a hybrid approach:
1. Starting position is hardcoded (always the same)
2. Pixel differencing detects WHICH squares changed
3. CNN classifies changed squares to determine the new piece (or empty)
4. python-chess validates which legal move matches

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
from chess_vision.board.warp import compute_homography, warp_board
from chess_vision.board.squares import extract_squares, square_index_to_name, name_to_square_index
from chess_vision.inference.onnx_runtime import ONNXClassifier
from chess_vision.inference.classify import _prepare_batch
from chess_vision.game.pixel_moves import PixelMoveDetector
from chess_vision.game.pgn import generate_pgn, save_pgn
from chess_vision.config import MODELS_DIR

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


def classify_square(
    square_img: np.ndarray,
    occ_model: ONNXClassifier,
    piece_model: ONNXClassifier,
    piece_class_order: list[str],
    input_size: int,
) -> str | None:
    """Classify a single square. Returns FEN char or None (empty)."""
    class_to_fen = {
        "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
        "white_rook": "R", "white_queen": "Q", "white_king": "K",
        "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
        "black_rook": "r", "black_queen": "q", "black_king": "k",
    }

    batch = _prepare_batch([square_img], input_size=input_size)
    occ_logit = occ_model.predict(batch)[0, 0]
    occ_prob = 1 / (1 + np.exp(-occ_logit))

    if occ_prob < 0.5:
        return None

    piece_logits = piece_model.predict(batch)[0]
    pred_class = piece_class_order[np.argmax(piece_logits)]
    return class_to_fen[pred_class]


def resolve_move_hybrid(
    changed_squares: list[str],
    board: chess.Board,
    squares: list[np.ndarray],
    occ_model: ONNXClassifier,
    piece_model: ONNXClassifier,
    piece_class_order: list[str],
    input_size: int,
) -> chess.Move | None:
    """Resolve a move using changed squares + CNN classification + legal move validation.

    Strategy:
    1. For each legal move, check if it touches the changed squares
    2. If multiple moves match, use CNN to classify the destination square
       to disambiguate
    """
    changed_set = set(changed_squares)
    candidates = []

    for move in board.legal_moves:
        move_squares = {chess.square_name(move.from_square), chess.square_name(move.to_square)}

        if board.is_castling(move):
            rank = "1" if board.turn == chess.WHITE else "8"
            if board.is_kingside_castling(move):
                move_squares.update({f"h{rank}", f"f{rank}"})
            else:
                move_squares.update({f"a{rank}", f"d{rank}"})

        if board.is_en_passant(move):
            cap_rank = "5" if board.turn == chess.WHITE else "4"
            cap_file = chess.square_name(move.to_square)[0]
            move_squares.add(f"{cap_file}{cap_rank}")

        overlap = len(move_squares & changed_set)

        if move_squares == changed_set:
            candidates.append((move, overlap, True))  # Perfect match
        elif overlap >= 2:
            candidates.append((move, overlap, False))

    if not candidates:
        return None

    # Perfect matches first
    perfect = [c for c in candidates if c[2]]
    if len(perfect) == 1:
        return perfect[0][0]

    # Multiple candidates: use CNN to classify destination square
    if len(candidates) >= 1:
        best_move = candidates[0][0]

        if len(candidates) > 1:
            # Use CNN on the destination square to disambiguate
            dest_name = chess.square_name(candidates[0][0].to_square)
            dest_idx = name_to_square_index(dest_name)
            dest_piece = classify_square(
                squares[dest_idx], occ_model, piece_model,
                piece_class_order, input_size,
            )

            # Find the candidate whose result matches CNN prediction
            for move, overlap, perfect in candidates:
                board.push(move)
                expected_piece = board.piece_at(move.to_square)
                board.pop()
                if expected_piece and expected_piece.symbol() == dest_piece:
                    return move

        return best_move

    return None


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
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--stability", type=int, default=4)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    # Load models (for disambiguation only)
    project_root = Path(__file__).resolve().parent.parent
    occ_path = project_root / "models/occupancy.onnx"
    piece_path = project_root / "models/piece.onnx"

    occ_model = None
    piece_model = None
    piece_class_order = []
    input_size = 100

    if occ_path.exists() and piece_path.exists():
        print("Loading CNN models (for disambiguation)...")
        occ_model = ONNXClassifier(occ_path)
        piece_model = ONNXClassifier(piece_path)
        occ_shape = occ_model.session.get_inputs()[0].shape
        input_size = occ_shape[2] if isinstance(occ_shape[2], int) else 100

        classes_file = project_root / "models/piece_classes.json"
        if classes_file.exists():
            piece_class_order = json.load(open(classes_file))
    else:
        print("No CNN models found. Using pixel-only mode (less accurate for ambiguous moves).")

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

    # STARTING POSITION IS KNOWN
    board = chess.Board()
    move_history: list[chess.Move] = []

    # Pixel detector for change detection
    detector = PixelMoveDetector(
        stability_frames=args.stability,
        change_threshold=args.threshold,
    )
    detector.set_reference(warped)

    last_move_san = ""

    print()
    print("=== RECORDING ===")
    print("Starting position assumed. Make moves and they'll be detected.")
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

            warped = warp_board(frame, H)
            squares = extract_squares(warped)
            changed = detector.detect_change(warped)

            if not args.no_display:
                debug = draw_debug(warped, board, changed, detector, last_move_san)
                cv2.imshow("Chess Vision", debug)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if changed is None:
                continue

            # Resolve the move
            if occ_model and piece_model:
                move = resolve_move_hybrid(
                    changed, board, squares,
                    occ_model, piece_model, piece_class_order, input_size,
                )
            else:
                # Pixel-only fallback
                from chess_vision.game.pixel_moves import PixelMoveDetector
                changed_set = set(changed)
                move = None
                for m in board.legal_moves:
                    ms = {chess.square_name(m.from_square), chess.square_name(m.to_square)}
                    if ms == changed_set or (len(ms & changed_set) >= 2):
                        move = m
                        break

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

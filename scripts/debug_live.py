"""Live debug view: shows what the model sees in real-time.

Displays:
- Raw camera feed
- Warped board (top-down view after homography)
- Per-square classification overlay (piece labels on each square)
- Occupancy heatmap (which squares the model thinks are occupied)

Usage:
    python scripts/debug_live.py
    python scripts/debug_live.py --select-corners
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.board.detect import select_corners
from chess_vision.board.warp import compute_homography, warp_board
from chess_vision.board.squares import extract_squares, square_index_to_name
from chess_vision.inference.onnx_runtime import ONNXClassifier
from chess_vision.inference.classify import classify_board, board_to_fen
from chess_vision.config import INPUT_SIZE

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


def draw_debug_board(warped: np.ndarray, board_state: dict, occ_probs: np.ndarray) -> np.ndarray:
    """Draw classification results on the warped board image."""
    h, w = warped.shape[:2]
    sq_h = h // 8
    sq_w = w // 8
    overlay = warped.copy()

    for rank in range(8):
        for file in range(8):
            row = 7 - rank
            x1 = file * sq_w
            y1 = row * sq_h
            x2 = x1 + sq_w
            y2 = y1 + sq_h

            sq_name = chr(ord("a") + file) + str(rank + 1)
            piece = board_state.get(sq_name)
            idx = rank * 8 + file
            prob = occ_probs[idx] if idx < len(occ_probs) else 0

            # Color border based on occupancy probability
            if prob > 0.5:
                # Green border = occupied
                color = (0, 255, 0)
                cv2.rectangle(overlay, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), color, 2)
            else:
                # Dim red border = empty
                color = (0, 0, 100)
                cv2.rectangle(overlay, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), color, 1)

            # Draw piece label
            if piece:
                # White text with black outline for visibility
                label = piece
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                tx = x1 + (sq_w - text_size[0]) // 2
                ty = y1 + (sq_h + text_size[1]) // 2
                cv2.putText(overlay, label, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(overlay, label, (tx, ty), font, font_scale, (255, 255, 255), thickness)

            # Draw occupancy probability
            prob_text = f"{prob:.0%}"
            cv2.putText(overlay, prob_text, (x1 + 2, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

            # Draw square name
            cv2.putText(overlay, sq_name, (x1 + 2, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Live debug view")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--select-corners", action="store_true")
    parser.add_argument("--profile", type=str, default=None)
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
        print(f"Model files not found. Run training first.")
        return

    print("Loading models...")
    occ_model = ONNXClassifier(occ_path)
    piece_model = ONNXClassifier(piece_path)

    # Detect input size from model
    occ_input_shape = occ_model.session.get_inputs()[0].shape
    input_size = occ_input_shape[2] if isinstance(occ_input_shape[2], int) else INPUT_SIZE
    print(f"Model input size: {input_size}x{input_size}")

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

    # Import the batch preparation function
    import json
    from chess_vision.inference.classify import _prepare_batch
    from chess_vision.models.piece import PIECE_CLASSES, PIECE_TO_FEN

    # Load class ordering from training
    classes_file = project_root / "models" / "piece_classes.json"
    if classes_file.exists():
        piece_class_order = json.load(open(classes_file))
        print(f"Using class ordering from {classes_file}")
    else:
        piece_class_order = PIECE_CLASSES

    class_to_fen = {
        "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
        "white_rook": "R", "white_queen": "Q", "white_king": "K",
        "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
        "black_rook": "r", "black_queen": "q", "black_king": "k",
    }

    print()
    print("=== LIVE DEBUG ===")
    print("Press Q to quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        warped = warp_board(frame, H)
        squares = extract_squares(warped)

        # Run occupancy model
        batch = _prepare_batch(squares, input_size=input_size)
        occ_logits = occ_model.predict(batch)
        occ_probs = 1 / (1 + np.exp(-occ_logits.flatten()))  # sigmoid

        # Run piece model on occupied squares
        occupied_indices = [i for i in range(64) if occ_probs[i] > 0.5]
        board_state = {}

        if occupied_indices:
            occ_images = [squares[i] for i in occupied_indices]
            piece_batch = _prepare_batch(occ_images, input_size=input_size)
            piece_logits = piece_model.predict(piece_batch)
            pred_classes = np.argmax(piece_logits, axis=1)

            for j, idx in enumerate(occupied_indices):
                class_name = piece_class_order[pred_classes[j]]
                sq_name = square_index_to_name(idx)
                board_state[sq_name] = class_to_fen[class_name]

        # Fill empty squares
        for i in range(64):
            name = square_index_to_name(i)
            if name not in board_state:
                board_state[name] = None

        # Draw debug overlay
        debug_board = draw_debug_board(warped, board_state, occ_probs)
        fen = board_to_fen(board_state)

        # Stats
        n_occupied = sum(1 for v in board_state.values() if v is not None)
        avg_occ_prob = np.mean(occ_probs)
        stats = f"FEN: {fen} | Occupied: {n_occupied}/64 | Avg occ prob: {avg_occ_prob:.2f}"

        # Show windows
        cv2.imshow("Debug: Warped Board + Classifications", debug_board)
        # Show raw camera feed scaled down
        small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.putText(small_frame, stats[:80], (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Debug: Camera Feed", small_frame)

        # Print FEN periodically
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final FEN: {fen}")


if __name__ == "__main__":
    main()

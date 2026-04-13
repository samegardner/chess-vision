"""Quick camera + board detection test.

Captures a frame, lets you click 4 corners, warps the board, and
saves the results so you can verify everything looks correct.

Usage:
    python scripts/test_camera.py [--camera 0] [--image test_frame.jpg]
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.board.detect import select_corners
from chess_vision.board.warp import compute_homography, warp_board
from chess_vision.board.squares import extract_squares, square_index_to_name


def main():
    parser = argparse.ArgumentParser(description="Test camera and board detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--image", type=str, default=None, help="Use existing image instead of camera")
    args = parser.parse_args()

    # Get frame
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not load {args.image}")
            return
        print(f"Loaded {args.image}: {frame.shape[1]}x{frame.shape[0]}")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Could not open camera {args.camera}")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture frame")
            return
        cv2.imwrite("test_frame.jpg", frame)
        print(f"Captured frame: {frame.shape[1]}x{frame.shape[0]} (saved to test_frame.jpg)")

    # Select corners
    print("\nClick the 4 corners of the chessboard (any order).")
    print("Press R to reset, Q to quit.")
    corners = select_corners(frame)
    print(f"Corners: {corners.astype(int).tolist()}")

    # Warp
    H = compute_homography(corners)
    warped = warp_board(frame, H)
    cv2.imwrite("test_warped.jpg", warped)
    print(f"Saved warped board: test_warped.jpg")

    # Extract and save a few squares
    squares = extract_squares(warped)
    print(f"Extracted {len(squares)} squares")

    for idx in [0, 4, 7, 56, 60, 63]:
        name = square_index_to_name(idx)
        cv2.imwrite(f"test_square_{name}.jpg", squares[idx])

    print("Saved sample squares: a1, e1, h1, a8, e8, h8")
    print("\nCheck test_warped.jpg to verify the board looks correct!")


if __name__ == "__main__":
    main()

"""Debug visualization: draw detected grid and piece labels on board image."""

import sys
from pathlib import Path

import cv2

from chess_vision.board.detect import detect_board
from chess_vision.board.warp import compute_homography, warp_board
from chess_vision.board.squares import extract_squares, square_index_to_name


def visualize(image_path: str) -> None:
    """Load an image, detect the board, and display annotated result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    corners = detect_board(image)
    if corners is None:
        print("Board not detected.")
        return

    # Draw corners on original image
    annotated = image.copy()
    for i, corner in enumerate(corners):
        cv2.circle(annotated, tuple(corner.astype(int)), 10, (0, 0, 255), -1)
        cv2.putText(
            annotated, str(i), tuple(corner.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )

    # Warp and show
    H = compute_homography(corners)
    warped = warp_board(image, H)

    cv2.imshow("Detected Corners", annotated)
    cv2.imshow("Warped Board", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_board.py <image_path>")
        sys.exit(1)
    visualize(sys.argv[1])

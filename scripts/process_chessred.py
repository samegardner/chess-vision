"""Process ChessReD dataset into per-square training images.

Uses the 2,078 images that have corner annotations (ChessReD2K subset).
Warps each board using the annotated corners, extracts 64 squares,
and saves them in ImageFolder format.

Usage:
    python scripts/process_chessred.py
    python scripts/process_chessred.py --output-dir data/chessred_squares
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.board.warp import compute_homography, warp_board
from chess_vision.board.squares import extract_squares, square_index_to_name

CHESSRED_DIR = Path(__file__).resolve().parent.parent / "data" / "chessred"

# ChessReD category ID -> folder name
CATEGORY_TO_FOLDER = {
    0: "white_pawn",
    1: "white_rook",
    2: "white_knight",
    3: "white_bishop",
    4: "white_queen",
    5: "white_king",
    6: "black_pawn",
    7: "black_rook",
    8: "black_knight",
    9: "black_bishop",
    10: "black_queen",
    11: "black_king",
    12: "empty",
}


def main():
    parser = argparse.ArgumentParser(description="Process ChessReD into training data")
    parser.add_argument("--output-dir", type=str, default="data/chessred_squares")
    parser.add_argument("--chessred-dir", type=str, default=str(CHESSRED_DIR))
    args = parser.parse_args()

    chessred_dir = Path(args.chessred_dir)
    output_dir = Path(args.output_dir)

    ann_path = chessred_dir / "annotations.json"
    if not ann_path.exists():
        print(f"annotations.json not found at {ann_path}")
        return

    print("Loading annotations...")
    ann = json.load(open(ann_path))

    # Build lookups
    images_by_id = {img["id"]: img for img in ann["images"]}
    categories = {cat["id"]: cat["name"] for cat in ann["categories"]}

    # Build per-image piece annotations: image_id -> {square_name: category_id}
    pieces_by_image: dict[int, dict[str, int]] = {}
    for piece_ann in ann["annotations"]["pieces"]:
        img_id = piece_ann["image_id"]
        if img_id not in pieces_by_image:
            pieces_by_image[img_id] = {}
        sq = piece_ann["chessboard_position"]
        pieces_by_image[img_id][sq] = piece_ann["category_id"]

    # Build corner lookup: image_id -> corners
    corners_by_image: dict[int, np.ndarray] = {}
    for corner_ann in ann["annotations"]["corners"]:
        img_id = corner_ann["image_id"]
        c = corner_ann["corners"]
        # Order: top-left, top-right, bottom-right, bottom-left
        corners_by_image[img_id] = np.array([
            c["top_left"],
            c["top_right"],
            c["bottom_right"],
            c["bottom_left"],
        ], dtype=np.float32)

    print(f"Images with corners: {len(corners_by_image)}")
    print(f"Images with piece annotations: {len(pieces_by_image)}")

    # Process only images that have both corners and piece annotations
    valid_ids = set(corners_by_image.keys()) & set(pieces_by_image.keys())
    print(f"Images to process: {len(valid_ids)}")

    total_saved = 0
    skipped = 0

    for img_id in tqdm(sorted(valid_ids), desc="Processing"):
        img_info = images_by_id[img_id]
        img_path = chessred_dir / img_info["path"]

        if not img_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        corners = corners_by_image[img_id]
        H = compute_homography(corners)
        warped = warp_board(img, H)
        squares = extract_squares(warped, padding=0.5)

        piece_map = pieces_by_image[img_id]

        for i, sq_img in enumerate(squares):
            sq_name = square_index_to_name(i)
            cat_id = piece_map.get(sq_name, 12)  # Default to empty
            folder = CATEGORY_TO_FOLDER[cat_id]

            class_dir = output_dir / folder
            class_dir.mkdir(parents=True, exist_ok=True)

            filename = f"cr_{img_id:05d}_{sq_name}.png"
            cv2.imwrite(str(class_dir / filename), sq_img)
            total_saved += 1

    print()
    print(f"Done! Saved {total_saved} squares to {output_dir}/")
    print(f"Skipped {skipped} images (not found or unreadable)")
    print()

    # Summary
    for class_dir in sorted(output_dir.iterdir()):
        if class_dir.is_dir():
            n = len(list(class_dir.glob("*.png")))
            print(f"  {class_dir.name}: {n}")


if __name__ == "__main__":
    main()

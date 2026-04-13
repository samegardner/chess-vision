"""Download and process training datasets into ImageFolder format.

Chesscog: Synthetic renders from OSF (project xf3ka).
    Each sample is {id}.png + {id}.json with corners and FEN.
    We warp each board and extract per-square crops.

ChessReD: Real smartphone images from 4TU.ResearchData.
    annotations.json + image archive with per-square piece labels.

Output format (ImageFolder):
    data/processed/occupancy/{split}/{empty,occupied}/*.png
    data/processed/pieces/{split}/{piece_class}/*.png
"""

import json
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import chess
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_vision.config import DATA_DIR

CHESSCOG_OSF_PROJECT = "xf3ka"
CHESSRED_ANNOTATIONS_URL = "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/3cae6364-daca-4967-b426-1e4b68cdb64c"
CHESSRED_IMAGES_URL = "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/6329e969-616e-48e3-b893-a0379d1c15ba"

SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE  # Extra margin for padding

# Piece class names matching our model's PIECE_CLASSES
FEN_TO_CLASS = {
    "P": "white_pawn", "N": "white_knight", "B": "white_bishop",
    "R": "white_rook", "Q": "white_queen", "K": "white_king",
    "p": "black_pawn", "n": "black_knight", "b": "black_bishop",
    "r": "black_rook", "q": "black_queen", "k": "black_king",
}


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or url}...")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 // total_size
            print(f"\r  {pct}%", end="", flush=True)

    urlretrieve(url, str(dest), reporthook=reporthook)
    print()


def _sort_corners(corners: np.ndarray) -> np.ndarray:
    """Order corners as: top-left, top-right, bottom-right, bottom-left."""
    corners = corners.reshape(4, 2).astype(np.float32)
    s = corners.sum(axis=1)
    d = np.diff(corners, axis=1).flatten()
    return np.array([
        corners[np.argmin(s)],
        corners[np.argmin(d)],
        corners[np.argmax(s)],
        corners[np.argmax(d)],
    ], dtype=np.float32)


def _warp_board(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp board image to a canonical square grid with margin."""
    src = _sort_corners(corners)
    dst = np.array([
        [SQUARE_SIZE, SQUARE_SIZE],
        [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],
        [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE],
        [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return cv2.warpPerspective(img, H, (IMG_SIZE, IMG_SIZE))


def _crop_square(warped: np.ndarray, square: int) -> np.ndarray:
    """Crop a square from the warped board image with context padding.

    Square indexing matches python-chess: a1=0, b1=1, ..., h8=63.
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    row = 7 - rank
    col = file

    # Crop with 1 square of padding on each side (matching chesscog)
    y1 = int(SQUARE_SIZE * (row + 0.5))
    y2 = int(SQUARE_SIZE * (row + 2.5))
    x1 = int(SQUARE_SIZE * (col + 0.5))
    x2 = int(SQUARE_SIZE * (col + 2.5))

    return warped[y1:y2, x1:x2]


def _fen_to_square_map(fen: str) -> dict[int, str | None]:
    """Parse FEN position string to {square_index: piece_char_or_None}."""
    board = chess.Board(fen)
    result = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        result[sq] = piece.symbol() if piece else None
    return result


def _save_square(img: np.ndarray, path: Path) -> None:
    """Save a square image as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def process_chesscog(raw_dir: Path, output_dir: Path) -> None:
    """Process chesscog raw renders into ImageFolder format.

    Raw format: {split}/{id}.png + {id}.json
    JSON has: {"fen": "...", "corners": [[x,y], ...], ...}
    """
    for split in ["train", "val", "test"]:
        split_dir = raw_dir / split
        if not split_dir.exists():
            print(f"  Skipping {split} (not found)")
            continue

        json_files = sorted(split_dir.glob("*.json"))
        print(f"  Processing {split}: {len(json_files)} samples")

        for json_path in tqdm(json_files, desc=f"  {split}", leave=False):
            img_path = json_path.with_suffix(".png")
            if not img_path.exists():
                continue

            with open(json_path) as f:
                label = json.load(f)

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            corners = np.array(label["corners"], dtype=np.float32)
            fen = label["fen"]

            warped = _warp_board(img, corners)
            square_map = _fen_to_square_map(fen)

            for sq in chess.SQUARES:
                crop = _crop_square(warped, sq)
                piece_char = square_map[sq]
                sample_id = f"{json_path.stem}_{chess.square_name(sq)}"

                # Save for occupancy dataset
                occ_class = "occupied" if piece_char else "empty"
                _save_square(
                    crop,
                    output_dir / "occupancy" / split / occ_class / f"{sample_id}.png",
                )

                # Save for piece dataset (only occupied squares)
                if piece_char:
                    piece_class = FEN_TO_CLASS[piece_char]
                    _save_square(
                        crop,
                        output_dir / "pieces" / split / piece_class / f"{sample_id}.png",
                    )


def process_chessred(raw_dir: Path, output_dir: Path) -> None:
    """Process ChessReD dataset into ImageFolder format.

    Raw format: annotations.json + images/ directory
    annotations.json has image paths, piece annotations per square, and splits.
    """
    ann_path = raw_dir / "annotations.json"
    if not ann_path.exists():
        print("  annotations.json not found, skipping ChessReD")
        return

    with open(ann_path) as f:
        data = json.load(f)

    images_df = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Build per-image annotation lookup
    annotations = {}
    for ann in data["annotations"]["pieces"]:
        img_id = ann["image_id"]
        if img_id not in annotations:
            annotations[img_id] = {}
        annotations[img_id][ann["square"]] = ann["category_id"]

    for split_name, split_info in data["splits"].items():
        img_ids = split_info["image_ids"]
        print(f"  Processing {split_name}: {len(img_ids)} images")

        for img_id in tqdm(img_ids, desc=f"  {split_name}", leave=False):
            img_info = images_df.get(img_id)
            if not img_info:
                continue

            img_path = raw_dir / img_info["path"]
            if not img_path.exists():
                continue

            # ChessReD images come with corner annotations
            if "corners" not in img_info:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            corners = np.array(img_info["corners"], dtype=np.float32)
            warped = _warp_board(img, corners)

            img_anns = annotations.get(img_id, {})
            for sq_name, cat_id in img_anns.items():
                sq = chess.parse_square(sq_name)
                crop = _crop_square(warped, sq)
                cat_name = categories[cat_id]
                sample_id = f"chessred_{img_id}_{sq_name}"

                # Determine occupancy
                is_empty = cat_name.lower() in ("empty", "none", "")
                occ_class = "empty" if is_empty else "occupied"
                _save_square(
                    crop,
                    output_dir / "occupancy" / split_name / occ_class / f"{sample_id}.png",
                )

                # Save piece crop if occupied
                if not is_empty:
                    # Map category name to our class names
                    piece_class = _normalize_category(cat_name)
                    if piece_class:
                        _save_square(
                            crop,
                            output_dir / "pieces" / split_name / piece_class / f"{sample_id}.png",
                        )


def _normalize_category(cat_name: str) -> str | None:
    """Normalize ChessReD category names to our piece class names."""
    # ChessReD uses names like "white pawn", "black knight", etc.
    normalized = cat_name.lower().replace(" ", "_").replace("-", "_")
    if normalized in FEN_TO_CLASS.values():
        return normalized
    # Try common variations
    mappings = {
        "white_rook": "white_rook", "white_knight": "white_knight",
        "white_bishop": "white_bishop", "white_queen": "white_queen",
        "white_king": "white_king", "white_pawn": "white_pawn",
        "black_rook": "black_rook", "black_knight": "black_knight",
        "black_bishop": "black_bishop", "black_queen": "black_queen",
        "black_king": "black_king", "black_pawn": "black_pawn",
    }
    return mappings.get(normalized)


def download_chesscog(output_dir: Path = DATA_DIR / "chesscog") -> None:
    """Download chesscog synthetic dataset from OSF."""
    print("Downloading chesscog dataset from OSF...")
    print("  Note: This requires the 'osfclient' package.")
    print("  Install with: pip install osfclient")
    print(f"  Manual download: osf -p {CHESSCOG_OSF_PROJECT} clone {output_dir}")
    print()
    print("  Alternative: download manually from https://osf.io/xf3ka/")
    print(f"  Extract train/val/test directories to: {output_dir}")

    try:
        import osfclient.cli
        from types import SimpleNamespace
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(project=CHESSCOG_OSF_PROJECT, output=tmp, username=None)
            osfclient.cli.clone(args)
            tmp_path = Path(tmp) / "osfstorage"
            if tmp_path.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                # Extract zip files
                for split in ["val", "test"]:
                    zf = tmp_path / f"{split}.zip"
                    if zf.exists():
                        print(f"  Extracting {split}.zip")
                        with zipfile.ZipFile(zf) as z:
                            z.extractall(output_dir)
                # Train is split across train.zip + train.z01
                train_zip = tmp_path / "train.zip"
                if train_zip.exists():
                    print("  Extracting train.zip (may need manual merge if split)")
                    try:
                        with zipfile.ZipFile(train_zip) as z:
                            z.extractall(output_dir)
                    except zipfile.BadZipFile:
                        print("  Train archive is a split zip. Merge manually:")
                        print(f"  zip -s 0 {train_zip} --out {output_dir}/train_full.zip")
                        shutil.copy2(train_zip, output_dir / "train.zip")
                        z01 = tmp_path / "train.z01"
                        if z01.exists():
                            shutil.copy2(z01, output_dir / "train.z01")
    except ImportError:
        print("  osfclient not installed. Skipping automatic download.")
        print("  Install with: pip install osfclient")


def download_chessred(output_dir: Path = DATA_DIR / "chessred") -> None:
    """Download ChessReD dataset from 4TU.ResearchData."""
    print("Downloading ChessReD dataset...")
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_dest = output_dir / "annotations.json"
    _download_file(CHESSRED_ANNOTATIONS_URL, ann_dest, "annotations.json")

    images_dest = output_dir / "images.zip"
    _download_file(CHESSRED_IMAGES_URL, images_dest, "images archive")

    if images_dest.exists() and not (output_dir / "images").exists():
        print("  Extracting images...")
        with zipfile.ZipFile(images_dest) as z:
            z.extractall(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and process chess datasets")
    parser.add_argument(
        "--dataset",
        choices=["chesscog", "chessred", "all"],
        default="all",
        help="Which dataset to download/process",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip download, only process existing raw data",
    )
    args = parser.parse_args()

    processed_dir = DATA_DIR / "processed"

    if args.dataset in ("chesscog", "all"):
        if not args.process_only:
            download_chesscog()
        chesscog_raw = DATA_DIR / "chesscog"
        if chesscog_raw.exists() and any(chesscog_raw.rglob("*.json")):
            print("\nProcessing chesscog into ImageFolder format...")
            process_chesscog(chesscog_raw, processed_dir)
        else:
            print("\nNo chesscog data found to process.")

    if args.dataset in ("chessred", "all"):
        if not args.process_only:
            download_chessred()
        chessred_raw = DATA_DIR / "chessred"
        if (chessred_raw / "annotations.json").exists():
            print("\nProcessing ChessReD into ImageFolder format...")
            process_chessred(chessred_raw, processed_dir)
        else:
            print("\nNo ChessReD data found to process.")

    print("\nDone. Processed datasets at:", processed_dir)

"""Download training datasets (chesscog synthetic + ChessReD real images)."""

from pathlib import Path

from chess_vision.config import DATA_DIR


def download_chesscog(output_dir: Path = DATA_DIR / "chesscog") -> None:
    """Download the chesscog synthetic dataset from OSF."""
    raise NotImplementedError


def download_chessred(output_dir: Path = DATA_DIR / "chessred") -> None:
    """Download the ChessReD real-world dataset."""
    raise NotImplementedError


if __name__ == "__main__":
    print("Downloading chesscog dataset...")
    download_chesscog()
    print("Downloading ChessReD dataset...")
    download_chessred()
    print("Done.")

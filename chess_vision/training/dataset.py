"""Dataset classes for training data.

All datasets output per-square images in ImageFolder format:
    data/processed/{occupancy,pieces}/{split}/{class_name}/*.png

Occupancy classes: "empty", "occupied"
Piece classes: "white_pawn", "white_knight", ..., "black_king" (12 total)
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


# Label mappings
OCCUPANCY_CLASSES = ["empty", "occupied"]

PIECE_CLASSES = [
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
]

# FEN char to piece class name
FEN_TO_PIECE_CLASS = {
    "P": "white_pawn",
    "N": "white_knight",
    "B": "white_bishop",
    "R": "white_rook",
    "Q": "white_queen",
    "K": "white_king",
    "p": "black_pawn",
    "n": "black_knight",
    "b": "black_bishop",
    "r": "black_rook",
    "q": "black_queen",
    "k": "black_king",
}


def create_occupancy_dataset(root: Path, split: str = "train", transform=None) -> ImageFolder:
    """Load the occupancy classification dataset (ImageFolder format).

    Expected directory structure:
        root/{split}/empty/*.png
        root/{split}/occupied/*.png
    """
    return ImageFolder(root=str(root / split), transform=transform)


def create_piece_dataset(root: Path, split: str = "train", transform=None) -> ImageFolder:
    """Load the piece classification dataset (ImageFolder format).

    Expected directory structure:
        root/{split}/white_pawn/*.png
        root/{split}/white_knight/*.png
        ...
        root/{split}/black_king/*.png
    """
    return ImageFolder(root=str(root / split), transform=transform)


class CalibrationDataset(Dataset):
    """Loads labeled squares from calibration photos.

    Expected directory structure:
        squares_dir/empty/*.png
        squares_dir/white_pawn/*.png
        ...
    """

    def __init__(self, squares_dir: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        # Build class list from subdirectories
        for class_dir in sorted(squares_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.classes.append(class_name)
                self.class_to_idx[class_name] = len(self.classes) - 1

                for img_path in sorted(class_dir.glob("*.png")):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

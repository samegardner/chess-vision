"""Dataset classes for training data."""

from pathlib import Path

from torch.utils.data import Dataset


class ChesscogDataset(Dataset):
    """Loads chesscog synthetic square images with piece labels."""

    def __init__(self, root_dir: Path, split: str = "train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class ChessRedDataset(Dataset):
    """Loads ChessReD real square images with piece labels."""

    def __init__(self, root_dir: Path, split: str = "train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class CalibrationDataset(Dataset):
    """Loads labeled squares from calibration photos with augmentation."""

    def __init__(self, squares_dir: Path, transform=None):
        self.squares_dir = squares_dir
        self.transform = transform
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

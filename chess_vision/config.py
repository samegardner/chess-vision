from pathlib import Path

import yaml

# Project root (directory containing chess_vision/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Default paths
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
PROFILES_DIR = ROOT_DIR / "profiles"
CONFIG_DIR = ROOT_DIR / "config"

# Board detection defaults
WARP_SIZE = 800
SQUARE_SIZE = WARP_SIZE // 8  # 100px per square
CONTEXT_PAD = 0.5  # 50% padding around each square crop

# CNN defaults
OCCUPANCY_THRESHOLD = 0.5
PIECE_CONFIDENCE_THRESHOLD = 0.6
INPUT_SIZE = 224  # ImageNet standard

# Move detection defaults
DIFF_THRESHOLD = 30
STABILITY_FRAMES = 5
FRAME_INTERVAL = 0.5

# Camera defaults
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080


def load_config(path: Path | None = None) -> dict:
    """Load configuration from YAML file, falling back to defaults."""
    config_path = path or CONFIG_DIR / "default.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

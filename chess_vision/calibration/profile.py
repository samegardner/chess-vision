"""Calibration profile management."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import yaml

from chess_vision.config import PROFILES_DIR


@dataclass
class CalibrationProfile:
    name: str
    created_at: str
    occupancy_model: str  # Relative path to fine-tuned ONNX model
    piece_model: str  # Relative path to fine-tuned ONNX model
    square_images_dir: str  # Relative path to saved calibration squares
    notes: str = ""


def save_profile(profile: CalibrationProfile, profiles_dir: Path = PROFILES_DIR) -> Path:
    """Save a calibration profile to YAML."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profiles_dir / f"{profile.name}.yaml"
    with open(path, "w") as f:
        yaml.dump(asdict(profile), f, default_flow_style=False)
    return path


def load_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> CalibrationProfile:
    """Load a calibration profile by name."""
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return CalibrationProfile(**data)


def list_profiles(profiles_dir: Path = PROFILES_DIR) -> list[CalibrationProfile]:
    """List all saved calibration profiles."""
    if not profiles_dir.exists():
        return []
    profiles = []
    for path in sorted(profiles_dir.glob("*.yaml")):
        if path.name == "example.yaml":
            continue
        with open(path) as f:
            data = yaml.safe_load(f)
        if data:
            profiles.append(CalibrationProfile(**data))
    return profiles

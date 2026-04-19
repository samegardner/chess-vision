"""Tests for helpers in scripts/record_game.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.record_game import align_to_existing  # noqa: E402


def test_align_recovers_identity():
    corners = np.array([[0, 100], [0, 0], [100, 0], [100, 100]], dtype=np.float32)
    aligned = align_to_existing(corners.copy(), corners)
    assert np.allclose(aligned, corners)


def test_align_undoes_180_flip():
    """If auto_detect_corners returns the board flipped 180 (white/black
    swapped), align_to_existing must rotate it back to the original
    orientation, not accept the flip."""
    corners = np.array([[0, 100], [0, 0], [100, 0], [100, 100]], dtype=np.float32)
    flipped = np.array([[100, 0], [100, 100], [0, 100], [0, 0]], dtype=np.float32)  # 2 cyclic
    aligned = align_to_existing(flipped, corners)
    assert np.allclose(aligned, corners)


def test_align_picks_nearest_rotation_with_jitter():
    """Tiny jitter on each corner should still be recognized as the same
    orientation, even if a different rotation has nominally similar distance.
    """
    corners = np.array([[10, 110], [12, 8], [108, 6], [112, 109]], dtype=np.float32)
    # Same orientation but shifted 3px
    new = corners + np.array([3, -2], dtype=np.float32)
    aligned = align_to_existing(new, corners)
    # Per-corner shift after alignment should match the input shift
    per_corner = np.linalg.norm(aligned - corners, axis=1)
    assert np.all(per_corner < 5)

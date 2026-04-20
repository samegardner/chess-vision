"""Tests for the starting-position-based orientation logic.

The bug we're guarding against: YOLO occasionally misclassifies the King
as a Queen (both are tall back-rank pieces). The old chirality heuristic
relied on Q/K alone, so a single bad classification flipped a-h files.
The new logic scores all 8 possible orientations against the standard
starting position, using all 32 detected pieces as voters.
"""

import chess
import numpy as np

from chess_vision.board.auto_corners import (
    _all_orientations, _score_orientation, _match_to_existing_labels,
)


def _starting_detections(corners: np.ndarray, image_shape: tuple) -> list[dict]:
    """Build a list of fake YOLO detections that exactly match the
    starting position relative to the given corners. Each detection is
    placed at its square's center."""
    from chess_vision.inference.yolo_detect import compute_square_centers
    centers = compute_square_centers(corners, image_shape)
    board = chess.Board()
    dets = []
    for sq, piece in board.piece_map().items():
        cx, cy = centers[sq]
        # Anchor formula: cy + h/2 - w/3. We want the anchor at (cx, cy).
        # If h=80, w=50: h/2 - w/3 = 40 - 17 = 23. So box cy = cy - 23.
        dets.append({
            "cx": float(cx),
            "cy": float(cy) - 23,
            "w": 50.0,
            "h": 80.0,
            "class_name": piece.symbol(),
            "class_id": 0,
            "confidence": 0.85,
            "scores": [0.0] * 12,
        })
    return dets


def test_score_correct_orientation_wins():
    """The orientation matching the actual starting position should
    score MUCH higher than any other."""
    corners = np.array([
        [0, 800],    # a1
        [0, 0],      # a8
        [800, 0],    # h8
        [800, 800],  # h1
    ], dtype=np.float32)
    shape = (1000, 1000, 3)
    dets = _starting_detections(corners, shape)

    # The correct orientation: should score = sum of all 32 confidences = 32 * 0.85 = 27.2
    correct_score = _score_orientation(corners, dets, shape)
    assert correct_score > 25, f"Correct orientation scored {correct_score}"


def test_mirror_chirality_loses_against_correct():
    """Mirroring a-h (the bug Sam hit) should score lower than correct,
    because Queen and King end up on each other's expected squares."""
    corners = np.array([
        [0, 800],    # a1
        [0, 0],      # a8
        [800, 0],    # h8
        [800, 800],  # h1
    ], dtype=np.float32)
    mirrored = corners[::-1]  # [h1, h8, a8, a1] - swaps a-h
    shape = (1000, 1000, 3)
    dets = _starting_detections(corners, shape)

    correct = _score_orientation(corners, dets, shape)
    wrong = _score_orientation(mirrored, dets, shape)
    assert correct > wrong, f"correct={correct} should beat mirror={wrong}"


def test_all_orientations_returns_eight():
    """Sanity: 4 cyclic positions * 2 chiralities = 8 total."""
    out = _all_orientations((0, 0), (1, 0), (1, 1), (0, 1))
    assert len(out) == 8
    # Each must be a permutation of the 4 input corners
    inputs = {(0, 0), (1, 0), (1, 1), (0, 1)}
    for orient in out:
        assert {tuple(c) for c in orient} == inputs


def test_match_to_existing_keeps_orientation():
    """Mid-game C-key path: hull corners are unlabeled; existing corners
    have labels. Each label should snap to the nearest unused hull corner."""
    existing = np.array([
        [10, 100], [10, 10], [100, 10], [100, 100],   # a1, a8, h8, h1
    ], dtype=np.float32)
    # New hull, slightly shifted
    new_hull = [
        [12, 102], [12, 12], [102, 12], [102, 102],
    ]
    matched = _match_to_existing_labels(new_hull, existing)
    # Each label position should match the nearest new hull corner (in order).
    expected = np.array([
        [12, 102], [12, 12], [102, 12], [102, 102],
    ], dtype=np.float32)
    assert np.allclose(matched, expected)


def test_match_to_existing_handles_shuffled_hull_order():
    """Hull corners may come in TL/TR/BR/BL order while existing is
    a1/a8/h8/h1; matching should still find the right pairing."""
    existing = np.array([
        [10, 100], [10, 10], [100, 10], [100, 100],
    ], dtype=np.float32)
    # Hull comes in a different order - matching should still pair correctly
    new_hull = [
        [102, 12], [12, 12], [102, 102], [12, 102],   # shuffled
    ]
    matched = _match_to_existing_labels(new_hull, existing)
    assert tuple(matched[0]) == (12, 102)   # a1 nearest
    assert tuple(matched[1]) == (12, 12)    # a8 nearest
    assert tuple(matched[2]) == (102, 12)   # h8 nearest
    assert tuple(matched[3]) == (102, 102)  # h1 nearest

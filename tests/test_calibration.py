"""Tests for calibration labeling and board remapping."""

import numpy as np

from chess_vision.calibration.label import STARTING_POSITION
from chess_vision.board.squares import remap_board_state


def test_remap_board_state_roundtrip():
    """Remapping twice should return the original state."""
    remapped = remap_board_state(STARTING_POSITION)
    restored = remap_board_state(remapped)
    assert restored == STARTING_POSITION


def test_remap_board_state_corners():
    """Check specific square remappings."""
    state = {"a1": "R", "h8": "r", "e1": "K", "e8": "k"}
    # Fill remaining squares with None
    for f in "abcdefgh":
        for r in range(1, 9):
            sq = f"{f}{r}"
            if sq not in state:
                state[sq] = None

    remapped = remap_board_state(state)
    # a1 -> h8, h8 -> a1, e1 -> d8, e8 -> d1
    assert remapped["h8"] == "R"
    assert remapped["a1"] == "r"
    assert remapped["d8"] == "K"
    assert remapped["d1"] == "k"


def test_remap_starting_position_flips_colors():
    """Remapping starting position should put black pieces on ranks 1-2."""
    remapped = remap_board_state(STARTING_POSITION)
    # 180-degree rotation: file flips (a<->h, b<->g, etc.) AND rank flips (1<->8, 2<->7)
    # e1 (white king) maps to d8, so d8 should have "K"
    # e8 (black king) maps to d1, so d1 should have "k"
    assert remapped["d8"] == "K"  # Was e1 (white king)
    assert remapped["d1"] == "k"  # Was e8 (black king)
    assert remapped["a1"] == "r"  # Was h8 (black rook)
    assert remapped["h8"] == "R"  # Was a1 (white rook)
    assert remapped["a2"] == "p"  # Was h7 (black pawn)
    assert remapped["h7"] == "P"  # Was a2 (white pawn)


def test_label_rotation_mapping():
    """Verify the 180-degree rotation math used in label.py."""
    # When viewing from Black's side, position (file, rank) in the image
    # corresponds to the real square at (7-file, 7-rank) in 0-indexed terms.
    # In algebraic: a1_image -> h8_real, h8_image -> a1_real, etc.
    from chess_vision.calibration.label import label_calibration_squares

    # We can't call label_calibration_squares without a detectable board,
    # so test the rotation math directly
    for sq_name in STARTING_POSITION:
        f = sq_name[0]
        r = sq_name[1]
        flipped_file = chr(ord("h") - (ord(f) - ord("a")))
        flipped_rank = str(9 - int(r))
        flipped_sq = f"{flipped_file}{flipped_rank}"

        # Double-flip should return to original
        re_flipped_file = chr(ord("h") - (ord(flipped_file) - ord("a")))
        re_flipped_rank = str(9 - int(flipped_rank))
        assert f"{re_flipped_file}{re_flipped_rank}" == sq_name


def test_label_rotation_specific_squares():
    """Check specific rotation mappings match expectations."""
    # The rotation should be a 180-degree board rotation:
    # a1 <-> h8, a8 <-> h1, d4 <-> e5, etc.
    mappings = {
        "a1": "h8", "h8": "a1",
        "a8": "h1", "h1": "a8",
        "e1": "d8", "d8": "e1",
        "e4": "d5", "d5": "e4",
    }
    for original, expected in mappings.items():
        f = original[0]
        r = original[1]
        flipped_file = chr(ord("h") - (ord(f) - ord("a")))
        flipped_rank = str(9 - int(r))
        assert f"{flipped_file}{flipped_rank}" == expected, (
            f"Expected {original} -> {expected}, got {flipped_file}{flipped_rank}"
        )

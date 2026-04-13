"""Tests for move detection."""

from chess_vision.game.moves import MoveDetector


def test_no_change_returns_none(starting_board_state):
    detector = MoveDetector(stability_frames=3)
    detector.set_initial_board(starting_board_state)
    assert detector.detect_change(starting_board_state) is None


def test_stable_change_detected(starting_board_state):
    detector = MoveDetector(stability_frames=3)
    detector.set_initial_board(starting_board_state)

    # Simulate e2-e4
    new_state = dict(starting_board_state)
    new_state["e2"] = None
    new_state["e4"] = "P"

    # Not stable yet (need 3 frames)
    assert detector.detect_change(new_state) is None
    assert detector.detect_change(new_state) is None

    # 3rd frame: stable
    changed = detector.detect_change(new_state)
    assert changed is not None
    assert set(changed) == {"e2", "e4"}

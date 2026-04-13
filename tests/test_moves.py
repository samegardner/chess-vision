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


def test_multi_move_sequence(starting_board_state):
    """After detecting a move, the detector should track from the new state."""
    detector = MoveDetector(stability_frames=2)
    detector.set_initial_board(starting_board_state)

    # Move 1: e2-e4
    state_after_e4 = dict(starting_board_state)
    state_after_e4["e2"] = None
    state_after_e4["e4"] = "P"

    assert detector.detect_change(state_after_e4) is None
    changed = detector.detect_change(state_after_e4)
    assert changed is not None
    assert set(changed) == {"e2", "e4"}

    # Move 2: e7-e5 (detected relative to new state, not original)
    state_after_e5 = dict(state_after_e4)
    state_after_e5["e7"] = None
    state_after_e5["e5"] = "p"

    assert detector.detect_change(state_after_e5) is None
    changed = detector.detect_change(state_after_e5)
    assert changed is not None
    assert set(changed) == {"e7", "e5"}


def test_transient_noise_resets(starting_board_state):
    """A briefly flickering square should not trigger a move detection."""
    detector = MoveDetector(stability_frames=3)
    detector.set_initial_board(starting_board_state)

    # Frame 1-2: noise on a1
    noisy = dict(starting_board_state)
    noisy["a1"] = None
    assert detector.detect_change(noisy) is None
    assert detector.detect_change(noisy) is None

    # Frame 3: noise clears, board is back to normal
    assert detector.detect_change(starting_board_state) is None

    # Now a real move happens: e2-e4
    real_move = dict(starting_board_state)
    real_move["e2"] = None
    real_move["e4"] = "P"

    assert detector.detect_change(real_move) is None
    assert detector.detect_change(real_move) is None
    changed = detector.detect_change(real_move)
    assert changed is not None
    assert set(changed) == {"e2", "e4"}

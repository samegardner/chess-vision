"""Tests for perspective warping."""

import numpy as np

from chess_vision.board.warp import order_corners, compute_homography, warp_board


def test_order_corners_already_ordered():
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    ordered = order_corners(corners)
    np.testing.assert_array_almost_equal(ordered[0], [0, 0])    # top-left
    np.testing.assert_array_almost_equal(ordered[1], [100, 0])  # top-right
    np.testing.assert_array_almost_equal(ordered[2], [100, 100])  # bottom-right
    np.testing.assert_array_almost_equal(ordered[3], [0, 100])  # bottom-left


def test_order_corners_shuffled():
    corners = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
    ordered = order_corners(corners)
    np.testing.assert_array_almost_equal(ordered[0], [0, 0])
    np.testing.assert_array_almost_equal(ordered[1], [100, 0])
    np.testing.assert_array_almost_equal(ordered[2], [100, 100])
    np.testing.assert_array_almost_equal(ordered[3], [0, 100])


def test_compute_homography_identity():
    """Square corners mapping to same-size target should be near-identity."""
    corners = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
    H = compute_homography(corners, target_size=800)
    assert H.shape == (3, 3)
    # Should be close to identity
    np.testing.assert_array_almost_equal(H, np.eye(3), decimal=3)


def test_warp_board_output_size():
    """Warped output should have the correct dimensions."""
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    corners = np.array([[50, 50], [750, 50], [750, 550], [50, 550]], dtype=np.float32)
    H = compute_homography(corners, target_size=400)
    warped = warp_board(image, H, target_size=400)
    assert warped.shape == (400, 400, 3)


def test_warp_preserves_content():
    """A white square region in the source should appear in the warped output."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    # Draw a white filled rectangle in the board area
    image[50:350, 50:350] = 255

    corners = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype=np.float32)
    H = compute_homography(corners, target_size=300)
    warped = warp_board(image, H, target_size=300)

    # Center of warped image should be white
    center_val = warped[150, 150]
    assert np.all(center_val > 200)

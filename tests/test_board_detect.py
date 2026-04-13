"""Tests for board detection."""

import cv2
import numpy as np
import pytest

from chess_vision.board.detect import (
    _preprocess,
    detect_lines,
    find_intersections,
    _line_intersection,
    _merge_similar_lines,
    detect_board,
)


def _make_checkerboard(size=800, squares=8):
    """Generate a synthetic checkerboard image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // squares
    for r in range(squares):
        for c in range(squares):
            if (r + c) % 2 == 0:
                img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = [200, 200, 200]
            else:
                img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = [50, 50, 50]
    return img


def _make_board_on_background(board_size=400, img_size=600):
    """Generate a checkerboard centered on a larger background."""
    bg = np.full((img_size, img_size, 3), 128, dtype=np.uint8)
    board = _make_checkerboard(board_size)
    offset = (img_size - board_size) // 2
    bg[offset : offset + board_size, offset : offset + board_size] = board
    return bg


class TestPreprocess:
    def test_returns_single_channel(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        edges = _preprocess(img)
        assert len(edges.shape) == 2

    def test_edge_detection_on_checkerboard(self):
        board = _make_checkerboard(400)
        edges = _preprocess(board)
        # Checkerboard should produce many edge pixels
        assert np.count_nonzero(edges) > 100


class TestLineIntersection:
    def test_perpendicular_lines(self):
        # Vertical line at x=50: rho=50, theta=0
        # Horizontal line at y=100: rho=100, theta=pi/2
        pt = _line_intersection(50, 0, 100, np.pi / 2)
        assert pt is not None
        np.testing.assert_array_almost_equal(pt, [50, 100], decimal=3)

    def test_parallel_lines_return_none(self):
        pt = _line_intersection(50, 0, 100, 0)
        assert pt is None


class TestMergeSimilarLines:
    def test_merges_close_lines(self):
        lines = np.array([
            [100, 1.0],
            [105, 1.02],
            [300, 1.0],
        ])
        merged = _merge_similar_lines(lines, rho_threshold=20, theta_threshold=0.1)
        assert len(merged) == 2  # First two should merge

    def test_keeps_distant_lines(self):
        lines = np.array([
            [100, 1.0],
            [300, 1.0],
            [500, 1.0],
        ])
        merged = _merge_similar_lines(lines, rho_threshold=20, theta_threshold=0.1)
        assert len(merged) == 3


class TestFindIntersections:
    def test_basic_grid(self):
        # 2 vertical lines and 2 horizontal lines = 4 intersections
        vertical = [[50, 0.0], [150, 0.0]]
        horizontal = [[50, np.pi / 2], [150, np.pi / 2]]
        pts = find_intersections(vertical, horizontal, (200, 200))
        assert len(pts) == 4

    def test_filters_out_of_bounds(self):
        # Line intersection that falls way outside the image
        far_line_a = [[5000, 0.0]]
        far_line_b = [[50, np.pi / 2]]
        pts = find_intersections(far_line_a, far_line_b, (200, 200))
        # Intersection at (5000, 50) should be filtered out
        assert len(pts) == 0


class TestDetectBoard:
    def test_on_clean_checkerboard(self):
        """Board detection on a clean synthetic checkerboard."""
        board = _make_board_on_background(board_size=500, img_size=700)
        corners = detect_board(board)
        assert corners is not None, "Board detection failed on clean synthetic checkerboard"
        assert corners.shape == (4, 2)
        # Corners should be roughly at the board boundaries (100-600 range)
        for corner in corners:
            assert 50 < corner[0] < 650
            assert 50 < corner[1] < 650

    def test_on_blank_image_returns_none(self):
        """Blank image should fail detection gracefully."""
        blank = np.full((600, 600, 3), 128, dtype=np.uint8)
        corners = detect_board(blank)
        assert corners is None

    def test_detect_lines_returns_two_groups(self):
        """Line detection on a checkerboard should find two line groups."""
        board = _make_checkerboard(600)
        group_a, group_b = detect_lines(board)
        assert len(group_a) > 0, "No lines detected in group A"
        assert len(group_b) > 0, "No lines detected in group B"
        # The two groups should have different dominant angles
        avg_theta_a = np.mean([l[1] for l in group_a])
        avg_theta_b = np.mean([l[1] for l in group_b])
        assert abs(avg_theta_a - avg_theta_b) > 0.3  # At least ~17 degrees apart

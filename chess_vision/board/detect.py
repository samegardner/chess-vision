"""Board detection using Hough lines + clustering + intersection finding."""

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale, blur, and run Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholds based on image intensity
    median = np.median(blurred)
    low = int(max(0, 0.5 * median))
    high = int(min(255, 1.5 * median))
    return cv2.Canny(blurred, low, high)


def detect_lines(image: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect and cluster lines into two dominant groups (horizontal/vertical).

    Uses the standard Hough transform, then agglomerative clustering on line
    angles to separate into two groups. Within each group, merges similar lines.

    Returns:
        (group_a, group_b) where each is a list of (rho, theta) arrays.
        group_a has the smaller average theta (more vertical).
    """
    edges = _preprocess(image)

    # Standard Hough transform
    raw_lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)
    if raw_lines is None or len(raw_lines) < 2:
        return [], []

    lines = raw_lines[:, 0, :]  # shape (N, 2), each row is (rho, theta)

    # Normalize: ensure rho >= 0 (flip rho sign and shift theta by pi if needed)
    for i in range(len(lines)):
        if lines[i, 0] < 0:
            lines[i, 0] = -lines[i, 0]
            lines[i, 1] -= np.pi

    # Cluster by angle into 2 groups
    thetas = lines[:, 1].reshape(-1, 1)
    # Use sin/cos features so angles near 0 and pi cluster together
    features = np.column_stack([np.sin(2 * thetas), np.cos(2 * thetas)])

    if len(lines) < 2:
        return [], []

    clustering = AgglomerativeClustering(n_clusters=2)
    labels = clustering.fit_predict(features)

    group_a = lines[labels == 0]
    group_b = lines[labels == 1]

    # Merge similar lines within each group
    group_a = _merge_similar_lines(group_a)
    group_b = _merge_similar_lines(group_b)

    # Sort so group_a is the one with smaller average theta (more vertical)
    avg_a = np.mean(group_a[:, 1]) if len(group_a) > 0 else 0
    avg_b = np.mean(group_b[:, 1]) if len(group_b) > 0 else 0
    if avg_a > avg_b:
        group_a, group_b = group_b, group_a

    return group_a.tolist(), group_b.tolist()


def _merge_similar_lines(
    lines: np.ndarray, rho_threshold: float = 20, theta_threshold: float = 0.1
) -> np.ndarray:
    """Merge lines that are very close in (rho, theta) space.

    Groups lines that are within rho_threshold and theta_threshold of each other,
    then averages each group into a single representative line.
    """
    if len(lines) == 0:
        return lines

    # Sort by rho for stable merging
    sorted_idx = np.argsort(lines[:, 0])
    lines = lines[sorted_idx]

    merged = []
    used = set()
    for i in range(len(lines)):
        if i in used:
            continue
        group = [lines[i]]
        used.add(i)
        for j in range(i + 1, len(lines)):
            if j in used:
                continue
            if (
                abs(lines[j, 0] - lines[i, 0]) < rho_threshold
                and abs(lines[j, 1] - lines[i, 1]) < theta_threshold
            ):
                group.append(lines[j])
                used.add(j)
        merged.append(np.mean(group, axis=0))

    return np.array(merged)


def _line_intersection(
    rho1: float, theta1: float, rho2: float, theta2: float
) -> np.ndarray | None:
    """Compute intersection point of two lines in (rho, theta) form.

    Returns (x, y) or None if lines are nearly parallel.
    """
    ct1, st1 = np.cos(theta1), np.sin(theta1)
    ct2, st2 = np.cos(theta2), np.sin(theta2)

    det = ct1 * st2 - ct2 * st1
    if abs(det) < 1e-6:
        return None

    x = (rho1 * st2 - rho2 * st1) / det
    y = (rho2 * ct1 - rho1 * ct2) / det
    return np.array([x, y])


def find_intersections(
    group_a: list, group_b: list, image_shape: tuple[int, ...]
) -> np.ndarray:
    """Find all intersection points between two line groups.

    Filters to points within image bounds.

    Returns:
        (N, 2) array of intersection points.
    """
    h, w = image_shape[:2]
    margin = 0.1  # Allow points slightly outside the image
    points = []

    for line_a in group_a:
        rho_a, theta_a = line_a[0], line_a[1]
        for line_b in group_b:
            rho_b, theta_b = line_b[0], line_b[1]
            pt = _line_intersection(rho_a, theta_a, rho_b, theta_b)
            if pt is not None:
                x, y = pt
                if -margin * w < x < (1 + margin) * w and -margin * h < y < (1 + margin) * h:
                    points.append(pt)

    if not points:
        return np.array([]).reshape(0, 2)
    return np.array(points)


def _find_grid_corners(
    intersections: np.ndarray, expected_grid: int = 9
) -> np.ndarray | None:
    """From intersection points, find the 4 outer corners of the board.

    Tries to identify a grid of ~9x9 intersection points (the internal
    corners of an 8x8 chessboard) and returns the 4 extreme corners.
    If a clean grid isn't found, falls back to the convex hull extremes.

    Returns:
        (4, 2) array of corner points, or None if insufficient points.
    """
    if len(intersections) < 4:
        return None

    # Use the convex hull to find the 4 extreme points
    hull = cv2.convexHull(intersections.astype(np.float32))
    hull = hull.reshape(-1, 2)

    if len(hull) < 4:
        return None

    # Find the 4 points that maximize the enclosed area
    # Approximate: use the points with extreme sum/diff (same logic as order_corners)
    s = hull.sum(axis=1)
    d = np.diff(hull, axis=1).flatten()

    tl = hull[np.argmin(s)]   # top-left: smallest x+y
    br = hull[np.argmax(s)]   # bottom-right: largest x+y
    tr = hull[np.argmin(d)]   # top-right: smallest y-x
    bl = hull[np.argmax(d)]   # bottom-left: largest y-x

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Expand corners outward by half a grid cell to get the board edges
    # (intersections are at internal grid lines, not the board border)
    if len(intersections) >= expected_grid * 2:
        corners = _expand_corners(corners, intersections, expected_grid)

    return corners


def _expand_corners(
    corners: np.ndarray,
    intersections: np.ndarray,
    expected_grid: int = 9,
) -> np.ndarray:
    """Expand corners from internal grid intersections to board edges.

    The detected intersections are at the grid lines between squares,
    so the actual board border is roughly half a cell width further out.
    """
    # Estimate cell size from the bounding rect of intersections
    x_min, y_min = intersections.min(axis=0)
    x_max, y_max = intersections.max(axis=0)
    cell_w = (x_max - x_min) / (expected_grid - 2)  # 7 gaps for 9 lines on 8 squares
    cell_h = (y_max - y_min) / (expected_grid - 2)

    # Expand each corner outward by half a cell
    center = corners.mean(axis=0)
    expanded = corners.copy()
    for i in range(4):
        direction = corners[i] - center
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        expanded[i] = corners[i] + direction * np.array([cell_w, cell_h]) * 0.5

    return expanded


def find_corners(group_a: list, group_b: list, image_shape: tuple[int, ...]) -> np.ndarray | None:
    """Find 4 board corner points from two groups of lines.

    Returns ndarray of shape (4, 2) or None if detection fails.
    """
    intersections = find_intersections(group_a, group_b, image_shape)
    return _find_grid_corners(intersections)


def detect_board(image: np.ndarray) -> np.ndarray | None:
    """Detect chessboard in image and return 4 corner points.

    Returns ndarray of shape (4, 2) or None if detection fails.
    """
    group_a, group_b = detect_lines(image)

    if len(group_a) < 2 or len(group_b) < 2:
        return None

    return find_corners(group_a, group_b, image.shape)

"""Auto-detect board corners using ChessCam's xcorners model + piece-based orientation.

Pipeline:
1. Run piece detector to find the board region
2. Crop to piece region, run xcorners model to find grid intersection points
3. Find the 4 outermost xcorner points (convex hull extremes)
4. Extrapolate outward by half a cell to get the actual board edges
5. Determine orientation (which corner is a1) from white/black piece positions
"""

import cv2
import numpy as np
import onnxruntime as ort

import chess

from chess_vision.inference.yolo_detect import (
    letterbox_resize, MODEL_WIDTH, MODEL_HEIGHT,
    compute_square_centers, compute_board_quad, point_in_quad,
)


class XCornerDetector:
    """Detects grid intersection points using ChessCam's xcorners model."""

    def __init__(self, model_path: str, conf_threshold: float = 0.2):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect xcorner points. Returns (N, 2) array in image pixel space."""
        h_orig, w_orig = image.shape[:2]
        padded, scale, pad_x, pad_y = letterbox_resize(image, MODEL_WIDTH, MODEL_HEIGHT)
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float16)

        outputs = self.session.run(None, {"images": blob})[0]
        preds = outputs[0].astype(np.float32)

        points = []
        for i in range(preds.shape[1]):
            cx, cy, w, h, conf = preds[:, i]
            if conf < self.conf_threshold:
                continue
            real_cx = (float(cx) - pad_x) / scale
            real_cy = (float(cy) - pad_y) / scale
            points.append([real_cx, real_cy])

        return np.array(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)


def _find_board_corners_from_xcorners(xcorners: np.ndarray) -> np.ndarray | None:
    """Find the 4 board corners from xcorner grid intersection points.

    The xcorners form a 7x7 grid of internal intersections. The board edges
    are half a cell beyond the outermost xcorners. We find the 4 extreme
    points, then estimate cell size and extrapolate outward.
    """
    if len(xcorners) < 20:
        return None

    # Find convex hull extremes (same sum/diff trick as corner ordering)
    hull = cv2.convexHull(xcorners.reshape(-1, 1, 2)).reshape(-1, 2)
    if len(hull) < 4:
        return None

    s = hull.sum(axis=1)
    d = np.diff(hull, axis=1).flatten()

    tl = hull[np.argmin(s)]   # Top-left (smallest x+y)
    br = hull[np.argmax(s)]   # Bottom-right (largest x+y)
    tr = hull[np.argmin(d)]   # Top-right (smallest y-x)
    bl = hull[np.argmax(d)]   # Bottom-left (largest y-x)

    # Estimate cell size from the span of xcorner points
    # Internal grid is 7x7 (6 gaps in each direction)
    # Use the distance between TL-TR and TL-BL divided by 6
    top_width = np.linalg.norm(tr - tl)
    left_height = np.linalg.norm(bl - tl)
    cell_w = top_width / 6
    cell_h = left_height / 6

    # Extrapolate each corner outward by 1 cell (half cell to edge + some margin)
    # Direction vectors along each edge
    top_dir = (tr - tl) / (np.linalg.norm(tr - tl) + 1e-8)
    left_dir = (bl - tl) / (np.linalg.norm(bl - tl) + 1e-8)
    right_dir = (br - tr) / (np.linalg.norm(br - tr) + 1e-8)
    bottom_dir = (br - bl) / (np.linalg.norm(br - bl) + 1e-8)

    # Use the perspective transform to extrapolate properly.
    # Fit a homography from the xcorner extremes to an ideal 6x6 grid,
    # then inverse-transform the board corners (at -1 and 7 in grid space).
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=np.float32)

    try:
        H = cv2.getPerspectiveTransform(dst, src)
        # Board corners in grid space are at (-1, -1), (7, -1), (7, 7), (-1, 7)
        grid_corners = np.array([[[-1, -1]], [[7, -1]], [[7, 7]], [[-1, 7]]], dtype=np.float32)
        board_pts = cv2.perspectiveTransform(grid_corners, H).reshape(4, 2)
        board_tl, board_tr, board_br, board_bl = board_pts
    except Exception:
        # Fallback to simple expansion
        expand = 1.2
        board_tl = tl - top_dir * cell_w * expand - left_dir * cell_h * expand
        board_tr = tr + top_dir * cell_w * expand - right_dir * cell_h * expand
        board_br = br + bottom_dir * cell_w * expand + right_dir * cell_h * expand
        board_bl = bl - bottom_dir * cell_w * expand + left_dir * cell_h * expand

    return np.array([board_tl, board_tr, board_br, board_bl], dtype=np.float32)


def _all_orientations(tl, tr, br, bl) -> list[list]:
    """Enumerate every valid (a1, a8, h8, h1) labeling of 4 board corners.

    For each of the 4 corners that could be a1, there are 2 possible
    chiralities depending on which adjacent corner is a8 (CW vs CCW
    around the board). 4 starting points x 2 directions = 8 orientations.
    """
    cw = [
        [tl, tr, br, bl],   # a1=TL, a8=TR (CW)
        [tr, br, bl, tl],   # a1=TR, a8=BR
        [br, bl, tl, tr],   # a1=BR, a8=BL
        [bl, tl, tr, br],   # a1=BL, a8=TL
    ]
    ccw = [
        [tl, bl, br, tr],   # a1=TL, a8=BL (CCW)
        [tr, tl, bl, br],   # a1=TR, a8=TL
        [br, tr, tl, bl],   # a1=BR, a8=TR
        [bl, br, tr, tl],   # a1=BL, a8=BR
    ]
    return cw + ccw


def _score_orientation(
    corners: np.ndarray,
    piece_detections: list[dict],
    image_shape: tuple,
    min_conf: float = 0.3,
) -> float:
    """Score an (a1, a8, h8, h1) orientation by how well YOLO detections
    match the standard starting position.

    For every detection, find the square it'd be assigned to under this
    orientation, look up what piece SHOULD be there in the starting
    position, and add the detection's confidence if the class matches.
    Pieces detected on empty squares (ranks 3-6) and class mismatches
    contribute zero. Robust against any single piece being misclassified
    by YOLO, because the scoring uses all 32 pieces as voters.
    """
    centers = compute_square_centers(corners, image_shape)
    quad = compute_board_quad(corners)
    starting = chess.Board()

    score = 0.0
    for det in piece_detections:
        if det["confidence"] < min_conf:
            continue
        ax = det["cx"]
        ay = det["cy"] + det["h"] / 2 - det["w"] / 3
        if not point_in_quad(np.array([ax, ay]), quad):
            continue
        dists = np.sqrt(
            (centers[:, 0] - ax) ** 2 + (centers[:, 1] - ay) ** 2
        )
        sq_idx = int(np.argmin(dists))
        expected = starting.piece_at(sq_idx)
        if expected is None:
            continue
        if det["class_name"] == expected.symbol():
            score += det["confidence"]
    return score


def _match_to_existing_labels(hull_corners: list, existing: np.ndarray) -> np.ndarray:
    """Assign [a1, a8, h8, h1] labels to hull_corners by matching each
    existing labeled corner to its nearest hull corner. Used by mid-game
    C-key re-detection to keep orientation stable across a small board
    bump without re-running starting-position heuristics (which don't
    work once pieces have moved)."""
    hull_arr = np.array(hull_corners, dtype=np.float32)
    result = np.zeros((4, 2), dtype=np.float32)
    used = set()
    for i, label_pos in enumerate(existing):
        best_idx = -1
        best_dist = float("inf")
        for j in range(len(hull_arr)):
            if j in used:
                continue
            d = float(np.linalg.norm(hull_arr[j] - label_pos))
            if d < best_dist:
                best_dist = d
                best_idx = j
        result[i] = hull_arr[best_idx]
        used.add(best_idx)
    return result


def auto_detect_corners(
    piece_detections: list[dict],
    xcorner_detector: XCornerDetector,
    image: np.ndarray,
    existing_corners: np.ndarray | None = None,
) -> np.ndarray | None:
    """Auto-detect board corners and orientation.

    Returns (4, 2) corners in [a1, a8, h8, h1] order, or None if failed.

    If existing_corners is given, each label is matched to the nearest
    newly-detected corner - stable for mid-game re-detection. Otherwise
    we score all 8 orientations against the standard starting position,
    which is the right choice at launch.
    """
    # Step 1: Crop to piece region
    good_pieces = [d for d in piece_detections if d["confidence"] > 0.2]
    if len(good_pieces) < 8:
        return None

    all_cx = [d["cx"] for d in good_pieces]
    all_cy = [d["cy"] for d in good_pieces]
    margin = 80
    x1 = max(0, int(min(all_cx)) - margin)
    y1 = max(0, int(min(all_cy)) - margin)
    x2 = min(image.shape[1], int(max(all_cx)) + margin)
    y2 = min(image.shape[0], int(max(all_cy)) + margin)
    cropped = image[y1:y2, x1:x2]

    # Step 2: Detect xcorners
    xcorners_local = xcorner_detector.detect(cropped)
    if len(xcorners_local) < 20:
        return None

    # Map to full image coordinates
    xcorners = xcorners_local + np.array([x1, y1], dtype=np.float32)

    # Step 3: Find board corners from xcorner extremes
    board_corners = _find_board_corners_from_xcorners(xcorners)
    if board_corners is None:
        return None

    tl, tr, br, bl = board_corners

    # Mid-game re-detection: match each existing labeled corner to the
    # nearest new hull corner. Preserves orientation, doesn't depend on
    # starting-position assumptions.
    if existing_corners is not None:
        return _match_to_existing_labels([tl, tr, br, bl], existing_corners)

    # Initial detection: pick the orientation that best matches the
    # standard starting position. Replaces the older centroid + Q/K
    # heuristic (which broke whenever YOLO confused King and Queen,
    # since both are tall pieces on the back rank). The new approach
    # uses all detected pieces as voters - the only way for it to
    # mis-orient is for several class predictions to coincidentally
    # match a wrong layout, which is overwhelmingly unlikely.
    if len(piece_detections) < 8:
        return None

    candidates = _all_orientations(tl, tr, br, bl)
    scored = []
    for orient in candidates:
        c = np.array(orient, dtype=np.float32)
        score = _score_orientation(c, piece_detections, image.shape)
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])

    # If the top two scores tie, we don't trust the result - return None
    # so the caller can fall back to a saved corners.json or manual click.
    if len(scored) >= 2 and scored[0][0] - scored[1][0] < 0.5:
        return None

    return scored[0][1]

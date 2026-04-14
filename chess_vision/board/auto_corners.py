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

from chess_vision.inference.yolo_detect import letterbox_resize, MODEL_WIDTH, MODEL_HEIGHT


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


def auto_detect_corners(
    piece_detections: list[dict],
    xcorner_detector: XCornerDetector,
    image: np.ndarray,
) -> np.ndarray | None:
    """Auto-detect board corners and orientation.

    Returns (4, 2) corners in [a1, a8, h8, h1] order, or None if failed.
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

    # Step 4: Determine orientation from piece colors
    white_pieces = [d for d in piece_detections if d["class_name"].isupper() and d["confidence"] > 0.3]
    black_pieces = [d for d in piece_detections if d["class_name"].islower() and d["confidence"] > 0.3]

    if len(white_pieces) < 4 or len(black_pieces) < 4:
        return None

    white_centroid = np.mean([[d["cx"], d["cy"]] for d in white_pieces], axis=0)
    black_centroid = np.mean([[d["cx"], d["cy"]] for d in black_pieces], axis=0)

    # Try all 4 rotations: [a1, a8, h8, h1]
    # a1/h1 edge should be near white, a8/h8 edge near black
    rotations = [
        [bl, tl, tr, br],
        [br, bl, tl, tr],
        [tr, br, bl, tl],
        [tl, tr, br, bl],
    ]

    best_rotation = None
    best_score = -float("inf")

    for rot in rotations:
        a1, a8, h8, h1 = rot
        white_edge = (np.array(a1) + np.array(h1)) / 2
        black_edge = (np.array(a8) + np.array(h8)) / 2
        score = -(np.linalg.norm(white_centroid - white_edge) + np.linalg.norm(black_centroid - black_edge))
        if score > best_score:
            best_score = score
            best_rotation = rot

    if best_rotation is None:
        return None

    # Step 5: Disambiguate a-file vs h-file using Queen position
    a1, a8, h8, h1 = best_rotation
    white_queens = [d for d in piece_detections if d["class_name"] == "Q" and d["confidence"] > 0.3]

    if white_queens:
        queen_pos = np.array([white_queens[0]["cx"], white_queens[0]["cy"]])
        a1_pos = np.array(a1)
        h1_pos = np.array(h1)
        if np.linalg.norm(queen_pos - h1_pos) < np.linalg.norm(queen_pos - a1_pos):
            best_rotation = [h1, h8, a8, a1]

    return np.array(best_rotation, dtype=np.float32)

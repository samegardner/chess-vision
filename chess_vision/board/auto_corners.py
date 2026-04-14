"""Auto-detect board corners using ChessCam's xcorners model + piece-based orientation.

Pipeline (matching ChessCam's findCorners.tsx):
1. Run piece detector to find the board region
2. Crop to piece region, run xcorners model to find grid intersection points
3. Delaunay triangulation on xcorner points to form quads
4. Score each quad by how well all xcorners align to a 7x7 grid
5. Best quad defines the board plane, extrapolate to board edges
6. Determine orientation (which corner is a1) from white/black piece positions
"""

import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial import Delaunay

from chess_vision.inference.yolo_detect import letterbox_resize, MODEL_WIDTH, MODEL_HEIGHT


class XCornerDetector:
    """Detects grid intersection points using ChessCam's xcorners model."""

    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect xcorner points in an image.

        Returns (N, 2) array of (x, y) coordinates in image pixel space.
        """
        h_orig, w_orig = image.shape[:2]
        padded, scale, pad_x, pad_y = letterbox_resize(image, MODEL_WIDTH, MODEL_HEIGHT)
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float16)

        outputs = self.session.run(None, {"images": blob})[0]
        preds = outputs[0].astype(np.float32)  # (5, N)

        points = []
        for i in range(preds.shape[1]):
            cx, cy, w, h, conf = preds[:, i]
            if conf < self.conf_threshold:
                continue
            real_cx = (float(cx) - pad_x) / scale
            real_cy = (float(cy) - pad_y) / scale
            points.append([real_cx, real_cy])

        return np.array(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)


def _find_best_quad(xcorners: np.ndarray) -> np.ndarray | None:
    """Find the 4 xcorner points that best define the board grid.

    Uses Delaunay triangulation to form quads, then scores each quad
    by how well all xcorner points align to a 7x7 grid when warped
    through the quad's perspective transform.

    Returns (4, 2) array of the best quad's corner points, or None.
    """
    if len(xcorners) < 10:
        return None

    try:
        tri = Delaunay(xcorners)
    except Exception:
        return None

    # Form quads from adjacent triangle pairs
    quads = []
    simplices = tri.simplices
    neighbors = tri.neighbors

    for i, simplex in enumerate(simplices):
        for j, neighbor_idx in enumerate(neighbors[i]):
            if neighbor_idx == -1 or neighbor_idx <= i:
                continue
            neighbor = simplices[neighbor_idx]
            # Merge the two triangles into a quad (4 unique points)
            pts = list(set(simplex) | set(neighbor))
            if len(pts) == 4:
                quad_pts = xcorners[pts]
                # Order the quad points (convex hull order)
                hull = cv2.convexHull(quad_pts.reshape(-1, 1, 2))
                if len(hull) == 4:
                    quads.append(hull.reshape(4, 2))

    if not quads:
        return None

    # Score each quad
    best_score = -1
    best_quad = None

    for quad in quads:
        score = _score_quad(quad, xcorners)
        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad


def _score_quad(quad: np.ndarray, xcorners: np.ndarray) -> float:
    """Score a quad by how well xcorners align to a 7x7 grid when warped.

    Higher score = more xcorners land close to ideal grid positions.
    """
    # Warp xcorners through the quad's perspective transform to a unit square
    dst = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float32)

    # Order quad points consistently (TL, TR, BR, BL by sum/diff)
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).flatten()
    ordered = np.array([
        quad[np.argmin(s)],
        quad[np.argmin(d)],
        quad[np.argmax(s)],
        quad[np.argmax(d)],
    ], dtype=np.float32)

    try:
        H = cv2.getPerspectiveTransform(ordered, dst)
    except Exception:
        return -1

    # Warp all xcorner points
    pts = xcorners.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    # Try different integer offsets to find the best alignment
    best_alignment = 0
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            shifted = warped - np.array([ox, oy])
            # Distance to nearest integer grid point
            rounded = np.round(shifted)
            dists = np.sqrt(np.sum((shifted - rounded) ** 2, axis=1))
            # Count points that are close to a grid position (within 0.3)
            close = np.sum(dists < 0.3)
            if close > best_alignment:
                best_alignment = close

    return best_alignment


def _extrapolate_to_board_edges(quad: np.ndarray, xcorners: np.ndarray) -> np.ndarray:
    """Extrapolate from internal grid points to the actual board edges.

    The quad and xcorners are at grid intersections (internal lines).
    The board edges are half a cell further out on each side.
    """
    # Order the quad
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).flatten()
    ordered = np.array([
        quad[np.argmin(s)],
        quad[np.argmin(d)],
        quad[np.argmax(s)],
        quad[np.argmax(d)],
    ], dtype=np.float32)

    dst = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(ordered, dst)

    # Find the best offset
    pts = xcorners.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    best_offset = np.array([0.0, 0.0])
    best_count = 0
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            shifted = warped - np.array([ox, oy])
            rounded = np.round(shifted)
            dists = np.sqrt(np.sum((shifted - rounded) ** 2, axis=1))
            count = np.sum(dists < 0.3)
            if count > best_count:
                best_count = count
                best_offset = np.array([float(ox), float(oy)])

    # Board corners in the warped grid space (with offset)
    # Grid goes 0-6 for internal points. Board edges are at -0.5 and 6.5
    # But we need to account for the offset
    board_corners_grid = np.array([
        best_offset + [-0.5, -0.5],  # TL
        best_offset + [6.5, -0.5],   # TR
        best_offset + [6.5, 6.5],    # BR
        best_offset + [-0.5, 6.5],   # BL
    ], dtype=np.float32)

    # Inverse transform back to image space
    H_inv = cv2.getPerspectiveTransform(dst, ordered)
    board_corners_img = cv2.perspectiveTransform(
        board_corners_grid.reshape(-1, 1, 2), H_inv
    ).reshape(4, 2)

    return board_corners_img


def auto_detect_corners(
    piece_detections: list[dict],
    xcorner_detector: XCornerDetector,
    image: np.ndarray,
) -> np.ndarray | None:
    """Auto-detect board corners and orientation.

    Args:
        piece_detections: Raw YOLO piece detections (for orientation + ROI).
        xcorner_detector: XCorner model instance.
        image: Full camera frame.

    Returns:
        (4, 2) corners in [a1, a8, h8, h1] order, or None if failed.
    """
    # Step 1: Determine crop region from piece detections
    if len(piece_detections) < 8:
        return None

    all_cx = [d["cx"] for d in piece_detections if d["confidence"] > 0.2]
    all_cy = [d["cy"] for d in piece_detections if d["confidence"] > 0.2]
    if len(all_cx) < 8:
        return None

    margin = 50
    x1 = max(0, int(min(all_cx)) - margin)
    y1 = max(0, int(min(all_cy)) - margin)
    x2 = min(image.shape[1], int(max(all_cx)) + margin)
    y2 = min(image.shape[0], int(max(all_cy)) + margin)

    cropped = image[y1:y2, x1:x2]

    # Step 2: Detect xcorners in the cropped region
    xcorners_local = xcorner_detector.detect(cropped)
    if len(xcorners_local) < 10:
        return None

    # Map back to full image coordinates
    xcorners = xcorners_local + np.array([x1, y1], dtype=np.float32)

    # Step 3: Find the best quad
    quad = _find_best_quad(xcorners)
    if quad is None:
        return None

    # Step 4: Extrapolate to board edges
    board_corners = _extrapolate_to_board_edges(quad, xcorners)
    # board_corners is in [TL, TR, BR, BL] order (by sum/diff)

    # Step 5: Determine orientation from piece colors
    # White pieces should be near one edge, black near the opposite
    white_pieces = [d for d in piece_detections if d["class_name"].isupper() and d["confidence"] > 0.3]
    black_pieces = [d for d in piece_detections if d["class_name"].islower() and d["confidence"] > 0.3]

    if len(white_pieces) < 4 or len(black_pieces) < 4:
        return None

    white_centroid = np.mean([[d["cx"], d["cy"]] for d in white_pieces], axis=0)
    black_centroid = np.mean([[d["cx"], d["cy"]] for d in black_pieces], axis=0)

    tl, tr, br, bl = board_corners

    # Try all 4 rotations of corners as [a1, a8, h8, h1]
    # a1/h1 should be near white, a8/h8 near black
    rotations = [
        [bl, tl, tr, br],  # BL=a1, TL=a8, TR=h8, BR=h1
        [br, bl, tl, tr],  # BR=a1, BL=a8, TL=h8, TR=h1
        [tr, br, bl, tl],  # TR=a1, BR=a8, BL=h8, TL=h1
        [tl, tr, br, bl],  # TL=a1, TR=a8, BR=h8, BL=h1
    ]

    best_rotation = None
    best_score = -float("inf")

    for rot in rotations:
        a1, a8, h8, h1 = rot
        # White edge midpoint (a1-h1)
        white_edge = (np.array(a1) + np.array(h1)) / 2
        # Black edge midpoint (a8-h8)
        black_edge = (np.array(a8) + np.array(h8)) / 2

        # Score: white centroid close to white edge, black to black edge
        w_dist = np.linalg.norm(white_centroid - white_edge)
        b_dist = np.linalg.norm(black_centroid - black_edge)
        score = -(w_dist + b_dist)

        if score > best_score:
            best_score = score
            best_rotation = rot

    if best_rotation is None:
        return None

    # Now determine a-file vs h-file using King/Queen positions
    a1, a8, h8, h1 = best_rotation
    white_queens = [d for d in piece_detections if d["class_name"] == "Q" and d["confidence"] > 0.3]
    white_kings = [d for d in piece_detections if d["class_name"] == "K" and d["confidence"] > 0.3]

    if white_queens and white_kings:
        queen_pos = np.array([white_queens[0]["cx"], white_queens[0]["cy"]])
        a1_pos = np.array(a1)
        h1_pos = np.array(h1)

        # Queen should be closer to a1 side (d1) than h1 side
        q_to_a1 = np.linalg.norm(queen_pos - a1_pos)
        q_to_h1 = np.linalg.norm(queen_pos - h1_pos)

        if q_to_h1 < q_to_a1:
            # Queen is closer to what we called h1, so swap a-file and h-file
            best_rotation = [h1, h8, a8, a1]

    return np.array(best_rotation, dtype=np.float32)

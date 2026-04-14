"""YOLO-based piece detection using ChessCam's pretrained model.

Matches ChessCam's preprocessing exactly: letterbox resize with gray
padding (114), no NMS (let square-level max aggregate), bottom-width/3
anchor point for piece-to-square mapping, out-of-board filtering.
"""

import cv2
import numpy as np
import onnxruntime as ort

# ChessCam class ordering (lowercase = black, uppercase = white)
YOLO_CLASSES = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]

MODEL_WIDTH = 480
MODEL_HEIGHT = 288


def letterbox_resize(image: np.ndarray, target_w: int, target_h: int, pad_value: int = 114):
    """Resize preserving aspect ratio, pad with gray (matching ChessCam's detect.tsx).

    Returns (resized_image, scale, pad_x, pad_y) where scale/pad are needed
    to map coordinates back to original space.
    """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Pad to target size
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return padded, scale, pad_x, pad_y


def point_in_quad(point: np.ndarray, quad: np.ndarray) -> bool:
    """Check if a point is inside a quadrilateral using cross-product test."""
    px, py = point
    n = len(quad)
    for i in range(n):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < 0:
            return False
    return True


class YoloPieceDetector:
    """Detects pieces using ChessCam's LeYOLO model + EMA smoothing."""

    def __init__(self, model_path: str, conf_threshold: float = 0.25, ema_decay: float = 0.5):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.conf_threshold = conf_threshold
        self.ema_decay = ema_decay

        # State matrix: 64 squares x 12 classes, smoothed over time
        self.state = np.zeros((64, 12), dtype=np.float32)
        self.initialized = False

    def detect_raw(self, image: np.ndarray, crop_region: tuple | None = None) -> list[dict]:
        """Run YOLO on an image, return raw detections.

        All coordinates returned in original image pixel space.
        No NMS applied (matching ChessCam's approach).
        """
        offset_x, offset_y = 0, 0
        if crop_region is not None:
            x1, y1, x2, y2 = crop_region
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
            image = image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        h_orig, w_orig = image.shape[:2]

        # Letterbox resize (matching ChessCam's preprocessing)
        padded, scale, pad_x, pad_y = letterbox_resize(image, MODEL_WIDTH, MODEL_HEIGHT)
        blob = padded.astype(np.float32) / 255.0  # float32, not float16
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 288, 480)
        # Model expects float16 input
        blob = blob.astype(np.float16)

        # Run inference
        outputs = self.session.run(None, {"images": blob})[0]  # (1, 16, N)
        preds = outputs[0].astype(np.float32)  # (16, N)

        detections = []
        for i in range(preds.shape[1]):
            cx, cy, w, bh = preds[0, i], preds[1, i], preds[2, i], preds[3, i]
            all_scores = preds[4:, i]  # 12 class scores
            max_score = float(np.max(all_scores))

            if max_score < self.conf_threshold:
                continue

            # Undo letterbox: remove padding, then unscale
            real_cx = (float(cx) - pad_x) / scale
            real_cy = (float(cy) - pad_y) / scale
            real_w = float(w) / scale
            real_h = float(bh) / scale

            # Add crop offset
            real_cx += offset_x
            real_cy += offset_y

            detections.append({
                "cx": real_cx,
                "cy": real_cy,
                "w": real_w,
                "h": real_h,
                "scores": all_scores.tolist(),  # All 12 class scores (not just argmax)
                "class_id": int(np.argmax(all_scores)),
                "class_name": YOLO_CLASSES[int(np.argmax(all_scores))],
                "confidence": max_score,
            })

        return detections

    def detections_to_board(
        self,
        detections: list[dict],
        square_centers: np.ndarray,
        board_quad: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map detections to a 64x12 update matrix.

        Uses ALL class scores per detection (not just argmax).
        Filters out detections outside the board boundary.
        Uses ChessCam's anchor: bottom of box minus width/3.
        """
        update = np.zeros((64, 12), dtype=np.float32)

        for det in detections:
            # ChessCam anchor: (center_x, bottom - width/3)
            piece_x = det["cx"]
            piece_y = det["cy"] + det["h"] / 2 - det["w"] / 3

            # Filter out-of-board detections
            if board_quad is not None:
                if not point_in_quad(np.array([piece_x, piece_y]), board_quad):
                    continue

            # Find nearest square
            dists = np.sqrt(
                (square_centers[:, 0] - piece_x) ** 2
                + (square_centers[:, 1] - piece_y) ** 2
            )
            nearest_sq = int(np.argmin(dists))

            # Update with ALL class scores (max per class per square)
            scores = det["scores"]
            for cls_idx in range(12):
                update[nearest_sq, cls_idx] = max(
                    update[nearest_sq, cls_idx], scores[cls_idx]
                )

        return update

    def update_state(self, update: np.ndarray) -> None:
        """Apply EMA smoothing: state = decay * state + (1-decay) * update."""
        if not self.initialized:
            self.state = update.copy()
            self.initialized = True
        else:
            self.state = self.ema_decay * self.state + (1 - self.ema_decay) * update


def compute_crop_region(corners: np.ndarray, padding: float = 0.15) -> tuple[int, int, int, int]:
    """Compute a bounding box around the board corners with padding."""
    corners = corners.reshape(4, 2)
    xs = corners[:, 0]
    ys = corners[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    w = x2 - x1
    h = y2 - y1
    x1 -= w * padding
    y1 -= h * padding * 2  # More padding on top for tall pieces
    x2 += w * padding
    y2 += h * padding * 1.5  # More padding on bottom for close-to-camera pieces
    return (int(x1), int(y1), int(x2), int(y2))


def compute_board_quad(corners: np.ndarray) -> np.ndarray:
    """Get the board quadrilateral for point-in-quad filtering.

    Corners are in [a1, a8, h8, h1] order from select_corners.
    """
    return corners.reshape(4, 2).astype(np.float32)


def compute_square_centers(corners: np.ndarray, image_shape: tuple) -> np.ndarray:
    """Compute the 64 square center positions in image space.

    Corners must be in [a1, a8, h8, h1] order (from select_corners).
    Uses homography to correctly handle perspective distortion.

    Indexed 0=a1, 1=b1, ..., 63=h8.
    """
    corners = corners.reshape(4, 2).astype(np.float32)
    a1, a8, h8, h1 = corners

    # Ideal board: 800x800, a1 at bottom-left, h8 at top-right
    # a1=(0,800), a8=(0,0), h8=(800,0), h1=(800,800)
    ideal_corners = np.array([
        [0, 800],    # a1
        [0, 0],      # a8
        [800, 0],    # h8
        [800, 800],  # h1
    ], dtype=np.float32)

    H, _ = cv2.findHomography(ideal_corners, corners)

    centers = np.zeros((64, 2), dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            # a-file = x=50, h-file = x=750
            # rank 1 = y=750 (bottom), rank 8 = y=50 (top)
            ideal_x = (file + 0.5) * 100
            ideal_y = (7 - rank + 0.5) * 100

            pt = np.array([[[ideal_x, ideal_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, H)
            centers[rank * 8 + file] = transformed[0, 0]

    return centers

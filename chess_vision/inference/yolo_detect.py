"""YOLO-based piece detection using ChessCam's pretrained model.

Detects chess pieces with bounding boxes, maps them to board squares
using the perspective transform, and maintains a smoothed state matrix
via exponential moving average.
"""

import cv2
import numpy as np
import onnxruntime as ort

# ChessCam class ordering (lowercase = black, uppercase = white)
YOLO_CLASSES = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]

# Map to standard FEN chars
YOLO_TO_FEN = {
    "b": "b", "k": "k", "n": "n", "p": "p", "q": "q", "r": "r",
    "B": "B", "K": "K", "N": "N", "P": "P", "Q": "Q", "R": "R",
}

MODEL_WIDTH = 480
MODEL_HEIGHT = 288


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

        Args:
            image: Full camera frame (BGR).
            crop_region: Optional (x1, y1, x2, y2) to crop before detection.
                         Coordinates returned are mapped back to full image space.

        Returns list of dicts with keys: cx, cy, w, h, class_id, class_name, confidence
        All coordinates are in original image pixel space.
        """
        # Crop to board region if provided (makes pieces larger in the input)
        offset_x, offset_y = 0, 0
        if crop_region is not None:
            x1, y1, x2, y2 = crop_region
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
            image = image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        h_orig, w_orig = image.shape[:2]

        # Preprocess: resize to model input, normalize to 0-1, CHW format
        resized = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))
        blob = resized.astype(np.float16) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 288, 480)

        # Run inference
        outputs = self.session.run(None, {"images": blob})[0]  # (1, 16, N)
        preds = outputs[0].astype(np.float32)  # (16, N)

        boxes = []
        scores = []
        class_ids = []
        scale_x = w_orig / MODEL_WIDTH
        scale_y = h_orig / MODEL_HEIGHT

        for i in range(preds.shape[1]):
            cx, cy, w, bh = preds[0, i], preds[1, i], preds[2, i], preds[3, i]
            class_scores = preds[4:, i]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            boxes.append([
                float(cx * scale_x), float(cy * scale_y),
                float(w * scale_x), float(bh * scale_y),
            ])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        # Non-Maximum Suppression
        boxes_arr = np.array(boxes)
        # Convert cx,cy,w,h to x1,y1,x2,y2 for NMS
        x1 = boxes_arr[:, 0] - boxes_arr[:, 2] / 2
        y1 = boxes_arr[:, 1] - boxes_arr[:, 3] / 2
        x2 = boxes_arr[:, 0] + boxes_arr[:, 2] / 2
        y2 = boxes_arr[:, 1] + boxes_arr[:, 3] / 2
        nms_boxes = np.stack([x1, y1, x2, y2], axis=1).tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, self.conf_threshold, 0.5)
        if len(indices) == 0:
            return []

        detections = []
        for idx in indices.flatten():
            detections.append({
                "cx": boxes[idx][0] + offset_x,
                "cy": boxes[idx][1] + offset_y,
                "w": boxes[idx][2],
                "h": boxes[idx][3],
                "class_id": class_ids[idx],
                "class_name": YOLO_CLASSES[class_ids[idx]],
                "confidence": scores[idx],
            })

        return detections

    def detections_to_board(
        self,
        detections: list[dict],
        square_centers: np.ndarray,
    ) -> np.ndarray:
        """Map detections to a 64x12 update matrix.

        Args:
            detections: Raw YOLO detections with cx, cy coordinates.
            square_centers: (64, 2) array of square center coordinates in image space.

        Returns:
            (64, 12) matrix where each cell is the max confidence of that
            class being detected on that square.
        """
        update = np.zeros((64, 12), dtype=np.float32)

        for det in detections:
            # Use bottom-center of box as the piece position
            # (pieces are taller than their base, so center is above the square)
            piece_x = det["cx"]
            piece_y = det["cy"] + det["h"] * 0.15  # Shift down slightly

            # Find nearest square
            dists = np.sqrt(
                (square_centers[:, 0] - piece_x) ** 2
                + (square_centers[:, 1] - piece_y) ** 2
            )
            nearest_sq = int(np.argmin(dists))

            # Update with max confidence for this class on this square
            update[nearest_sq, det["class_id"]] = max(
                update[nearest_sq, det["class_id"]], det["confidence"]
            )

        return update

    def update_state(self, update: np.ndarray) -> None:
        """Apply EMA smoothing: state = decay * state + (1-decay) * update."""
        if not self.initialized:
            self.state = update.copy()
            self.initialized = True
        else:
            self.state = self.ema_decay * self.state + (1 - self.ema_decay) * update

    def get_board_state(self) -> dict[str, str | None]:
        """Read the current smoothed state as a board state dict."""
        board_state = {}
        for i in range(64):
            file = chr(ord("a") + i % 8)
            rank = str(i // 8 + 1)
            sq_name = file + rank

            max_score = float(np.max(self.state[i]))
            if max_score > 0.3:  # Occupied threshold
                class_id = int(np.argmax(self.state[i]))
                board_state[sq_name] = YOLO_TO_FEN[YOLO_CLASSES[class_id]]
            else:
                board_state[sq_name] = None

        return board_state


def compute_crop_region(corners: np.ndarray, padding: float = 0.15) -> tuple[int, int, int, int]:
    """Compute a bounding box around the board corners with padding.

    Returns (x1, y1, x2, y2) crop region.
    """
    from chess_vision.board.warp import order_corners
    ordered = order_corners(corners)
    xs = ordered[:, 0]
    ys = ordered[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    w = x2 - x1
    h = y2 - y1
    # Add padding (pieces extend above the board edge)
    x1 -= w * padding
    y1 -= h * padding * 2  # More padding on top (pieces are tall)
    x2 += w * padding
    y2 += h * padding
    return (int(x1), int(y1), int(x2), int(y2))


def compute_square_centers(corners: np.ndarray, image_shape: tuple) -> np.ndarray:
    """Compute the 64 square center positions in image space.

    Args:
        corners: (4, 2) board corners (TL, TR, BR, BL after ordering).
        image_shape: (height, width, ...) of the image.

    Returns:
        (64, 2) array of (x, y) center coordinates for each square.
        Indexed 0=a1, 1=b1, ..., 63=h8.
    """
    from chess_vision.board.warp import order_corners

    ordered = order_corners(corners)  # TL, TR, BR, BL
    tl, tr, br, bl = ordered

    centers = np.zeros((64, 2), dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            # Bilinear interpolation across the quadrilateral
            # rank 0 = rank 1 (bottom), rank 7 = rank 8 (top)
            # In the image: rank 1 is at the bottom (BL-BR edge)
            u = (file + 0.5) / 8  # Horizontal fraction
            v = (rank + 0.5) / 8  # Vertical fraction (0=bottom, 1=top)

            # Interpolate: bottom edge = BL to BR, top edge = TL to TR
            bottom = bl + u * (br - bl)
            top = tl + u * (tr - tl)
            center = bottom + v * (top - bottom)

            idx = rank * 8 + file
            centers[idx] = center

    return centers

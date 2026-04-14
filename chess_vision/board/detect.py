"""Manual board corner selection via OpenCV window."""

import cv2
import numpy as np


def select_corners(image: np.ndarray) -> np.ndarray:
    """Select 4 board corners in a specific order: a1, a8, h8, h1.

    The user clicks the corners in this order:
    1. a1 (white's queenside rook, closest to the 'A' and '1' labels)
    2. a8 (black's queenside rook)
    3. h8 (black's kingside rook)
    4. h1 (white's kingside rook)

    Returns ndarray of shape (4, 2) with corners in [a1, a8, h8, h1] order.
    """
    corners: list[tuple[int, int]] = []
    display = image.copy()
    window_name = "Click corners: a1, a8, h8, h1 (R=reset, Q=quit)"

    labels = ["a1 (White QR)", "a8 (Black QR)", "h8 (Black KR)", "h1 (White KR)"]

    h, w = image.shape[:2]
    scale = min(1.0, 1200 / w, 800 / h)
    if scale < 1.0:
        display_size = (int(w * scale), int(h * scale))
    else:
        display_size = None
        scale = 1.0

    def _draw():
        nonlocal display
        display = image.copy()
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(display, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(
                display, labels[i], (cx + 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
        if len(corners) >= 2:
            for i in range(len(corners) - 1):
                cv2.line(display, corners[i], corners[i + 1], (0, 255, 0), 2)
            if len(corners) == 4:
                cv2.line(display, corners[3], corners[0], (0, 255, 0), 2)

        if len(corners) < 4:
            msg = f"Click {labels[len(corners)]}"
        else:
            msg = "Done!"
        cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def _on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            real_x = int(x / scale)
            real_y = int(y / scale)
            corners.append((real_x, real_y))
            _draw()

    _draw()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_click)

    while True:
        show = display
        if display_size:
            show = cv2.resize(display, display_size)
        cv2.imshow(window_name, show)

        if len(corners) == 4:
            break

        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            corners.clear()
            _draw()
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Corner selection cancelled")

    cv2.destroyAllWindows()
    return np.array(corners, dtype=np.float32)

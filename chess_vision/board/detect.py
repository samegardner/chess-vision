"""Manual board corner selection via OpenCV window."""

import cv2
import numpy as np


def select_corners(image: np.ndarray) -> np.ndarray:
    """Manually select 4 board corners by clicking on the image.

    Opens an OpenCV window. Click the 4 corners of the chessboard
    (any order, they get reordered automatically).

    Press 'r' to reset and start over. Window closes after 4 clicks.

    Returns ndarray of shape (4, 2) with corner coordinates.
    """
    corners: list[tuple[int, int]] = []
    display = image.copy()
    window_name = "Click 4 board corners (R to reset, Q to quit)"

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
                display, str(i + 1), (cx + 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
        if len(corners) >= 2:
            for i in range(len(corners) - 1):
                cv2.line(display, corners[i], corners[i + 1], (0, 255, 0), 2)
            if len(corners) == 4:
                cv2.line(display, corners[3], corners[0], (0, 255, 0), 2)

        remaining = 4 - len(corners)
        msg = f"Click {remaining} more corner(s)" if remaining > 0 else "Done!"
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

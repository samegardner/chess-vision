"""Guided calibration photo capture."""

import cv2
import numpy as np

from chess_vision.inference.camera import Camera


def capture_calibration_photos(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Interactive capture of 2 calibration photos.

    Displays camera preview. User presses SPACE to capture each photo.
    First photo: white pieces facing camera.
    Second photo: board rotated 180 degrees (black pieces facing camera).

    Returns:
        (white_side_photo, black_side_photo)
    """
    photos = []
    prompts = [
        "Set up starting position with WHITE pieces facing the camera. Press SPACE to capture.",
        "Rotate board 180 degrees (BLACK pieces facing camera). Press SPACE to capture.",
    ]

    for prompt in prompts:
        print(prompt)
        while True:
            frame = camera.capture()
            display = frame.copy()
            cv2.putText(
                display, prompt[:80], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                photos.append(frame)
                print("  Captured!")
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("Calibration cancelled")

    cv2.destroyAllWindows()
    return photos[0], photos[1]

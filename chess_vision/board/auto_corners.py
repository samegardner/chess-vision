"""Auto-detect board corners and orientation from piece positions.

Uses the YOLO piece detector on the starting position to find where
white and black pieces are, then computes the 4 board corners in
[a1, a8, h8, h1] order from the rook positions.
"""

import numpy as np


def auto_detect_corners(detections: list[dict], conf_threshold: float = 0.3) -> np.ndarray | None:
    """Detect board corners from YOLO detections of the starting position.

    Finds the 4 rooks (2 white "R", 2 black "r") and uses King/Queen
    positions to determine which rook is on a1 vs h1.

    Args:
        detections: Raw YOLO detections with cx, cy, w, h, class_name, confidence.
        conf_threshold: Minimum confidence to consider a detection.

    Returns:
        (4, 2) array of corners in [a1, a8, h8, h1] order, or None if detection fails.
    """
    # Filter by confidence
    dets = [d for d in detections if d["confidence"] >= conf_threshold]

    # Group by class
    white_rooks = [d for d in dets if d["class_name"] == "R"]
    black_rooks = [d for d in dets if d["class_name"] == "r"]
    white_kings = [d for d in dets if d["class_name"] == "K"]
    black_kings = [d for d in dets if d["class_name"] == "k"]
    white_queens = [d for d in dets if d["class_name"] == "Q"]

    # Need at least 2 white rooks and 2 black rooks
    if len(white_rooks) < 2 or len(black_rooks) < 2:
        return None

    # Sort rooks by confidence, take top 2 of each
    white_rooks = sorted(white_rooks, key=lambda d: -d["confidence"])[:2]
    black_rooks = sorted(black_rooks, key=lambda d: -d["confidence"])[:2]

    # Compute centroids of white and black pieces to determine which side is which
    white_pieces = [d for d in dets if d["class_name"].isupper()]
    black_pieces = [d for d in dets if d["class_name"].islower()]

    if len(white_pieces) < 4 or len(black_pieces) < 4:
        return None

    white_centroid = np.mean([[d["cx"], d["cy"]] for d in white_pieces], axis=0)
    black_centroid = np.mean([[d["cx"], d["cy"]] for d in black_pieces], axis=0)

    # The two white rooks define the a1-h1 edge (white's back rank)
    # The two black rooks define the a8-h8 edge (black's back rank)
    wr1, wr2 = white_rooks[0], white_rooks[1]
    br1, br2 = black_rooks[0], black_rooks[1]

    # Now determine which white rook is a1 and which is h1
    # The Queen ("Q") is on d1, closer to a1 than to h1
    # The King ("K") is on e1, closer to h1 than to a1
    if white_kings and white_queens:
        king = white_kings[0]
        queen = white_queens[0]
        # Queen is on the a-file side, King on the h-file side
        # Find which rook is closer to the queen (that's a1) and which to the king (that's h1)
        wr1_to_queen = _dist(wr1, queen)
        wr2_to_queen = _dist(wr2, queen)
        if wr1_to_queen < wr2_to_queen:
            a1_rook, h1_rook = wr1, wr2
        else:
            a1_rook, h1_rook = wr2, wr1
    else:
        # Fallback: use left-right ordering relative to white centroid direction
        # This is less reliable but works if K/Q aren't detected
        if wr1["cx"] < wr2["cx"]:
            a1_rook, h1_rook = wr1, wr2
        else:
            a1_rook, h1_rook = wr2, wr1

    # Similarly for black rooks: queen side (d8) is a8, king side (e8) is h8
    if black_kings:
        bk = black_kings[0]
        br1_to_king = _dist(br1, bk)
        br2_to_king = _dist(br2, bk)
        if br1_to_king < br2_to_king:
            h8_rook, a8_rook = br1, br2
        else:
            h8_rook, a8_rook = br2, br1
    else:
        # Fallback: a8 is on the same side as a1
        a1_pos = np.array([a1_rook["cx"], a1_rook["cy"]])
        br1_pos = np.array([br1["cx"], br1["cy"]])
        br2_pos = np.array([br2["cx"], br2["cy"]])
        if np.linalg.norm(br1_pos - a1_pos) < np.linalg.norm(br2_pos - a1_pos):
            a8_rook, h8_rook = br1, br2
        else:
            a8_rook, h8_rook = br2, br1

    # Compute corner positions from rook bounding boxes
    # Each rook's bounding box gives us a point near the corner
    # Extrapolate slightly outward to get the actual board corner
    a1 = _rook_to_corner(a1_rook, white_centroid)
    h1 = _rook_to_corner(h1_rook, white_centroid)
    a8 = _rook_to_corner(a8_rook, black_centroid)
    h8 = _rook_to_corner(h8_rook, black_centroid)

    corners = np.array([a1, a8, h8, h1], dtype=np.float32)
    return corners


def _dist(d1: dict, d2: dict) -> float:
    return ((d1["cx"] - d2["cx"]) ** 2 + (d1["cy"] - d2["cy"]) ** 2) ** 0.5


def _rook_to_corner(rook: dict, centroid: np.ndarray) -> np.ndarray:
    """Extrapolate from a rook's bounding box center to the board corner.

    The rook sits one square in from the corner. We push outward from
    the board center (approximated by the centroid) by about half a
    square width.
    """
    pos = np.array([rook["cx"], rook["cy"]])

    # Direction from board center toward this rook
    direction = pos - centroid
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Push outward by roughly half a rook bounding box width
    offset = max(rook["w"], rook["h"]) * 0.6
    corner = pos + direction * offset

    return corner

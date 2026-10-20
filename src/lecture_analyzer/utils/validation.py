"""Numerical and spatial validation helpers."""

import math
from typing import Optional, Tuple


def is_finite_number(val: Optional[float]) -> bool:
    """Returns True if value is a valid finite float, False otherwise."""
    if val is None:
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamps a numeric value within [min_val, max_val]."""
    return max(min_val, min(max_val, val))


def validate_bbox(
    bbox: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int
) -> Optional[Tuple[int, int, int, int]]:
    """Validates and clips a bounding box (x1, y1, x2, y2) to image dimensions.

    Returns None if bounding box is degenerate or completely out of bounds.
    """
    x1, y1, x2, y2 = bbox
    if not (is_finite_number(x1) and is_finite_number(y1) and is_finite_number(x2) and is_finite_number(y2)):
        return None

    ix1 = max(0, min(frame_w - 1, int(round(x1))))
    iy1 = max(0, min(frame_h - 1, int(round(y1))))
    ix2 = max(0, min(frame_w - 1, int(round(x2))))
    iy2 = max(0, min(frame_h - 1, int(round(y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    return ix1, iy1, ix2, iy2


def compute_iou(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float]
) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

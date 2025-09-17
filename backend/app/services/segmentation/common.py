from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, TypeAlias

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class RowSlices:
    """Per-row slices prepared for OCR."""

    bounds: tuple[int, int]  # (y1, y2)
    hex: Box  # detected hex marker box
    row: np.ndarray  # original row crop
    left: np.ndarray  # trimmed left part
    mid: np.ndarray  # mid part (score column)
    right: np.ndarray  # trimmed right part
    mid_bounds: tuple[int, int]  # (mx1, mx2)
    x_mid_global: Optional[int]  # global x_mid used for this row


def clip(v: int, lo: int, hi: int) -> int:
    """Clamp integer value into [lo, hi]."""
    return max(lo, min(int(v), hi))


def y_bounds(H: int, y_range: Tuple[float, float]) -> Tuple[int, int]:
    """
    Convert fractional y-range into pixel bounds.
    Fallback to full height if the result is degenerate.
    """
    y1 = clip(int(y_range[0] * H), 0, H - 1)
    y2 = clip(int(y_range[1] * H), 1, H)
    if y2 <= y1 + 1:
        y1, y2 = 0, H
    return y1, y2


ColorPlanes: TypeAlias = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


def prepare_color_planes(img_bgr: np.ndarray) -> ColorPlanes:
    """Return HSV, Lab, and Lab channels (L, a, b) for reuse."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    return hsv, lab, L, a, b


def prepare_gray_grad(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return gray image and Sobel magnitude (float32)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)  # float32
    return gray, mag

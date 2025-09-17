from __future__ import annotations

import cv2
import numpy as np

from app.utils.profiling import profiled

from .common import prepare_color_planes
from .constants import (
    HSV_S_MAX,
    HSV_V_MAX,
    LAB_CHROMA_MAX,
    LAB_L_MAX,
    TRIM_HCLOSE_FRAC,
)
from .se_cache import se_ellipse, se_rect


@profiled("seg.mask_dark")
def mask_dark(
    img_bgr: np.ndarray,
    *,
    hsv: np.ndarray | None = None,
    L: np.ndarray | None = None,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """Mask for dark/low-saturation regions (prefers text/background on dark UI)."""
    if hsv is None or L is None or a is None or b is None:
        hsv, _, Lc, ac, bc = prepare_color_planes(img_bgr)
        L, a, b = Lc, ac, bc

    # HSV: dark & desaturated
    m_hsv = cv2.inRange(hsv, (0, 0, 0), (179, HSV_S_MAX, HSV_V_MAX))

    # Lab chroma distance from neutral (128,128)
    da = cv2.absdiff(a, 128).astype(np.float32)
    db = cv2.absdiff(b, 128).astype(np.float32)
    chroma_sq = da * da + db * db
    mC = cv2.threshold(
        chroma_sq, float(LAB_CHROMA_MAX * LAB_CHROMA_MAX), 255, cv2.THRESH_BINARY_INV
    )[1].astype(np.uint8)

    # Lightness: keep darker
    mL = cv2.threshold(L, LAB_L_MAX, 255, cv2.THRESH_BINARY_INV)[1]

    m = cv2.bitwise_and(mL, mC)
    m = cv2.bitwise_or(m_hsv, m)

    H, W = img_bgr.shape[:2]
    k = max(2, int(min(H, W) * 0.008))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se_ellipse(k), 1)

    # Remove tiny components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    area_min = int(0.00003 * H * W)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            out[labels == i] = 255
    return out


def mask_bright_digits(
    img_bgr: np.ndarray,
    *,
    L: np.ndarray | None = None,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """Mask for bright, low-chroma digit-like pixels (white numeric labels)."""
    if L is None or a is None or b is None:
        _, _, Lc, ac, bc = prepare_color_planes(img_bgr)
        L, a, b = Lc, ac, bc

    da = cv2.absdiff(a, 128).astype(np.float32)
    db = cv2.absdiff(b, 128).astype(np.float32)
    chroma_sq = da * da + db * db

    mL = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY)[1]
    mC = cv2.threshold(chroma_sq, float(25 * 25), 255, cv2.THRESH_BINARY_INV)[1].astype(
        np.uint8
    )
    m = cv2.bitwise_and(mL.astype(np.uint8), mC)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se_ellipse(3), iterations=1)
    return m


@profiled("seg.mask_text_for_trim")
def mask_text_for_trim(
    img_bgr: np.ndarray,
    *,
    dark_mask: np.ndarray | None = None,
    hsv: np.ndarray | None = None,
    bright_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Consolidated “text-like” mask for horizontal trimming."""
    if hsv is None:
        hsv, _, _, _, _ = prepare_color_planes(img_bgr)

    dark = dark_mask if dark_mask is not None else mask_dark(img_bgr, hsv=hsv)
    bright = bright_mask if bright_mask is not None else mask_bright_digits(img_bgr)

    # Warm tones help keep informative pixels
    warm1 = cv2.inRange(hsv, (0, 70, 80), (35, 255, 255))
    warm2 = cv2.inRange(hsv, (140, 60, 80), (179, 255, 255))

    union = cv2.bitwise_or(dark, bright)
    union = cv2.bitwise_or(union, warm1)
    union = cv2.bitwise_or(union, warm2)

    H, W = union.shape[:2]
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, se_ellipse(3), 1)
    kx = max(3, int(W * TRIM_HCLOSE_FRAC) | 1)
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, se_rect(kx, 1), 1)
    return union

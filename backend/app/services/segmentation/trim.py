from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .common import clip
from .constants import ANCHOR_GAP_PX, MASK_TRIM_PAD_PX, MIN_TRIM_WIDTH


def anchored_span(
    cols_ok: np.ndarray, side: str, W: int, gap_allow: int
) -> Optional[Tuple[int, int]]:
    """
    Expand a continuous (with small gaps) span from an anchor toward both sides.
    Anchor: last 'ok' col (left) or first 'ok' col (right).
    """
    if not np.any(cols_ok):
        return None
    idx = np.where(cols_ok)[0]
    anchor = idx[-1] if side == "left" else idx[0]

    left = anchor
    gaps = 0
    for i in range(anchor - 1, -1, -1):
        if cols_ok[i]:
            left = i
            gaps = 0
        else:
            gaps += 1
            if gaps > gap_allow:
                break
    right = anchor
    gaps = 0
    for i in range(anchor + 1, W):
        if cols_ok[i]:
            right = i
            gaps = 0
        else:
            gaps += 1
            if gaps > gap_allow:
                break
    return (left, right)


def trim_by_mask_horizontal(
    img_slice: np.ndarray,
    mask_slice: np.ndarray,
    pad_px: int = MASK_TRIM_PAD_PX,
    min_width: int = MIN_TRIM_WIDTH,
    side: str = "left",
    smooth_ksz: int = 9,
    thr_frac_of_peak: float = 0.12,
    thr_abs_frac_h: float = 0.06,
    gap_allow: int = ANCHOR_GAP_PX,
    *,
    edge_mag_slice: np.ndarray | None = None,
) -> np.ndarray:
    """
    Trim horizontally using a text-like mask; fallback to edge energy when needed.
    The span grows around an anchor and tolerates small gaps to keep characters intact.
    """
    if img_slice.size == 0 or mask_slice.size == 0:
        return img_slice

    Hs, Ws = mask_slice.shape[:2]
    col = (mask_slice > 0).sum(axis=0, dtype=np.int32).astype(np.float32)
    if smooth_ksz % 2 == 0:
        smooth_ksz += 1
    if smooth_ksz > 1:
        col = cv2.GaussianBlur(col.reshape(1, -1), (smooth_ksz, 1), 0).ravel()

    peak = float(col.max())
    if peak > 1e-6:
        thr_abs = thr_abs_frac_h * Hs
        thr = max(thr_frac_of_peak * peak, thr_abs)
        cols_ok = col >= thr
        if not np.any(cols_ok):
            cols_ok = col >= max(0.06 * peak, 0.5 * thr_abs)
        span = anchored_span(cols_ok, side, Ws, gap_allow)
        if span is not None:
            i1, i2 = span
            x1 = max(0, i1 - pad_px)
            x2 = min(Ws, i2 + 1 + pad_px)
            if (x2 - x1) >= min_width:
                return img_slice[:, x1:x2]

    # Fallback: edge energy
    if edge_mag_slice is None:
        gray = cv2.cvtColor(img_slice, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
    else:
        mag = edge_mag_slice

    col_e = mag.mean(axis=0).astype(np.float32)
    if smooth_ksz % 2 == 0:
        smooth_ksz += 1
    col_e = cv2.GaussianBlur(col_e.reshape(1, -1), (smooth_ksz, 1), 0).ravel()
    peak_e = float(col_e.max())
    if peak_e > 1e-6:
        thr_e = 0.18 * peak_e
        cols_ok_e = col_e >= thr_e
        span = anchored_span(cols_ok_e, side, Ws, gap_allow)
        if span is not None:
            i1, i2 = span
            x1 = max(0, i1 - pad_px)
            x2 = min(Ws, i2 + 1 + pad_px)
            if (x2 - x1) >= min_width:
                return img_slice[:, x1:x2]
    return img_slice


def mid_bounds_from_global_x(W: int, x_mid_global: int, width: int) -> tuple[int, int]:
    """Turn a mid x and desired width into safe [x1, x2) bounds within [0, W)."""
    width = max(1, int(width))
    c = clip(x_mid_global, 0, W - 1)
    mx1 = clip(c - width // 2, 0, W - 1)
    mx2 = clip(mx1 + width, 1, W)
    if mx2 - mx1 < width:
        mx1 = clip(mx2 - width, 0, W - 1)
    return mx1, mx2

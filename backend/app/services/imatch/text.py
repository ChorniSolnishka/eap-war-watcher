from __future__ import annotations

"""Text-specific primitives: ink mask, profiles, similarities, coverage."""
import cv2
import numpy as np

_KER_OPEN_CLOSE = np.ones((2, 2), np.uint8)
_KER_DILATE_1x3 = np.ones((1, 3), np.uint8)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of likely text “ink” using Otsu ∧ adaptive threshold."""
    g_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr_otsu = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr_adap = cv2.adaptiveThreshold(
        g_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    m = cv2.bitwise_and(thr_otsu, thr_adap)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _KER_OPEN_CLOSE)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _KER_OPEN_CLOSE)
    return m


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks after gentle dilation to reduce brittleness."""
    A = (a > 0).astype(np.uint8)
    B = (b > 0).astype(np.uint8)
    A = cv2.dilate(A, _KER_DILATE_1x3, iterations=1)
    B = cv2.dilate(B, _KER_DILATE_1x3, iterations=1)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum() + 1e-9
    return float(inter / union)


def column_profile(mask: np.ndarray) -> np.ndarray:
    """L2-normalized column-sum profile (shape [W])."""
    col = mask.sum(axis=0).astype(np.float32)
    n = np.linalg.norm(col) + 1e-9
    return col / n


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity with safe normalization."""
    return float(
        np.dot(u, v) / ((np.linalg.norm(u) + 1e-9) * (np.linalg.norm(v) + 1e-9))
    )


def shift_array(x: np.ndarray, s: int) -> np.ndarray:
    """Integer shift with zero padding."""
    n = len(x)
    y = np.zeros(n, dtype=x.dtype)
    if s >= 0:
        y[s:] = x[: n - s]
    else:
        y[: n + s] = x[-s:]
    return y


def best_shifted_cos(
    pa: np.ndarray, pb: np.ndarray, max_shift: int = 40
) -> tuple[float, int]:
    """Max cosine over integer shifts in [-max_shift, max_shift]."""
    best, s_best = -1.0, 0
    for s in range(-max_shift, max_shift + 1):
        cs = cosine(pa, shift_array(pb, s))
        if cs > best:
            best, s_best = cs, s
    return best, s_best


def coverage(mask: np.ndarray) -> float:
    """Fraction of ‘ink’ pixels in the mask."""
    return float((mask > 0).sum()) / mask.size

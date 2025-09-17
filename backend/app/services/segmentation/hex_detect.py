from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from app.utils.profiling import profiled

from .common import clip, y_bounds
from .constants import (
    CENTER_BAND,
    HEX_AR_RANGE,
    HEX_H_RANGE,
    ROW_EXPAND_FRAC,
    ROW_FWHM_K,
    ROW_MIN_PAD_PX,
    WORK_BAND_HALF,
)


def find_score_column_x(
    mask: np.ndarray, y_range: Tuple[float, float] | None = None
) -> int:
    """Estimate x of the central score column by column-sum peak within a safe band."""
    H, W = mask.shape
    slice_ = (
        mask[y_bounds(H, y_range)[0] : y_bounds(H, y_range)[1], :] if y_range else mask
    )
    col = slice_.sum(axis=0, dtype=np.int32).astype(np.float32)
    k = max(5, int(0.01 * W) | 1)
    col = cv2.GaussianBlur(col.reshape(1, -1), (k, 1), 0).ravel()
    x1 = int(CENTER_BAND[0] * W)
    x2 = int(CENTER_BAND[1] * W)
    return int(np.argmax(col[x1:x2]) + x1)


def _cluster_by_y(
    boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Merge overlapping boxes into one per row by proximity along Y."""
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2.0)
    cys = [y + h / 2.0 for (_, y, _, h) in boxes]
    hs = [h for (_, _, _, h) in boxes]
    med_h = float(np.median(hs))
    med_sep = float(np.median(np.diff(cys))) if len(cys) >= 2 else med_h * 1.1
    thr = max(0.38 * med_h, 0.55 * med_sep)

    clusters: List[List[Tuple[int, int, int, int]]] = [[boxes[0]]]
    prev_cy = cys[0]
    for b, cy in zip(boxes[1:], cys[1:]):
        new_row = (cy - prev_cy) > thr
        if not new_row:
            pl = clusters[-1][-1]
            y_prev, h_prev = pl[1], pl[3]
            y_cur, h_cur = b[1], b[3]
            top = max(y_prev, y_cur)
            bottom = min(y_prev + h_prev, y_cur + h_cur)
            overlap = bottom - top
            if overlap < 0.15 * min(h_prev, h_cur):
                new_row = True
        clusters.append([b]) if new_row else clusters[-1].append(b)
        prev_cy = cy
    merged = [max(cl, key=lambda x: x[3]) for cl in clusters]
    return merged


def _row_peaks_near_xmid_band(
    band_mask: np.ndarray,
    x_mid: int,
    full_W: int,
    y1: int,
    *,
    min_peak_frac: float = 0.22,
) -> List[Tuple[int, int, int, int]]:
    """Pick row-like horizontal energy peaks inside a band and expand to FWHM."""
    Hband, Wband = band_mask.shape[:2]
    prof = band_mask.sum(axis=1, dtype=np.int32).astype(np.float32)
    k = max(5, int(0.015 * Hband) | 1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, k), 0).ravel()
    if prof.max() < 1e-6:
        return []
    thr = max(min_peak_frac * float(prof.max()), 2.0)

    # Estimate row spacing from autocorrelation
    ac = np.correlate(prof - prof.mean(), prof - prof.mean(), mode="full")
    ac = ac[ac.size // 2 :]
    min_sep = max(6, int(0.025 * Hband))
    if ac.size > min_sep + 3:
        j = np.argmax(ac[min_sep : min_sep + int(0.12 * Hband)]) + min_sep
        est_sep = max(min_sep, int(j))
    else:
        est_sep = max(min_sep, int(0.06 * Hband))

    peaks: List[int] = []
    last = -(10**9)
    for i in range(1, prof.size - 1):
        if prof[i] > thr and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]:
            if peaks and (i - last) < est_sep // 2:
                if prof[i] > prof[peaks[-1]]:
                    peaks[-1] = i
                    last = i
            else:
                peaks.append(i)
                last = i

    boxes: List[Tuple[int, int, int, int]] = []
    for p in peaks:
        half = prof[p] * ROW_FWHM_K
        up = p
        while up + 1 < prof.size and prof[up + 1] >= half:
            up += 1
        dn = p
        while dn - 1 >= 0 and prof[dn - 1] >= half:
            dn -= 1
        span = up - dn + 1
        pad_y = max(ROW_MIN_PAD_PX, int(ROW_EXPAND_FRAC * span))
        dn = max(0, dn - pad_y)
        up = min(prof.size - 1, up + pad_y)
        yy1 = y1 + dn
        yy2 = y1 + up + 1
        h = max(1, yy2 - yy1)
        cx = x_mid
        w = max(6, int(0.006 * full_W))
        boxes.append((cx - w // 2, yy1, w, h))
    return boxes


@profiled("seg.detect_hexes_in_band")
def detect_hexes_in_band(
    img: np.ndarray,
    mask_dark_only: np.ndarray,
    x_mid: int,
    y_range: Tuple[float, float],
    *,
    bright_mask: np.ndarray | None = None,
) -> List[Tuple[int, int, int, int]]:
    """Detect hex-shaped row markers in a working band around the mid column."""
    from .masks import mask_bright_digits  # avoid cycles

    bright = bright_mask if bright_mask is not None else mask_bright_digits(img)
    H, W = mask_dark_only.shape[:2]
    dx = int(WORK_BAND_HALF * W)
    x1, x2 = clip(x_mid - dx, 0, W - 1), clip(x_mid + dx, 0, W - 1)
    y1, y2 = y_bounds(H, y_range)

    # Operate on band to save memory/time
    union_band = cv2.bitwise_or(mask_dark_only[y1:y2, x1:x2], bright[y1:y2, x1:x2])

    boxes_cnt = _detect_hexes_in_band_impl_band(
        union_band,
        x1,
        y1,
        H,
        W,
        hex_h_range=HEX_H_RANGE,
        hex_ar_range=HEX_AR_RANGE,
    )
    boxes_prof = _row_peaks_near_xmid_band(union_band, x_mid, W, y1)

    if not boxes_cnt and boxes_prof:
        return _cluster_by_y(boxes_prof)

    # Merge contour-based and profile-based boxes per row
    merged: List[Tuple[int, int, int, int]] = []
    used_prof = np.zeros(len(boxes_prof), dtype=bool)
    for bc in boxes_cnt:
        x, y, w, h = bc
        cy = y + h / 2.0
        idx = []
        for i, bp in enumerate(boxes_prof):
            if used_prof[i]:
                continue
            py = bp[1] + bp[3] / 2.0
            if abs(py - cy) <= max(0.45 * h, 0.35 * bp[3]):
                idx.append(i)
        if len(idx) >= 2:
            for i in idx:
                used_prof[i] = True
                merged.append(boxes_prof[i])
        elif len(idx) == 1:
            used_prof[idx[0]] = True
            merged.append(boxes_prof[idx[0]])
        else:
            merged.append(bc)
    for i, bp in enumerate(boxes_prof):
        if not used_prof[i]:
            merged.append(bp)
    return _cluster_by_y(merged)


def _detect_hexes_in_band_impl_band(
    union_band: np.ndarray,
    offset_x: int,
    offset_y: int,
    H: int,
    W: int,
    *,
    hex_h_range: Tuple[float, float],
    hex_ar_range: Tuple[float, float],
) -> List[Tuple[int, int, int, int]]:
    """Contour-based hex detection inside a band.
    Coordinates are offset to full image."""
    cnts, _ = cv2.findContours(union_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (hex_h_range[0] * H <= h <= hex_h_range[1] * H):
            continue
        ar = w / (h + 1e-6)
        if not (hex_ar_range[0] <= ar <= hex_ar_range[1]):
            continue
        cx = offset_x + x + w / 2
        cy = offset_y + y + h / 2
        if not (0 <= cx <= W - 1):
            continue
        if not (0 <= cy <= H - 1):
            continue
        cand.append((offset_x + x, offset_y + y, w, h))
    return _cluster_by_y(cand)


def refine_x_mid_from_boxes(x_mid: int, boxes: List[Tuple[int, int, int, int]]) -> int:
    """Refine mid-column by median of candidate centers; keep input if no boxes."""
    if not boxes:
        return x_mid
    cxs = [x + w / 2 for (x, y, w, h) in boxes]
    return int(np.median(cxs))

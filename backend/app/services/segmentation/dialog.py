from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .common import clip
from .constants import (
    BLUE_MASKS,
    DIALOG_CLOSE_K,
    DIALOG_PAD,
    MIN_DIALOG_H_FRAC,
    MIN_DIALOG_W_FRAC,
)
from .se_cache import se_rect


def _largest_reasonable_box(
    mask: np.ndarray, W: int, H: int
) -> Optional[Tuple[int, int, int, int]]:
    """Pick the largest dialog-like contour by geometric sanity checks."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    minW = int(MIN_DIALOG_W_FRAC * W)
    minH = int(MIN_DIALOG_H_FRAC * H)
    best = None
    best_area = -1.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < minW or h < minH:
            continue
        box_area = float(w * h)
        cnt_area = float(cv2.contourArea(c))
        if box_area <= 1:
            continue
        extent = cnt_area / box_area
        if extent < 0.65:
            continue
        if cnt_area > best_area:
            best_area = cnt_area
            best = (x, y, w, h)
    return best


def extract_attack_info_window(
    img_bgr: np.ndarray, dbg_dir: Optional[Path] = None
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """
    Localize the blue attack dialog and return a padded ROI around it.
    Returns:
        (roi_bgr, bbox) where bbox=(x,y,w,h) in original coords.
        If not found: (img, None).
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_all = np.zeros((H, W), np.uint8)
    for lo, hi in BLUE_MASKS:
        mask_all = cv2.bitwise_or(mask_all, cv2.inRange(hsv, lo, hi))
    k = se_rect(DIALOG_CLOSE_K, DIALOG_CLOSE_K)
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, k, iterations=2)
    box = _largest_reasonable_box(mask_all, W, H)
    if dbg_dir:
        dbg_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg_dir / "dialog_mask.png"), mask_all)
    if box is None:
        return img_bgr, None
    x, y, w, h = box
    x1 = clip(x - DIALOG_PAD, 0, W - 1)
    y1 = clip(y - DIALOG_PAD, 0, H - 1)
    x2 = clip(x + w + DIALOG_PAD, 1, W)
    y2 = clip(y + h + DIALOG_PAD, 1, H)
    roi = img_bgr[y1:y2, x1:x2]
    return roi, (x1, y1, x2 - x1, y2 - y1)

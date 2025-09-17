from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from app.utils.profiling import profiled

from .common import clip
from .constants import (
    GLOBAL_ROW_HEIGHT_GAIN,
    MID_PAD_X,
    MIN_MID_W_FRAC,
    MIN_MID_W_ROW,
    ROW_PAD_Y,
)
from .trim import mid_bounds_from_global_x, trim_by_mask_horizontal


@profiled("seg.segment_and_cut")
def segment_and_cut(
    img: np.ndarray,
    hex_boxes: list[tuple[int, int, int, int]],
    mask_for_trim: np.ndarray,
    x_mid_global: Optional[int] = None,
    lock_mid_to_global: bool = True,
    fixed_mid_w: Optional[int] = None,
    *,
    edge_mag_full: np.ndarray | None = None,
) -> list[Dict]:
    """
    Slice the ROI into per-row dicts with {left, mid, right} parts ready for OCR.
    Uses hex heights to estimate row height, trims left/right by text-like masks
    with a robust fallback to edge energy.
    """
    H, W = img.shape[:2]
    if not hex_boxes:
        return []

    heights = [h for (_, _, _, h) in hex_boxes]
    heights.sort()
    if len(heights) >= 4:
        k = max(1, len(heights) // 4)
        core = heights[k:-k]
        target_row_h = int(np.median(core))
    else:
        target_row_h = int(np.median(heights))
    target_row_h = int(target_row_h * GLOBAL_ROW_HEIGHT_GAIN)

    if hex_boxes:
        hh_med = int(np.median([h for (_, _, _, h) in hex_boxes]))
        min_by_hex = int(hh_med * (1 + 2 * ROW_PAD_Y))
        target_row_h = max(target_row_h, min_by_hex)

    target_row_h = max(1, min(target_row_h, H))

    base_min_mid_w = max(int(MIN_MID_W_FRAC * W), int(MIN_MID_W_ROW * target_row_h))
    mid_w_for_all = base_min_mid_w if fixed_mid_w is None else max(1, int(fixed_mid_w))

    rows: list[Dict] = []
    for hx, hy, hw, hh in hex_boxes:
        cy = hy + hh / 2.0
        y1 = int(round(cy - target_row_h / 2.0))
        y2 = y1 + target_row_h
        off_top = max(0, -y1)
        off_bottom = max(0, y2 - H)
        y1 += off_bottom - off_top
        y1 = clip(y1, 0, H - 1)
        y2 = clip(y1 + target_row_h, 1, H)

        row_img = img[y1:y2, :]
        row_mask = mask_for_trim[y1:y2, :]
        row_mag = None if edge_mag_full is None else edge_mag_full[y1:y2, :]

        # Determine mid bounds either from global hint or from hex box vicinity
        if lock_mid_to_global and (x_mid_global is not None):
            mx1, mx2 = mid_bounds_from_global_x(W, x_mid_global, mid_w_for_all)
        else:
            mx1 = clip(hx - int(W * MID_PAD_X / 2), 0, W - 1)
            mx2 = clip(hx + hw + int(W * MID_PAD_X / 2), 0, W - 1)
            if (mx2 - mx1) < base_min_mid_w:
                c = (mx1 + mx2) // 2
                mx1 = clip(c - base_min_mid_w // 2, 0, W - 1)
                mx2 = clip(mx1 + base_min_mid_w, 1, W)

        left_img = row_img[:, 0:mx1]
        mid_img = row_img[:, mx1:mx2]
        right_img = row_img[:, mx2:W]

        left_mask = row_mask[:, 0:mx1]
        right_mask = row_mask[:, mx2:W]

        left_trimmed = trim_by_mask_horizontal(
            left_img,
            left_mask,
            side="left",
            edge_mag_slice=None if row_mag is None else row_mag[:, 0:mx1],
        )
        right_trimmed = trim_by_mask_horizontal(
            right_img,
            right_mask,
            side="right",
            edge_mag_slice=None if row_mag is None else row_mag[:, mx2:W],
        )

        rows.append(
            {
                "bounds": (y1, y2),
                "hex": (hx, hy, hw, hh),
                "row": row_img,
                "left": left_trimmed,
                "mid": mid_img,
                "right": right_trimmed,
                "mid_bounds": (mx1, mx2),
                "x_mid_global": x_mid_global,
            }
        )
    return rows

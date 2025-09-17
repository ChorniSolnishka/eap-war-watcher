from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.utils.profiling import profiled

from .common import prepare_color_planes, prepare_gray_grad
from .constants import (
    CONTENT_Y_RANGE,
    FALLBACK_CONTENT_Y_RANGE,
    LOCK_MID_TO_GLOBAL,
)
from .debug import save_debug
from .dialog import extract_attack_info_window
from .hex_detect import (
    detect_hexes_in_band,
    find_score_column_x,
    refine_x_mid_from_boxes,
)
from .masks import mask_bright_digits, mask_dark, mask_text_for_trim
from .rows import segment_and_cut


class DarkSegmenter:
    """
    End-to-end dark-theme segmenter with small cross-call memory.

    Usage:
        seg = DarkSegmenter()
        rows = seg.run_segmentation_on_roi(full_bgr, debug_dir)

    Memory:
        - mid column fraction and width fraction are remembered across calls
          to stabilize cuts on video/sequence.
    """

    def __init__(self, *, lock_mid_to_global: bool = LOCK_MID_TO_GLOBAL) -> None:
        self._mid_frac: Optional[float] = None
        self._mid_w_frac: Optional[float] = None
        self._lock_mid = lock_mid_to_global

    def reset_memory(self) -> None:
        """Forget cached mid-column hints (use between different battles/datasets)."""
        self._mid_frac = None
        self._mid_w_frac = None

    @profiled("seg.run_segmentation_on_roi")
    def run_segmentation_on_roi(
        self, full_bgr: np.ndarray, debug_dir: Path
    ) -> List[Dict]:
        """
        ROI segmentation pipeline (optimized, thresholds preserved):

        - Compute HSV/Lab once and reuse.
        - mask_text_for_trim reuses mask_dark/bright if provided.
        - detect_hexes_in_band operates on a band around x_mid (not whole frame).
        - Compute Sobel magnitude once and reuse during trimming.
        - Remember mid position & width (as fractions) to stabilize subsequent frames.
        """
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 1) Dialog window → ROI
        roi_bgr, _ = extract_attack_info_window(full_bgr, dbg_dir=debug_dir)
        H, W = roi_bgr.shape[:2]

        # 2) Shared preprocessing for ROI
        hsv, lab, L, a, b = prepare_color_planes(roi_bgr)
        _, edge_mag = prepare_gray_grad(roi_bgr)

        # 3) Single-pass masks
        dark = mask_dark(roi_bgr, hsv=hsv, L=L, a=a, b=b)
        bright = mask_bright_digits(roi_bgr, L=L, a=a, b=b)
        trim_mask = mask_text_for_trim(
            roi_bgr, dark_mask=dark, hsv=hsv, bright_mask=bright
        )

        # 4) Find x_mid and rows
        y_range = CONTENT_Y_RANGE
        use_memory = self._mid_frac is not None

        if use_memory:
            mid_frac = self._mid_frac
            x_mid = int(round(mid_frac * W))
            fixed_mid_w = (
                int(round((self._mid_w_frac or 0.0) * W))
                if self._mid_w_frac is not None
                else None
            )

            hex_boxes = detect_hexes_in_band(
                roi_bgr, dark, x_mid, y_range=y_range, bright_mask=bright
            )
            if len(hex_boxes) < 12:
                y_range = FALLBACK_CONTENT_Y_RANGE
                hex_boxes = detect_hexes_in_band(
                    roi_bgr, dark, x_mid, y_range=y_range, bright_mask=bright
                )
        else:
            fixed_mid_w = None
            x_mid = find_score_column_x(dark, y_range=y_range)
            hex_boxes = detect_hexes_in_band(
                roi_bgr, dark, x_mid, y_range=y_range, bright_mask=bright
            )

            if hex_boxes:
                x_mid_ref = refine_x_mid_from_boxes(x_mid, hex_boxes)
                if abs(x_mid_ref - x_mid) > 2:
                    x_mid = x_mid_ref
                    hex_boxes = detect_hexes_in_band(
                        roi_bgr, dark, x_mid, y_range=y_range, bright_mask=bright
                    )

            if len(hex_boxes) < 12:
                y_range = FALLBACK_CONTENT_Y_RANGE
                x_mid = find_score_column_x(dark, y_range=y_range)
                hex_boxes = detect_hexes_in_band(
                    roi_bgr, dark, x_mid, y_range=y_range, bright_mask=bright
                )
                if hex_boxes:
                    x_mid = refine_x_mid_from_boxes(x_mid, hex_boxes)
                    hex_boxes = detect_hexes_in_band(
                        roi_bgr, dark, x_mid, y_range=y_range, bright_mask=bright
                    )

        if not hex_boxes:
            save_debug(
                roi_bgr,
                dark,
                x_mid,
                [],
                [],
                debug_dir,
                y_range=y_range,
                mask_trim=trim_mask,
            )
            return []

        # 5) Row slicing
        rows = segment_and_cut(
            roi_bgr,
            hex_boxes,
            trim_mask,
            x_mid_global=x_mid,
            lock_mid_to_global=self._lock_mid,
            fixed_mid_w=fixed_mid_w,
            edge_mag_full=edge_mag,
        )

        # 6) Update mid "memory" once we have a stable row
        if not use_memory and rows:
            mx1, mx2 = rows[0].get("mid_bounds", (0, 0))
            mid_w = max(1, int(mx2 - mx1))
            self._mid_frac = (x_mid / float(W)) if W > 0 else None
            self._mid_w_frac = (mid_w / float(W)) if W > 0 else None

        # 7) Debug outputs
        save_debug(
            roi_bgr,
            dark,
            x_mid,
            hex_boxes,
            rows,
            debug_dir,
            y_range=y_range,
            mask_trim=trim_mask,
        )
        return rows

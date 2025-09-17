from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .common import y_bounds


def save_debug(
    img: np.ndarray,
    mask_dark_used: np.ndarray,
    x_mid: int,
    hex_boxes: List[Tuple[int, int, int, int]],
    rows: List[Dict],
    out_dir: Path,
    y_range: Tuple[float, float] | None = None,
    mask_trim: np.ndarray | None = None,
) -> None:
    """Dump visual debug artifacts: masks, bands, hex boxes, row slices, etc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = img.shape[:2]
    dbg = img.copy()
    if y_range is not None:
        y1, y2 = y_bounds(H, y_range)
        cv2.line(dbg, (0, y1), (W - 1, y1), (255, 128, 0), 2)
        cv2.line(dbg, (0, y2), (W - 1, y2), (255, 128, 0), 2)
    cv2.line(dbg, (x_mid, 0), (x_mid, H - 1), (0, 255, 255), 2)
    for x, y, w, h in hex_boxes:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for i, r in enumerate(rows, 1):
        y1r, y2r = r["bounds"]
        mx1, mx2 = r.get("mid_bounds", (0, 0))
        cv2.rectangle(dbg, (0, y1r), (W - 1, y2r), (255, 0, 0), 2)
        cv2.rectangle(dbg, (mx1, y1r), (mx2, y2r), (0, 255, 255), 2)
        cv2.putText(
            dbg,
            f"row {i}",
            (10, max(25, y1r + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    cv2.imwrite(str(out_dir / "mask_dark.png"), mask_dark_used)
    if mask_trim is not None:
        cv2.imwrite(str(out_dir / "mask_trim.png"), mask_trim)
    cv2.imwrite(str(out_dir / "debug.png"), dbg)

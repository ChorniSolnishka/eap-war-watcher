from __future__ import annotations

import logging
from typing import Tuple, TypeAlias

import cv2

logger = logging.getLogger(__name__)

ScaleRet: TypeAlias = Tuple[float, float, int, int]


def calc_image_scale(image_path: str, max_w_px: int, max_h_px: int) -> ScaleRet:
    """
    Compute uniform scale (no upscaling) and original size for an image
    to fit into a cell. Returns: (sx, sy, orig_w, orig_h).
    """
    try:
        im = cv2.imread(image_path)
        if im is None:
            return 1.0, 1.0, 0, 0
        h, w = im.shape[:2]
        if w == 0 or h == 0:
            return 1.0, 1.0, w, h
        sx = max_w_px / w
        sy = max_h_px / h
        s = min(sx, sy, 1.0)
        return float(s), float(s), int(w), int(h)
    except Exception as e:
        logger.warning("Failed to read/scale image %s: %s", image_path, e)
        return 1.0, 1.0, 0, 0

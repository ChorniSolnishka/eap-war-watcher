from __future__ import annotations

import cv2
import numpy as np

_SE_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


def se_ellipse(k: int) -> np.ndarray:
    """Return cached elliptical structuring element k×k."""
    key = ("ell", k, k)
    se = _SE_CACHE.get(key)
    if se is None:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        _SE_CACHE[key] = se
    return se


def se_rect(w: int, h: int) -> np.ndarray:
    """Return cached rectangular structuring element w×h."""
    key = ("rect", w, h)
    se = _SE_CACHE.get(key)
    if se is None:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
        _SE_CACHE[key] = se
    return se

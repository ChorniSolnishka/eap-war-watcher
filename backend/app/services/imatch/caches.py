from __future__ import annotations

"""
Small hand-rolled caches:
- path->BGR image
- path->descriptor (hash, profile, W,H, gray256x64)
- rotation cache for 256×64 gray
- boolean verdict cache for (query,candidate,param_signature)
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .constants import (
    IMATCH_DESC_LRU_MAX,
    IMATCH_IMG_LRU_MAX,
    IMATCH_ROT_CACHE_MAX,
    IMATCH_VERDICT_CACHE_MAX,
)

_IMG_CACHE: Dict[str, np.ndarray] = {}
_DESC_CACHE: Dict[str, Tuple[int, np.ndarray, int, int, np.ndarray]] = {}
_ROT_CACHE: Dict[tuple[int, float], np.ndarray] = {}
_VERDICT_CACHE: Dict[tuple, bool] = {}


def get_img_bgr(path: str) -> Optional[np.ndarray]:
    """Read BGR image with a tiny LRU-like cache."""
    im = _IMG_CACHE.get(path)
    if im is not None:
        return im
    im = cv2.imread(path)
    if im is None:
        return None
    if len(_IMG_CACHE) >= IMATCH_IMG_LRU_MAX:
        _IMG_CACHE.clear()
    _IMG_CACHE[path] = im
    return im


def get_desc(path: str) -> Optional[Tuple[int, np.ndarray, int, int, np.ndarray]]:
    """Get candidate descriptor from cache, computing if needed."""
    d = _DESC_CACHE.get(path)
    return d


def put_desc(
    path: str, desc: Tuple[int, np.ndarray, int, int, np.ndarray]
) -> Tuple[int, np.ndarray, int, int, np.ndarray]:
    if len(_DESC_CACHE) >= IMATCH_DESC_LRU_MAX:
        _DESC_CACHE.clear()
    _DESC_CACHE[path] = desc
    return desc


def rot_cache_get(key: tuple[int, float]) -> Optional[np.ndarray]:
    return _ROT_CACHE.get(key)


def rot_cache_put(key: tuple[int, float], img: np.ndarray) -> None:
    if len(_ROT_CACHE) >= IMATCH_ROT_CACHE_MAX:
        _ROT_CACHE.clear()
    _ROT_CACHE[key] = img


def verdict_get(key: tuple) -> Optional[bool]:
    return _VERDICT_CACHE.get(key)


def verdict_put(key: tuple, val: bool) -> None:
    if len(_VERDICT_CACHE) >= IMATCH_VERDICT_CACHE_MAX:
        _VERDICT_CACHE.clear()
    _VERDICT_CACHE[key] = val

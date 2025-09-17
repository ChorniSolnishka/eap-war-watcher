from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class _BatchCache:
    __slots__ = ("by_id", "limit", "created")

    def __init__(self, limit: int = 1024):
        self.by_id: Dict[int, Dict[Tuple[Any, ...], np.ndarray]] = {}
        self.limit = limit
        self.created = time.time()

    def get(self, arr: np.ndarray, key: Tuple[Any, ...]) -> Optional[np.ndarray]:
        d = self.by_id.get(id(arr))
        if not d:
            return None
        return d.get(key)

    def put(self, arr: np.ndarray, key: Tuple[Any, ...], val: np.ndarray) -> None:
        d = self.by_id.get(id(arr))
        if d is None:
            d = {}
            self.by_id[id(arr)] = d
        if len(d) >= self.limit:
            d.clear()
        d[key] = val


_tls = threading.local()


class batch:
    """Контекст: кэш препроцессинга в рамках одного батча (одного matсh_or_new)."""

    def __init__(self) -> None:
        self._old = None

    def __enter__(self):
        self._old = getattr(_tls, "cache", None)
        _tls.cache = _BatchCache()
        return self

    def __exit__(self, exc_type, exc, tb):
        _tls.cache = self._old


def batch_scoped(fn):
    """Декоратор: оборачивает вызов в batch-контекст кэша."""

    def w(*a, **kw):
        with batch():
            return fn(*a, **kw)

    w.__name__ = getattr(fn, "__name__", "wrapped")
    w.__qualname__ = getattr(fn, "__qualname__", w.__name__)
    w.__doc__ = getattr(fn, "__doc__", None)
    return w


def _ensure_contiguous(a, dtype=None):
    """Гарантируем np.ndarray (uint8 по умолчанию) и C_CONTIGUOUS layout."""
    if a is None:
        raise ValueError("img_cache: got None as image")

    arr = a if isinstance(a, np.ndarray) else np.asarray(a)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)

    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def cvtColor(src_bgr, code):
    src_bgr = _ensure_contiguous(src_bgr, np.uint8)
    return cv2.cvtColor(src_bgr, code)


def GaussianBlur(src, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT):
    src = _ensure_contiguous(src, np.uint8)
    return cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)


def resize(src, dsize, fx=0, fy=0, interpolation=cv2.INTER_LINEAR):
    src = _ensure_contiguous(src, np.uint8)
    return cv2.resize(src, dsize, fx=fx, fy=fy, interpolation=interpolation)


def matchTemplate(image, templ, method=cv2.TM_CCOEFF_NORMED, mask=None):
    image = _ensure_contiguous(image, np.uint8)
    templ = _ensure_contiguous(templ, np.uint8)
    if mask is not None:
        mask = _ensure_contiguous(mask, np.uint8)
    return cv2.matchTemplate(image, templ, method, mask=mask)

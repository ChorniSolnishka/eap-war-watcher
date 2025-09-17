from __future__ import annotations

"""Resizing + perceptual hash helpers."""
import cv2
import numpy as np


def resize_gray(img, w: int = 256, h: int = 64) -> np.ndarray:
    """BGR → gray, resized to (w,h) with AREA interpolation."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)


def dhash64(gray: np.ndarray) -> int:
    """Vectorized 64-bit dHash on grayscale image."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = (small[:, 1:] > small[:, :-1]).astype(np.uint8)
    bits = np.packbits(diff, axis=None, bitorder="big")
    return int.from_bytes(bytes(bits.tolist()), byteorder="big", signed=False)


def image_hash64(bgr) -> int:
    """Convenience: compute 64-bit dHash on BGR image."""
    return dhash64(resize_gray(bgr))

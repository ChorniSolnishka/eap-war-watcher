from __future__ import annotations

"""Alignment & similarity primitives (phase/ECC/rotations/Sobel/NCC)."""
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from .caches import rot_cache_get, rot_cache_put

# try enable OpenCV opts
try:
    cv2.setUseOptimized(True)
except Exception:
    pass


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation on zero-mean float32 arrays."""
    A = a.astype(np.float32)
    A -= A.mean()
    B = b.astype(np.float32)
    B -= B.mean()
    denom = (np.linalg.norm(A) * np.linalg.norm(B)) + 1e-12
    return float((A * B).sum() / denom)


def crop_center(gray: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """Central crop by fraction on both axes."""
    H, W = gray.shape[:2]
    f = max(0.1, min(1.0, float(frac)))
    w = int(W * f)
    h = int(H * f)
    x1 = (W - w) // 2
    y1 = (H - h) // 2
    return gray[y1 : y1 + h, x1 : x1 + w]


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    """Edge magnitude normalized by std; emphasizes strokes."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag / (mag.std() + 1e-6)


def rotate_gray(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around center with border replicate; keeps shape."""
    H, W = gray.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        gray, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def rotate_gray_cached(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate 256×64 gray using a tiny cache keyed by buffer address and angle."""
    try:
        ptr = int(gray.__array_interface__["data"][0])
    except Exception:
        ptr = id(gray)
    key = (ptr, float(angle_deg))
    im = rot_cache_get(key)
    if im is not None:
        return im
    im = rotate_gray(gray, angle_deg)
    rot_cache_put(key, im)
    return im


@lru_cache(maxsize=8)
def _hann_window(w: int, h: int) -> np.ndarray:
    """Hann window for phase correlation; OpenCV expects (width,height)."""
    return cv2.createHanningWindow((w, h), cv2.CV_32F)


def phase_estimate_shift(
    gray_ref: np.ndarray, gray_mov: np.ndarray
) -> tuple[float, float]:
    """Subpixel (dx,dy) via phase correlation with Hann window."""
    a = gray_ref.astype(np.float32)
    b = gray_mov.astype(np.float32)
    win = _hann_window(a.shape[1], a.shape[0])
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return dx, dy


def phase_align(
    gray_ref: np.ndarray, gray_mov: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Translate moving image by phase-correlation shift and return aligned copy."""
    dx, dy = phase_estimate_shift(gray_ref, gray_mov)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(
        gray_mov,
        M,
        (gray_mov.shape[1], gray_mov.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, dx, dy


def ecc_align(
    gray_ref: np.ndarray,
    gray_mov: np.ndarray,
    motion: str = "euclidean",
    max_iter: int = 200,
    eps: float = 1e-5,
    gauss_size: int = 5,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """ECC alignment with motion model selection and safe failure."""
    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    motion_type = motion_map.get(motion.lower(), cv2.MOTION_EUCLIDEAN)

    H, W = gray_ref.shape[:2]
    ref = gray_ref.astype(np.float32) / 255.0
    mov = gray_mov.astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(max_iter),
        float(eps),
    )
    try:
        cc, warp = cv2.findTransformECC(
            ref, mov, warp, motion_type, criteria, None, gaussFiltSize=int(gauss_size)
        )
        aligned = cv2.warpAffine(
            gray_mov,
            warp,
            (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, float(cc)
    except cv2.error:
        return None, None

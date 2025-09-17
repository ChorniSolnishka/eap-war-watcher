from __future__ import annotations

"""Composite verifier (phase + ECC + text heuristics).
Precomputes a query context to reuse across candidates."""
from dataclasses import dataclass

import numpy as np

from .alignment import (
    crop_center,
    ecc_align,
    ncc,
    phase_align,
    rotate_gray_cached,
    sobel_mag,
)
from .text import best_shifted_cos, column_profile, cosine, coverage, ink_mask, mask_iou


@dataclass(frozen=True)
class QCtx:
    """Immutable bundle of query precomputations."""

    ga: np.ndarray  # gray 256×64
    ga_center: np.ndarray  # central crop
    sobel_ga: np.ndarray  # edge magnitude
    ma: np.ndarray  # ink mask
    pa: np.ndarray  # column profile


def make_qctx(q_gray: np.ndarray, *, center_frac: float = 0.9) -> QCtx:
    """Precompute query features used across many candidate checks."""
    ga_center = crop_center(q_gray, center_frac)
    sobel = sobel_mag(q_gray)
    m = ink_mask(q_gray)
    p = column_profile(m)
    return QCtx(q_gray, ga_center, sobel, m, p)


def verify_with_qctx(
    qctx: QCtx,
    gb: np.ndarray,
    *,
    method: str = "both",
    ncc_thr: float = 0.82,
    ncc_center_thr: float = 0.86,
    edge_ncc_thr: float = 0.70,
    center_frac: float = 0.9,
    max_shift: float = 10.0,
    ecc_cc_min: float = 0.0,
    ecc_rot_candidates: tuple[float, ...] = (-2.0, 0.0, 2.0),
    ecc_gauss_sizes: tuple[int, ...] = (3, 5, 7),
    ecc_kinds: tuple[str, ...] = ("euclidean", "affine"),
    ecc_max_iter: int = 200,
    ecc_eps: float = 1e-5,
    ink_iou_thr: float = 0.76,
    profile_cos_thr: float = 0.93,
    profile_shifted_thr: float = 0.99,
    coverage_delta_max: float = 0.06,
) -> bool:
    """Return True if candidate `gb` matches the query represented by `qctx`."""
    ga = qctx.ga
    ga_center = qctx.ga_center
    sobel_ga = qctx.sobel_ga
    ma = qctx.ma
    pa = qctx.pa

    def _text_checks(ga_aligned: np.ndarray) -> tuple[bool, bool]:
        mb = ink_mask(ga_aligned)
        pb = column_profile(mb)
        iou = mask_iou(ma, mb)
        cos = cosine(pa, pb)
        text_ok = (iou >= ink_iou_thr) and (cos >= profile_cos_thr)
        best_cos, _ = best_shifted_cos(pa, pb, max_shift=40)
        cov_ok = abs(coverage(ma) - coverage(mb)) <= coverage_delta_max
        text_strong = (best_cos >= profile_shifted_thr) and cov_ok
        return text_ok or text_strong, text_strong

    def _accept(ga_aligned: np.ndarray) -> bool:
        full_ok = ncc(ga, ga_aligned) >= ncc_thr
        center_ok = (
            ncc(ga_center, crop_center(ga_aligned, center_frac)) >= ncc_center_thr
        )
        edge_ok = ncc(sobel_ga, sobel_mag(ga_aligned)) >= edge_ncc_thr
        text_ok, text_strong = _text_checks(ga_aligned)
        if text_strong:
            return True
        if center_ok and edge_ok and text_ok:
            return True
        if full_ok and text_ok and (center_ok or edge_ok):
            return True
        return False

    if method in ("phase", "both"):
        aligned, dx, dy = phase_align(ga, gb)
        if abs(dx) <= max_shift and abs(dy) <= max_shift and _accept(aligned):
            return True
        if method == "phase":
            return False

    for ang in ecc_rot_candidates:
        gb_r = rotate_gray_cached(gb, ang) if abs(ang) > 1e-6 else gb
        for gsz in ecc_gauss_sizes:
            for kind in ecc_kinds:
                aligned, cc = ecc_align(
                    ga,
                    gb_r,
                    motion=kind,
                    max_iter=ecc_max_iter,
                    eps=ecc_eps,
                    gauss_size=gsz,
                )
                if aligned is None:
                    continue
                if ecc_cc_min > 0.0 and (cc is None or cc < ecc_cc_min):
                    continue
                if _accept(aligned):
                    return True
    return False

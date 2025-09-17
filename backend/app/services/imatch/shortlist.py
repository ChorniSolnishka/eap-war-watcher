from __future__ import annotations

"""Descriptors and ranking for shortlist generation."""
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .caches import get_desc, get_img_bgr, put_desc
from .hashing import image_hash64, resize_gray
from .text import column_profile, cosine, ink_mask


def desc_for_query(bgr) -> Tuple[int, np.ndarray, np.ndarray]:
    """(dhash64, profile[256], gray256×64) for the query image."""
    h = image_hash64(bgr)
    g = resize_gray(bgr, 256, 64)
    prof = column_profile(ink_mask(g))
    return h, prof, g


def _build_or_get_candidate_desc(
    path: str,
) -> Optional[Tuple[int, np.ndarray, int, int, np.ndarray]]:
    d = get_desc(path)
    if d is not None:
        return d
    img = get_img_bgr(path)
    if img is None:
        return None
    h = image_hash64(img)
    g = resize_gray(img, 256, 64)
    H, W = img.shape[:2]
    prof = column_profile(ink_mask(g))
    return put_desc(path, (h, prof, int(W), int(H), g))


def hamming64(a: int, b: int) -> int:
    """Popcount of a XOR b."""
    x = a ^ b
    try:
        return x.bit_count()  # py3.8+
    except AttributeError:
        cnt = 0
        while x:
            x &= x - 1
            cnt += 1
        return cnt


def gather_metrics(
    existing: Sequence[Tuple[int, str]],
    *,
    q_hash: int,
    q_prof: np.ndarray,
    qW: int,
    qH: int,
    max_wdiff: int,
    use_ar_filter: int,
    max_ar_diff: float,
) -> List[Tuple[int, int, float, str, np.ndarray, int]]:
    """
    Compute shortlist metrics for all candidates, applying geometry gates.
    Returns list of tuples: (pid, hamDist, profCos, path, cand_gray, cand_hash)
    """
    seen_paths: set[str] = set()
    uniq: list[tuple[int, str]] = []
    for pid, p in existing:
        if p not in seen_paths:
            seen_paths.add(p)
            uniq.append((pid, p))

    out: List[Tuple[int, int, float, str, np.ndarray, int]] = []
    for pid, path in uniq:
        d = _build_or_get_candidate_desc(path)
        if d is None:
            continue
        ch, cprof, cW, cH, cgray = d
        if abs(int(cW) - int(qW)) > max_wdiff:
            continue
        if use_ar_filter:
            q_ar = qW / max(1e-6, qH)
            c_ar = cW / max(1e-6, cH)
            if abs(q_ar - c_ar) > max_ar_diff:
                continue
        out.append(
            (
                int(pid),
                hamming64(q_hash, ch),
                float(cosine(q_prof, cprof)),
                path,
                cgray,
                int(ch),
            )
        )
    return out

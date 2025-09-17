from __future__ import annotations

"""
Public API surface for image matching.

- match_or_new(existing, query_bgr, *, scale=1.0, threshold=None) -> Optional[int]
- verify_gray(img1_bgr, img2_bgr, *, scale=1.0) -> float
- image_hash64(bgr) -> int
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from app.utils.img_cache import batch_scoped
from app.utils.profiling import profiled

from .alignment import crop_center, ncc
from .caches import verdict_get, verdict_put
from .constants import (
    IMATCH_COVERAGE_DMAX,
    IMATCH_ECC_CC_MIN,
    IMATCH_ECC_EPS,
    IMATCH_ECC_GS,
    IMATCH_ECC_KINDS,
    IMATCH_ECC_MAX_ITER,
    IMATCH_ECC_ROTS,
    IMATCH_ECC_TIERED,
    IMATCH_EDGE_NCC_THR,
    IMATCH_FAST_REJECT,
    IMATCH_FR_NCC_CTR,
    IMATCH_FR_PROF,
    IMATCH_INK_IOU_THR,
    IMATCH_MAX_AR_DIFF,
    IMATCH_MAX_CAND,
    IMATCH_MAX_SHIFT,
    IMATCH_MAX_WDIFF,
    IMATCH_NCC_CENTER_THR,
    IMATCH_NCC_THR,
    IMATCH_PROF_COS_THR,
    IMATCH_PROF_SHIFTED_THR,
    IMATCH_TIER1_GS,
    IMATCH_TIER1_KINDS,
    IMATCH_TIER1_ROTS,
    IMATCH_TIER_UP_CTR_NCC,
    IMATCH_TIER_UP_PROF,
    IMATCH_TOPK_HASH,
    IMATCH_TOPK_PROF,
    IMATCH_USE_AR_FILTER,
    IMATCH_VERIFY_METHOD,
)
from .hashing import resize_gray
from .shortlist import desc_for_query, gather_metrics
from .verify import make_qctx, verify_with_qctx


@profiled("image_matcher.verify_gray")
def verify_gray(
    img1_bgr: np.ndarray, img2_bgr: np.ndarray, *, scale: float = 1.0
) -> float:
    """Diagnostic: NCC of central crops on 256×64 grays."""
    g1 = resize_gray(img1_bgr, 256, 64)
    g2 = resize_gray(img2_bgr, 256, 64)
    return ncc(crop_center(g1, 0.9), crop_center(g2, 0.9))


def _param_sig(rot, gs, kinds) -> tuple:
    """Immutable signature of verification params (for verdict cache key)."""
    return (
        IMATCH_VERIFY_METHOD,
        IMATCH_NCC_THR,
        IMATCH_NCC_CENTER_THR,
        IMATCH_EDGE_NCC_THR,
        0.9,
        IMATCH_MAX_SHIFT,
        IMATCH_ECC_CC_MIN,
        tuple(rot),
        tuple(gs),
        tuple(kinds),
        IMATCH_ECC_MAX_ITER,
        IMATCH_ECC_EPS,
        IMATCH_INK_IOU_THR,
        IMATCH_PROF_COS_THR,
        IMATCH_PROF_SHIFTED_THR,
        IMATCH_COVERAGE_DMAX,
    )


@batch_scoped
@profiled("image_matcher.match_or_new")
def match_or_new(
    existing: Sequence[Tuple[int, str]],
    query_bgr: np.ndarray,
    *,
    scale: float = 1.0,
    threshold: Optional[float] = None,
) -> Optional[int]:
    """
    Main pipeline:
      1) dHash + text-profile → two shortlists.
      2) Merge/rank, then verify per candidate (phase + ECC + text checks).
    Speed-ups only; logic preserved. Returns player_id or None.
    """
    if query_bgr is None or np.asarray(query_bgr).size == 0:
        return None

    q_h, q_prof, q_gray = desc_for_query(query_bgr)
    qH, qW = query_bgr.shape[:2]

    # 1) metrics + geometry gates
    metrics = gather_metrics(
        existing,
        q_hash=q_h,
        q_prof=q_prof,
        qW=qW,
        qH=qH,
        max_wdiff=IMATCH_MAX_WDIFF,
        use_ar_filter=IMATCH_USE_AR_FILTER,
        max_ar_diff=IMATCH_MAX_AR_DIFF,
    )
    if not metrics:
        return None

    # 2) shortlist: ranks by hash & profile
    sorted_by_hash = sorted(metrics, key=lambda x: x[1])  # smaller Hamming first
    sorted_by_prof = sorted(metrics, key=lambda x: -x[2])  # larger cosine first

    top_hash = sorted_by_hash[: max(1, min(IMATCH_TOPK_HASH, len(metrics)))]
    top_prof = sorted_by_prof[: max(1, min(IMATCH_TOPK_PROF, len(metrics)))]

    ranks_h = {pid: r for r, (pid, *_) in enumerate(sorted_by_hash)}
    ranks_p = {pid: r for r, (pid, *_) in enumerate(sorted_by_prof)}
    pool_ids = {pid for pid, *_ in top_hash} | {pid for pid, *_ in top_prof}
    pool = [m for m in metrics if m[0] in pool_ids]
    pool.sort(key=lambda x: (ranks_h.get(x[0], 1e9) + ranks_p.get(x[0], 1e9)))
    pool = pool[:IMATCH_MAX_CAND]

    # 3) query precompute once
    qctx = make_qctx(q_gray, center_frac=0.9)

    # 4) verification params signature (affects verdict cache key)
    full_sig = _param_sig(IMATCH_ECC_ROTS, IMATCH_ECC_GS, IMATCH_ECC_KINDS)
    tier1_sig = _param_sig(
        IMATCH_TIER1_ROTS or (0.0,),
        IMATCH_TIER1_GS or (5,),
        IMATCH_TIER1_KINDS or ("euclidean",),
    )

    # 5) candidate verification loop
    for pid, _ham, prof_cos, _path, cand_gray, cand_hash in pool:

        # optional fast reject
        if IMATCH_FAST_REJECT:
            ctr_ncc = ncc(qctx.ga_center, crop_center(cand_gray, 0.9))
            if (ctr_ncc < IMATCH_FR_NCC_CTR) and (prof_cos < IMATCH_FR_PROF):
                continue

        if not IMATCH_ECC_TIERED:
            key = (int(q_h), int(cand_hash), full_sig)
            vc = verdict_get(key)
            if vc is True:
                return int(pid)
            if vc is not False:
                ok = verify_with_qctx(
                    qctx,
                    cand_gray,
                    method=IMATCH_VERIFY_METHOD,
                    ncc_thr=IMATCH_NCC_THR,
                    ncc_center_thr=IMATCH_NCC_CENTER_THR,
                    edge_ncc_thr=IMATCH_EDGE_NCC_THR,
                    center_frac=0.9,
                    max_shift=IMATCH_MAX_SHIFT,
                    ecc_cc_min=IMATCH_ECC_CC_MIN,
                    ecc_rot_candidates=IMATCH_ECC_ROTS,
                    ecc_gauss_sizes=IMATCH_ECC_GS,
                    ecc_kinds=IMATCH_ECC_KINDS,
                    ecc_max_iter=IMATCH_ECC_MAX_ITER,
                    ecc_eps=IMATCH_ECC_EPS,
                    ink_iou_thr=IMATCH_INK_IOU_THR,
                    profile_cos_thr=IMATCH_PROF_COS_THR,
                    profile_shifted_thr=IMATCH_PROF_SHIFTED_THR,
                    coverage_delta_max=IMATCH_COVERAGE_DMAX,
                )
                verdict_put(key, bool(ok))
                if ok:
                    return int(pid)
            continue

        # tiered
        key1 = (int(q_h), int(cand_hash), tier1_sig)
        vc1 = verdict_get(key1)
        if vc1 is True:
            return int(pid)
        if vc1 is not False:
            ok1 = verify_with_qctx(
                qctx,
                cand_gray,
                method=IMATCH_VERIFY_METHOD,
                ncc_thr=IMATCH_NCC_THR,
                ncc_center_thr=IMATCH_NCC_CENTER_THR,
                edge_ncc_thr=IMATCH_EDGE_NCC_THR,
                center_frac=0.9,
                max_shift=IMATCH_MAX_SHIFT,
                ecc_cc_min=IMATCH_ECC_CC_MIN,
                ecc_rot_candidates=IMATCH_TIER1_ROTS or (0.0,),
                ecc_gauss_sizes=IMATCH_TIER1_GS or (5,),
                ecc_kinds=IMATCH_TIER1_KINDS or ("euclidean",),
                ecc_max_iter=IMATCH_ECC_MAX_ITER,
                ecc_eps=IMATCH_ECC_EPS,
                ink_iou_thr=IMATCH_INK_IOU_THR,
                profile_cos_thr=IMATCH_PROF_COS_THR,
                profile_shifted_thr=IMATCH_PROF_SHIFTED_THR,
                coverage_delta_max=IMATCH_COVERAGE_DMAX,
            )
            verdict_put(key1, bool(ok1))
            if ok1:
                return int(pid)

        ctr_ncc = ncc(qctx.ga_center, crop_center(cand_gray, 0.9))
        if (ctr_ncc >= IMATCH_TIER_UP_CTR_NCC) or (prof_cos >= IMATCH_TIER_UP_PROF):
            key2 = (int(q_h), int(cand_hash), full_sig)
            vc2 = verdict_get(key2)
            if vc2 is True:
                return int(pid)
            if vc2 is not False:
                ok2 = verify_with_qctx(
                    qctx,
                    cand_gray,
                    method=IMATCH_VERIFY_METHOD,
                    ncc_thr=IMATCH_NCC_THR,
                    ncc_center_thr=IMATCH_NCC_CENTER_THR,
                    edge_ncc_thr=IMATCH_EDGE_NCC_THR,
                    center_frac=0.9,
                    max_shift=IMATCH_MAX_SHIFT,
                    ecc_cc_min=IMATCH_ECC_CC_MIN,
                    ecc_rot_candidates=IMATCH_ECC_ROTS,
                    ecc_gauss_sizes=IMATCH_ECC_GS,
                    ecc_kinds=IMATCH_ECC_KINDS,
                    ecc_max_iter=IMATCH_ECC_MAX_ITER,
                    ecc_eps=IMATCH_ECC_EPS,
                    ink_iou_thr=IMATCH_INK_IOU_THR,
                    profile_cos_thr=IMATCH_PROF_COS_THR,
                    profile_shifted_thr=IMATCH_PROF_SHIFTED_THR,
                    coverage_delta_max=IMATCH_COVERAGE_DMAX,
                )
                verdict_put(key2, bool(ok2))
                if ok2:
                    return int(pid)

    return None

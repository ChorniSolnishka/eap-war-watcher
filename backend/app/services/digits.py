"""
Digit Recognition Service

This module provides specialized OCR functionality for extracting numeric values
from image crops. It uses multiple image preprocessing variants and ensemble
voting to improve digit recognition accuracy.

"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

import cv2
import numpy as np

from app.services.ocr_provider import OCRProvider
from app.utils.profiling import profiled

logger = logging.getLogger(__name__)

# Global OCR provider instance
_OCR = OCRProvider()

# Character substitution mapping for common OCR errors
CHAR_SUBSTITUTIONS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "B": "8",
        "ß": "8",
        "g": "9",
        "q": "9",
    }
)

# Valid score range for game damage values
MIN_SCORE = 0
MAX_SCORE = 80


def _prepare_image_variants(bgr: np.ndarray) -> List[np.ndarray]:
    """
    Prepare multiple image variants for ensemble OCR processing.

    Creates different preprocessing variants of the input image to improve
    OCR accuracy through ensemble voting.

    Args:
        bgr: Input image in BGR format

    Returns:
        List of preprocessed image variants
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    base = cv2.bilateralFilter(gray, 5, 30, 30)
    base = cv2.resize(base, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    base = cv2.copyMakeBorder(base, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    variant_base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4))
    enhanced_gray = clahe.apply(gray)

    _, otsu = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variant_otsu = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    variant_otsu_inv = cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR)

    adaptive = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    )
    variant_adaptive = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

    return [variant_base, variant_otsu, variant_otsu_inv, variant_adaptive]


def _extract_candidates(results: List[Tuple[str, float]]) -> List[Tuple[int, float]]:
    """
    Extract numeric candidates from OCR results with confidence weighting.

    Processes OCR results to find valid numeric values within the expected range,
    applying character substitutions and confidence weighting.

    Args:
        results: List of (text, confidence) tuples from OCR

    Returns:
        List of (value, weight) tuples for voting
    """
    candidates: List[Tuple[int, float]] = []
    for text, confidence in results:
        if not text:
            continue
        corrected_text = text.translate(CHAR_SUBSTITUTIONS)
        for match in re.finditer(r"\d+", corrected_text):
            digit_string = match.group(0)
            try:
                value = int(digit_string)
            except ValueError:
                continue
            if MIN_SCORE <= value <= MAX_SCORE:
                base_confidence = max(0.0, min(1.0, confidence))
                length_bonus = 0.15 * min(len(digit_string), 2)
                weight = base_confidence + length_bonus
                candidates.append((value, weight))
    return candidates


def _vote_on_candidates(candidates: List[Tuple[int, float]]) -> int | None:
    """
    Perform weighted voting on numeric candidates.

    Args:
        candidates: List of (value, weight) tuples

    Returns:
        Most likely numeric value or None if no valid candidates
    """
    if not candidates:
        return None
    score_agg: dict[int, float] = {}
    for value, weight in candidates:
        score_agg[value] = score_agg.get(value, 0.0) + weight
    return int(max(score_agg.items(), key=lambda kv: kv[1])[0])


@profiled("digits.read_score_digits")
def read_score_digits(crop_bgr: np.ndarray) -> int | None:
    """
    Extract numeric score from image crop using ensemble OCR.

    Performs OCR on multiple preprocessed variants of the input image and uses
    weighted voting to determine the most likely numeric value.

    Args:
        crop_bgr: Input image crop in BGR format

    Returns:
        Extracted numeric value or None if no valid digits found
    """
    variants = _prepare_image_variants(crop_bgr)

    # IMPORTANT: restrict OCR to digits to reduce noise
    results = _OCR.read_ensemble(crop_bgr, variants=variants, only_digits=True, psm=7)
    if not results:
        results = _OCR.read(crop_bgr, only_digits=True, psm=7)

    candidates = _extract_candidates(results)
    return _vote_on_candidates(candidates)

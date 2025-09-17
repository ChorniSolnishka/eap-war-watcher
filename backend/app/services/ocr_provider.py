"""
OCR Provider Service

This module provides OCR functionality using Tesseract as the primary backend.
It handles text extraction from images with support for digit-only recognition
and ensemble processing of multiple image variants.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try dynamic import so mypy doesn't require stubs for pytesseract.
pytesseract: Any | None
try:
    pytesseract = importlib.import_module("pytesseract")
    _HAVE_TESS = True
except Exception:  # pragma: no cover
    pytesseract = None
    _HAVE_TESS = False


def _configure_tesseract_path() -> None:
    """
    Configure Tesseract executable path for Windows systems.

    Searches for tesseract.exe in common installation directories
    and sets the path if found.
    """
    if not _HAVE_TESS or pytesseract is None:
        return

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.getenv("TESSERACT_PATH", ""),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info("Tesseract configured with path: %s", path)
            break
    else:
        logger.warning("Tesseract executable not found in standard locations")


class OCRProvider:
    """
    OCR Provider using Tesseract as the primary backend.

    Provides text extraction from images with support for:
      - Digit-only recognition
      - Ensemble processing of multiple image variants
      - Configurable PSM (Page Segmentation Mode)
    """

    def __init__(self) -> None:
        """Initialize the OCR provider with Tesseract backend."""
        self._impls: list[tuple[str, object]] = []
        self._kind: str | None = None
        self._impl: object | None = None

        _configure_tesseract_path()
        self._init_implementations()

        if self._kind == "tesseract":
            self._log_tesseract_info()

    def _log_tesseract_info(self) -> None:
        """Log Tesseract configuration and version information."""
        try:
            if _HAVE_TESS and pytesseract is not None:
                logger.info(
                    "Tesseract configured - CMD: %s, Version: %s",
                    getattr(pytesseract.pytesseract, "tesseract_cmd", "Not set"),
                    str(pytesseract.get_tesseract_version()),
                )
        except Exception as e:  # pragma: no cover
            logger.warning("Tesseract present but version check failed: %s", e)

    def info(self) -> dict:
        """
        Get diagnostic information about the active OCR backend.

        Returns:
            Dictionary containing OCR configuration and status information.
        """
        info = {
            "mode": "tesseract",
            "primary": self._kind,
            "implementations": [name for name, _ in self._impls],
            "tesseract_cmd": None,
            "tesseract_version": None,
        }

        if self._kind == "tesseract" and _HAVE_TESS and pytesseract is not None:
            try:
                info["tesseract_cmd"] = getattr(
                    pytesseract.pytesseract, "tesseract_cmd", None
                )
                info["tesseract_version"] = str(pytesseract.get_tesseract_version())
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to get Tesseract info: %s", e)

        return info

    def _init_implementations(self) -> None:
        """Initialize available OCR implementations."""
        if not _HAVE_TESS:
            raise RuntimeError(
                "Tesseract (pytesseract) is not installed or unavailable."
            )

        # Marker for actual call in _run_single
        self._impls = [("tesseract", "tesseract")]
        self._kind, self._impl = self._impls[0]

    def _run_single(
        self,
        kind: str,
        impl: object,  # placeholder; kept for symmetry / future backends
        crop_bgr: np.ndarray,
        *,
        digits_only: bool = True,
        psm: int = 7,
    ) -> List[Tuple[str, float]]:
        """
        Run OCR on a single image using the specified backend.

        Args:
            kind: Backend type identifier.
            impl: Backend implementation object (unused placeholder).
            crop_bgr: Input image in BGR format.
            digits_only: Restrict recognition to digits only.
            psm: Page Segmentation Mode for Tesseract.

        Returns:
            List of (text, confidence) tuples.
        """
        if kind != "tesseract" or not _HAVE_TESS or pytesseract is None:
            return []

        # Convert BGR to RGB for Tesseract
        rgb_image = crop_bgr[:, :, ::-1]

        # Configure Tesseract parameters
        char_whitelist = "0123456789" if digits_only else ""
        config = f"--oem 1 --psm {psm}"
        if char_whitelist:
            config += f" -c tessedit_char_whitelist={char_whitelist}"

        try:
            text = pytesseract.image_to_string(rgb_image, config=config)
        except Exception as e:
            logger.warning("OCR processing failed: %s", e)
            text = ""

        text = (text or "").strip()

        # Heuristic confidence (pytesseract doesn't expose real confidence here)
        return [(text, 0.95)] if text else []

    def read(
        self,
        crop_bgr: np.ndarray,
        *,
        only_digits: bool = False,
        psm: int = 7,
    ) -> List[Tuple[str, float]]:
        """
        Perform single OCR pass on an image.

        Args:
            crop_bgr: Input image in BGR format.
            only_digits: Restrict recognition to digits only.
            psm: Page Segmentation Mode.

        Returns:
            List of (text, confidence) tuples.
        """
        if self._kind is None or self._impl is None:
            return []
        return self._run_single(
            self._kind, self._impl, crop_bgr, digits_only=only_digits, psm=psm
        )

    def read_ensemble(
        self,
        crop_bgr: np.ndarray,
        variants: list[np.ndarray] | None = None,
        *,
        only_digits: bool = False,
        psm: int = 7,
    ) -> List[Tuple[str, float]]:
        """
        Perform OCR on multiple image variants for improved accuracy.

        Processes all provided image variants and returns combined results.
        Early-exits if a confident short numeric string is found.
        """
        images = variants if variants is not None else [crop_bgr]
        results: List[Tuple[str, float]] = []

        def _is_high_confidence_digit(text: str, conf: float) -> bool:
            txt = (text or "").strip()
            return txt.isdigit() and len(txt) <= 2 and conf >= 0.90

        for image in images:
            try:
                result = self._run_single(
                    "tesseract",
                    "tesseract",
                    image,
                    digits_only=only_digits,
                    psm=psm,
                )
            except Exception as e:
                logger.warning("OCR ensemble processing failed for variant: %s", e)
                continue

            results.extend(result)

            # Early exit if high-confidence digit found
            for text, conf in result:
                if _is_high_confidence_digit(text, conf):
                    return results

        return results

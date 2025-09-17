"""
Color Side Detection Service

This module provides functionality to determine the side (ally/enemy) of players
based on color analysis of image crops. It uses LAB color space for accurate
color matching against known enemy color patterns.

"""

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Enemy color reference in RGB format
ENEMY_RGB = np.array([0xDE, 0x8D, 0xA0], dtype=np.uint8)  # #de8da0

# Color distance threshold for enemy detection
ENEMY_COLOR_THRESHOLD = 25.0


def _convert_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB color to LAB color space.

    Args:
        rgb: RGB color array

    Returns:
        LAB color array as float32
    """
    arr = rgb.reshape(1, 1, 3).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)


# Pre-computed enemy color in LAB space for efficiency
ENEMY_LAB = _convert_rgb_to_lab(ENEMY_RGB)


def side_from_crop(bgr: np.ndarray, debug: bool = False) -> Literal["ally", "enemy"]:
    """
    Determine player side (ally/enemy) based on color analysis of image crop.

    Analyzes the image crop for enemy-specific color patterns using LAB color space
    for accurate color matching. If any pixels match the enemy color pattern within
    the threshold, the player is classified as enemy, otherwise as ally.

    Args:
        bgr: Input image crop in BGR format
        debug: Enable debug logging

    Returns:
        "enemy" if enemy color patterns are detected, "ally" otherwise
    """
    # Convert BGR to LAB color space
    lab_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Calculate color distance to enemy reference in LAB space
    color_distance = np.linalg.norm(lab_image - ENEMY_LAB, axis=2)

    # Create mask for enemy-like pixels
    enemy_pixel_mask = color_distance < ENEMY_COLOR_THRESHOLD
    enemy_pixel_count = int(np.count_nonzero(enemy_pixel_mask))

    if debug:
        logger.debug("Color side detection - enemy-like pixels: %d", enemy_pixel_count)

    return "enemy" if enemy_pixel_count > 0 else "ally"

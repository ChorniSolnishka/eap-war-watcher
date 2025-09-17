from __future__ import annotations


def pixels_to_row_height(pixels: int) -> float:
    """Convert pixel height to Excel row height units."""
    return pixels * 0.75


def column_chars_to_pixels(characters: float) -> int:
    """Convert character count to pixel width."""
    return int(characters * 7 + 5)

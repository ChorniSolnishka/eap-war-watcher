from __future__ import annotations

from typing import Any, Dict, Tuple

from xlsxwriter import Workbook


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string."""
    r, g, b = [max(0, min(255, int(v))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def blend_rgb(
    color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float
) -> str:
    """Blend two RGB colors by factor t∈[0..1] and return hex."""
    t = max(0.0, min(1.0, t))
    r = int(round(_lerp(color1[0], color2[0], t)))
    g = int(round(_lerp(color1[1], color2[1], t)))
    b = int(round(_lerp(color1[2], color2[2], t)))
    return rgb_to_hex((r, g, b))


def luminance(hex_color: str) -> float:
    """Relative luminance (0..1) for hex color."""
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def fmt_with_bg(workbook: Workbook, cache: Dict[str, Any], bg_hex: str):
    """
    Create/reuse xlsxwriter Format with given bg color and auto font color.
    Cached by bg_hex.
    """
    if bg_hex in cache:
        return cache[bg_hex]
    font = "#FFFFFF" if luminance(bg_hex) < 0.5 else "#000000"
    fmt = workbook.add_format(
        {
            "border": 1,
            "text_wrap": True,
            "align": "center",
            "valign": "vcenter",
            "bg_color": bg_hex,
            "font_color": font,
        }
    )
    cache[bg_hex] = fmt
    return fmt

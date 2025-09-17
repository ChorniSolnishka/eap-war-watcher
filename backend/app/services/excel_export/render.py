from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from xlsxwriter import Workbook

from .colors import blend_rgb, fmt_with_bg
from .constants import (
    ATTACK_SLOTS,
    BASE_ROW_H_PX,
    DEFENCE_SLOTS,
    GRAD_GRN,
    GRAD_RED,
    NICK_COL_CHARS,
)
from .dims import column_chars_to_pixels, pixels_to_row_height
from .images import calc_image_scale
from .sequence import sum_attacks
from .status import attack_bg_by_status, defense_bg_by_status

DamageRef = Tuple[Optional[int], Optional[int]]


def _format_cell_text(
    damage: Optional[int], *, percent: Optional[float], status: Optional[str]
) -> str:
    """Multiline text shown inside a colored cell."""
    if damage is None:
        return ""
    lines = [str(damage)]
    if percent is not None:
        lines.append(f"{percent:.0f}%")
    if status:
        lines.append(status)
    return "\n".join(lines)


def render_report_sheet(
    *,
    wb: Workbook,
    players_sorted: List[Dict[str, Any]],
    attack_meta: Dict[tuple[int, int], Dict[str, Any]],
    defense_meta: Dict[tuple[int, int], Dict[str, Any]],
) -> None:
    """Render the single 'Отчёт' sheet into the provided workbook."""
    ws = wb.add_worksheet("Отчёт")

    header_fmt = wb.add_format(
        {
            "bold": True,
            "bg_color": "#D9E1F2",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )
    cell_txt = wb.add_format({"border": 1, "align": "center", "valign": "vcenter"})
    cell_sum = wb.add_format(
        {
            "num_format": "0",
            "bold": True,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )
    cell_multiline = wb.add_format(
        {"border": 1, "text_wrap": True, "align": "center", "valign": "vcenter"}
    )

    headers = (
        ["Ник", "Сумма атак"]
        + [f"Атака {i}" for i in range(1, ATTACK_SLOTS + 1)]
        + ["Сумма защит"]
        + [f"Защита {i}" for i in range(1, DEFENCE_SLOTS + 1)]
    )

    ws.set_column(0, 0, NICK_COL_CHARS)
    ws.set_column(1, 1, 12)
    ws.set_column(2, 1 + ATTACK_SLOTS, 18)
    sum_def_col = 2 + ATTACK_SLOTS
    ws.set_column(sum_def_col, sum_def_col, 14)
    def_first_col = sum_def_col + 1
    def_last_col = def_first_col + DEFENCE_SLOTS - 1
    ws.set_column(def_first_col, def_last_col, 18)

    for c, h in enumerate(headers):
        ws.write(0, c, h, header_fmt)

    ws_row_h = pixels_to_row_height(BASE_ROW_H_PX)
    nick_cell_w_px = column_chars_to_pixels(NICK_COL_CHARS)

    fmt_cache_attack: dict[str, Any] = {}
    fmt_cache_def: dict[str, Any] = {}

    row = 1
    for p in players_sorted:
        ws.set_row(row, ws_row_h)

        # Nick image or placeholder
        nick_img = p.get("name_img_path")
        if nick_img and os.path.isfile(nick_img):
            sx, sy, iw, ih = calc_image_scale(
                nick_img, max_w_px=nick_cell_w_px - 6, max_h_px=BASE_ROW_H_PX - 6
            )
            scaled_w = iw * sx
            scaled_h = ih * sy
            x_off = max(0, int((nick_cell_w_px - scaled_w) // 2))
            y_off = max(0, int((BASE_ROW_H_PX - scaled_h) // 2))
            ws.insert_image(
                row,
                0,
                nick_img,
                {
                    "x_offset": x_off,
                    "y_offset": y_off,
                    "x_scale": sx,
                    "y_scale": sy,
                    "object_position": 1,
                },
            )
        else:
            ws.write(row, 0, "(нет изображения)", cell_txt)

        attacks: List[DamageRef] = p.get("attacks", [])[:ATTACK_SLOTS]
        defences: List[DamageRef] = p.get("defences", [])[:DEFENCE_SLOTS]
        pid = int(p["gamer_id"])

        # sum attacks
        ws.write(row, 1, sum_attacks(attacks), cell_sum)

        # attacks cells
        col = 2
        for idx, (dmg, _target_crop) in enumerate(attacks):
            meta = attack_meta.get((pid, idx))
            percent = meta.get("percent") if meta else None
            status = meta.get("status") if meta else None

            if status:
                bg = attack_bg_by_status(status, percent)
                fmt = fmt_with_bg(wb, fmt_cache_attack, bg)
            else:
                # percent-only gradient (fallback)
                if percent is not None:
                    t = max(0.0, min(1.0, float(percent) / 100.0))
                    bg = blend_rgb(GRAD_RED, GRAD_GRN, t)
                    fmt = fmt_with_bg(wb, fmt_cache_attack, bg)
                else:
                    fmt = cell_multiline

            ws.write(
                row, col, _format_cell_text(dmg, percent=percent, status=status), fmt
            )
            col += 1

        # sum defences
        ws.write(row, 2 + ATTACK_SLOTS, sum(int(v or 0) for v, _ in defences), cell_sum)

        # defense cells
        col = 3 + ATTACK_SLOTS
        for idx, (dmg, _attacker_crop) in enumerate(defences):
            meta = defense_meta.get((pid, idx))
            percent = meta.get("percent") if meta else None
            status = meta.get("status") if meta else None

            if status:
                bg = defense_bg_by_status(status)
                fmt = fmt_with_bg(wb, fmt_cache_def, bg)
                ws.write(
                    row,
                    col,
                    _format_cell_text(dmg, percent=percent, status=status),
                    fmt,
                )
            else:
                ws.write(
                    row,
                    col,
                    _format_cell_text(dmg, percent=percent, status=None),
                    cell_multiline,
                )
            col += 1

        row += 1

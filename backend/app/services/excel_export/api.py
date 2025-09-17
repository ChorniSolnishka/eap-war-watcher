from __future__ import annotations

from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

from xlsxwriter import Workbook

from .render import render_report_sheet
from .sequence import build_sequence_context, max_def, sum_attacks

DamageRef = Tuple[Optional[int], Optional[int]]


def _recalc_base_hp(p: Dict[str, Any]) -> Optional[int]:
    """Fallback base HP from defense cells."""
    return max_def(p.get("defences", []))


def build_war_report_xlsx(
    players: List[Dict[str, Any]],
    buffer: BytesIO,
    *,
    my_alliance_name: Optional[str] = None,
    base_hp_by_crop: Optional[
        Dict[int, int]
    ] = None,  # kept for compatibility (not required)
    owner_by_crop: Optional[Dict[int, int]] = None,
    player_crop_id: Optional[Dict[int, int]] = None,
    save_base_hp: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Build the war report into an in-memory xlsx (buffer).
    Public signature preserved for backward compatibility.
    """
    # 1) Recalculate/propagate base HP and optionally persist it
    players_all = list(players)
    for p in players_all:
        new_base = _recalc_base_hp(p)
        p["base_hp"] = new_base
        if new_base and save_base_hp and p.get("gamer_id") is not None:
            try:
                save_base_hp(int(p["gamer_id"]), int(new_base))
            except Exception:
                pass

    # 2) Derive statuses & percents by sequencing
    attack_meta, defense_meta, base_hp_by_player = build_sequence_context(
        players_all,
        owner_by_crop=owner_by_crop or {},
        player_crop_id=player_crop_id or {},
    )

    # 3) Filter and sort for display
    if my_alliance_name:
        players_visible = [
            p for p in players_all if p.get("alliance_name") == my_alliance_name
        ]
    else:
        players_visible = players_all
    players_sorted = sorted(
        players_visible, key=lambda x: sum_attacks(x.get("attacks", [])), reverse=True
    )

    # 4) Render workbook
    wb = Workbook(buffer, {"in_memory": True})
    render_report_sheet(
        wb=wb,
        players_sorted=players_sorted,
        attack_meta=attack_meta,
        defense_meta=defense_meta,
    )
    wb.close()

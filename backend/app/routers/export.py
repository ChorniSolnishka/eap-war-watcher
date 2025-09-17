from __future__ import annotations

"""
Export Router

Endpoints to generate and download the XLSX war report. The report is built
in-memory, saved under `storage/exports/`, and returned as a file download.
"""

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.alliance import Alliance
from app.models.player import Player
from app.models.screenshot_crop import ScreenshotCrop
from app.models.war import War
from app.services.export_xlsx import build_war_report_xlsx

router = APIRouter(prefix="/export", tags=["export"])

ATTACK_SLOTS = 6
DEFENCE_SLOTS = 30


def _max_def_from_player(p: Player) -> Optional[int]:
    """
    Return the player's maximum defence value across all defence slots.

    Args:
        p: Player ORM object.

    Returns:
        The maximum defence value, or None if not available.
    """
    vals = [getattr(p, f"defence_{i}") for i in range(1, DEFENCE_SLOTS + 1)]
    vals = [v for v in vals if isinstance(v, int)]
    return max(vals) if vals else None


@router.get("/wars/{war_id}/xlsx")
def export_war_xlsx(war_id: int, db: Session = Depends(get_db)):
    """
    Build and return the XLSX report for a given war.

    The function collects players/crops, infers base HP where needed,
    updates stored base HP, saves the workbook to disk, and returns it.

    Args:
        war_id: War identifier.
        db: SQLAlchemy session (dependency).

    Returns:
        FileResponse with the generated XLSX.

    Raises:
        HTTPException: If war/alliance/players are missing or report fails.
    """
    war = db.get(War, war_id)
    if not war:
        raise HTTPException(404, "War not found")

    my_alliance: Alliance | None = db.get(Alliance, war.alliance_id)
    if not my_alliance:
        raise HTTPException(404, "Alliance not found for this war")
    my_alliance_name = my_alliance.name

    # Players (ordered by alliance then gamer_id)
    rows = db.execute(
        select(Player, Alliance.name.label("alliance_name"))
        .join(Alliance, Alliance.id == Player.alliance_id)
        .where(Player.war_id == war_id)
        .order_by(Alliance.name, Player.gamer_id)
    ).all()
    if not rows:
        raise HTTPException(404, "No players for this war_id")

    # Map folder path -> player
    path_to_player: Dict[str, Player] = {}
    for r in rows:
        p = cast(Player, r[0])
        if p.name_folder:
            path_to_player[p.name_folder] = p

    # Crop metadata for export
    base_hp_by_crop: Dict[int, int] = {}
    owner_by_crop: Dict[int, int] = {}
    player_crop_id: Dict[int, int] = {}

    crop_rows = db.scalars(
        select(ScreenshotCrop).where(ScreenshotCrop.war_id == war_id)
    ).all()
    for crop in crop_rows:
        owner = path_to_player.get(crop.path)
        if not owner:
            continue

        owner_by_crop[crop.id] = owner.gamer_id

        # Canonical player's crop
        if crop.path == owner.name_folder and owner.gamer_id not in player_crop_id:
            player_crop_id[owner.gamer_id] = crop.id

        hp = owner.base_hp
        if not isinstance(hp, int) or hp <= 0:
            hp = _max_def_from_player(owner) or 0
        if hp > 0:
            base_hp_by_crop[crop.id] = hp

    players_payload: List[dict] = []
    for r in rows:
        p = cast(Player, r[0])
        alliance_name: str = r[1]
        players_payload.append(
            {
                "gamer_id": p.gamer_id,
                "alliance_name": alliance_name,
                "war_id": p.war_id,
                "name_img_path": p.name_folder,
                "base_hp": p.base_hp,
                "attacks": [
                    (getattr(p, f"attack_{i}"), getattr(p, f"attack_{i}_sc_id"))
                    for i in range(1, ATTACK_SLOTS + 1)
                ],
                "defences": [
                    (getattr(p, f"defence_{i}"), getattr(p, f"defence_{i}_sc_id"))
                    for i in range(1, DEFENCE_SLOTS + 1)
                ],
                "attacks_sec": [
                    getattr(p, f"attack_{i}_sec_id") for i in range(1, ATTACK_SLOTS + 1)
                ],
                "defences_sec": [
                    getattr(p, f"defence_{i}_sec_id")
                    for i in range(1, DEFENCE_SLOTS + 1)
                ],
            }
        )

    def save_base_hp(gamer_id: int, base_hp: int) -> None:
        """
        Persist updated base HP for a player if it has changed.

        Args:
            gamer_id: Player's ID.
            base_hp: New base HP value.
        """
        pl = db.get(Player, gamer_id)
        if not pl:
            return
        if pl.base_hp != base_hp:
            pl.base_hp = base_hp
            db.add(pl)

    # Build workbook in-memory
    buf = BytesIO()
    build_war_report_xlsx(
        players_payload,
        buf,
        my_alliance_name=my_alliance_name,
        base_hp_by_crop=base_hp_by_crop,
        owner_by_crop=owner_by_crop,
        player_crop_id=player_crop_id,
        save_base_hp=save_base_hp,
    )
    try:
        db.commit()
    except Exception:
        db.rollback()

    buf.seek(0)
    data = buf.getvalue()
    if not data:
        raise HTTPException(500, "Failed to build XLSX")

    # Save to disk and return via FileResponse (reliable on Windows)
    exports_dir = Path(settings.STORAGE_ROOT) / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    report_path = exports_dir / f"war_{war_id}.xlsx"
    with open(report_path, "wb") as f:
        f.write(data)

    return FileResponse(
        path=str(report_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"war_{war_id}_report.xlsx",
        headers={"Cache-Control": "no-store"},
    )

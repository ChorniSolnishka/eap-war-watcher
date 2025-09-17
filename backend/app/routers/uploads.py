from __future__ import annotations

"""
Combined router with preview URLs (absolute), using debug.png.

Endpoints:
1) POST /uploads/war/screenshots
   - Create war + upload screenshots.
   - Response includes `previews` with absolute URLs to debug.png for
     the latest and previous screenshots (if exists).

2) POST /uploads/{war_id}/screenshots/{screen_id}/process
   - Accepts rows to skip on the latest screenshot (1-based indices).
   - Returns processing result and previews (absolute URLs).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.alliance import Alliance
from app.models.screenshot import Screenshot
from app.models.war import War
from app.schemas.war import WarOut
from app.services.pipelines.uploads_pipeline import UploadsProcessor

router = APIRouter(prefix="/uploads", tags=["uploads"])

# Single shared processor to reuse heavy objects in memory
_processor = UploadsProcessor(storage_root=Path(settings.STORAGE_ROOT))


# ---------------------- helpers ----------------------


def _get_or_create_alliance(db: Session, name: str) -> Alliance:
    """Get existing alliance by name or create a new one."""
    al = db.scalar(select(Alliance).where(Alliance.name == name))
    if al:
        return al
    al = Alliance(name=name)
    db.add(al)
    db.flush()
    return al


def _to_static_url(fs_path: Path, request: Request) -> str:
    """
    Convert a filesystem path within STORAGE_ROOT into an absolute URL under STATIC_URL.
    Example:
      fs_path = /.../storage/debug_seg_dark/w17/s42/debug.png
      -> /static/debug_seg_dark/w17/s42/debug.png -> http://<host>/static/...
    """
    fs_path = fs_path.resolve()
    storage_root = Path(settings.STORAGE_ROOT).resolve()

    try:
        rel = fs_path.relative_to(storage_root)
    except Exception:
        # Not under storage root; return as-is (file://) to avoid crash
        return fs_path.as_uri()

    web_path = f"{settings.STATIC_URL.rstrip('/')}/{str(rel).replace(os.sep, '/')}"
    base = str(request.base_url)  # e.g. "http://localhost:8000/"
    return urljoin(base, web_path.lstrip("/"))


def _ensure_debug_and_preview(
    request: Request,
    war_id: int,
    sc: Screenshot,
) -> Dict:
    """
    Run segmentation to ensure debug.png exists and build a preview dict
    with absolute debug URL.
    """
    # Read image
    img = cv2.imread(sc.folder)
    if img is None:
        raise RuntimeError(f"Screenshot image missing or unreadable: {sc.folder}")

    # Ensure debug is generated
    dbg_dir = _processor.layout.debug_seg_dir(war_id, sc.screen_id)
    rows = _processor.segmenter.run_segmentation_on_roi(img, debug_dir=dbg_dir)
    dbg_png = dbg_dir / "debug.png"

    return {
        "screen_id": sc.screen_id,
        "rows_detected": len(rows),
        "debug_url": _to_static_url(dbg_png, request) if dbg_png.exists() else None,
    }


def _latest_two_screens(db: Session, war_id: int) -> List[Screenshot]:
    """Return the two latest screenshots (latest first)."""
    rows = db.scalars(
        select(Screenshot)
        .where(Screenshot.war_id == war_id)
        .order_by(Screenshot.screen_id.desc())
        .limit(2)
    ).all()
    return rows


# ---------------------- schemas ----------------------


class ProcessPayload(BaseModel):
    """1-based row indices to skip on the latest screenshot."""

    skip_rows: List[int] = []


# ---------------------- routes ----------------------


@router.post("/war/screenshots")
async def create_war_and_upload_screenshots(
    request: Request,
    my_alliance: str = Form(...),
    enemy_alliance: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """
    Create a war and upload screenshots. Returns war block, upload payload,
    and previews (latest & previous) with absolute debug.png URLs.
    """
    # 1) war + alliances
    try:
        my_al = _get_or_create_alliance(db, my_alliance)
        enemy_al = _get_or_create_alliance(db, enemy_alliance)
        war = War(alliance_id=my_al.id, war_enemy_name=enemy_al.name)
        db.add(war)
        db.commit()
        db.refresh(war)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create war: {e}")

    # 2) uploads
    try:
        uploads_payload = _processor.persist_uploads(db=db, war=war, files=files)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}")

    # 3) previews (ensure debug.png for last & prev)
    last_two = _latest_two_screens(db, war.war_id)
    latest_preview = None
    previous_preview = None
    try:
        if len(last_two) >= 1:
            latest_preview = _ensure_debug_and_preview(request, war.war_id, last_two[0])
        if len(last_two) >= 2:
            previous_preview = _ensure_debug_and_preview(
                request, war.war_id, last_two[1]
            )
    except FileNotFoundError:
        # If something is missing, return without previews
        # rather than failing the entire request
        latest_preview = latest_preview or None
        previous_preview = previous_preview or None
    except RuntimeError:
        latest_preview = latest_preview or None
        previous_preview = previous_preview or None

    war_block = WarOut(
        war_id=war.war_id,
        my_alliance_id=my_al.id,
        enemy_alliance_id=enemy_al.id,
    ).model_dump()

    return {
        "war": war_block,
        "uploads": uploads_payload,
        "previews": {
            "latest": latest_preview,
            "previous": previous_preview,
        },
    }


@router.post("/{war_id}/screenshots/{screen_id}/process")
def process_all_after_confirmation_with_previews(
    request: Request,
    war_id: int,
    screen_id: int,  # must be the latest screen id for this war
    payload: ProcessPayload,
    db: Session = Depends(get_db),
):
    """
    Accept rows to skip on the latest screenshot; return processing result
    and previews (absolute debug.png URLs for latest & previous).
    """
    war = db.get(War, war_id)
    if not war:
        raise HTTPException(404, "War not found")

    # determine latest & previous
    last_two = _latest_two_screens(db, war_id)
    if not last_two:
        raise HTTPException(404, "No screenshots for this war")
    last_sc = last_two[0]
    prev_sc: Optional[Screenshot] = last_two[1] if len(last_two) >= 2 else None

    if last_sc.screen_id != screen_id:
        raise HTTPException(
            400,
            f"Provided screen_id is not the latest for this war. "
            f"Latest is {last_sc.screen_id}.",
        )

    # previews, ensure debug
    latest_preview = _ensure_debug_and_preview(request, war_id, last_sc)
    prev_preview = (
        _ensure_debug_and_preview(request, war_id, prev_sc) if prev_sc else None
    )

    # processing
    enemy = db.scalar(select(Alliance).where(Alliance.name == war.war_enemy_name))
    if not enemy:
        raise HTTPException(400, "Enemy alliance not found")

    ally_id = war.alliance_id
    enemy_id = enemy.id
    skip: Set[int] = {int(i) for i in payload.skip_rows if isinstance(i, int)}

    try:
        processing_result = _processor.process_all_after_confirmation(
            db=db,
            war=war,
            last_screen_id=screen_id,
            skip_rows_last=skip,
            ally_id=ally_id,
            enemy_id=enemy_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    return {
        "war_id": war.war_id,
        "previews": {
            "latest": latest_preview,
            "previous": prev_preview,
        },
        "processing": processing_result,
    }

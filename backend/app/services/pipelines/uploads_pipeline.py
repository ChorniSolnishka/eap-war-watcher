from __future__ import annotations

"""
High-level processing pipeline for uploaded war screenshots.

This module keeps FastAPI routers slim by encapsulating:
- filesystem layout helpers,
- segmentation (DarkSegmenter),
- per-row cropping & saving,
- player matching/creation, score assignment, and war counters update,
- "process all in order" orchestration.

"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from sqlalchemy import asc, select
from sqlalchemy.orm import Session

from app.models.player import Player
from app.models.screenshot import Screenshot
from app.models.screenshot_crop import ScreenshotCrop
from app.models.war import War
from app.services.color_side import side_from_crop
from app.services.digits import read_score_digits
from app.services.image_matcher import match_or_new
from app.services.segmentation import DarkSegmenter

# ---------- dataclasses & helpers ----------


@dataclass(frozen=True)
class DirLayout:
    """Resolves storage directories for war assets and debug artifacts."""

    storage_root: Path

    def war_dir(self, war_id: int) -> Path:
        d = self.storage_root / "screens" / f"war_{war_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def crops_dir(self, war_id: int, screen_id: int, row: int) -> Path:
        d = (
            self.storage_root
            / "crops"
            / f"war_{war_id}"
            / f"screen_{screen_id}"
            / f"row_{row}"
        )
        d.mkdir(parents=True, exist_ok=True)
        return d

    def debug_seg_dir(self, war_id: int, screen_id: int) -> Path:
        d = self.storage_root / "debug_seg_dark" / f"w{war_id}" / f"s{screen_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def debug_by_id_root(self, war_id: int) -> Path:
        d = self.storage_root / "debug_by_id" / f"w{war_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def debug_digits_root(self, war_id: int) -> Path:
        d = self.storage_root / "debug_digits" / f"w{war_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass(frozen=True)
class ProcessedScreen:
    """Summary info for a processed screenshot."""

    screen_id: int
    rows_detected: int
    skipped_rows: Tuple[int, ...]


# ---------- core processor ----------


class UploadsProcessor:
    """
    Stateful processor for uploads:
    - keeps a single DarkSegmenter instance (mid-column memory across frames),
    - exposes small methods the router can call.
    """

    ATTACK_SLOTS = 6
    DEFENCE_SLOTS = 30

    def __init__(self, *, storage_root: Path) -> None:
        self.layout = DirLayout(storage_root=storage_root)
        self.segmenter = DarkSegmenter()

    # ---- public API for the router ----

    def persist_uploads(
        self,
        *,
        db: Session,
        war: War,
        files: Sequence,  # List[UploadFile]-like (has .filename and .file)
    ) -> Dict:
        """
        Save images to disk, create Screenshot rows.
        Run segmentation to generate debug previews.
        Returns a JSON-serializable summary.
        """
        saved: List[Dict] = []
        latest_preview: Optional[Dict] = None

        for uf in files:
            screen_dir = self.layout.war_dir(war.war_id)
            dst = screen_dir / uf.filename
            with dst.open("wb") as f:
                shutil.copyfileobj(uf.file, f)

            sc = Screenshot(war_id=war.war_id, folder=str(dst))
            db.add(sc)
            db.flush()  # screen_id is assigned

            img = cv2.imread(str(dst))
            if img is None:
                continue

            rows = self.segmenter.run_segmentation_on_roi(
                img, debug_dir=self.layout.debug_seg_dir(war.war_id, sc.screen_id)
            )
            dbg_png = self.layout.debug_seg_dir(war.war_id, sc.screen_id) / "debug.png"

            latest_preview = {
                "screen_id": sc.screen_id,
                "path": str(dst),
                "rows_detected": len(rows),
                "debug_overview": str(dbg_png) if dbg_png.exists() else None,
            }
            saved.append({"screen_id": sc.screen_id, "path": str(dst)})

        db.commit()
        return {
            "war_id": war.war_id,
            "uploaded": len(saved),
            "screens": saved,
            "latest_screen": latest_preview,
        }

    def latest_screen_info(self, *, db: Session, war_id: int) -> Dict:
        """Return summary for the latest screenshot of a war
        With on-the-fly segmentation preview."""
        sc = db.scalar(
            select(Screenshot)
            .where(Screenshot.war_id == war_id)
            .order_by(Screenshot.screen_id.desc())
        )
        if not sc:
            raise FileNotFoundError("No screenshots for this war")

        img = cv2.imread(sc.folder)
        if img is None:
            raise RuntimeError("Latest screenshot image missing or unreadable")

        rows = self.segmenter.run_segmentation_on_roi(
            img, debug_dir=self.layout.debug_seg_dir(war_id, sc.screen_id)
        )
        dbg_png = self.layout.debug_seg_dir(war_id, sc.screen_id) / "debug.png"

        return {
            "war_id": war_id,
            "screen_id": sc.screen_id,
            "path": sc.folder,
            "rows_detected": len(rows),
            "row_indices": list(range(1, len(rows) + 1)),
            "debug_overview": str(dbg_png) if dbg_png.exists() else None,
        }

    def process_all_after_confirmation(
        self,
        *,
        db: Session,
        war: War,
        last_screen_id: int,
        skip_rows_last: Set[int],
        ally_id: int,
        enemy_id: int,
    ) -> Dict:
        """
        Process ALL screenshots in ascending order,
        skipping given rows on the last screen.
        Updates global attack/defence sequence counters in `war`.
        """
        summary = self._process_all_screens_in_order(
            db=db,
            war=war,
            last_screen_id=last_screen_id,
            skip_rows_last=skip_rows_last,
            ally_id=ally_id,
            enemy_id=enemy_id,
        )
        return {
            "war_id": war.war_id,
            "last_screen_id": last_screen_id,
            "attack_seq_counter_final": war.attack_seq_counter,
            "defence_seq_counter_final": war.defence_seq_counter,
            "summary": summary,
        }

    # ---- internal helpers ----

    @staticmethod
    def _roi(img: np.ndarray) -> np.ndarray:
        """Identity ROI (hook for future cropping if needed)."""
        return img

    def _save_unique(self, dst_dir: Path, base_name: str, img_bgr: np.ndarray) -> Path:
        dst_dir.mkdir(parents=True, exist_ok=True)
        out = dst_dir / f"{base_name}.png"
        if out.exists():
            k = 1
            while True:
                out2 = dst_dir / f"{base_name}__{k}.png"
                if not out2.exists():
                    out = out2
                    break
                k += 1
        cv2.imwrite(str(out), img_bgr)
        return out

    def _save_player_debug_crop(
        self,
        *,
        war_id: int,
        gamer_id: int,
        screen_id: int,
        row_idx: int,
        side: str,
        crop_bgr: np.ndarray,
    ) -> None:
        root = self.layout.debug_by_id_root(war_id)
        dst_dir = root / f"id_{gamer_id}"
        base = f"s{screen_id}_r{row_idx}_{side}"
        self._save_unique(dst_dir, base, crop_bgr)

    def _save_digit_debug(
        self,
        *,
        war_id: int,
        val: Optional[int],
        screen_id: int,
        row_idx: int,
        crop_bgr: np.ndarray,
    ) -> None:
        root = self.layout.debug_digits_root(war_id)
        prefix = "NA" if (val is None) else str(val)
        base = f"{prefix}__s{screen_id}_r{row_idx}"
        self._save_unique(root, base, crop_bgr)

    # ---- DB helpers ----

    @staticmethod
    def _assign_next(
        player: Player,
        *,
        is_attack: bool,
        score: Optional[int],
        sc_id: int,
        seq_id: Optional[int],
    ) -> bool:
        """
        Find the first empty slot; set the score (or 0), *_sc_id and *_sec_id.
        Returns True if a slot was filled.
        """
        if is_attack:
            for i in range(1, UploadsProcessor.ATTACK_SLOTS + 1):
                val = getattr(player, f"attack_{i}")
                if val is None:
                    setattr(player, f"attack_{i}", score if score is not None else 0)
                    setattr(player, f"attack_{i}_sc_id", sc_id)
                    setattr(player, f"attack_{i}_sec_id", seq_id)
                    return True
        else:
            for i in range(1, UploadsProcessor.DEFENCE_SLOTS + 1):
                val = getattr(player, f"defence_{i}")
                if val is None:
                    setattr(player, f"defence_{i}", score if score is not None else 0)
                    setattr(player, f"defence_{i}_sc_id", sc_id)
                    setattr(player, f"defence_{i}_sec_id", seq_id)
                    return True
        return False

    @staticmethod
    def _get_canonical_crop_id_for_player(db: Session, player: Player) -> Optional[int]:
        if not player or not player.name_folder:
            return None
        row = db.scalar(
            select(ScreenshotCrop).where(ScreenshotCrop.path == player.name_folder)
        )
        return row.id if row else None

    def _create_crop_row(
        self,
        db: Session,
        *,
        war_id: int,
        screen_id: int,
        row_idx: int,
        kind: str,
        path: Path,
    ) -> int:
        row = ScreenshotCrop(
            war_id=war_id,
            screen_id=screen_id,
            row_idx=row_idx,
            kind=kind,
            path=str(path),
        )
        db.add(row)
        db.flush()
        return row.id

    # ---- core per-screen processing ----

    def _process_screen_with_rows(
        self,
        *,
        db: Session,
        war: War,
        war_id: int,
        screen_id: int,
        rows: List[dict],
        ally_id: int,
        enemy_id: int,
        skip_rows: Set[int],
    ) -> None:
        existing_by_alliance: Dict[int, List[Tuple[int, str]]] = {
            ally_id: [
                (p.gamer_id, p.name_folder)
                for p in db.scalars(
                    select(Player).where(
                        Player.war_id == war_id, Player.alliance_id == ally_id
                    )
                ).all()
            ],
            enemy_id: [
                (p.gamer_id, p.name_folder)
                for p in db.scalars(
                    select(Player).where(
                        Player.war_id == war_id, Player.alliance_id == enemy_id
                    )
                ).all()
            ],
        }

        for idx, r in enumerate(rows, start=1):
            if idx in skip_rows:
                continue

            crops_path = self.layout.crops_dir(war_id, screen_id, idx)
            left_path = crops_path / "left.png"
            mid_path = crops_path / "score.png"
            right_path = crops_path / "right.png"

            cv2.imwrite(str(left_path), r["left"])
            cv2.imwrite(str(mid_path), r["mid"])
            cv2.imwrite(str(right_path), r["right"])

            left_side = side_from_crop(r["left"])
            right_side = side_from_crop(r["right"])
            score = read_score_digits(r["mid"])
            try:
                self._save_digit_debug(
                    war_id=war_id,
                    val=score,
                    screen_id=screen_id,
                    row_idx=idx,
                    crop_bgr=r["mid"],
                )
            except Exception:
                pass

            # LEFT (attacker)
            l_alliance = ally_id if left_side == "ally" else enemy_id
            l_match_id = match_or_new(existing_by_alliance[l_alliance], r["left"])
            if l_match_id is None:
                lp = Player(
                    name_folder=str(left_path), alliance_id=l_alliance, war_id=war_id
                )
                db.add(lp)
                db.flush()
                l_crop_id = self._create_crop_row(
                    db,
                    war_id=war_id,
                    screen_id=screen_id,
                    row_idx=idx,
                    kind="left",
                    path=left_path,
                )
                existing_by_alliance[l_alliance].append((lp.gamer_id, lp.name_folder))
                l_player = lp
                try:
                    self._save_player_debug_crop(
                        war_id=war_id,
                        gamer_id=lp.gamer_id,
                        screen_id=screen_id,
                        row_idx=idx,
                        side="left",
                        crop_bgr=r["left"],
                    )
                except Exception:
                    pass
            else:
                l_player = db.get(Player, l_match_id)
                canon_id = self._get_canonical_crop_id_for_player(db, l_player)
                if canon_id is None:
                    canon_id = self._create_crop_row(
                        db,
                        war_id=war_id,
                        screen_id=screen_id,
                        row_idx=idx,
                        kind="left",
                        path=left_path,
                    )
                l_crop_id = canon_id
                try:
                    self._save_player_debug_crop(
                        war_id=war_id,
                        gamer_id=l_player.gamer_id,
                        screen_id=screen_id,
                        row_idx=idx,
                        side="left",
                        crop_bgr=r["left"],
                    )
                except Exception:
                    pass

            # RIGHT (defender)
            r_alliance = ally_id if right_side == "ally" else enemy_id
            r_match_id = match_or_new(existing_by_alliance[r_alliance], r["right"])
            if r_match_id is None:
                rp = Player(
                    name_folder=str(right_path), alliance_id=r_alliance, war_id=war_id
                )
                db.add(rp)
                db.flush()
                r_crop_id = self._create_crop_row(
                    db,
                    war_id=war_id,
                    screen_id=screen_id,
                    row_idx=idx,
                    kind="right",
                    path=right_path,
                )
                existing_by_alliance[r_alliance].append((rp.gamer_id, rp.name_folder))
                r_player = rp
                try:
                    self._save_player_debug_crop(
                        war_id=war_id,
                        gamer_id=rp.gamer_id,
                        screen_id=screen_id,
                        row_idx=idx,
                        side="right",
                        crop_bgr=r["right"],
                    )
                except Exception:
                    pass
            else:
                r_player = db.get(Player, r_match_id)
                canon_id = self._get_canonical_crop_id_for_player(db, r_player)
                if canon_id is None:
                    canon_id = self._create_crop_row(
                        db,
                        war_id=war_id,
                        screen_id=screen_id,
                        row_idx=idx,
                        kind="right",
                        path=right_path,
                    )
                r_crop_id = canon_id
                try:
                    self._save_player_debug_crop(
                        war_id=war_id,
                        gamer_id=r_player.gamer_id,
                        screen_id=screen_id,
                        row_idx=idx,
                        side="right",
                        crop_bgr=r["right"],
                    )
                except Exception:
                    pass

            # --- assign order and bump war counters ---
            # attacker (left side)
            next_attack_seq = war.attack_seq_counter + 1
            if self._assign_next(
                l_player,
                is_attack=True,
                score=score,
                sc_id=r_crop_id,
                seq_id=next_attack_seq,
            ):
                war.attack_seq_counter = next_attack_seq

            # defender (right side)
            next_defence_seq = war.defence_seq_counter + 1
            if self._assign_next(
                r_player,
                is_attack=False,
                score=score,
                sc_id=l_crop_id,
                seq_id=next_defence_seq,
            ):
                war.defence_seq_counter = next_defence_seq

    def _process_all_screens_in_order(
        self,
        *,
        db: Session,
        war: War,
        last_screen_id: int,
        skip_rows_last: Set[int],
        ally_id: int,
        enemy_id: int,
    ) -> List[Dict]:
        """Iterate all screenshots for war (ascending by screen_id) and process."""
        screens = db.scalars(
            select(Screenshot)
            .where(Screenshot.war_id == war.war_id)
            .order_by(asc(Screenshot.screen_id))
        ).all()
        if not screens:
            raise FileNotFoundError("No screenshots to process")

        processed: List[Dict] = []
        for sc in screens:
            img = cv2.imread(sc.folder)
            if img is None:
                continue

            rows = self.segmenter.run_segmentation_on_roi(
                self._roi(img),
                debug_dir=self.layout.debug_seg_dir(war.war_id, sc.screen_id),
            )
            skip = skip_rows_last if sc.screen_id == last_screen_id else set()

            self._process_screen_with_rows(
                db=db,
                war=war,
                war_id=war.war_id,
                screen_id=sc.screen_id,
                rows=rows,
                ally_id=ally_id,
                enemy_id=enemy_id,
                skip_rows=skip,
            )
            processed.append(
                {
                    "screen_id": sc.screen_id,
                    "rows": len(rows),
                    "skipped": sorted(list(skip)),
                }
            )

        db.commit()
        return processed

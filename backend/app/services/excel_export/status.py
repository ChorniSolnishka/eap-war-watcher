from __future__ import annotations

from typing import Optional

from .colors import blend_rgb
from .constants import (
    GRAD_GRN,
    GRAD_RED,
    GRAY,
    GREEN_LIGHT,
    GREEN_SOFT,
    GREEN_STRONG,
    ORANGE,
    ORANGE_LIGHT,
    RED_STRONG,
)


def attack_status_step(base_hp: int, cur_hp: int, dmg: int) -> tuple[int, str]:
    """
    Compute next HP and status of a single attack step.
    Statuses: "шотнул" / "добил" / "не шотнул" / "не добил"
    """
    if base_hp <= 0:
        return cur_hp, ""
    dmg = max(0, int(dmg))
    remained = max(0, cur_hp - dmg)
    if cur_hp == base_hp and remained == 0:
        return base_hp, "шотнул"
    if cur_hp < base_hp and remained == 0:
        return base_hp, "добил"
    if cur_hp == base_hp and remained > 0:
        return remained, "не шотнул"
    return remained, "не добил"


def attack_bg_by_status(status: str, percent: Optional[float]) -> str:
    if status == "шотнул":
        return GREEN_STRONG
    if status == "не шотнул":
        p = max(0.0, min(100.0, float(percent or 0.0))) / 100.0
        return blend_rgb(GRAD_RED, GRAD_GRN, p)
    if status == "добил":
        p = float(percent or 0.0)
        if p >= 80.0:
            return GREEN_SOFT
        if p <= 20.0:
            return ORANGE_LIGHT
        return GREEN_LIGHT
    if status == "не добил":
        return GRAY
    return GRAY


def defense_bg_by_status(status: str) -> str:
    if status == "шотнул":
        return RED_STRONG
    if status == "добил":
        return ORANGE
    return GREEN_SOFT

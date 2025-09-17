from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .constants import ATTACK_SLOTS, DEFENCE_SLOTS
from .status import attack_status_step

DamageRef = Tuple[Optional[int], Optional[int]]  # (damage, crop_id)


def max_def(vals: List[DamageRef]) -> Optional[int]:
    """Max defense value or None."""
    return max([int(v or 0) for v, _ in vals], default=0) or None


def sum_attacks(vals: List[DamageRef]) -> int:
    """Sum of attack damages (None treated as 0)."""
    return sum(int(v or 0) for v, _ in vals)


def build_sequence_context(
    players: List[Dict],
    *,
    owner_by_crop: Dict[int, int] | None,
    player_crop_id: Dict[int, int] | None,
) -> tuple[
    Dict[
        tuple[int, int], Dict[str, object]
    ],  # attack_meta[(pid, slot)] -> {percent, status}
    Dict[
        tuple[int, int], Dict[str, object]
    ],  # defense_meta[(pid, slot)] -> {percent, status}
    Dict[int, int],  # base_hp_by_player
]:
    """
    Replay attacks in ascending sec_id to derive statuses and percents for both
    attack cells and mirrored defense cells.
    """
    owner_by_crop = owner_by_crop or {}
    player_crop_id = player_crop_id or {}

    # base / current HP
    base_hp_by_player: Dict[int, int] = {}
    for p in players:
        pid = int(p["gamer_id"])
        base = p.get("base_hp")
        if base is None:
            base = max_def(p.get("defences", []))
        base_hp_by_player[pid] = int(base or 0)
    current_hp = dict(base_hp_by_player)

    # index defense slots by (slot_index, *_sc_id)
    def_sc_index: Dict[int, List[Tuple[int, Optional[int]]]] = {}
    consumed_def_slot: Dict[int, set] = {}
    for p in players:
        pid = int(p["gamer_id"])
        ds: List[DamageRef] = p.get("defences", [])[:DEFENCE_SLOTS]
        def_sc_index[pid] = [(j, ds[j][1]) for j in range(len(ds))]
        consumed_def_slot[pid] = set()

    # group attacks by sec_id
    sec_events: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for p in players:
        pid = int(p["gamer_id"])
        attacks: List[DamageRef] = p.get("attacks", [])[:ATTACK_SLOTS]
        secs: List[Optional[int]] = p.get("attacks_sec", [None] * ATTACK_SLOTS)
        for idx in range(min(len(attacks), ATTACK_SLOTS)):
            dmg, target_crop_id = attacks[idx]
            sec = secs[idx] if idx < len(secs) else None
            if not isinstance(sec, int):
                continue
            if dmg is None or target_crop_id is None:
                continue
            defender_pid = owner_by_crop.get(int(target_crop_id))
            if defender_pid is None:
                continue
            sec_events.setdefault(sec, []).append(
                (pid, idx, int(defender_pid), int(dmg))
            )

    attack_meta: Dict[tuple[int, int], Dict[str, object]] = {}
    defense_meta: Dict[tuple[int, int], Dict[str, object]] = {}

    for sec in sorted(sec_events.keys()):
        group = sorted(sec_events[sec], key=lambda t: (t[0], t[1]))
        for attacker_pid, attacker_slot_idx, defender_pid, dmg in group:
            base_def = base_hp_by_player.get(defender_pid, 0)
            cur_def = current_hp.get(defender_pid, base_def)

            next_hp, status = attack_status_step(base_def, cur_def, dmg)
            current_hp[defender_pid] = next_hp

            pct_attack = (
                (float(dmg) / float(base_def) * 100.0) if base_def > 0 else None
            )
            attack_meta[(attacker_pid, attacker_slot_idx)] = {
                "status": status,
                "percent": pct_attack,
            }

            # mirror status into defender slot matched by attacker's canonical crop id
            atk_crop_id = player_crop_id.get(attacker_pid)
            if atk_crop_id is not None:
                for j, sc_id in def_sc_index.get(defender_pid, []):
                    if j in consumed_def_slot[defender_pid]:
                        continue
                    if sc_id == atk_crop_id:
                        consumed_def_slot[defender_pid].add(j)
                        base_att = base_hp_by_player.get(attacker_pid, 0)
                        pct_def = (
                            (float(dmg) / float(base_att) * 100.0)
                            if base_att > 0
                            else None
                        )
                        defense_meta[(defender_pid, j)] = {
                            "status": status,
                            "percent": pct_def,
                        }
                        break

    return attack_meta, defense_meta, base_hp_by_player

# mechanics.py
# Phase-by-phase evaluator for 40k-8e style attacks using simple dice lists.
# Inputs: lists of dice you captured via your app (per phase).
# Output: hits, wounds, failed saves, damage.

from dataclasses import dataclass
from typing import List, Dict, Optional

# ---------- core thresholds ----------
def to_wound_threshold(strength: int, toughness: int) -> int:
    if strength >= 2 * toughness: return 2
    if strength > toughness:      return 3
    if strength == toughness:     return 4
    if strength * 2 <= toughness: return 6
    return 5

def save_threshold(armor_save: int, ap: int, cover_mod: int = 0, invuln: Optional[int] = None) -> int:
    # AP worsens the save (higher target). Cover improves (lower target).
    normal = max(2, min(6, armor_save - cover_mod + max(0, ap)))
    if invuln is not None:
        return min(normal, invuln)
    return normal

# ---------- helpers ----------
def count_successes(dice: List[int], target: int, mod: int = 0) -> int:
    t = max(2, min(6, target - mod))
    return sum(1 for d in dice if d >= t)

def apply_reroll_ones(dice: List[int], rerolled: List[int]) -> List[int]:
    # You must provide the numbers from re-rolled dice (your app can capture a second image for rerolls).
    out = []
    rr_iter = iter(rerolled)
    for d in dice:
        if d == 1:
            try:
                out.append(next(rr_iter))
            except StopIteration:
                out.append(d)  # no replacement provided
        else:
            out.append(d)
    return out

# ---------- configuration ----------
@dataclass
class Weapon:
    name: str
    type: str
    attacks_per_model: int
    rapid_fire_multiplier: int
    strength: int
    ap: int
    damage: int

@dataclass
class UnitConfig:
    unit: str
    models: int
    bs: int
    weapons: List[Weapon]
    abilities: Dict  # {"reroll_hit_ones":bool, "hit_mod":int, "wound_mod":int}

# ---------- main API ----------
def evaluate_attack_sequence(
    cfg: UnitConfig,
    weapon_idx: int,
    target_toughness: int,
    target_save: int,
    *,
    cover_mod: int = 0,
    invuln: Optional[int] = None,
    # phase dice (each provided by your app)
    hit_rolls: List[int],
    hit_rerolls_ones: Optional[List[int]] = None,
    wound_rolls: List[int] = None,
    save_rolls: List[int] = None,
) -> Dict:
    w = cfg.weapons[weapon_idx]

    # hits
    hr = hit_rolls[:]  # copy
    if cfg.abilities.get("reroll_hit_ones", False) and hit_rerolls_ones is not None:
        hr = apply_reroll_ones(hr, hit_rerolls_ones)
    hits = count_successes(hr, cfg.bs, cfg.abilities.get("hit_mod", 0))

    # wounds
    tw = to_wound_threshold(w.strength, target_toughness)
    wounds = count_successes(wound_rolls or [], tw, cfg.abilities.get("wound_mod", 0))

    # failed saves (i.e., unsaved wounds)
    sv_t = save_threshold(target_save, w.ap, cover_mod, invuln)
    failed_saves = 0
    if save_rolls is not None:
        # saves succeed on >= sv_t; we need the number that FAIL
        succ = sum(1 for d in save_rolls if d >= sv_t)
        # clamp by wounds to avoid overcounting
        failed_saves = max(0, min(wounds, (len(save_rolls) - succ)))

    # damage = unsaved * damage
    # If you roll damage per shot, extend this: pass damage_rolls and compute per-hit
    damage = failed_saves * max(1, w.damage)

    return {
        "unit": cfg.unit,
        "weapon": w.name,
        "bs_target": cfg.bs,
        "hits": hits,
        "wound_target": tw,
        "wounds": wounds,
        "save_target": sv_t,
        "failed_saves": failed_saves,
        "damage": damage
    }

# ---------- loader ----------
import json
def load_unit_config(path: str) -> UnitConfig:
    j = json.load(open(path, "r"))
    weapons = [Weapon(**w) for w in j["weapons"]]
    return UnitConfig(
        unit=j["unit"],
        models=int(j["models"]),
        bs=int(j["bs"]),
        weapons=weapons,
        abilities=j.get("abilities", {})
    )
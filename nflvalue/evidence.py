"""Phase 8.2 -- evidence strength for every driver on a card.

The governing idea: **calibrate the reader's confidence, do not persuade.**
This tool has not proven edge (real-closing-line backtests lose, resolved CLV
is n=0, the kill-check exists to say NO-GO), so an interface that makes a thin
case feel overwhelming is the most expensive bug it could ship. Everything here
exists to make a 70%-on-n=20 claim LOOK weaker than a 62%-on-n=800 claim.

Grades
------
``strong``      passes the ACCURACY_PROTOCOL gate (n>=100 exposed and >=100
                matched control, interval excludes the null) AND has a
                season-forward replication.
``moderate``    passes the gate, no replication yet.
``thin``        n below the gate, or the interval spans the null.
``unproven``    never gated -- measured at approximately zero, or never tested.

``thin`` and ``unproven`` are not softer words for "probably fine". They are
rendered at the same visual weight as ``strong`` (see the renderer), because a
reader who skims past uncertainty is exactly the failure mode.

Intervals
---------
Two kinds of claim need two kinds of interval, and conflating them is a
statistics error dressed up as a UI decision:

* **Proportions** (hit rates, band accuracy) -> Wilson score interval. Reused
  from :mod:`nflvalue.top_bets`, not reimplemented.
* **Continuous effects** (a 0.734 volume multiplier, a 9.1 yards/target level)
  -> bootstrap percentile interval, resampled in TEAM-SEASON BLOCKS because
  player-weeks within a team-season are not independent (ACCURACY_PROTOCOL
  requires clustering uncertainty by team-season).

Where an interval cannot be honestly recomputed
-----------------------------------------------
Three shipped constants (the absence matrix, the backup-QB efficiency
multiplier, the reallocation efficiency slope) were measured in earlier phases
against raw outcome data, and the cached artifacts retain the point estimate
and n but NOT the per-observation residuals needed to resample them. They are
therefore published with ``interval=None`` and
``interval_status="not_recomputed"``, which the renderer must show as an
explicit disclosure.

That is a deliberate deviation from "every figure carries n AND a CI": the
alternative was to invent an interval, and a fabricated confidence bound on a
money-adjacent card is worse than an admitted gap. Recomputing them from
``historical/`` is real work and is registered as a follow-up, not quietly
skipped.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from .top_bets import wilson_lower_bound

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_PATH = os.path.join(ROOT, "data", "evidence_reference.json")

#: ACCURACY_PROTOCOL matched-control gate.
GATE_N = 100
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260718          # determinism: same inputs -> same interval

GRADES = ("strong", "moderate", "thin", "unproven")


@dataclass
class Evidence:
    """Empirical support for one driver."""
    claim: str
    n: Optional[int]
    effect: Optional[float]
    effect_unit: str
    interval: Optional[List[float]]        # [lo, hi]
    interval_kind: Optional[str]           # "wilson" | "bootstrap_team_season"
    interval_status: str                   # "computed" | "not_recomputed" | "not_applicable"
    grade: str
    source: str
    replicated: bool = False
    note: Optional[str] = None

    def __post_init__(self):
        if self.grade not in GRADES:
            raise ValueError(f"unknown evidence grade {self.grade!r}")

    @property
    def spans_null(self) -> Optional[bool]:
        """Does the interval include 'no effect'? None when unknown.

        The null is 1.0 for a multiplier and 0.0 for a difference, so the
        caller states which via ``effect_unit``.
        """
        if not self.interval:
            return None
        null = 1.0 if self.effect_unit == "multiplier" else 0.0
        return self.interval[0] <= null <= self.interval[1]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["spans_null"] = self.spans_null
        return d


def grade_from(n: Optional[int], interval: Optional[Sequence[float]],
               effect_unit: str, replicated: bool = False,
               measured_zero: bool = False) -> str:
    """Assign a grade from sample size and interval. No judgement calls."""
    if measured_zero:
        return "unproven"
    if n is None or n < GATE_N:
        return "thin"
    if interval:
        null = 1.0 if effect_unit == "multiplier" else 0.0
        if interval[0] <= null <= interval[1]:
            return "thin"        # measured, but indistinguishable from nothing
        return "strong" if replicated else "moderate"
    # n clears the gate but no interval exists to rule out the null.
    return "moderate" if replicated else "thin"


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #
def wilson_interval(wins: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """Two-sided Wilson score interval for a proportion."""
    if n <= 0:
        return None
    lo = wilson_lower_bound(wins, n, z)
    phat = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    hi = min(1.0, (centre + margin) / denom)
    return [round(lo, 4), round(hi, 4)]


def block_bootstrap_mean(values: Sequence[float], blocks: Sequence,
                         draws: int = BOOTSTRAP_DRAWS,
                         seed: int = BOOTSTRAP_SEED,
                         alpha: float = 0.05) -> Optional[List[float]]:
    """Percentile CI for a mean, resampling whole TEAM-SEASON blocks.

    Player-weeks inside a team-season share a scheme, a quarterback and an
    injury history, so treating them as independent draws would shrink the
    interval by roughly sqrt(block size) and make every driver look better
    evidenced than it is. Resampling blocks keeps the dependence.

    Deterministic: seeded, and the block order is sorted before sampling.
    """
    if not values or len(values) != len(blocks):
        return None
    grouped: Dict = {}
    for v, b in zip(values, blocks):
        if v is None or not math.isfinite(float(v)):
            continue
        grouped.setdefault(b, []).append(float(v))
    keys = sorted(grouped, key=str)
    if len(keys) < 2:
        return None

    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(draws):
        pool: List[float] = []
        for _ in range(len(keys)):
            pool.extend(grouped[keys[rng.randrange(len(keys))]])
        if pool:
            means.append(sum(pool) / len(pool))
    if not means:
        return None
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return [round(lo, 4), round(hi, 4)]


# --------------------------------------------------------------------------- #
# Position/market reference levels -- what makes a LEVEL directional
# --------------------------------------------------------------------------- #
def build_reference_levels(frame, through_season: Optional[int] = None) -> Dict:
    """Measured league reference for each (market, position) level.

    A catch rate of 0.568 is not an argument for the under until you know the
    league's WR catch rate. This computes those references from the candidate
    corpus so a level can carry a direction that is *measured* rather than
    asserted -- and, crucially, so a card can show "below the position
    average" as a genuine counter-argument.

    ``through_season`` restricts to strictly-prior seasons. Callers explaining
    a live week must pass it: a reference computed including the week being
    explained would be a leak, even though it only affects presentation.
    """
    import pandas as pd  # local: keeps the module importable without pandas

    df = frame
    if through_season is not None:
        df = df[df["season"] < through_season]
    if df.empty:
        return {"references": {}, "through_season": through_season, "n": 0}

    pos_cols = [c for c in df.columns if c.startswith("pos_")]

    def _pos(row) -> str:
        for c in pos_cols:
            if row.get(c):
                return c[4:]
        return "UNK"

    out: Dict[str, Dict] = {}
    for (market,), grp in df.groupby(["market"]):
        for pos in ("WR", "TE", "RB", "QB"):
            col = f"pos_{pos}"
            if col not in grp.columns:
                continue
            sub = grp[grp[col] == 1]
            if len(sub) < GATE_N:
                continue
            for field_name in ("proj_efficiency", "proj_volume"):
                if field_name not in sub.columns:
                    continue
                vals = sub[field_name].dropna()
                if len(vals) < GATE_N:
                    continue
                blocks = sub.loc[vals.index].apply(
                    lambda r: f"{r['season']}", axis=1).tolist()
                ci = block_bootstrap_mean(vals.tolist(), blocks)
                out[f"{market}|{pos}|{field_name}"] = {
                    "mean": round(float(vals.mean()), 4),
                    "median": round(float(vals.median()), 4),
                    "n": int(len(vals)),
                    "ci": ci,
                }
    return {"references": out, "through_season": through_season,
            "n": int(len(df)),
            "note": ("league reference levels by market and position; "
                     "prior seasons only when through_season is set")}


def load_reference_levels(path: str = REFERENCE_PATH) -> Dict:
    if not os.path.exists(path):
        return {"references": {}}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reference_for(refs: Dict, market: str, pos: Optional[str],
                  field_name: str = "proj_efficiency") -> Optional[Dict]:
    return (refs.get("references") or {}).get(f"{market}|{pos}|{field_name}")


# --------------------------------------------------------------------------- #
# The registry: measured support for each driver key
# --------------------------------------------------------------------------- #
def _absence_matrix() -> Dict:
    path = os.path.join(ROOT, "data", "absence_matrix.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def evidence_for(key: str, contribution: Optional[Dict] = None,
                 refs: Optional[Dict] = None) -> Evidence:
    """Empirical support for one ledger driver key."""
    contribution = contribution or {}

    if key in ("team_volume", "baseline_volume", "usage_share"):
        return Evidence(
            claim="Rolling usage measured from prior weeks only",
            n=None, effect=contribution.get("multiplier"),
            effect_unit="level", interval=None, interval_kind=None,
            interval_status="not_applicable",
            grade="moderate",
            source="nflverse play-by-play, 8-game rolling, shift(1)",
            note=("This is an observed usage level, not a claimed effect. Its "
                  "uncertainty is the player's own week-to-week variance, "
                  "which the projection carries in the SD, not here."))

    if key == "efficiency":
        return Evidence(
            claim="Rolling efficiency, shrunk to the position mean on small samples",
            n=None, effect=contribution.get("multiplier"), effect_unit="level",
            interval=None, interval_kind=None, interval_status="not_applicable",
            grade="moderate",
            source="nflverse play-by-play, 8-game rolling, shift(1)",
            note="Observed level, not a claimed effect.")

    if key == "game_script":
        return Evidence(
            claim="Pre-game spread tilts run/pass volume",
            n=None, effect=contribution.get("multiplier"),
            effect_unit="multiplier", interval=None, interval_kind=None,
            interval_status="not_recomputed",
            grade="thin",
            source="nflverse schedule spread_line; capped +/-12%",
            note=("The +/-12% cap is a modelling choice, not a measured "
                  "effect size with a published interval."))

    if key == "opp_factor":
        return Evidence(
            claim="Opponent yards/EPA allowed to this position",
            n=None, effect=contribution.get("multiplier"),
            effect_unit="multiplier", interval=None, interval_kind=None,
            interval_status="not_recomputed",
            grade="thin",
            source="nflverse play-by-play, rolling defence-vs-position",
            note="Rolling defensive factor; interval not recomputed in Phase 8.")

    if key == "realloc_mult":
        return Evidence(
            claim="Teammate out -> this player's usage share rises",
            n=297, effect=contribution.get("multiplier"),
            effect_unit="multiplier", interval=None,
            interval_kind=None, interval_status="not_recomputed",
            grade="moderate",
            source="measured 2019-2025, n=297 absent player-weeks",
            note=("Volume boost is capped at x1.35 and halved when the basis "
                  "is a proportional guess rather than an observed "
                  "with/without split."))

    if key == "realloc_eff_mult":
        return Evidence(
            claim=("Players absorbing extra volume lose about 31% efficiency "
                   "per opportunity"),
            n=297, effect=contribution.get("multiplier"),
            effect_unit="multiplier", interval=None,
            interval_kind=None, interval_status="not_recomputed",
            grade="moderate",
            source="measured 2019-2025, n=297 absent player-weeks",
            note=("This CUTS the projection -- it is the model arguing "
                  "against its own volume bump."))

    if key == "backup_qb_adj":
        return Evidence(
            claim=("With a backup QB, pass-family volume is flat but "
                   "efficiency drops ~8%"),
            n=162, effect=contribution.get("multiplier"),
            effect_unit="multiplier", interval=None,
            interval_kind=None, interval_status="not_recomputed",
            grade="moderate",
            source="measured 2019-2025, n=162 backup-QB team-weeks",
            note=("The 'backups mean more handoffs' intuition was tested and "
                  "is empirically false: rush volume moved x0.98."))

    if key == "absence_qb_mult":
        matrix = _absence_matrix()
        ns = [c.get("n_absent") for c in (matrix.get("matrix") or {}).values()
              if c.get("n_absent")]
        return Evidence(
            claim="QB passing output falls when a skill-position leader sits",
            n=min(ns) if ns else None, effect=contribution.get("multiplier"),
            effect_unit="multiplier", interval=None,
            interval_kind=None, interval_status="not_recomputed",
            grade="moderate",
            source=f"data/absence_matrix.json ({matrix.get('provenance', 'pooled 2019-2025')})",
            note="n shown is the smallest cell in the matrix, not the largest.")

    return Evidence(
        claim=f"Unregistered driver {key!r}", n=None, effect=None,
        effect_unit="unknown", interval=None, interval_kind=None,
        interval_status="not_applicable", grade="unproven",
        source="unknown",
        note="No measured support is registered for this driver.")


#: Narrative factors that were TESTED and came back at approximately zero.
#: They are surfaced deliberately: a reader who has heard the birthday story
#: should see that it was measured, not that it was ignored.
MEASURED_ZERO = {
    "birthday": Evidence(
        claim="Playing in one's birthday week", n=2275, effect=0.363,
        effect_unit="proportion",
        interval=wilson_interval(int(round(0.363 * 2275)), 2275),
        interval_kind="wilson", interval_status="computed",
        grade="unproven",
        source="2019-2025 candidates, n=73,925 baseline",
        note=("36.3% over-rate against a 36.6% baseline. Measured at "
              "approximately zero and given no weight.")),
    "revenge": Evidence(
        claim="Facing a former team", n=1111, effect=0.337,
        effect_unit="proportion",
        interval=wilson_interval(int(round(0.337 * 1111)), 1111),
        interval_kind="wilson", interval_status="computed",
        grade="unproven",
        source="2019-2025 candidates, n=73,925 baseline",
        note=("33.7% over-rate against a 36.6% baseline -- if anything "
              "slightly negative. Given no weight.")),
    "defensive_outs": Evidence(
        claim="Two or more opposing secondary players out",
        n=3540, effect=0.439, effect_unit="proportion",
        interval=wilson_interval(int(round(0.439 * 3540)), 3540),
        interval_kind="wilson", interval_status="computed",
        grade="moderate",
        source="2019-2025 pass-family candidate rows",
        note=("43.9% over-rate against 41.4% baseline -- real and "
              "directionally sensible. Enters via the ML ranker (ordering "
              "only), never the projected number.")),
}


def attach_evidence(ledger_dict: Dict, refs: Optional[Dict] = None) -> Dict:
    """Attach an :class:`Evidence` block to every contribution in a ledger."""
    for c in ledger_dict.get("contributions", []):
        c["evidence"] = evidence_for(c["key"], c, refs).to_dict()
    ledger_dict["measured_zero"] = {k: v.to_dict() for k, v in MEASURED_ZERO.items()}
    return ledger_dict

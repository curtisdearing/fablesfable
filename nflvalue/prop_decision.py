"""Fail-closed contracts for live prop decisions.

Two separate, load-bearing questions live here:

1. ROSTER ELIGIBILITY -- is this player on THIS team's active roster right
   now? Live candidates are carry-forward history (``candidates.py``), and
   history is not membership: a retired, released, or traded player keeps a
   row on his old team. The gate consumes a dated roster snapshot
   (``sources.active_roster``) and classifies every candidate as one of
   ``active | practice_squad | reserve | inactive | released | retired |
   traded_away | team_changed | ambiguous | unknown``. Only ``active`` (and,
   at T-90, a practice-squad player the event roster shows as active --
   a standard elevation) reaches the ranker. This is deliberately NOT a
   recency cutoff on play history: a player returning from IR is ``active``
   the day he is activated, however long he sat.

   Roster eligibility, injury availability (``sources.availability``:
   OK/RISK/OUT from ESPN), and "we don't know" are three different states
   and are reported separately.

2. DECISION PROBABILITY -- the only probability eligible for a displayed or
   actionable prop decision is re-derived from the published mean, residual
   SD, line, and distribution family. The ML ranker is ordinal only. Note the
   word chosen: this is the DISTRIBUTION probability. Agreement with the
   distribution is a self-consistency check, not empirical calibration;
   calibration is a held-out claim made elsewhere (analysis/eval_harness.py)
   or not at all.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from .projection import p_over

ACTIVE_ROSTER_MAX_AGE_HOURS = 48
PROBABILITY_TOLERANCE = 0.0005  # persisted projection probabilities are rounded to 4 dp

#: nflverse status code -> eligibility class. Kept here (not only in the
#: source module) so the gate's vocabulary is closed and testable.
STATUS_CLASS: Dict[str, str] = {
    "ACT": "active", "DEV": "practice_squad",
    "RES": "reserve", "PUP": "reserve", "NON": "reserve", "SUS": "reserve", "EXE": "reserve",
    "INA": "inactive", "CUT": "released", "RET": "retired",
    "TRD": "traded_away", "TRC": "traded_away",
}
ELIGIBLE_CLASSES = {"active"}
DISCRETE_DISTS = {"negbinom", "poisson"}


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #
def _parse_stamp(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


# --------------------------------------------------------------------------- #
# Roster snapshot validation
# --------------------------------------------------------------------------- #
def active_roster_gate(active_player_ids: Optional[Iterable[str]], fetched_at: Optional[str], *,
                       now: Optional[dt.datetime] = None,
                       max_age_hours: float = ACTIVE_ROSTER_MAX_AGE_HOURS) -> dict:
    """Low-level freshness check on an id set + stamp (no coverage semantics).

    Kept for callers that only hold ids; :func:`validate_roster_snapshot` is
    the production gate.
    """
    ids = set(active_player_ids or [])
    stamp = _parse_stamp(fetched_at)
    now = now or dt.datetime.now(dt.timezone.utc)
    if not ids or stamp is None:
        return {"publish": False, "reason": "active roster data missing"}
    age = now - stamp.astimezone(dt.timezone.utc)
    if age > dt.timedelta(hours=max_age_hours) or age < dt.timedelta(hours=-1):
        return {"publish": False, "reason": "active roster data stale or invalid"}
    return {"publish": True, "reason": None, "n_active_players": len(ids)}


def validate_roster_snapshot(roster: Optional[Dict], *, season: int, week: int,
                             slate_teams: Optional[Iterable[str]] = None,
                             now: Optional[dt.datetime] = None,
                             max_age_hours: float = ACTIVE_ROSTER_MAX_AGE_HOURS) -> dict:
    """Is this roster snapshot usable as THE roster for (season, week)?

    Fails closed, with one explicit reason, on: no snapshot / no rows; no
    parseable ``snapshot_at``; a snapshot older than ``max_age_hours`` or
    dated in the future; a snapshot for another season; a snapshot whose
    week is neither ``week`` nor ``week-1`` (the asset rolls to the new week
    during the week -- yesterday's roster labeled week-1 is still the current
    roster, but a week-2-old label is not); or a slate team with no roster
    rows at all (partial coverage would silently empty that team's board).

    Returns ``{"publish", "reason", "n_rows", "n_teams", "week",
    "snapshot_at", "age_hours", "missing_teams"}``.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    out = {"publish": False, "reason": None, "n_rows": 0, "n_teams": 0,
           "week": None, "snapshot_at": None, "age_hours": None, "missing_teams": []}
    if not roster or not isinstance(roster, dict):
        out["reason"] = "active roster data missing"
        return out
    rows = roster.get("rows") or []
    out["n_rows"] = len(rows)
    if not rows:
        out["reason"] = "active roster data missing (snapshot carries no rows)"
        return out
    stamp = _parse_stamp(roster.get("snapshot_at") or roster.get("fetched_at"))
    if stamp is None:
        out["reason"] = "active roster snapshot has no parseable timestamp"
        return out
    out["snapshot_at"] = stamp.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    age = now - stamp.astimezone(dt.timezone.utc)
    out["age_hours"] = round(age.total_seconds() / 3600.0, 2)
    if age < dt.timedelta(hours=-1):
        out["reason"] = f"active roster snapshot is future-dated ({out['snapshot_at']})"
        return out
    if age > dt.timedelta(hours=max_age_hours):
        out["reason"] = (f"active roster snapshot stale: {out['age_hours']}h old "
                         f"> {max_age_hours:.0f}h")
        return out
    r_season = roster.get("season")
    if r_season is not None and int(r_season) != int(season):
        out["reason"] = f"active roster snapshot is for season {r_season}, not {season}"
        return out
    weeks = {int(r["week"]) for r in rows if r.get("week") is not None}
    r_week = roster.get("week")
    r_week = int(r_week) if r_week is not None else (max(weeks) if weeks else None)
    out["week"] = r_week
    if r_week is None or r_week > int(week) or r_week < int(week) - 1:
        out["reason"] = (f"active roster snapshot covers week {r_week}, "
                         f"not week {week} (or {int(week) - 1})")
        return out
    teams = {str(r.get("team")).upper() for r in rows if r.get("team")}
    out["n_teams"] = len(teams)
    if slate_teams:
        missing = sorted({str(t).upper() for t in slate_teams} - teams)
        out["missing_teams"] = missing
        if missing:
            out["reason"] = ("active roster snapshot has no rows for slate team(s) "
                             + ", ".join(missing))
            return out
    out["publish"] = True
    return out


# --------------------------------------------------------------------------- #
# Per-player eligibility
# --------------------------------------------------------------------------- #
def roster_index(rows: Iterable[Dict]) -> Dict[str, Dict]:
    """{player_id: {"team", "status", "class", "ambiguous", "n_rows"}}.

    A player quoted on two different teams in the same snapshot (mid-week
    trade with both rows present) is ``ambiguous`` and fails closed. Identical
    duplicate rows collapse. A ``traded_away`` row (TRD/TRC on the departing
    club) never wins against an ``active`` row on the new club.
    """
    idx: Dict[str, Dict] = {}
    for r in rows or []:
        pid = r.get("player_id")
        if not pid:
            continue
        pid = str(pid)
        team = str(r.get("team")).upper() if r.get("team") else None
        status = str(r.get("status")).upper() if r.get("status") else None
        cls = STATUS_CLASS.get(status, "unknown") if status else "unknown"
        cur = idx.get(pid)
        if cur is None:
            idx[pid] = {"team": team, "status": status, "class": cls,
                        "ambiguous": False, "n_rows": 1, "name": r.get("name")}
            continue
        cur["n_rows"] += 1
        if cur["team"] == team and cur["status"] == status:
            continue  # exact duplicate
        # a departing-club row loses to any other row for the same player
        if cur["class"] == "traded_away" and cls != "traded_away":
            cur.update({"team": team, "status": status, "class": cls})
            continue
        if cls == "traded_away":
            continue
        if cur["team"] != team:
            cur["ambiguous"] = True
    return idx


def classify_candidate(player_id: str, team: Optional[str], index: Dict[str, Dict],
                       t90_active_names: Optional[Set[str]] = None,
                       name: Optional[str] = None) -> Dict:
    """One candidate row -> {"eligibility", "detail", "roster_team", "status"}."""
    entry = index.get(str(player_id))
    team_u = str(team).upper() if team else None
    if entry is None:
        return {"eligibility": "unknown", "detail": "not in roster snapshot",
                "roster_team": None, "status": None}
    base = {"roster_team": entry["team"], "status": entry["status"]}
    if entry.get("ambiguous"):
        return {**base, "eligibility": "ambiguous",
                "detail": "conflicting team rows in the same snapshot"}
    cls = entry["class"]
    if cls == "active" and team_u and entry["team"] and entry["team"] != team_u:
        return {**base, "eligibility": "team_changed",
                "detail": f"roster says {entry['team']}, candidate row says {team_u}"}
    if cls == "practice_squad":
        key = _norm(name) if name else None
        if t90_active_names and key and key in t90_active_names:
            return {**base, "eligibility": "active",
                    "detail": "practice squad, elevated (active on event roster)"}
        return {**base, "eligibility": "practice_squad", "detail": "practice squad (DEV)"}
    return {**base, "eligibility": cls, "detail": f"status {entry['status']}"}


def _norm(name: Optional[str]) -> str:
    try:
        from .sources.availability import normalize_name
        return normalize_name(name)
    except Exception:  # noqa: BLE001 -- keep this module importable standalone
        return str(name or "").lower().strip()


def apply_roster_eligibility(candidates: pd.DataFrame, roster: Dict,
                             t90_active_names: Optional[Set[str]] = None):
    """Keep only roster-eligible candidates; return (kept, diagnostic).

    ``diagnostic``: {"n_in", "n_kept", "n_excluded", "by_reason": {...},
    "excluded": [{player_id, name, team, eligibility, detail}, ...]}.
    Nothing is dropped silently -- every exclusion is listed with its reason.
    """
    diag = {"n_in": 0, "n_kept": 0, "n_excluded": 0, "by_reason": {}, "excluded": []}
    if candidates is None or candidates.empty:
        return candidates, diag
    idx = roster_index((roster or {}).get("rows") or [])
    keep: List[bool] = []
    seen_excl = set()
    for r in candidates.itertuples(index=False):
        d = r._asdict()
        cls = classify_candidate(d.get("player_id"), d.get("team"), idx,
                                 t90_active_names=t90_active_names, name=d.get("name"))
        ok = cls["eligibility"] in ELIGIBLE_CLASSES
        keep.append(ok)
        if not ok:
            key = (d.get("player_id"), d.get("team"))
            if key not in seen_excl:
                seen_excl.add(key)
                diag["excluded"].append({
                    "player_id": d.get("player_id"), "name": d.get("name"),
                    "team": d.get("team"), "eligibility": cls["eligibility"],
                    "detail": cls["detail"], "roster_team": cls.get("roster_team"),
                    "status": cls.get("status")})
                diag["by_reason"][cls["eligibility"]] = \
                    diag["by_reason"].get(cls["eligibility"], 0) + 1
    diag["n_in"] = int(len(candidates))
    diag["n_kept"] = int(sum(keep))
    diag["n_excluded"] = diag["n_in"] - diag["n_kept"]
    kept = candidates[pd.Series(keep, index=candidates.index)].reset_index(drop=True)
    return kept, diag


def filter_to_active_roster(candidates: pd.DataFrame, active_player_ids: Iterable[str]) -> pd.DataFrame:
    """Id-only membership filter (no team check). Prefer apply_roster_eligibility."""
    if candidates is None or candidates.empty:
        return candidates
    return candidates[candidates["player_id"].isin(set(active_player_ids))].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Decision probability (distribution-derived; NOT a calibration claim)
# --------------------------------------------------------------------------- #
def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def probability_from_projection(candidate: dict) -> Optional[float]:
    """P(over) re-derived from mean/sd/line/dist; None when it cannot be."""
    try:
        value = p_over(float(candidate["mean"]), float(candidate["sd"]),
                       float(candidate["line"]), str(candidate["dist"]))
    except (KeyError, TypeError, ValueError):
        return None
    return float(value) if _finite(value) and 0.0 <= value <= 1.0 else None


#: How far the pooled SD is assumed to be from a player's true SD when
#: measuring whether an edge survives that uncertainty. Not a fitted number:
#: it is a deliberately mild stress, and a real per-player SD will miss the
#: pooled one by far more than this for a QB at either tail.
SD_STRESS_FRACTION = 0.25


def sd_stress(candidate: dict, fraction: float = SD_STRESS_FRACTION) -> Optional[float]:
    """How much P(over) moves when SD is scaled by 1 +/- ``fraction``.

    ``candidates.market_residual_sd`` fits ONE residual SD per market across
    every player, so every quarterback on a board carries the identical
    passing-yards SD (97.375 on 2026 Week 1 -- D.Maye and S.Darnold alike).
    A pooled SD that wide pins P(over) near 0.5 and, differenced against a
    real book, manufactures a small "edge" out of a quantity the model never
    estimated for that player.

    This returns the half-width of the probability interval the pooled SD
    leaves open. Compared against the edge, it answers the only question that
    matters for ACTION: is this edge bigger than our ignorance about SD?
    Returns None when the probability cannot be re-derived.
    """
    base = probability_from_projection(candidate)
    if base is None:
        return None
    try:
        sd = float(candidate["sd"])
    except (KeyError, TypeError, ValueError):
        return None
    if not _finite(sd) or sd <= 0:
        return None
    moved = []
    for scale in (1.0 - float(fraction), 1.0 + float(fraction)):
        probe = dict(candidate)
        probe["sd"] = sd * scale
        p = probability_from_projection(probe)
        if p is not None:
            moved.append(abs(float(p) - float(base)))
    return max(moved) if moved else None


def edge_survives_sd_uncertainty(edge: Optional[float], stress: Optional[float]) -> Optional[bool]:
    """Is ``edge`` larger than the probability swing a mis-specified SD buys?

    None when either input is unknown -- absence of the check is not a pass.
    """
    if edge is None or stress is None:
        return None
    return abs(float(edge)) > float(stress)


def side_probabilities(candidate: dict) -> Dict[str, Optional[float]]:
    """{p_over, p_under, p_push} from the distribution.

    A count market (negbinom/poisson) quoted at an INTEGER line can push.
    ``p_over`` is P(X > line) as everywhere else; ``p_under`` is P(X < line),
    i.e. the push mass is NOT credited to the under; ``p_push`` is P(X = line).
    Continuous families and half-point lines have ``p_push = 0``.
    """
    po = probability_from_projection(candidate)
    if po is None:
        return {"p_over": None, "p_under": None, "p_push": None}
    p_push = 0.0
    try:
        line = float(candidate["line"])
        dist = str(candidate.get("dist"))
    except (KeyError, TypeError, ValueError):
        return {"p_over": po, "p_under": 1.0 - po, "p_push": 0.0}
    if dist in DISCRETE_DISTS and _finite(line) and float(line).is_integer():
        # P(X >= line) = P(X > line - 1)
        p_ge = p_over(float(candidate["mean"]), float(candidate["sd"]), line - 1.0, dist)
        if _finite(p_ge):
            p_push = max(0.0, min(1.0, float(p_ge) - po))
    return {"p_over": po, "p_under": max(0.0, 1.0 - po - p_push), "p_push": p_push}


def probability_coherent(candidate: dict, probability: Optional[float] = None,
                          tolerance: float = PROBABILITY_TOLERANCE) -> bool:
    """Reject a stored P(over) that drifts from the stated distribution."""
    probability = probability if probability is not None else probability_from_projection(candidate)
    try:
        supplied = float(candidate.get("p_over"))
    except (TypeError, ValueError):
        return False
    if not _finite(supplied):
        return False
    return probability is not None and abs(supplied - probability) <= tolerance


def valid_price(value) -> bool:
    """A decimal price is usable only if it is a finite number above 1.0."""
    return _finite(value) and float(value) > 1.0


def valid_book_count(n_books) -> Optional[int]:
    """Distinct-book count as a non-negative int, or None when malformed.

    Only a real integer (or an integral float, as SQLite/pandas round-trip
    it) counts; bools, strings, NaN, negatives and fractions are malformed.
    """
    if isinstance(n_books, bool) or n_books is None:
        return None
    try:
        import numpy as _np
        if isinstance(n_books, _np.generic):
            n_books = n_books.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(n_books, int):
        return n_books if n_books >= 0 else None
    if isinstance(n_books, float):
        if not math.isfinite(n_books) or not n_books.is_integer() or n_books < 0:
            return None
        return int(n_books)
    return None

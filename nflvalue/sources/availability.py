"""Availability resolver: who is actually playing, and who absorbs the usage.

PHASE1_HANDSOFF_DESIGN.md H1/H2: the nflverse injury feed is DEAD (2025+),
so ESPN's undocumented endpoints are the only free injury source, and true
availability isn't known until ~90 minutes before kickoff. This module
implements both clocks:

    clock="wed"   -- Wednesday provisional: league-wide team-injuries feed
                     (Out/Doubtful/Questionable) -> OK | RISK | OUT
    clock="t90"   -- pre-kick final: per-event competitor roster ``active``
                     flag OVERRIDES the Wednesday read (inactive -> OUT).
                     This feed's population time is not guaranteed (the
                     single riskiest link in the chain) -- callers must gate
                     it through nflvalue.freshness rather than trusting it.

Every resolved status carries {source, timestamp, matched_by} so nothing is
asserted without provenance. ESPN rows are matched to nflverse gsis ids by
normalized name + team; UNMATCHED rows are returned separately, never
silently dropped or guessed (H3: fail loud).

Usage reallocation (H8) estimates, from historical with/without splits, how a
team's usage shifts when a player sits. When no absent-week sample exists the
estimate is an explicitly-flagged proportional guess (low_confidence=True) --
the honest answer is "we don't know, here's a weak prior," not a made-up split.

Recorded fixtures for offline tests: tests/fixtures/espn_*.json.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from ._http import get_json
from ..freshness import stamp_now

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

# Reuse the existing app's team map (build_ratings.ABBR: abbr -> display name),
# inverted -- ESPN reports "Arizona Cardinals", nflverse uses "ARI".
try:  # pragma: no cover - trivial import plumbing
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    from build_ratings import ABBR as _ABBR
except Exception:  # noqa: BLE001 -- keep this module importable standalone
    _ABBR = {}
DISPLAY_TO_ABBR: Dict[str, str] = {}
for _ab, _disp in _ABBR.items():
    # keep the canonical (modern) abbr for a display name: first writer wins,
    # dict order in build_ratings lists the modern code before legacy aliases.
    DISPLAY_TO_ABBR.setdefault(_disp, _ab)

# ESPN status text -> our three-state availability
_OUT_STATUSES = {
    "out", "injured reserve", "ir", "doubtful", "physically unable to perform",
    "pup", "suspension", "reserve/suspended", "non football injury",
    "reserve", "practice squad injured",
}
_RISK_STATUSES = {"questionable", "day-to-day"}


class EspnSchemaError(RuntimeError):
    """An ESPN payload no longer matches the recorded shape (H3: fail loud)."""


# --------------------------------------------------------------------------- #
# Name matching (deterministic, conservative)
# --------------------------------------------------------------------------- #
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: Optional[str]) -> str:
    """'Odell Beckham Jr.' -> 'odell beckham'; punctuation/case/suffix-proof."""
    if not name:
        return ""
    s = re.sub(r"[^a-z\s]", "", str(name).lower().replace(".", " "))
    parts = [p for p in s.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def _espn_id_from_links(athlete: Dict) -> Optional[str]:
    for link in athlete.get("links") or []:
        m = re.search(r"/id/(\d+)", str(link.get("href", "")))
        if m:
            return m.group(1)
    return None


# --------------------------------------------------------------------------- #
# Parsing (pure -- fixture-testable offline)
# --------------------------------------------------------------------------- #
def normalize_status(status_raw: Optional[str]) -> str:
    s = str(status_raw or "").strip().lower()
    if s in _OUT_STATUSES:
        return "OUT"
    if s in _RISK_STATUSES:
        return "RISK"
    return "OK"


def parse_team_injuries(raw: Dict) -> List[Dict]:
    """League-wide injuries payload -> flat rows.

    Row: {team, team_display, name, espn_id, position, status_raw, status,
          date, comment}. ``team`` is the nflverse abbr ('' if the display
    name isn't recognized -- visible, not dropped).
    """
    if not isinstance(raw, dict) or "injuries" not in raw:
        raise EspnSchemaError("injuries payload missing top-level 'injuries' key")
    rows: List[Dict] = []
    for team_block in raw.get("injuries", []):
        display = team_block.get("displayName") or (team_block.get("team") or {}).get("displayName") or ""
        abbr = DISPLAY_TO_ABBR.get(display, "")
        for it in team_block.get("injuries", []) or []:
            ath = it.get("athlete") or {}
            rows.append({
                "team": abbr,
                "team_display": display,
                "name": ath.get("displayName") or ath.get("fullName") or "",
                "espn_id": ath.get("id") or _espn_id_from_links(ath),
                "position": ((ath.get("position") or {}).get("abbreviation")
                             if isinstance(ath.get("position"), dict) else ath.get("position")) or "",
                "status_raw": it.get("status") or "",
                "status": normalize_status(it.get("status")),
                "date": it.get("date") or "",
                "comment": (it.get("shortComment") or "")[:400],
            })
    return rows


def parse_event_roster(raw: Dict) -> List[Dict]:
    """Per-event competitor roster payload -> rows with the pre-kick active flag.

    Row: {espn_id, name, active, did_not_play, starter, jersey}.
    """
    if not isinstance(raw, dict) or "entries" not in raw:
        raise EspnSchemaError("event roster payload missing 'entries'")
    rows = []
    for e in raw.get("entries", []):
        rows.append({
            "espn_id": str(e.get("playerId", "")) or None,
            "name": e.get("displayName") or "",
            "active": bool(e.get("active", False)),
            "did_not_play": bool(e.get("didNotPlay", False)),
            "starter": bool(e.get("starter", False)),
            "jersey": e.get("jersey"),
        })
    return rows


# --------------------------------------------------------------------------- #
# Fetch (thin, timestamped)
# --------------------------------------------------------------------------- #
def fetch_team_injuries() -> Dict:
    """League-wide injuries -> {"rows": [...], "fetched_at": iso, "n_teams": int}."""
    raw = get_json(f"{SITE}/injuries")
    rows = parse_team_injuries(raw)
    return {"rows": rows, "fetched_at": stamp_now(),
            "n_teams": len(raw.get("injuries", []))}


def fetch_event_rosters(event_id: str) -> Dict:
    """T-90 actives for one event -> {"rows": [...], "fetched_at": iso}.

    Resolves the event's two competitor team ids via the site summary, then
    pulls each competitor's core-API roster (the payload carrying ``active``).
    """
    summary = get_json(f"{SITE}/summary", params={"event": event_id})
    comps = (((summary.get("header") or {}).get("competitions")) or [{}])[0]
    competitors = comps.get("competitors") or []
    if not competitors:
        raise EspnSchemaError(f"event {event_id}: no competitors in summary header")
    rows: List[Dict] = []
    for c in competitors:
        team_id = (c.get("team") or {}).get("id") or c.get("id")
        if not team_id:
            raise EspnSchemaError(f"event {event_id}: competitor without a team id")
        raw = get_json(f"{CORE}/events/{event_id}/competitions/{event_id}/competitors/{team_id}/roster")
        team_rows = parse_event_roster(raw)
        abbr = (c.get("team") or {}).get("abbreviation", "")
        for r in team_rows:
            r["team"] = abbr
        rows.extend(team_rows)
    return {"rows": rows, "fetched_at": stamp_now()}


# --------------------------------------------------------------------------- #
# Resolution: our players x ESPN feeds -> OK | RISK | OUT (with provenance)
# --------------------------------------------------------------------------- #
def resolve_statuses(
    players: pd.DataFrame,
    injury_rows: Iterable[Dict],
    inactive_rows: Optional[Iterable[Dict]] = None,
    clock: str = "wed",
    injuries_fetched_at: Optional[str] = None,
    inactives_fetched_at: Optional[str] = None,
) -> Dict:
    """Resolve availability for each projected player.

    ``players``: DataFrame with columns [player_id, player_name, team]
    (nflverse gsis ids + abbrs -- e.g. a slice of ``player_week``).

    Returns::

        {"statuses": {player_id: {status, status_raw, source, timestamp,
                                   matched_by, comment}},
         "unmatched_espn_rows": [...]}   # ESPN said something about a player
                                         # we couldn't match -- kept visible

    clock="wed": statuses from the league injuries feed only.
    clock="t90": additionally require ``inactive_rows``; a player present in
    the event roster with active=False is OUT regardless of the Wednesday
    read; active=True upgrades a Wednesday OUT/RISK back to OK only if the
    injury status wasn't OUT-final (IR/suspension stays OUT).
    """
    if clock not in ("wed", "t90"):
        raise ValueError(f"clock must be 'wed' or 't90', got {clock!r}")
    if clock == "t90" and inactive_rows is None:
        raise ValueError("clock='t90' requires inactive_rows (per-event actives); "
                         "refusing to silently fall back to the stale Wednesday read")

    inj_ts = injuries_fetched_at or stamp_now()
    ina_ts = inactives_fetched_at or stamp_now()

    # index ESPN injury rows by (normalized name, team) and by name alone
    by_name_team: Dict[Tuple[str, str], Dict] = {}
    by_name: Dict[str, List[Dict]] = {}
    for r in injury_rows:
        key = normalize_name(r.get("name"))
        if not key:
            continue
        by_name_team[(key, r.get("team") or "")] = r
        by_name.setdefault(key, []).append(r)

    ina_by_name: Dict[str, Dict] = {}
    for r in inactive_rows or []:
        key = normalize_name(r.get("name"))
        if key:
            ina_by_name[key] = r

    statuses: Dict[str, Dict] = {}
    matched_keys: Set[str] = set()
    for p in players.itertuples(index=False):
        pid = getattr(p, "player_id")
        pname = normalize_name(getattr(p, "player_name", ""))
        pteam = getattr(p, "team", "") or ""

        row = by_name_team.get((pname, pteam))
        matched_by = "name+team" if row is not None else None
        if row is None:
            cands = by_name.get(pname, [])
            if len(cands) == 1:  # unambiguous name-only match (team moved/renamed)
                row, matched_by = cands[0], "name_only"

        status, status_raw, source, ts, comment = "OK", "", "none(no injury listed)", inj_ts, ""
        if row is not None:
            matched_keys.add(pname)
            status, status_raw = row["status"], row["status_raw"]
            source, ts, comment = "espn_team_injuries", inj_ts, row.get("comment", "")

        if clock == "t90":
            ina = ina_by_name.get(pname)
            if ina is not None:
                if not ina.get("active", False):
                    status, status_raw = "OUT", (status_raw or "") + "|inactive_t90"
                    source, ts = "espn_event_roster", ina_ts
                elif status != "OUT" or "reserve" not in (status_raw or "").lower():
                    # confirmed active pre-kick clears a Wed Questionable/Out
                    # (but never un-OUTs an IR/suspension designation)
                    if status in ("RISK", "OUT"):
                        status, status_raw = "OK", (status_raw or "") + "|active_t90"
                        source, ts = "espn_event_roster", ina_ts

        statuses[pid] = {"status": status, "status_raw": status_raw, "source": source,
                         "timestamp": ts, "matched_by": matched_by or "unmatched",
                         "comment": comment}

    unmatched = [r for k, rows_ in by_name.items() if k not in matched_keys for r in rows_]
    return {"statuses": statuses, "unmatched_espn_rows": unmatched}


# --------------------------------------------------------------------------- #
# Usage reallocation from historical with/without splits (H8)
# --------------------------------------------------------------------------- #
_ROLE_FAMILY = {  # role -> (numerator actual col, team denominator col)
    "WR": ("targets", "team_pass_att"),
    "TE": ("targets", "team_pass_att"),
    "RB": ("carries", "team_rush_att"),
}
# Phase 6.2: the red-zone analog -- when a leader sits, his GOAL-LINE/RZ work
# is redistributed too, which is what anytime-TD props actually price
_ROLE_RZ_FAMILY = {
    "WR": ("rz_tgt", "team_rz_tgt"),
    "TE": ("rz_tgt", "team_rz_tgt"),
    "RB": ("rz_car", "team_rz_car"),
}


def reallocate_usage(player_week: pd.DataFrame, season: int, week: int,
                     out_player_id: str, window_weeks: int = 17,
                     min_absent_games: int = 2) -> Dict:
    """Estimate teammates' usage-share boost when ``out_player_id`` sits.

    Walk-forward: uses only rows strictly BEFORE (season, week). Splits the
    trailing window into team-games the OUT player played vs. missed and
    compares each same-family teammate's usage share across the two sets.

    Returns::

        {"out_player_id", "role", "team", "basis": "with_without"|"proportional_guess",
         "absent_games": int, "played_games": int, "low_confidence": bool,
         "boosts": {player_id: {"name", "share_with", "share_without",
                                "share_delta"}}}

    QBs are not reallocated (a backup QB is a different projection problem,
    not a share shift) -- returns basis="not_supported".
    Committees and role-specific usage make even the split-based number a
    rough estimate; with fewer than ``min_absent_games`` absent games it
    degrades to redistributing the OUT player's own trailing share across
    same-role teammates proportionally, flagged low_confidence (H8: flag the
    guess, don't dress it up).
    """
    pw = player_week
    hist = pw[(pw["season"] < season) | ((pw["season"] == season) & (pw["week"] < week))].copy()
    prow = hist[hist["player_id"] == out_player_id].sort_values(["season", "week"]).tail(1)
    if prow.empty:
        return {"out_player_id": out_player_id, "basis": "unknown_player",
                "low_confidence": True, "boosts": {}}
    role = prow.iloc[0]["role"]
    team = prow.iloc[0]["team"]
    if role not in _ROLE_FAMILY:
        return {"out_player_id": out_player_id, "role": role, "team": team,
                "basis": "not_supported", "low_confidence": True, "boosts": {}}
    num_col, den_col = _ROLE_FAMILY[role]

    team_hist = hist[hist["team"] == team].copy()
    team_hist["_gkey"] = list(zip(team_hist["season"], team_hist["week"]))
    game_keys = sorted(team_hist["_gkey"].unique())[-window_weeks:]
    team_hist = team_hist[team_hist["_gkey"].isin(game_keys)]

    played_games = set(team_hist.loc[(team_hist["player_id"] == out_player_id)
                                     & (team_hist[num_col] > 0), "_gkey"])
    absent_games = [g for g in game_keys if g not in played_games]

    # teammates sharing the usage family (targets: WR+TE together; carries: RB)
    fam_roles = [r for r, (n, _) in _ROLE_FAMILY.items() if n == num_col]
    mates = team_hist[(team_hist["role"].isin(fam_roles))
                      & (team_hist["player_id"] != out_player_id)]

    rz_num, rz_den = _ROLE_RZ_FAMILY[role]
    have_rz_cols = rz_num in team_hist.columns and rz_den in team_hist.columns

    def _share(frame: pd.DataFrame, n=num_col, d=den_col) -> pd.Series:
        den = frame[d].replace(0, pd.NA)
        return (frame[n] / den).astype(float)

    boosts: Dict[str, Dict] = {}
    if len(absent_games) >= min_absent_games:
        for pid, grp in mates.groupby("player_id"):
            with_g = grp[grp["_gkey"].isin(played_games)]
            without_g = grp[grp["_gkey"].isin(absent_games)]
            if with_g.empty or without_g.empty:
                continue
            sw = float(_share(with_g).mean(skipna=True) or 0.0)
            so = float(_share(without_g).mean(skipna=True) or 0.0)
            if pd.isna(sw) or pd.isna(so):
                continue
            boosts[pid] = {"name": grp.iloc[-1]["player_name"],
                           "share_with": round(sw, 4), "share_without": round(so, 4),
                           "share_delta": round(so - sw, 4)}
            # Phase 6.2: goal-line/red-zone share shift (drives anytime-TD
            # reallocation). NaN-safe: weeks where the team had no RZ play
            # simply drop out of the mean; both sides empty -> no rz keys.
            if have_rz_cols:
                rzw = _share(with_g, rz_num, rz_den).mean(skipna=True)
                rzo = _share(without_g, rz_num, rz_den).mean(skipna=True)
                if pd.notna(rzw) and pd.notna(rzo):
                    boosts[pid].update({
                        "rz_share_with": round(float(rzw), 4),
                        "rz_share_without": round(float(rzo), 4),
                        "rz_share_delta": round(float(rzo - rzw), 4)})
        return {"out_player_id": out_player_id, "role": role, "team": team,
                "basis": "with_without", "absent_games": len(absent_games),
                "played_games": len(played_games), "low_confidence": False,
                "boosts": boosts}

    # not enough absent games -> proportional redistribution, flagged as a guess
    out_share_hist = team_hist[(team_hist["player_id"] == out_player_id)]
    out_share = float(_share(out_share_hist).mean(skipna=True) or 0.0) if not out_share_hist.empty else 0.0
    out_rz_share = (float(_share(out_share_hist, rz_num, rz_den).mean(skipna=True) or 0.0)
                    if have_rz_cols and not out_share_hist.empty else 0.0)
    recent = mates[mates["_gkey"].isin(game_keys[-4:])]
    mate_share = {pid: float(_share(g).mean(skipna=True) or 0.0)
                  for pid, g in recent.groupby("player_id")}
    total = sum(v for v in mate_share.values() if v > 0)
    for pid, s in mate_share.items():
        if s <= 0 or total <= 0:
            continue
        boosts[pid] = {"name": str(recent[recent["player_id"] == pid].iloc[-1]["player_name"]),
                       "share_with": round(s, 4), "share_without": None,
                       "share_delta": round(out_share * (s / total), 4)}
        if have_rz_cols and pd.notna(out_rz_share) and out_rz_share > 0:
            g = recent[recent["player_id"] == pid]
            rzw = _share(g, rz_num, rz_den).mean(skipna=True)
            if pd.notna(rzw):
                boosts[pid].update({
                    "rz_share_with": round(float(rzw), 4), "rz_share_without": None,
                    "rz_share_delta": round(out_rz_share * (s / total), 4)})
    return {"out_player_id": out_player_id, "role": role, "team": team,
            "basis": "proportional_guess", "absent_games": len(absent_games),
            "played_games": len(played_games), "low_confidence": True,
            "boosts": boosts}

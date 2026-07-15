"""Deterministic context features: birthdays, revenge games, defensive injuries.

The user's ask — "base it on birthdays, revenge games, injuries on the
defense" — done the defensible way: each is computed from FACTS (DOBs, roster
history, injury reports), not news text, and each enters the system in two
places with different rules:

  1. as ML-RANKER FEATURES, where the classifier learns their weight from
     outcomes (a useless birthday feature earns ~zero splits — the evidence
     gate is the loss function itself);
  2. as CONTEXT-PANEL lines + `context_ledger` tags, keeping them visible and
     separately testable by `context_study` (the deterministic composite
     still never scores them — the locked spec holds there).

Walk-forward discipline per feature:
  birthday      DOB is immutable; "birthday within ±5 days of the game date"
                uses only the schedule. No leakage possible.
  revenge       former teams = teams the player was rostered on in weeks
                STRICTLY BEFORE this one (min 3 roster-weeks, excluding his
                current team). Roster history is factual and prior.
  def injuries  count of the OPPONENT defense's players listed Out/Doubtful
                on the official injury report FOR THAT WEEK — pre-game
                information by construction (reports precede kickoff);
                split into total front-seven-plus-secondary and DB-only
                (secondary outs matter most for passing markets).
  opp_epa       the defense's rolling EPA-allowed factor vs the player's
                role (already walk-forward in `opp_pos_def`; previously
                computed but unused by the ranker).

Data (cached under historical/, refreshed by ingest):
  players_meta.parquet   gsis_id -> birth_date (nflreadpy load_players)
  injuries.parquet       official injury reports by (season, week, team)
                         — empirically available 2019-2025 via nflreadpy
                         despite the design doc's "dead post-2024" note;
                         ESPN live feed remains the T-90 backstop.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")
PLAYERS_META = os.path.join(HIST, "players_meta.parquet")
INJURIES = os.path.join(HIST, "injuries.parquet")

DB_POS = {"CB", "S", "SS", "FS", "DB"}
DEF_POS = DB_POS | {"DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "EDGE", "DL"}
OUT_STATUSES = {"Out", "Doubtful"}
BIRTHDAY_WINDOW_DAYS = 5
MIN_ROSTER_WEEKS_FOR_REVENGE = 3


# --------------------------------------------------------------------------- #
# Caches (network only on miss/refresh; loaders are offline)
# --------------------------------------------------------------------------- #
def load_players_meta(refresh: bool = False) -> pd.DataFrame:
    if os.path.exists(PLAYERS_META) and not refresh:
        return pd.read_parquet(PLAYERS_META)
    import nflreadpy as nfl
    pl = nfl.load_players().to_pandas()
    meta = (pl[["gsis_id", "birth_date"]].dropna(subset=["gsis_id"])
            .rename(columns={"gsis_id": "player_id"})
            .drop_duplicates(subset=["player_id"]))
    meta.to_parquet(PLAYERS_META, index=False)
    return meta


def load_injury_history(seasons: List[int], refresh: bool = False) -> pd.DataFrame:
    # Refresh means replace the requested seasons, never erase other seasons.
    # The prior implementation initialized an empty cache on refresh and could
    # silently destroy years of injury history during a one-season live pull.
    cached = pd.read_parquet(INJURIES) if os.path.exists(INJURIES) else pd.DataFrame()
    have = set(cached["season"].unique()) if len(cached) else set()
    missing = list(seasons) if refresh else [s for s in seasons if s not in have]
    if missing:
        import nflreadpy as nfl
        try:
            pulled = nfl.load_injuries(seasons=missing).to_pandas()
            keep = pulled[["season", "week", "team", "gsis_id", "position",
                           "report_status", "full_name"]].copy()
            if len(cached):
                cached = cached[~cached["season"].isin(missing)]
                cached = pd.concat([cached, keep], ignore_index=True)
            else:
                cached = keep
            cached = cached.drop_duplicates(subset=["season", "week", "team", "gsis_id"])
            cached.to_parquet(INJURIES, index=False)
        except Exception as exc:  # noqa: BLE001 -- fail loud in log, serve cache
            print(f"[context_features] injury history fetch failed for {missing}: {exc}")
    return cached[cached["season"].isin(seasons)].reset_index(drop=True) if len(cached) else cached


# --------------------------------------------------------------------------- #
# The pack: everything precomputed once per run/replay
# --------------------------------------------------------------------------- #
class ContextPack:
    def __init__(self, rosters: pd.DataFrame, seasons: List[int],
                 opd: Optional[pd.DataFrame] = None,
                 players_meta: Optional[pd.DataFrame] = None,
                 injuries: Optional[pd.DataFrame] = None):
        meta = players_meta if players_meta is not None else load_players_meta()
        self.dob: Dict[str, dt.date] = {}
        for r in meta.itertuples(index=False):
            try:
                self.dob[r.player_id] = pd.Timestamp(r.birth_date).date()
            except (TypeError, ValueError):
                continue

        inj = injuries if injuries is not None else load_injury_history(seasons)
        self.def_out: Dict[Tuple[int, int, str], Tuple[int, int]] = {}
        if len(inj):
            d = inj[inj["report_status"].isin(OUT_STATUSES)
                    & inj["position"].isin(DEF_POS)]
            for (s, w, t), grp in d.groupby(["season", "week", "team"]):
                total = int(len(grp))
                dbs = int(grp["position"].isin(DB_POS).sum())
                self.def_out[(int(s), int(w), str(t))] = (total, dbs)

        # roster stints ordered by (season, week) for walk-forward former-team lookup
        r = rosters.sort_values(["player_id", "season", "week"])
        self.stints: Dict[str, List[Tuple[int, int, str]]] = {}
        for pid, grp in r.groupby("player_id"):
            self.stints[pid] = list(zip(grp["season"].astype(int),
                                        grp["week"].astype(int), grp["team"]))

        self.opp_epa: Dict[Tuple[int, int, str, str], float] = {}
        if opd is not None and "roll_epa_allowed_factor" in opd.columns:
            for r_ in opd.itertuples(index=False):
                v = r_.roll_epa_allowed_factor
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    self.opp_epa[(int(r_.season), int(r_.week), r_.defteam, r_.role)] = float(v)

    # -- feature computations ------------------------------------------------ #
    def is_birthday_week(self, player_id: str, gameday: Optional[str]) -> int:
        dob = self.dob.get(player_id)
        if dob is None or not gameday:
            return 0
        try:
            gd = pd.Timestamp(gameday).date()
        except (TypeError, ValueError):
            return 0
        try:
            bday = dob.replace(year=gd.year)
        except ValueError:            # Feb 29
            bday = dt.date(gd.year, 2, 28)
        delta = min(abs((gd - bday).days),
                    abs((gd - bday.replace(year=gd.year - 1)).days) if gd.month == 1 else 999,
                    abs((bday.replace(year=gd.year + 1) - gd).days) if gd.month == 12 else 999)
        return int(delta <= BIRTHDAY_WINDOW_DAYS)

    def former_teams(self, player_id: str, season: int, week: int,
                     current_team: str) -> Set[str]:
        counts: Dict[str, int] = {}
        for (s, w, t) in self.stints.get(player_id, []):
            if (s, w) >= (season, week):
                break
            counts[t] = counts.get(t, 0) + 1
        return {t for t, n in counts.items()
                if n >= MIN_ROSTER_WEEKS_FOR_REVENGE and t != current_team}

    def revenge_game(self, player_id: str, season: int, week: int,
                     team: str, opponent: str) -> int:
        return int(opponent in self.former_teams(player_id, season, week, team))

    def defense_outs(self, season: int, week: int, opponent: str) -> Tuple[int, int]:
        return self.def_out.get((season, week, opponent), (0, 0))


def attach(cands: pd.DataFrame, pack: Optional["ContextPack"]) -> pd.DataFrame:
    """Stamp the four context features onto a candidate frame. ``pack=None``
    (or missing inputs) stamps neutral values so downstream code never
    branches — absent data reads as 'no signal', never crashes."""
    cands = cands.copy()
    if pack is None:
        cands["is_birthday_week"] = 0
        cands["revenge_game"] = 0
        cands["def_out_total"] = np.nan
        cands["def_out_db"] = np.nan
        cands["opp_epa_factor"] = np.nan
        return cands
    bday, rev, dtot, ddb, epa = [], [], [], [], []
    for r in cands.itertuples(index=False):
        gameday = getattr(r, "gameday", None)
        bday.append(pack.is_birthday_week(r.player_id, gameday))
        rev.append(pack.revenge_game(r.player_id, int(r.season), int(r.week),
                                     r.team, r.defteam))
        t, d = pack.defense_outs(int(r.season), int(r.week), r.defteam)
        dtot.append(t)
        ddb.append(d)
        epa.append(pack.opp_epa.get((int(r.season), int(r.week), r.defteam,
                                     getattr(r, "pos", None)), np.nan))
    cands["is_birthday_week"] = bday
    cands["revenge_game"] = rev
    cands["def_out_total"] = dtot
    cands["def_out_db"] = ddb
    cands["opp_epa_factor"] = epa
    return cands


def panel_items(lean: Dict) -> List[str]:
    """Deterministic context lines for the report panel (display-only there;
    also what context_ledger's keyword tagger picks up)."""
    items = []
    if lean.get("is_birthday_week"):
        items.append("birthday week (computed from DOB, nflverse)")
    if lean.get("revenge_game"):
        items.append("revenge game: previously rostered by this opponent")
    d = lean.get("def_out_total")
    if d is not None and not (isinstance(d, float) and np.isnan(d)) and d >= 2:
        db = lean.get("def_out_db") or 0
        items.append(f"opponent defense lists {int(d)} Out/Doubtful"
                     + (f" ({int(db)} in the secondary)" if db else ""))
    return items

"""Observed participation evidence: historical snap counts, and an honest route blocker.

What is free, and what it measures
----------------------------------
* ``snap_counts_<season>`` (nflverse, from Pro-Football-Reference): per player per game
  ``offense_snaps`` / ``offense_pct`` (share of the team's offensive snaps), plus defense
  and special teams.  Identity is ``pfr_player_id``; it links to gsis only through a
  unique ``pfr_id`` in the supplied players table.
* ``pbp_participation_<season>`` (nflverse, FTN-sourced from 2023): on-field players per
  play plus a ``route`` column that is the route of the TARGETED receiver only (one per
  pass play).  It is not per-player routes run, and no 2026 file is published.  So
  routes are unavailable from free data; targets are never used as a route proxy.

This module ingests snap counts only with the pinned columns, the requested season,
weeks strictly before the target week, and per-week game coverage stated -- never a
zero for a player/game that is simply absent.  Observed snaps are context
(``role_usage``, ``measurement_kind="observed"``): they describe past participation,
not expected future workload.  ``expected_workload`` stays unavailable until a model
passes validation (Opportunity G2 failed).  Nothing here is fetched at import.

Public entrypoints::

    load_snap_counts(frame, *, season, target_week, players=None, source=None,
                     schedule_games=None) -> {"rows": DataFrame, "receipt": dict}
    snap_records(loaded, *, player_id, team, game_id, as_of, last_n=3) -> [record]
    route_availability(season, participation=None, published_assets=()) -> dict
    expected_workload(player_id) -> dict   # always unavailable (shadow) for now
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional

import pandas as pd

SNAP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/snap_counts/"
            "snap_counts_{season}.parquet")
PARTICIPATION_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                     "pbp_participation/pbp_participation_{season}.parquet")
SNAP_COLUMNS = ("game_id", "season", "game_type", "week", "player", "pfr_player_id",
                "position", "team", "opponent", "offense_snaps", "offense_pct")
SNAP_DEFINITION = ("offense_snaps = offensive snaps the player was on the field for "
                   "(Pro-Football-Reference via nflverse); offense_pct = share of the "
                   "team's offensive snaps in that game. Not routes, not targets.")


class ParticipationError(ValueError):
    """Input fails the pinned column / season / week / identity contract."""


def frame_sha256(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def load_snap_counts(frame: pd.DataFrame, *, season: int, target_week: int,
                     players: Optional[pd.DataFrame] = None, source: Optional[Dict] = None,
                     schedule_games: Optional[Dict[int, Iterable[str]]] = None) -> Dict:
    """Validate and link a snap-count frame for use BEFORE ``target_week``.

    Raises ``ParticipationError`` on missing pinned columns, another season, or any
    row at/after ``target_week`` (a future or same-week row cannot be pre-game
    evidence).  ``players``: nflverse players table with ``pfr_id`` and ``gsis_id``;
    rows link only on a unique pfr id.  ``schedule_games``: {week: game_ids} to state
    per-week coverage; without it coverage is reported as "not checked"."""
    missing = [c for c in SNAP_COLUMNS if c not in frame.columns]
    if missing:
        raise ParticipationError(f"snap counts missing pinned columns: {missing}")
    df = frame[list(SNAP_COLUMNS)].copy()
    seasons = set(df["season"].dropna().astype(int))
    if seasons != {int(season)}:
        raise ParticipationError(f"snap counts seasons {sorted(seasons)} != {season}")
    late = df[df["week"].astype(int) >= int(target_week)]
    if len(late):
        raise ParticipationError(f"{len(late)} snap rows at/after target week {target_week} "
                                 f"(weeks {sorted(set(late['week'].astype(int)))})")
    df = df[df["game_type"] == "REG"]
    df["player_id"] = None
    link = {"linked": 0, "unlinked": int(len(df)), "ambiguous_pfr_ids": 0}
    if players is not None and {"pfr_id", "gsis_id"} <= set(players.columns):
        p = players.dropna(subset=["pfr_id", "gsis_id"])
        counts = p.groupby("pfr_id")["gsis_id"].nunique()
        uniq = p[p["pfr_id"].isin(counts[counts == 1].index)].drop_duplicates("pfr_id")
        df["player_id"] = df["pfr_player_id"].map(dict(zip(uniq["pfr_id"], uniq["gsis_id"])))
        link = {"linked": int(df["player_id"].notna().sum()),
                "unlinked": int(df["player_id"].isna().sum()),
                "ambiguous_pfr_ids": int((counts > 1).sum())}
    per_week = {}
    for wk, g in df.groupby("week"):
        have = set(g["game_id"])
        want = set((schedule_games or {}).get(int(wk), ()))
        per_week[int(wk)] = {"games_with_rows": len(have),
                             "games_scheduled": len(want) if want else None,
                             "missing_games": sorted(want - have) if want else "not checked"}
    receipt = {"source": dict(source or {}), "season": int(season), "target_week": int(target_week),
               "weeks": sorted(per_week), "per_week": per_week, "n_rows": int(len(df)),
               "columns": list(SNAP_COLUMNS), "definition": SNAP_DEFINITION,
               "identity": link, "sha256": frame_sha256(df.drop(columns=["player_id"]))}
    return {"rows": df.reset_index(drop=True), "receipt": receipt}


def snap_records(loaded: Dict, *, player_id: str, team: str, game_id: str, as_of,
                 last_n: int = 3) -> List[Dict]:
    """Observed recent offensive snaps for one linked player, as a context record.

    A week the team played but the player has no row is reported as "no row", never
    as zero snaps; a player with no linked rows gets an unavailable record."""
    df, rc = loaded["rows"], loaded["receipt"]
    mine = df[(df["player_id"] == player_id) & (df["team"] == team)].sort_values("week")
    base = dict(factor_id=f"observed_snaps:{player_id}:{game_id}", category="role_usage",
                entity_type="player", entity_id=player_id, team=team, game_id=game_id,
                as_of=as_of, source_url=(rc.get("source") or {}).get("url"),
                source_title="nflverse snap_counts (Pro-Football-Reference)",
                fetched_at=(rc.get("source") or {}).get("fetched_at"),
                published_at=(rc.get("source") or {}).get("last_modified"))
    if mine.empty:
        return [{**base, "measurement_kind": "unavailable", "verified": False, "populated": False,
                 "observation": "No linked snap-count rows before this week",
                 "reason_not_applied": "no rows for this player/team (not zero snaps)"}]
    team_weeks = sorted(set(df[df["team"] == team]["week"].astype(int)))[-last_n:]
    parts = []
    for wk in team_weeks:
        r = mine[mine["week"] == wk]
        parts.append(f"W{wk}: {int(r['offense_snaps'].iloc[0])} snaps "
                     f"({float(r['offense_pct'].iloc[0]):.0%})" if len(r) else f"W{wk}: no row")
    return [{**base, "measurement_kind": "observed", "verified": True, "populated": True,
             "value": [p for p in parts], "unit": "offense_snaps",
             "observation": "Observed offensive snaps, " + "; ".join(parts),
             "reason_not_applied": "observed past participation; not routes and not a forecast "
                                   "of this week's workload",
             "uncertainty": SNAP_DEFINITION}]


def route_availability(season: int, participation: Optional[pd.DataFrame] = None,
                       published_assets: Iterable[str] = ()) -> Dict:
    """Exact blocker for per-player routes from free data."""
    reasons = []
    name = f"pbp_participation_{season}.parquet"
    if published_assets and name not in set(published_assets):
        reasons.append(f"nflverse has not published {name}")
    if participation is not None:
        if "route" not in participation.columns:
            reasons.append("participation file has no 'route' column")
        else:
            reasons.append("participation 'route' is one value per play (the targeted "
                           "receiver's route), not routes run by every player on the field")
    else:
        reasons.append("participation 'route' column (2023+) is the targeted receiver's route "
                       "only; per-player routes run are not in any free nflverse/official file")
    return {"available": False, "measure": "routes_run", "season": int(season),
            "blocker": "; ".join(reasons),
            "not_used": "targets are not a route proxy; paid charting is out of scope"}


def expected_workload(player_id: str) -> Dict:
    return {"player_id": player_id, "status": "shadow_unavailable", "value": None,
            "reason": "no expected-workload model has passed validation (Opportunity G2 "
                      "failed); observed snaps are context only"}

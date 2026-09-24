"""Official (sack- and two-point-excluded) QB pass attempts, as a target and as a feature.

``features._passer_week`` sums nflverse ``pass_attempt``, which is 1 on every
sack and on two-point passing tries.  That sack-inclusive count stays as the
model's existing ``pass_attempts`` / ``roll_pass_attempts`` (team dropback
volume and trained ranker inputs read it and are NOT changed here).  Books settle
``pass_attempts`` on the OFFICIAL stat, so this module derives, from the SAME
play-by-play frame the player-week table is built from:

* ``pass_attempts_official``  = pass_attempts - sacks - non-sack two-point tries
  for that passer-week.  A row whose (season, week, team) has no play-by-play at
  all is left NaN, never silently 0.  No ``sack`` column -> raise.
* ``roll_pass_attempts_official`` = the same prior-games-only rolling transform
  (``features._rolling_shifted``) over each player's rows, so it is clock-valid
  exactly as ``roll_pass_attempts`` is.
* ``asof_roll_official`` = the value a pregame (not-yet-played) row would carry,
  equal to the as-played row's value whenever the player does play that week.

Two-point tries: ``two_point_attempt == 1`` when that column exists, otherwise a
pass play with no ``down`` (the cached play-by-play has no two_point column).
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import features as F

OFFICIAL_COL = "pass_attempts_official"
ROLL_OFFICIAL_COL = "roll_pass_attempts_official"
QB_PBP_COLUMNS = ["season", "week", "game_id", "season_type", "posteam", "pass_attempt",
                  "passer_player_id", "sack", "down"]


def load_qb_pbp(hist: Optional[str] = None) -> pd.DataFrame:
    """REG-season play-by-play with the columns the official count needs."""
    from . import ingest
    hist = hist or ingest.HIST
    paths = [os.path.join(hist, "historical_pbp.parquet")]
    paths += [os.path.join(hist, f"pbp_{s}.parquet") for s in sorted(ingest.extra_seasons_on_disk())]
    import pyarrow.parquet as pq
    frames = []
    for p in paths:
        names = set(pq.read_schema(p).names)
        cols = [c for c in QB_PBP_COLUMNS + ["two_point_attempt"] if c in names]
        frames.append(pd.read_parquet(p, columns=cols))
    df = pd.concat(frames, ignore_index=True)
    return df[df["season_type"] == "REG"].reset_index(drop=True)


def passer_deductions(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, player_id): sacks and non-sack two-point pass tries."""
    if "sack" not in pbp.columns:
        raise ValueError("play-by-play lacks 'sack': official pass attempts cannot be derived")
    p = pbp[pbp["pass_attempt"] == 1].dropna(subset=["passer_player_id"])
    sack = p["sack"].fillna(0) == 1
    if "two_point_attempt" in p.columns:
        two = p["two_point_attempt"].fillna(0) == 1
    elif "down" in p.columns:
        two = p["down"].isna()
    else:
        raise ValueError("play-by-play lacks 'two_point_attempt' and 'down': two-point tries unidentifiable")
    d = pd.DataFrame({"season": p["season"], "week": p["week"], "player_id": p["passer_player_id"],
                      "sacks": sack.astype(float), "two_pt": (two & ~sack).astype(float)})
    return d.groupby(["season", "week", "player_id"], as_index=False)[["sacks", "two_pt"]].sum()


def covered_team_weeks(pbp: pd.DataFrame) -> pd.DataFrame:
    return pbp[["season", "week", "posteam"]].dropna().drop_duplicates().rename(columns={"posteam": "team"})


def official_attempts_raw(pw: pd.DataFrame, pbp: pd.DataFrame) -> np.ndarray:
    """Official attempts aligned with ``pw`` rows (the production builder's hook).

    NaN -- unresolved, never 0 and never the sack-inclusive count -- when the
    play-by-play cannot identify sacks/two-point tries, when the row's
    (season, week, team) has no play-by-play, or when ``pass_attempts`` is NaN.
    """
    n = len(pw)
    if "sack" not in pbp.columns or ("two_point_attempt" not in pbp.columns and "down" not in pbp.columns):
        return np.full(n, np.nan)
    keys = pw[["season", "week", "player_id", "team"]].reset_index(drop=True)
    m = keys.merge(passer_deductions(pbp), on=["season", "week", "player_id"], how="left")
    m = m.merge(covered_team_weeks(pbp).assign(_covered=True), on=["season", "week", "team"], how="left")
    base = pw["pass_attempts"].astype(float).to_numpy()
    off = base - m["sacks"].fillna(0.0).to_numpy() - m["two_pt"].fillna(0.0).to_numpy()
    return np.where(m["_covered"].fillna(False).astype(bool).to_numpy(), off, np.nan)


def add_official_columns(pw: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """``pw`` (as built by features.build_player_week from ``pbp``) plus the official columns.

    Explicit research entry point: raises when the play-by-play has no ``sack``.
    """
    if "sack" not in pbp.columns:
        raise ValueError("play-by-play lacks 'sack': official pass attempts cannot be derived")
    out = pw.drop(columns=[OFFICIAL_COL, ROLL_OFFICIAL_COL], errors="ignore").copy()
    out[OFFICIAL_COL] = official_attempts_raw(out, pbp)
    order = out.sort_values(["player_id", "season", "week"], kind="mergesort").index
    rolled = out.loc[order].groupby("player_id")[OFFICIAL_COL].transform(F._rolling_shifted)
    out[ROLL_OFFICIAL_COL] = rolled.reindex(out.index)
    out.index = pw.index
    return out


def asof_roll_official(pw_off: pd.DataFrame, season: int, week: int,
                       player_ids: Optional[Iterable[str]] = None) -> pd.Series:
    """player_id -> pregame ``roll_pass_attempts_official`` at (season, week).

    Uses only rows strictly before (season, week): the player's history plus one
    placeholder row, rolled with the same transform (placeholder raw value NaN).
    """
    key = pw_off["season"] * 100 + pw_off["week"]
    hist = pw_off[key < season * 100 + week]
    if player_ids is not None:
        hist = hist[hist["player_id"].isin(set(player_ids))]
    out = {}
    for pid, g in hist.sort_values(["season", "week"], kind="mergesort").groupby("player_id"):
        s = pd.concat([g[OFFICIAL_COL].astype(float), pd.Series([np.nan])], ignore_index=True)
        out[pid] = float(F._rolling_shifted(s).iloc[-1])
    return pd.Series(out, dtype=float)

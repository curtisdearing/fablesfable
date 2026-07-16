#!/usr/bin/env python3
"""One-off backfill: stamp player_depth_rank onto an existing data/ml_frame.parquet.

New frames get the column from ml_test.py's frame build (DepthPack attach in
_frame_loop). The stored frame predates the feature; this reproduces the
exact same values from the cached sources (weekly rosters + player_week
usage), reconstructing each row's team from that week's roster restricted to
the game_id's two clubs. Idempotent: re-running overwrites the column.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nflvalue import ingest
from nflvalue.depth_features import DEEP_SENTINEL, DepthPack
from nflvalue.features import build_player_week
from nflvalue.sources import rosters as roster_source

FRAME = ROOT / "data" / "ml_frame.parquet"


def team_of_player(frame: pd.DataFrame, rosters: pd.DataFrame) -> pd.Series:
    parts = frame["game_id"].str.split("_", expand=True)
    away, home = parts[2], parts[3]
    ros = rosters[["season", "week", "team", "player_id"]].drop_duplicates().copy()
    ros["season"] = ros["season"].astype(int)
    ros["week"] = ros["week"].astype(int)
    probe = frame[["season", "week", "player_id"]].copy()
    probe["away_team"], probe["home_team"] = away, home
    m = probe.reset_index().merge(ros, on=["season", "week", "player_id"], how="left")
    m = m[(m["team"] == m["away_team"]) | (m["team"] == m["home_team"])]
    m = m.sort_values(["index", "team"]).drop_duplicates("index").set_index("index")
    team = m["team"].reindex(frame.index)
    if team.isna().any():          # trade-week stragglers: latest prior snapshot
        need = probe[team.isna()]
        fb = (need.reset_index()
              .merge(ros.rename(columns={"week": "week_r"}),
                     on=["season", "player_id"], how="left"))
        fb = fb[(fb["week_r"] <= fb["week"])
                & ((fb["team"] == fb["away_team"]) | (fb["team"] == fb["home_team"]))]
        fb = (fb.sort_values(["index", "week_r"]).drop_duplicates("index", keep="last")
              .set_index("index"))
        team.loc[fb.index] = fb["team"]
    return team


def main() -> None:
    pbp = ingest.load_all_pbp()
    seasons = sorted(int(s) for s in pbp["season"].unique())
    rosters = roster_source.fetch_rosters_weekly(seasons)
    pw = build_player_week(pbp, rosters=rosters)
    pack = DepthPack(rosters, pw)

    frame = pd.read_parquet(FRAME)
    team = team_of_player(frame, rosters)
    unmatched = int(team.isna().sum())
    rank = np.full(len(frame), DEEP_SENTINEL)
    rows = zip(frame["season"].astype(int), frame["week"].astype(int),
               team.fillna(""), frame["player_id"])
    rank = np.array([pack.lookup(s, w, t, p) if t else DEEP_SENTINEL
                     for s, w, t, p in rows])
    frame["player_depth_rank"] = rank
    frame.to_parquet(FRAME, index=False)
    dist = pd.Series(rank).value_counts(normalize=True).round(4).to_dict()
    print(f"backfilled {len(frame):,} rows -> {FRAME} "
          f"(team unmatched: {unmatched}); rank distribution: {dist}")


if __name__ == "__main__":
    main()

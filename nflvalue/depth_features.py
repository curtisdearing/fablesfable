"""Pregame usage-depth context: ``player_depth_rank``.

Depth = each player's rank by prior-8-game usage among the players on that
week's roster at the same position (QB=pass_attempts, RB=carries,
WR/TE=targets), per (season, week, team, role). Ranks 1-3 are kept; deeper
players, players with no roster row, and unknown roles read as the DEEP
sentinel 9. The construction is byte-for-byte the audited factor-frame depth
logic (analysis/build_factor_frame.py::prior_depth): roster membership plus
STRICTLY-PRIOR weeks only, so a current-week stat line can never change the
current week's rank (tests pin equivalence against the analysis function).

Evidence (walk-forward GBDT, train < S, lean config set as baseline):
  2021-2024 pooled  top-5 65.15% -> 66.37% (+1.22pp), top-1 68.35% -> 69.64%
  (+1.29pp), log_loss 0.62811 -> 0.62548 (-0.00263); direction unanimous at
  seeds 7 / 1234. 2025 single-shot holdout: top-1 72.06% -> 73.90%, top-5
  67.94% -> 67.79%, log_loss 0.62429 -> 0.62683 (holdout mixed; all-OOS
  2021-2025 pooled +0.94pp top-5 / +1.40pp top-1 / -0.00161 log_loss).
The RAW official-absence flags this depth machinery also supports
(official_rb1_out / official_wr1_out / official_te1_out / opp_db_outs_2plus /
post_bye) FAILED the same pre-registered gate as ranker features (pooled
top-5 -0.40pp, log_loss +0.00211) and are deliberately NOT wired in.
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

FEATURES = ["player_depth_rank"]

DEPTH_METRIC = {"QB": "pass_attempts", "RB": "carries", "WR": "targets", "TE": "targets"}
RANK_CAP = 3          # ranks 1..3 carry signal; deeper is one "bench" bucket
DEEP_SENTINEL = 9.0   # not top-3 depth / not on that week's roster
PRIOR_GAMES = 8


def _history_index(pw: pd.DataFrame) -> Dict[Tuple, Tuple[List[int], List[float]]]:
    """(team, player, role) -> (sorted season*100+week keys, usage values)."""
    histories: Dict[Tuple, Tuple[List[int], List[float]]] = {}
    for role, metric in DEPTH_METRIC.items():
        subset = pw[pw["role"] == role].sort_values(["team", "player_id", "season", "week"])
        for (team, player), group in subset.groupby(["team", "player_id"], sort=False):
            keys = (group["season"].astype(int) * 100 + group["week"].astype(int)).tolist()
            histories[(team, player, role)] = (keys, group[metric].astype(float).tolist())
    return histories


def prior_depth_ranks(rosters: pd.DataFrame, pw: pd.DataFrame) -> pd.DataFrame:
    """Rank that week's roster using only each player's prior eight games.

    Identical semantics to analysis/build_factor_frame.py::prior_depth --
    kept in production form here so the live pipeline never imports the
    research package. tests/test_ml_depth_features.py pins the equivalence.
    """
    histories = _history_index(pw)
    roster = rosters.rename(columns={"position": "role"}).copy()
    roster = roster[roster["role"].isin(DEPTH_METRIC)].copy()
    scored = []
    for row in roster.itertuples(index=False):
        key = int(row.season) * 100 + int(row.week)
        history_keys, values = histories.get((row.team, row.player_id, row.role), ([], []))
        cutoff = bisect.bisect_left(history_keys, key)
        prior = values[max(0, cutoff - PRIOR_GAMES):cutoff]
        scored.append(float(np.mean(prior)) if prior else 0.0)
    roster["prior_depth_score"] = scored
    roster = roster.sort_values(
        ["season", "week", "team", "role", "prior_depth_score", "player_id"],
        ascending=[True, True, True, True, False, True],
    )
    roster["depth_rank"] = roster.groupby(["season", "week", "team", "role"]).cumcount() + 1
    return roster[["season", "week", "team", "player_id", "role",
                   "prior_depth_score", "depth_rank"]]


class DepthPack:
    """Precomputed (season, week, team, player) -> depth rank, walk-forward.

    ``rosters`` is the weekly roster feed (season, week, team, position,
    player_id); ``pw`` is features.build_player_week output (must carry
    role/team/usage columns). For a live week whose roster snapshot has not
    landed yet, ``attach`` falls back to the team's latest prior snapshot in
    the same season -- still strictly pregame.
    """

    def __init__(self, rosters: pd.DataFrame, pw: pd.DataFrame):
        ranked = prior_depth_ranks(rosters, pw)
        ranked = ranked[ranked["depth_rank"] <= RANK_CAP]
        self.rank: Dict[Tuple[int, int, str, str], int] = {
            (int(r.season), int(r.week), str(r.team), str(r.player_id)): int(r.depth_rank)
            for r in ranked.itertuples(index=False)}
        self._team_weeks: Dict[Tuple[int, str], List[int]] = {}
        for (s, w, t, _p) in self.rank:
            self._team_weeks.setdefault((s, t), [])
        for (s, w, t, _p) in self.rank:
            wk = self._team_weeks[(s, t)]
            if w not in wk:
                bisect.insort(wk, w)

    def lookup(self, season: int, week: int, team: str, player_id: str) -> float:
        got = self.rank.get((season, week, team, player_id))
        if got is not None:
            return float(got)
        weeks = self._team_weeks.get((season, team), [])
        if weeks and week not in weeks:
            # roster snapshot for this week not on disk yet (live edge):
            # latest prior snapshot the same season, still pregame
            i = bisect.bisect_right(weeks, week) - 1
            if i >= 0:
                got = self.rank.get((season, weeks[i], team, player_id))
                if got is not None:
                    return float(got)
        return DEEP_SENTINEL

    def attach(self, cands: pd.DataFrame) -> pd.DataFrame:
        cands = cands.copy()
        cands["player_depth_rank"] = [
            self.lookup(int(r.season), int(r.week), str(r.team), str(r.player_id))
            for r in cands[["season", "week", "team", "player_id"]].itertuples(index=False)]
        return cands


def attach_neutral(cands: pd.DataFrame) -> pd.DataFrame:
    """No roster/usage inputs: stamp NaN (GBDT treats it as native missing)."""
    cands = cands.copy()
    for f in FEATURES:
        cands[f] = np.nan
    return cands

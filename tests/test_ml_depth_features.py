"""player_depth_rank: prior-usage-only depth ranking, roster gating, live
fallback, neutral stamping, config resolution, and byte-equivalence with the
audited analysis/build_factor_frame.py depth logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import depth_features as dfm  # noqa: E402
from nflvalue import ml_ranker as mlr  # noqa: E402


def _pw(rows):
    base = {"pass_attempts": 0.0, "carries": 0.0, "targets": 0.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def _rosters(rows):
    return pd.DataFrame(rows)


def _rb(season, week, team, pid, carries):
    return {"season": season, "week": week, "team": team, "player_id": pid,
            "role": "RB", "carries": float(carries)}


def _ros(season, week, team, pid, pos="RB"):
    return {"season": season, "week": week, "team": team,
            "position": pos, "player_id": pid}


def test_rank_orders_by_prior_usage_only():
    # A carries 20/wk in weeks 1-4; B carries 5/wk. In week 5, B explodes for
    # 40 carries -- the CURRENT week's stat line must not touch week 5's rank.
    pw = _pw([_rb(2024, w, "BUF", "A", 20) for w in (1, 2, 3, 4)]
             + [_rb(2024, w, "BUF", "B", 5) for w in (1, 2, 3, 4)]
             + [_rb(2024, 5, "BUF", "B", 40)])
    ros = _rosters([_ros(2024, 5, "BUF", p) for p in ("A", "B")])
    pack = dfm.DepthPack(ros, pw)
    assert pack.lookup(2024, 5, "BUF", "A") == 1.0
    assert pack.lookup(2024, 5, "BUF", "B") == 2.0


def test_prior_window_is_eight_games():
    # 10 prior games: only the LAST eight count. A's last 8 average 10 vs
    # B's steady 12 -> B outranks A despite A's early monster games.
    pw = _pw([_rb(2024, w, "BUF", "A", 50) for w in (1, 2)]
             + [_rb(2024, w, "BUF", "A", 10) for w in range(3, 11)]
             + [_rb(2024, w, "BUF", "B", 12) for w in range(1, 11)])
    ros = _rosters([_ros(2024, 11, "BUF", p) for p in ("A", "B")])
    pack = dfm.DepthPack(ros, pw)
    assert pack.lookup(2024, 11, "BUF", "B") == 1.0
    assert pack.lookup(2024, 11, "BUF", "A") == 2.0


def test_roster_membership_gates_ranking():
    # A has the usage history but is NOT on the week-5 roster: B is rank 1,
    # A reads deep-bench sentinel.
    pw = _pw([_rb(2024, w, "BUF", "A", 20) for w in (1, 2, 3, 4)]
             + [_rb(2024, w, "BUF", "B", 5) for w in (1, 2, 3, 4)])
    ros = _rosters([_ros(2024, 5, "BUF", "B")])
    pack = dfm.DepthPack(ros, pw)
    assert pack.lookup(2024, 5, "BUF", "B") == 1.0
    assert pack.lookup(2024, 5, "BUF", "A") == dfm.DEEP_SENTINEL


def test_rank_cap_reads_deep_sentinel():
    pw = _pw([_rb(2024, w, "BUF", p, c) for w in (1, 2, 3)
              for p, c in (("A", 20), ("B", 15), ("C", 10), ("D", 5))])
    ros = _rosters([_ros(2024, 4, "BUF", p) for p in "ABCD"])
    pack = dfm.DepthPack(ros, pw)
    assert pack.lookup(2024, 4, "BUF", "C") == 3.0
    assert pack.lookup(2024, 4, "BUF", "D") == dfm.DEEP_SENTINEL


def test_live_week_falls_back_to_latest_prior_snapshot():
    # No roster snapshot for week 6 yet (live edge): week 5's ranks serve.
    pw = _pw([_rb(2024, w, "BUF", "A", 20) for w in (1, 2, 3, 4)])
    ros = _rosters([_ros(2024, 5, "BUF", "A")])
    pack = dfm.DepthPack(ros, pw)
    assert pack.lookup(2024, 6, "BUF", "A") == 1.0
    # ...but an unseen team still reads deep
    assert pack.lookup(2024, 6, "NYJ", "A") == dfm.DEEP_SENTINEL


def test_attach_and_attach_neutral():
    pw = _pw([_rb(2024, w, "BUF", "A", 20) for w in (1, 2, 3, 4)])
    ros = _rosters([_ros(2024, 5, "BUF", "A")])
    pack = dfm.DepthPack(ros, pw)
    cands = pd.DataFrame([
        {"season": 2024, "week": 5, "team": "BUF", "player_id": "A"},
        {"season": 2024, "week": 5, "team": "BUF", "player_id": "ZZ"},
    ])
    out = pack.attach(cands)
    assert out["player_depth_rank"].tolist() == [1.0, dfm.DEEP_SENTINEL]
    neutral = dfm.attach_neutral(cands)
    assert neutral["player_depth_rank"].isna().all()


def test_matches_analysis_prior_depth():
    # the audited research implementation and the production pack must rank
    # identically -- drift between them would silently fork the evidence
    from analysis.build_factor_frame import prior_depth
    rng = np.random.default_rng(20260716)
    rows, ros_rows = [], []
    for team in ("BUF", "MIA"):
        for pid in range(6):
            role = ["RB", "WR", "TE"][pid % 3]
            metric = {"RB": "carries", "WR": "targets", "TE": "targets"}[role]
            for w in range(1, 9):
                if rng.random() < 0.8:
                    rows.append({"season": 2024, "week": w, "team": team,
                                 "player_id": f"{team}{pid}", "role": role,
                                 "pass_attempts": 0.0, "carries": 0.0, "targets": 0.0,
                                 metric: float(rng.integers(0, 25))})
            for w in range(1, 10):
                ros_rows.append({"season": 2024, "week": w, "team": team,
                                 "position": role, "player_id": f"{team}{pid}"})
    pw, ros = pd.DataFrame(rows), pd.DataFrame(ros_rows)
    ours = dfm.prior_depth_ranks(ros, pw)
    theirs = prior_depth(ros, pw)
    merged = ours.merge(theirs, on=["season", "week", "team", "player_id", "role"],
                        suffixes=("_prod", "_analysis"))
    assert len(merged) == len(ours) == len(theirs)
    pd.testing.assert_series_equal(
        merged["depth_rank_prod"], merged["depth_rank_analysis"],
        check_names=False)
    pd.testing.assert_series_equal(
        merged["prior_depth_score_prod"], merged["prior_depth_score_analysis"],
        check_names=False)


def test_config_and_feature_space_resolve_player_depth_rank():
    assert "player_depth_rank" in mlr.NUMERIC_FEATURES
    cols = mlr.feature_columns()          # shipped config subset
    assert "player_depth_rank" in cols

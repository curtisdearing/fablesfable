"""The live board must apply the same opponent factor the backtest scores.

RED against 44de45e. ``build_opp_pos_def`` only emits rows for weeks that have
been PLAYED, so ``enumerate_candidates`` found no ``(season, week, opp, role)``
row for a live week and ``projection.project`` silently used
``opp_factor = 1.0`` for rushing_yards / receiving_yards / passing_yards --
while ``prop_backtest.py`` and every evaluation applied the prior-weeks-only
factor (range ~0.72-1.16). Measured on production 2026 Week 4: Hampton
rushing_yards served 59.034 against the as-played replay 49.48 (SEA RB factor
0.845); Worthy receiving_yards 34.371 vs 39.67 (LV WR factor 1.159).

Same class as the live team-week row (PR #30), same controlling property: an
as-of opponent row built with NO knowledge of week W carries exactly the
factors the as-played row for week W carries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflvalue.candidates import WeekInputs, enumerate_candidates
from nflvalue.features import (
    asof_opp_pos_def,
    build_opp_pos_def,
    build_player_week,
    build_team_week,
)

SEASON, WEEK = 2020, 8
FACTOR_COLS = ["roll_games", "roll_ypt_allowed_factor", "roll_ypc_allowed_factor",
               "roll_ypa_allowed_factor", "roll_epa_allowed_factor"]
YARDS_MARKETS = ("rushing_yards", "receiving_yards", "passing_yards")


def _truncated(pbp: pd.DataFrame) -> pd.DataFrame:
    keep = (pbp["season"] < SEASON) | ((pbp["season"] == SEASON) & (pbp["week"] < WEEK))
    return pbp[keep].copy()


@pytest.fixture(scope="module")
def opd_pair(pbp_fast):
    full = build_opp_pos_def(pbp_fast)
    trunc = build_opp_pos_def(_truncated(pbp_fast))
    return full, trunc


def test_asof_opp_rows_equal_the_as_played_rows_for_that_week(opd_pair):
    """The controlling property. Exact equality, every defteam, every role."""
    full, trunc = opd_pair
    assert trunc[(trunc["season"] == SEASON) & (trunc["week"] == WEEK)].empty
    played = full[(full["season"] == SEASON) & (full["week"] == WEEK)]
    asof = asof_opp_pos_def(trunc, SEASON, WEEK, defteams=sorted(played["defteam"].unique()))
    assert len(asof) == played["defteam"].nunique() * 4
    m = played.merge(asof, on=["defteam", "role"], suffixes=("_played", "_asof"))
    assert len(m) == len(played)
    for c in FACTOR_COLS:
        pd.testing.assert_series_equal(m[c + "_played"].reset_index(drop=True),
                                       m[c + "_asof"].reset_index(drop=True), check_names=False)
    # the placeholder never invents a stat
    assert asof["roll_games"].ge(1).all()


def test_asof_opp_rows_are_prior_weeks_only(opd_pair, pbp_fast):
    """Poison every play from week 8 on: the week-8 as-of rows must not move."""
    _full, trunc = opd_pair
    base = asof_opp_pos_def(trunc, SEASON, WEEK)
    poisoned = pbp_fast.copy()
    later = (poisoned["season"] == SEASON) & (poisoned["week"] >= WEEK)
    poisoned.loc[later, "passing_yards"] = 999.0
    poisoned.loc[later, "rushing_yards"] = 999.0
    poisoned.loc[later, "epa"] = 9.0
    alt = asof_opp_pos_def(build_opp_pos_def(poisoned), SEASON, WEEK)
    pd.testing.assert_frame_equal(base, alt)


def test_live_candidates_carry_the_opponent_factor(pbp_fast, schedules_fast, opd_pair):
    """Live path: cut the play-by-play before the target week and enumerate.

    Before the fix every yards row carried ``opp_factor == 1.0``; now each
    carries the prior-weeks-only factor from the full-history table.
    """
    full, trunc = opd_pair
    pbp = _truncated(pbp_fast)
    inputs = WeekInputs(pw=build_player_week(pbp), opd=trunc, tw=build_team_week(pbp),
                        schedules=schedules_fast.copy())
    df = enumerate_candidates(SEASON, WEEK, inputs=inputs, roster_mode="carry_forward")
    yards = df[df["market"].isin(YARDS_MARKETS)]
    assert len(yards) > 30
    assert (yards["opp_source"] == "asof").all()
    factors = yards["components"].map(lambda c: c["opp_factor"]).to_numpy(float)
    assert np.mean(np.isclose(factors, 1.0)) < 0.2, "live rows still default to opp_factor 1.0"
    played = full[(full["season"] == SEASON) & (full["week"] == WEEK)].set_index(["defteam", "role"])
    col = {"QB": "roll_ypa_allowed_factor", "WR": "roll_ypt_allowed_factor",
           "TE": "roll_ypt_allowed_factor", "RB": "roll_ypc_allowed_factor"}
    for r in yards.itertuples(index=False):
        key = (r.defteam, r.pos)
        if key in played.index:
            assert r.components["opp_factor"] == pytest.approx(float(played.loc[key, col[r.pos]]), abs=1e-4)
    # counting markets never use the factor
    assert df[~df["market"].isin(YARDS_MARKETS)]["opp_source"].isna().all()


def test_played_weeks_are_unchanged(pbp_fast, schedules_fast, opd_pair):
    full, _trunc = opd_pair
    inputs = WeekInputs(pw=build_player_week(pbp_fast), opd=full, tw=build_team_week(pbp_fast),
                        schedules=schedules_fast.copy())
    df = enumerate_candidates(SEASON, WEEK, inputs=inputs, roster_mode="as_played")
    yards = df[df["market"].isin(YARDS_MARKETS)]
    assert (yards["opp_source"] == "played").all()


# --------------------------------------------------------------------------- #
# Behavioural coverage: wholly unplayed, mixed, sparse-role, completed weeks.
# The mixed-week test FAILS BY ASSERTION on ca7c495 (first repair), whose
# whole-week guard dropped every as-of row once one game had been played.
# --------------------------------------------------------------------------- #
def _first_completed_game(schedules_fast):
    slate = schedules_fast[(schedules_fast["season"] == SEASON) & (schedules_fast["week"] == WEEK)]
    return str(slate.sort_values(["gameday", "gametime"]).iloc[0]["game_id"])


def _factor_col(pos):
    return {"QB": "roll_ypa_allowed_factor", "WR": "roll_ypt_allowed_factor",
            "TE": "roll_ypt_allowed_factor", "RB": "roll_ypc_allowed_factor"}[pos]


def _enumerate(pbp_prior, opd, schedules_fast):
    inputs = WeekInputs(pw=build_player_week(pbp_prior), opd=opd, tw=build_team_week(pbp_prior),
                        schedules=schedules_fast.copy())
    df = enumerate_candidates(SEASON, WEEK, inputs=inputs, roster_mode="carry_forward")
    return df[df["market"].isin(YARDS_MARKETS)].copy()


def test_mixed_week_keeps_asof_rows_for_the_unplayed_games(pbp_fast, schedules_fast, opd_pair):
    """One completed game must not strip the factor from every other game.

    Opponent table = prior weeks + the first completed game of the week;
    player/team inputs held constant (the reviewer's construction).
    """
    full, _ = opd_pair
    gid = _first_completed_game(schedules_fast)
    prior = _truncated(pbp_fast)
    one_game = pbp_fast[pbp_fast["game_id"] == gid]
    assert not one_game.empty
    opd_mixed = build_opp_pos_def(pd.concat([prior, one_game], ignore_index=True))
    assert ((opd_mixed["season"] == SEASON) & (opd_mixed["week"] == WEEK)).sum() > 0
    yards = _enumerate(prior, opd_mixed, schedules_fast)
    done = yards[yards["game_id"] == gid]
    rest = yards[yards["game_id"] != gid]
    assert len(rest) > 30
    assert (done["opp_source"] == "played").all()
    assert (rest["opp_source"] == "asof").all(), rest["opp_source"].value_counts().to_dict()
    # and the as-of factors are exactly the as-played ones for those keys
    played = full[(full["season"] == SEASON) & (full["week"] == WEEK)].set_index(["defteam", "role"])
    for r in rest.itertuples(index=False):
        assert r.components["opp_factor"] == pytest.approx(float(played.loc[(r.defteam, r.pos), _factor_col(r.pos)]), abs=1e-4)
    # paired against the wholly-unplayed enumeration, the remaining games' factors are unchanged
    base = _enumerate(prior, opd_pair[1], schedules_fast)
    keys = ["game_id", "player_id", "market"]
    m = base.merge(rest, on=keys, suffixes=("_b", "_a"))
    assert len(m) == len(rest)
    fb = m["components_b"].map(lambda c: c["opp_factor"]).to_numpy(float)
    fa = m["components_a"].map(lambda c: c["opp_factor"]).to_numpy(float)
    np.testing.assert_allclose(fa, fb)


def test_sparse_role_history_gives_a_justified_neutral_not_a_missing_row(pbp_fast, schedules_fast, opd_pair):
    """A defteam with NO prior rows for one role gets an as-of row whose factor
    is the league prior (1.0) with ``roll_games == 0`` -- present and labelled,
    never ``missing`` -- while its other roles keep informed factors."""
    _, trunc = opd_pair
    prior = _truncated(pbp_fast)
    slate = schedules_fast[(schedules_fast["season"] == SEASON) & (schedules_fast["week"] == WEEK)]
    team = str(slate.iloc[-1]["home_team"])
    sparse = trunc[~((trunc["defteam"] == team) & (trunc["role"] == "TE"))]
    asof = asof_opp_pos_def(sparse, SEASON, WEEK, defteams=[team]).set_index("role")
    assert asof.loc["TE", "roll_games"] == 0
    assert asof.loc["TE", "roll_ypt_allowed_factor"] == pytest.approx(1.0)
    assert asof.loc["WR", "roll_games"] >= 1
    yards = _enumerate(prior, sparse, schedules_fast)
    te = yards[(yards["defteam"] == team) & (yards["pos"] == "TE")]
    if len(te):
        assert (te["opp_source"] == "asof").all()
        assert (te["opp_roll_games"] == 0).all()
        assert te["components"].map(lambda c: c["opp_factor"]).eq(1.0).all()
    others = yards[(yards["defteam"] == team) & (yards["pos"] != "TE")]
    assert (others["opp_source"] == "asof").all()
    assert (others["opp_roll_games"] >= 1).all()
    assert (yards["opp_source"] != "missing").all()


def test_completed_week_matches_the_baseline_path(pbp_fast, schedules_fast, opd_pair):
    """Fully played week: every row is ``played`` and the factors are the ones
    the full-history table carries (what 44de45e already did)."""
    full, _ = opd_pair
    inputs = WeekInputs(pw=build_player_week(pbp_fast), opd=full, tw=build_team_week(pbp_fast),
                        schedules=schedules_fast.copy())
    df = enumerate_candidates(SEASON, WEEK, inputs=inputs, roster_mode="as_played")
    yards = df[df["market"].isin(YARDS_MARKETS)]
    assert (yards["opp_source"] == "played").all()
    played = full[(full["season"] == SEASON) & (full["week"] == WEEK)].set_index(["defteam", "role"])
    for r in yards.itertuples(index=False):
        assert r.components["opp_factor"] == pytest.approx(float(played.loc[(r.defteam, r.pos), _factor_col(r.pos)]), abs=1e-4)

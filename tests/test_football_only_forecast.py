"""The player forecast is football-only: book odds, consensus and game lines never move it.

RED against 87c4d13: the live path tilted volume by the sportsbook spread, so
perturbing ``spread_line`` changed projected means.  An offered threshold may
change P(over); it may not change the forecast (mean, sd).
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from nflvalue.candidates import WeekInputs, enumerate_candidates
from nflvalue.features import build_opp_pos_def, build_player_week, build_team_week

SEASON, WEEK = 2020, 8
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def inputs(pbp_fast, schedules_fast):
    pbp = pbp_fast[(pbp_fast["season"] < SEASON)
                   | ((pbp_fast["season"] == SEASON) & (pbp_fast["week"] < WEEK))]
    return WeekInputs(pw=build_player_week(pbp), opd=build_opp_pos_def(pbp),
                      tw=build_team_week(pbp), schedules=schedules_fast.copy())


def _run(inputs, schedules=None, prop_lines=None, **kw):
    inp = inputs if schedules is None else WeekInputs(inputs.pw, inputs.opd, inputs.tw, schedules)
    df = enumerate_candidates(SEASON, WEEK, inputs=inp, roster_mode="carry_forward",
                              prop_lines=prop_lines, **kw)
    return df.set_index(["game_id", "player_id", "market"]).sort_index()


def _lines(base, price_over, price_under, consensus, shift=0.0):
    rows = base.reset_index().head(40)
    return pd.DataFrame({
        "game_id": rows["game_id"], "market": rows["market"], "player_id": rows["player_id"],
        "point": rows["line"] + shift, "over_price": price_over, "under_price": price_under,
        "book": "fixture", "consensus_p_over": consensus, "n_books": 3})


def _perturbed_schedule(s):
    s = s.copy()
    mask = (s["season"] == SEASON) & (s["week"] == WEEK)
    s.loc[mask, "spread_line"] = -s.loc[mask, "spread_line"].fillna(0) + 7.5
    s.loc[mask, "total_line"] = s.loc[mask, "total_line"].fillna(44) + 10
    for c in ("home_moneyline", "away_moneyline"):
        if c in s:
            s.loc[mask, c] = -s.loc[mask, c]
    return s


def test_forecast_is_invariant_to_book_odds_consensus_and_game_lines(inputs):
    base = _run(inputs)
    assert len(base) > 50
    a = _run(inputs, prop_lines=_lines(base, 1.91, 1.91, 0.50))
    b = _run(inputs, schedules=_perturbed_schedule(inputs.schedules),
             prop_lines=_lines(base, 2.60, 1.45, 0.31))
    assert a.index.equals(b.index)
    for col in ("mean", "sd"):
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)
    # same threshold -> same probability, whatever the price or consensus
    pd.testing.assert_series_equal(a["p_over"], b["p_over"], check_names=False)


def test_threshold_moves_probability_not_the_forecast(inputs):
    base = _run(inputs)
    a = _run(inputs, prop_lines=_lines(base, 1.91, 1.91, 0.5))
    b = _run(inputs, prop_lines=_lines(base, 1.91, 1.91, 0.5, shift=+3.0))
    moved = a["line"] != b["line"]
    assert moved.sum() >= 20
    pd.testing.assert_series_equal(a["mean"], b["mean"], check_names=False)
    pd.testing.assert_series_equal(a["sd"], b["sd"], check_names=False)
    assert (b.loc[moved, "p_over"] < a.loc[moved, "p_over"]).all()


def test_spread_is_a_comparator_arm_never_the_primary(inputs):
    from nflvalue import football_forecast as ff
    assert ff.PRIMARY_MARGIN_SOURCE in ("neutral", "football")
    base = _run(inputs)
    assert set(base["margin_source"]) == {ff.PRIMARY_MARGIN_SOURCE}
    assert set(base["forecast_version"]) == {ff.FORECAST_VERSION}
    # the comparator arm really does read the book: that is why it is not primary
    s0 = _run(inputs, margin_source="spread")
    s1 = _run(inputs, schedules=_perturbed_schedule(inputs.schedules), margin_source="spread")
    assert not np.allclose(s0["mean"], s1["mean"])
    with pytest.raises(ValueError):
        _run(inputs, margin_source="consensus")


def test_football_margin_reads_scores_only_and_never_the_target_or_later(schedules_fast):
    from nflvalue.football_forecast import football_margins
    ref = football_margins(schedules_fast, SEASON, WEEK)
    assert len(ref) >= 26 and any(v != 0 for v in ref.values())
    s = _perturbed_schedule(schedules_fast)
    later = (s["season"] > SEASON) | ((s["season"] == SEASON) & (s["week"] > WEEK))
    s.loc[later, ["home_score", "away_score"]] = [99, 0]
    assert football_margins(s, SEASON, WEEK) == ref
    # as of each game's own date: its own score (and anything on or after that
    # date) never feeds its margin
    target = s[(s["season"] == SEASON) & (s["week"] == WEEK)]
    for g in target.itertuples():
        t = s.copy()
        t.loc[pd.to_datetime(t["gameday"]) >= pd.to_datetime(g.gameday), ["home_score", "away_score"]] = [0, 99]
        got = football_margins(t, SEASON, WEEK)
        assert got[g.home_team] == ref[g.home_team] and got[g.away_team] == ref[g.away_team]


def test_dispersion_primary_replaces_sd_and_shadow_only_reports():
    from nflvalue.football_forecast import dispersion_fields
    params = {"version": "t", "markets": {
        "receptions": {"a": 1.0, "b": 0.5, "sd_floor": 0.5},
        "passing_yards": {"a": 100.0, "b": 0.0, "sd_floor": 50.0}},
        "decisions": {"receptions": "D1_primary", "passing_yards": "D0_primary_D1_shadow"}}
    rec = {"mean": 4.0, "sd": 2.0, "line": 3.5, "dist": "negbinom", "p_over": 0.6}
    out = dispersion_fields("receptions", rec, params)
    assert out["dispersion_role"] == "primary" and out["sd"] == 2.0 == out["sd_conditional"]
    assert out["sd_pooled"] == 2.0 and out["p_over_pooled"] == 0.6
    qb = {"mean": 240.0, "sd": 90.0, "line": 230.5, "dist": "normal", "p_over": 0.54}
    out = dispersion_fields("passing_yards", qb, params)
    assert out["dispersion_role"] == "shadow" and "sd" not in out and out["sd_conditional"] == 100.0
    assert dispersion_fields("rushing_yards", qb, params)["dispersion_role"] is None
    assert dispersion_fields("receptions", rec, None)["dispersion_role"] is None


def test_shipped_dispersion_decisions_match_the_recorded_evaluation():
    disp = json.load(open(os.path.join(ROOT, "data", "dispersion_v1.json")))
    res = json.load(open(os.path.join(ROOT, "analysis", "football_only_results.json")))
    proto = os.path.join(ROOT, "analysis", "football_only_protocol.json")
    import hashlib
    sha = hashlib.sha256(open(proto, "rb").read()).hexdigest()
    assert disp["protocol_sha256"] == res["protocol_sha256"] == sha
    assert disp["decisions"] == {m: r["decision"] for m, r in res["dispersion"].items()}
    assert disp["fit_seasons"] == [2023, 2024]
    from nflvalue.football_forecast import PRIMARY_MARGIN_SOURCE
    want = "football" if res["margin"]["test"]["decision"].startswith("A2") else "neutral"
    assert PRIMARY_MARGIN_SOURCE == want

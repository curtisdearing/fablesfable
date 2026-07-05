"""Phase 6.2: red-zone path in anytime-TD, RZ-share reallocation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflvalue import projection
from nflvalue.candidates import apply_reallocation
from nflvalue.features import build_player_week, build_team_week
from nflvalue.projection import TD_BLEND_W, _rz_lambda, project


def _player(**over):
    base = {"player_id": "p1", "player_name": "P One", "role": "RB",
            "roll_games": 8.0, "roll_carries": 12.0, "roll_targets": 3.0,
            "roll_rush_td_rate": 0.05, "roll_rec_td_rate": 0.04,
            "roll_rz_tgt_share": 0.10, "roll_rz_carry_share": 0.50}
    base.update(over)
    return base


def _team(**over):
    base = {"roll_team_rz_tgt": 6.0, "roll_team_rz_car": 8.0,
            "league_rz_tgt_td_rate": 0.22, "league_rz_car_td_rate": 0.18}
    base.update(over)
    return base


def test_rz_lambda_math_and_missing_data_fallback():
    lam = _rz_lambda(_player(), _team())
    assert abs(lam - (6.0 * 0.10 * 0.22 + 8.0 * 0.50 * 0.18)) < 1e-9
    # missing league rates -> None (base path only)
    assert _rz_lambda(_player(), _team(league_rz_tgt_td_rate=np.nan)) is None
    assert _rz_lambda(_player(), None) is None
    # no shares at all -> None; one share -> the other term drops to 0
    assert _rz_lambda(_player(roll_rz_tgt_share=np.nan, roll_rz_carry_share=np.nan),
                      _team()) is None
    one = _rz_lambda(_player(roll_rz_tgt_share=np.nan), _team())
    assert abs(one - 8.0 * 0.50 * 0.18) < 1e-9


def test_anytime_td_blends_and_degrades():
    p = _player()
    base_lam = 12.0 * 0.05 + 3.0 * 0.04
    rz_lam = 6.0 * 0.10 * 0.22 + 8.0 * 0.50 * 0.18
    with_rz = project(p, "anytime_td", team_row=_team(), line=0.5)
    assert abs(with_rz["mean"] - ((1 - TD_BLEND_W) * base_lam + TD_BLEND_W * rz_lam)) < 1e-3
    assert with_rz["components"]["lam_rz"] is not None
    # no team row -> pure base path, unchanged from the Phase-1 formula
    without = project(p, "anytime_td", team_row=None, line=0.5)
    assert abs(without["mean"] - base_lam) < 1e-3
    assert without["components"]["lam_rz"] is None
    # measured verdict pinned: the opp RZ factor must NOT move the mean
    goalline_sieve = {"roll_rz_td_factor": 1.6}
    same = project(p, "anytime_td", team_row=_team(), opp_row=goalline_sieve, line=0.5)
    assert same["mean"] == with_rz["mean"]


def test_player_and_team_tables_carry_rz_columns(pbp_tiny):
    pw = build_player_week(pbp_tiny)
    tw = build_team_week(pbp_tiny)
    assert {"roll_rz_tgt_share", "roll_rz_carry_share", "roll_gl_carry_share",
            "rz_tgt", "rz_car", "gl_car"} <= set(pw.columns)
    assert {"roll_team_rz_tgt", "roll_team_rz_car",
            "league_rz_tgt_td_rate", "league_rz_car_td_rate"} <= set(tw.columns)
    lg = tw["league_rz_tgt_td_rate"].dropna()
    assert ((lg > 0.05) & (lg < 0.60)).all()   # a rate, not a count


def test_reallocation_routes_rz_share_to_anytime_td():
    cands = pd.DataFrame([
        {"player_id": "ben", "market": "anytime_td", "mean": 0.40, "sd": 0.65,
         "dist": "poisson", "line": 0.5, "p_over": 0.33, "p_under": 0.67},
        {"player_id": "ben", "market": "rushing_yards", "mean": 55.0, "sd": 20.0,
         "dist": "gamma", "line": 50.5, "p_over": 0.55, "p_under": 0.45},
    ])
    realloc = [{"out_player_id": "star", "role": "RB", "basis": "with_without",
                "boosts": {"ben": {"share_with": 0.4, "share_without": 0.44,
                                    "share_delta": 0.04,
                                    "rz_share_with": 0.30, "rz_share_without": 0.375,
                                    "rz_share_delta": 0.075}}}]
    out = apply_reallocation(cands, realloc)
    td = out[out["market"] == "anytime_td"].iloc[0]
    ry = out[out["market"] == "rushing_yards"].iloc[0]
    assert td["realloc_mult"] > 1.0          # TD market boosted via RZ share
    assert td["mean"] > 0.40
    assert td["p_over"] > 0.33               # p_over recomputed
    assert ry["realloc_mult"] > 1.0          # family market still boosted
    # the TD boost rides the RZ delta (1+.075/.30=1.25), the family boost the
    # volume delta (1+.04/.40=1.10) -- distinct shares, distinct multipliers
    assert abs(td["realloc_mult"] - 1.25) < 1e-9
    assert abs(ry["realloc_mult"] - 1.10) < 1e-9


def test_reallocation_without_rz_keys_leaves_td_untouched():
    cands = pd.DataFrame([
        {"player_id": "ben", "market": "anytime_td", "mean": 0.40, "sd": 0.65,
         "dist": "poisson", "line": 0.5, "p_over": 0.33, "p_under": 0.67}])
    realloc = [{"out_player_id": "star", "role": "RB", "basis": "with_without",
                "boosts": {"ben": {"share_with": 0.4, "share_without": 0.5,
                                    "share_delta": 0.1}}}]
    out = apply_reallocation(cands, realloc)
    assert out.iloc[0]["mean"] == 0.40

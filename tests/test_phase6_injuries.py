"""Phase 6.5: opponent-secondary absence factor, durability features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflvalue.candidates import (OPP_ABSENCE_FACTOR_CAP, OPP_DB_OUT_PASS_YDS,
                                 apply_opp_absence_factor)
from nflvalue.composite import score_candidate
from nflvalue.features import _early_exit_week, build_player_week


def _cands():
    return pd.DataFrame([
        {"season": 2024, "week": 5, "defteam": "KC", "market": "receiving_yards",
         "mean": 60.0, "sd": 20.0, "line": 55.5, "p_over": 0.6, "p_under": 0.4,
         "components": {"opp_factor": 1.0, "game_script": 1.0}},
        {"season": 2024, "week": 5, "defteam": "KC", "market": "rushing_yards",
         "mean": 70.0, "sd": 22.0, "line": 65.5, "p_over": 0.6, "p_under": 0.4,
         "components": {"opp_factor": 1.0, "game_script": 1.0}},
        {"season": 2024, "week": 5, "defteam": "SF", "market": "receiving_yards",
         "mean": 60.0, "sd": 20.0, "line": 55.5, "p_over": 0.6, "p_under": 0.4,
         "components": {"opp_factor": 1.0, "game_script": 1.0}},
    ])


def test_opp_absence_factor_stamping_and_cap():
    out = apply_opp_absence_factor(_cands(), {(2024, 5, "KC"): 2})
    rec = out[(out.defteam == "KC") & (out.market == "receiving_yards")].iloc[0]
    rush = out[(out.defteam == "KC") & (out.market == "rushing_yards")].iloc[0]
    clean = out[out.defteam == "SF"].iloc[0]
    expected = 1.0 + OPP_DB_OUT_PASS_YDS * 2 / 243.4
    assert abs(rec["opp_absence_factor"] - round(expected, 4)) < 1e-9
    assert rush["opp_absence_factor"] == 1.0        # rush cleared nothing
    assert clean["opp_absence_factor"] == 1.0
    assert rec["mean"] == 60.0                       # sub-score only, mean untouched
    hammered = apply_opp_absence_factor(_cands(), {(2024, 5, "KC"): 9})
    assert hammered.iloc[0]["opp_absence_factor"] == OPP_ABSENCE_FACTOR_CAP


def test_absence_factor_moves_composite_matchup_only():
    c = _cands().iloc[0].to_dict()
    base = score_candidate(c)
    juiced = score_candidate({**c, "opp_absence_factor": 1.08})
    assert juiced["matchup"] > base["matchup"]
    assert juiced["components"]["absence_sub"] > 0.5 == base["components"]["absence_sub"]


def test_early_exit_detection():
    n_h2 = 12
    pbp = pd.DataFrame({
        "season": [2023] * (3 + n_h2), "week": [1] * (3 + n_h2),
        "posteam": ["KC"] * (3 + n_h2),
        "qtr": [1, 1, 2] + [3, 4] * (n_h2 // 2),
        "receiver_player_id": ["A", "A", "A"] + (["B", None] * (n_h2 // 2)),
        "rusher_player_id": [None] * (3 + n_h2),
        "passer_player_id": [None] * (3 + n_h2),
    })
    ee = _early_exit_week(pbp)
    a = ee[ee.player_id == "A"].iloc[0]
    b = ee[ee.player_id == "B"].iloc[0]
    assert a["early_exit"] == 1.0     # 3 H1 touches, zero H2, team played H2
    assert b["early_exit"] == 0.0


def test_player_week_carries_early_exit_rate(pbp_tiny):
    pw = build_player_week(pbp_tiny)
    assert "roll_early_exit_rate" in pw.columns
    v = pw["roll_early_exit_rate"]
    assert v.notna().all() and ((v >= 0) & (v <= 1)).all()
    assert (v > 0).any()              # somebody left a game early in 2019

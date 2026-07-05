"""Phase 6.6: situational tags, revenge subtypes, historical study harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflvalue.composite import score_candidate
from nflvalue.context_study import study_frame
from nflvalue.situations import (DIVISIONS, TEAM_TZ_OFFSET, game_tags,
                                 revenge_subtype)


def _sched():
    return pd.DataFrame([
        {"game_id": "g_tnf", "season": 2024, "week": 5, "game_type": "REG",
         "home_team": "KC", "away_team": "DEN", "weekday": "Thursday",
         "gametime": "20:15", "home_rest": 4, "away_rest": 4},
        {"game_id": "g_sun_early", "season": 2024, "week": 5, "game_type": "REG",
         "home_team": "NYJ", "away_team": "SEA", "weekday": "Sunday",
         "gametime": "13:00", "home_rest": 7, "away_rest": 7},
        {"game_id": "g_snf", "season": 2024, "week": 5, "game_type": "REG",
         "home_team": "DAL", "away_team": "PHI", "weekday": "Sunday",
         "gametime": "20:20", "home_rest": 7, "away_rest": 7},
    ])


def test_game_tags_flags():
    gt = game_tags(_sched()).set_index(["game_id", "team"])
    assert gt.loc[("g_tnf", "KC"), "primetime"] == 1
    assert gt.loc[("g_tnf", "KC"), "short_week"] == 1
    assert gt.loc[("g_tnf", "KC"), "division_game"] == 1        # KC-DEN AFCW
    assert gt.loc[("g_snf", "DAL"), "primetime"] == 1
    assert gt.loc[("g_snf", "DAL"), "division_game"] == 1       # DAL-PHI NFCE
    assert gt.loc[("g_sun_early", "NYJ"), "primetime"] == 0
    # SEA (PT) at NYJ (ET), 1pm kickoff -> both travel flags
    assert gt.loc[("g_sun_early", "SEA"), "long_travel_2tz"] == 1
    assert gt.loc[("g_sun_early", "SEA"), "west_east_early"] == 1
    assert gt.loc[("g_sun_early", "NYJ"), "long_travel_2tz"] == 0
    assert len(DIVISIONS) >= 32 and len(TEAM_TZ_OFFSET) >= 32


def test_revenge_subtype_classification():
    stints = {"p1": [(2021, w, "KC") for w in range(1, 10)] + [(2022, 1, "DEN")],
              "p2": [(2021, w, "KC") for w in range(1, 10)] + [(2022, 1, "DEN")],
              "p3": [(2021, w, "KC") for w in range(1, 10)] + [(2022, 1, "DEN")],
              "p4": [(2021, 1, "KC"), (2022, 1, "DEN")]}
    trades = {("traded guy", 2021)}
    expiries = {"p2": {2021}}
    assert revenge_subtype("p1", "Traded Guy", 2022, 5, "DEN", "KC",
                           stints, trades, expiries) == "revenge_trade"
    assert revenge_subtype("p2", "Expired Guy", 2022, 5, "DEN", "KC",
                           stints, trades, expiries) == "revenge_fa"
    assert revenge_subtype("p3", "Cut Guy", 2022, 5, "DEN", "KC",
                           stints, trades, expiries) == "revenge_cut"
    # under the 3-week stint floor -> not revenge at all
    assert revenge_subtype("p4", "Brief Guy", 2022, 5, "DEN", "KC",
                           stints, trades, expiries) is None
    # facing a team he never played for -> None
    assert revenge_subtype("p1", "Traded Guy", 2022, 5, "DEN", "SF",
                           stints, trades, expiries) is None


def test_study_frame_machinery_and_gate():
    rng = np.random.default_rng(7)
    n = 4000
    graded = pd.DataFrame({
        "season": 2024, "week": rng.integers(1, 18, n), "player_id": [f"p{i}" for i in range(n)],
        "market": "receiving_yards", "hit": rng.random(n) < 0.5})
    # a tag over 200 rows engineered to hit 75% -- must clear; a 20-row tag must not
    strong_ids = graded.index[:200]
    graded.loc[strong_ids, "hit"] = rng.random(200) < 0.75
    tags = pd.concat([
        graded.loc[strong_ids, ["season", "week", "player_id", "market"]].assign(tag="strong"),
        graded.iloc[300:320][["season", "week", "player_id", "market"]].assign(tag="tiny"),
    ])
    res = study_frame(tags, graded)
    assert res["tags"]["strong"]["verdict"] == "PROMOTABLE"
    assert "proposed_mult" in res["tags"]["strong"]
    assert res["tags"]["tiny"]["verdict"] == "insufficient_n"


def test_situational_flags_never_touch_composite():
    c = {"market": "receiving_yards", "mean": 60.0, "sd": 10.0, "line": 55.5,
         "p_over": 0.67, "p_under": 0.33,
         "components": {"opp_factor": 1.1, "game_script": 1.02}}
    plain = score_candidate(c)
    flagged = score_candidate({**c, "primetime": 1, "division_game": 1,
                               "short_week": 1, "long_travel_2tz": 1,
                               "west_east_early": 1})
    assert plain["composite"] == flagged["composite"]

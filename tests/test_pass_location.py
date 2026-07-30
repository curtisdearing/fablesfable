"""Pass-location features: share math, as-of safety, def-EPA walk-forward,
matchup composition, missing-column degradation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import advanced_features as af  # noqa: E402


def _loc_pbp(rows):
    base = dict(season=2025, game_id="g", season_type="REG", posteam="AAA",
                defteam="BBB", epa=0.0)
    out = []
    for r in rows:
        d = dict(base)
        d.update(r)
        d.setdefault("pass", 1)
        out.append(d)
    return pd.DataFrame(out)


def test_player_location_shares_math():
    pbp = _loc_pbp([
        {"week": 1, "receiver_player_id": "R1", "pass_location": "middle"},
        {"week": 1, "receiver_player_id": "R1", "pass_location": "middle"},
        {"week": 1, "receiver_player_id": "R1", "pass_location": "left"},
        {"week": 1, "receiver_player_id": "R1", "pass_location": "right"},
    ])
    d = af.build_player_target_locations(pbp)
    row = d[(d.player_id == "R1") & (d.week == 1)].iloc[0]
    assert row["loc_middle_share"] == 0.5
    assert row["loc_left_share"] == 0.25


def test_player_location_rolls_across_weeks_equally_weighted():
    pbp = _loc_pbp(
        [{"week": 1, "receiver_player_id": "R1", "pass_location": "middle"}] +
        [{"week": 2, "receiver_player_id": "R1", "pass_location": "left"}])
    d = af.build_player_target_locations(pbp)
    wk2 = d[(d.player_id == "R1") & (d.week == 2)].iloc[0]
    assert wk2["loc_middle_share"] == 0.5           # mean of 1.0 and 0.0 weekly shares


def test_untargeted_and_unlocated_plays_ignored():
    pbp = _loc_pbp([
        {"week": 1, "receiver_player_id": "R1", "pass_location": "middle"},
        {"week": 1, "receiver_player_id": "R1", "pass_location": None},
        {"week": 1, "receiver_player_id": None, "pass_location": "left"},
        {"week": 1, "receiver_player_id": "R1", "pass_location": "left", "pass": 0},
    ])
    d = af.build_player_target_locations(pbp)
    row = d[(d.player_id == "R1") & (d.week == 1)].iloc[0]
    assert row["loc_middle_share"] == 1.0


def test_def_loc_epa_is_strictly_prior():
    """Week w's value must be a function of weeks < w only (shift+EWM)."""
    rows = []
    for wk, e in ((1, 0.5), (2, -0.5), (3, 99.0)):   # wild week-3 value
        rows.extend([{"week": wk, "receiver_player_id": "R1",
                      "pass_location": "middle", "epa": e}] * 3)
    d = af.build_def_loc_epa(_loc_pbp(rows))
    assert np.isnan(d[(2025, 1, "BBB")][1])          # nothing before week 1
    assert d[(2025, 2, "BBB")][1] == 0.5             # week 1 only
    w3 = d[(2025, 3, "BBB")][1]
    assert w3 < 1.0                                   # 99.0 is NOT visible at week 3


def test_matchup_epa_composition_and_fail_closed():
    pbp_hist = _loc_pbp([
        {"week": 1, "receiver_player_id": "R1", "pass_location": "middle"},
        {"week": 1, "receiver_player_id": "R1", "pass_location": "middle"},
    ])
    pack = object.__new__(af.AdvancedPack)            # skip heavy __init__
    pack.team = {}
    pack.rz = af.AsOfLookup(pd.DataFrame(columns=["player_id", "season", "week", "a", "b"]),
                            ["a", "b"])
    pack.ngs = af.AsOfLookup(pd.DataFrame(columns=["player_id", "season", "week", "a", "b", "c"]),
                             ["a", "b", "c"])
    pack.loc = af.AsOfLookup(af.build_player_target_locations(pbp_hist),
                             ["loc_middle_share", "loc_left_share"])
    pack.def_loc = {(2025, 2, "BBB"): (0.1, 0.4, -0.2)}
    pack.qbc, pack.contract, pack.ol_out, pack.dob, pack.weather = {}, {}, {}, {}, {}
    cands = pd.DataFrame([
        {"season": 2025, "week": 2, "player_id": "R1", "team": "AAA",
         "defteam": "BBB", "game_id": "g", "gameday": None},
        {"season": 2025, "week": 2, "player_id": "GHOST", "team": "AAA",
         "defteam": "BBB", "game_id": "g", "gameday": None},
    ])
    out = pack.attach(cands)
    r1 = out.iloc[0]
    # R1: mid=1.0, left=0.0, right=0.0 -> matchup = 0.4 (pure middle)
    assert r1["loc_middle_share"] == 1.0
    assert r1["loc_matchup_epa"] == 0.4
    ghost = out.iloc[1]                               # no history -> everything NaN
    assert np.isnan(ghost["loc_middle_share"])
    assert np.isnan(ghost["loc_matchup_epa"])


def test_degrades_when_pass_location_column_missing():
    pbp = _loc_pbp([{"week": 1, "receiver_player_id": "R1",
                     "pass_location": "middle"}]).drop(columns=["pass_location"])
    assert af.build_player_target_locations(pbp).empty
    assert af.build_def_loc_epa(pbp) == {}


def test_new_features_registered():
    for f in ("loc_middle_share", "loc_left_share", "loc_matchup_epa"):
        assert f in af.FEATURES

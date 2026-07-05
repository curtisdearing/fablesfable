"""Phase 6.1: depth/location shape tilts, archetype priors, fixed matchup
weighting, and the carry-forward (live-mode) as-of lookups."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nflvalue import projection
from nflvalue.composite import MATCHUP_SUB_WEIGHTS, score_candidate
from nflvalue.features import (ARCHETYPE_MIN_GAMES, RB_RECEIVING_MIX,
                               WR_DEEP_ADOT, _assign_archetype,
                               build_opp_pos_def, build_player_week)
from nflvalue.projection import _one_tilt, shape_tilts


# --------------------------------------------------------------------------- #
# Tilt math
# --------------------------------------------------------------------------- #
def test_tilt_is_exactly_neutral_when_profile_matches_league():
    assert _one_tilt(0.6, 0.6, 1.3, 0.8) == 1.0


def test_tilt_direction_short_profile_vs_short_funnel_defense():
    # defense is soft short (1.2) / stingy deep (0.9); a shorter-than-league
    # profile player should tilt UP, a deeper-than-league player DOWN
    up = _one_tilt(0.85, 0.60, 1.2, 0.9)
    down = _one_tilt(0.30, 0.60, 1.2, 0.9)
    assert up > 1.0 > down


def test_tilt_clipped_and_nan_safe():
    hi = _one_tilt(1.0, 0.0, 1.6, 0.6)
    lo = _one_tilt(0.0, 1.0, 1.6, 0.6)
    assert hi <= 1.0 + projection.TILT_CLIP + 1e-12
    assert lo >= 1.0 - projection.TILT_CLIP - 1e-12
    assert _one_tilt(float("nan"), 0.6, 1.2, 0.9) is None
    assert _one_tilt(0.5, None, 1.2, 0.9) is None


def test_shape_tilts_market_routing():
    player = {"roll_short_tgt_share": 0.9, "roll_mid_tgt_share": 0.5,
              "roll_short_pass_share": 0.9}
    opp = {"roll_shape_short": 1.2, "roll_shape_deep": 0.9,
           "roll_shape_mid": 1.1, "roll_shape_out": 0.95,
           "league_short_share": 0.6, "league_mid_share": 0.25}
    rec = shape_tilts(player, opp, "receiving_yards")
    assert set(rec) == {"depth", "location"}
    qb = shape_tilts(player, opp, "passing_yards")
    assert set(qb) == {"depth"}          # QBs tilt on throw depth only
    assert shape_tilts(player, opp, "rushing_yards") == {}
    assert shape_tilts(player, None, "receiving_yards") == {}


def test_project_applies_tilt_into_opp_factor_and_components():
    player = {"player_id": "x", "player_name": "X", "role": "WR",
              "roll_games": 8.0, "roll_target_share": 0.25, "roll_ypt": 8.0,
              "roll_targets": 6.0, "roll_short_tgt_share": 0.9,
              "roll_mid_tgt_share": 0.25}
    team = {"roll_team_pass_att": 32.0}
    opp = {"roll_ypt_allowed_factor": 1.0, "roll_shape_short": 1.2,
           "roll_shape_deep": 0.9, "roll_shape_mid": 1.0, "roll_shape_out": 1.0,
           "league_short_share": 0.6, "league_mid_share": 0.25}
    with_tilt = projection.project(player, "receiving_yards", team_row=team,
                                   opp_row=opp, line=50.5)
    assert "shape_tilts" in with_tilt["components"]
    assert with_tilt["components"]["opp_factor"] > 1.0   # short profile, short-soft D
    # flipping the module switch removes the tilt (the ablation path)
    projection.TILTS_ENABLED["depth"] = False
    projection.TILTS_ENABLED["location"] = False
    try:
        without = projection.project(player, "receiving_yards", team_row=team,
                                     opp_row=opp, line=50.5)
        assert without["components"]["opp_factor"] == 1.0
    finally:
        projection.TILTS_ENABLED["depth"] = True
        projection.TILTS_ENABLED["location"] = True


# --------------------------------------------------------------------------- #
# Archetypes
# --------------------------------------------------------------------------- #
def test_archetype_assignment_rules():
    pw = pd.DataFrame({
        "role":       ["RB", "RB", "WR", "WR", "TE", "QB", "RB"],
        "roll_games": [8, 8, 8, 8, 8, 8, ARCHETYPE_MIN_GAMES - 1],
        "roll_targets": [4.0, 1.0, 6.0, 6.0, 4.0, 0.0, 9.0],
        "roll_carries": [6.0, 14.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "roll_adot":  [1.0, 1.0, WR_DEEP_ADOT + 1, WR_DEEP_ADOT - 3, 7.0, np.nan, 1.0],
    })
    arch = _assign_archetype(pw).tolist()
    assert arch[0] == "RB_receiving"      # 4/(4+6)=.4 >= .35
    assert arch[1] == "RB_early_down"     # 1/15 < .35
    assert arch[2] == "WR_deep"
    assert arch[3] == "WR_short"
    assert arch[4] == "generic"           # TE: no honest free split
    assert arch[5] == "generic"           # QB
    assert arch[6] == "generic"           # under the trailing-games gate
    assert RB_RECEIVING_MIX == 0.35       # documented constant, pinned


def test_player_week_carries_archetype_and_profiles(pbp_tiny):
    pw = build_player_week(pbp_tiny)
    assert {"archetype", "roll_short_tgt_share", "roll_mid_tgt_share",
            "roll_short_pass_share"} <= set(pw.columns)
    assert set(pw["archetype"].unique()) <= {
        "generic", "RB_receiving", "RB_early_down", "WR_deep", "WR_short"}
    # profiles are shares
    for c in ("roll_short_tgt_share", "roll_mid_tgt_share", "roll_short_pass_share"):
        v = pw[c].dropna()
        assert ((v >= 0) & (v <= 1)).all(), c


# --------------------------------------------------------------------------- #
# Opponent table: full-grid rows (no missingness signal)
# --------------------------------------------------------------------------- #
def test_opp_shape_and_rz_exist_on_every_defense_week(pbp_tiny):
    opd = build_opp_pos_def(pbp_tiny)
    # every (season, week, defteam) row carries a value (trailing or neutral),
    # so NaN can never encode "nothing happened THIS week"
    for c in ("roll_shape_short", "roll_shape_deep", "roll_rz_td_factor"):
        assert opd[c].notna().all(), f"{c} has holes -- missingness leak risk"


# --------------------------------------------------------------------------- #
# Composite: fixed matchup weighting
# --------------------------------------------------------------------------- #
def _cand(**over):
    base = {"market": "receiving_yards", "mean": 60.0, "sd": 10.0, "line": 55.5,
            "p_over": 0.67, "p_under": 0.33,
            "components": {"opp_factor": 1.15, "game_script": 1.05}}
    base.update(over)
    return base


def test_matchup_weights_are_fixed_and_neutral_filled():
    assert abs(sum(MATCHUP_SUB_WEIGHTS.values()) - 1.0) < 1e-9
    # a candidate with NO epa data must score identically to one whose epa
    # factor is exactly neutral -- the dimension no longer drops out
    no_epa = score_candidate(_cand())
    neutral_epa = score_candidate(_cand(opp_epa_factor=1.0))
    assert no_epa["composite"] == neutral_epa["composite"]
    assert no_epa["components"]["epa_sub"] == 0.5
    # a soft-epa defense should move the matchup for an over
    soft = score_candidate(_cand(opp_epa_factor=1.12))
    assert soft["matchup"] > no_epa["matchup"]
    # reserved 6.5 dimension: present-but-neutral today
    assert no_epa["components"]["absence_sub"] == 0.5


# --------------------------------------------------------------------------- #
# Live-mode as-of lookups (the dead-opp-factor fix)
# --------------------------------------------------------------------------- #
def test_carry_forward_uses_asof_rows(pbp_tiny):
    from nflvalue.candidates import WeekInputs, build_week_inputs, enumerate_candidates
    from nflvalue.features import build_team_week
    inputs = WeekInputs(build_player_week(pbp_tiny), build_opp_pos_def(pbp_tiny),
                        build_team_week(pbp_tiny), _sched_2019())
    # week 18 of 2019 never happened (17-game era starts 2021): a "live" slate
    # for (2020, 1) has no exact rows -- as-of must serve the freshest priors
    row = inputs.opp_row_asof(2020, 1, "KC", "WR")
    assert row is not None and row["week"] == max(
        w for (s, w) in {(r.season, r.week) for r in inputs.opd[
            inputs.opd.defteam.eq("KC") & inputs.opd.role.eq("WR")].itertuples()} if s == 2019)
    assert inputs.team_row_asof(2020, 1, "KC") is not None
    # exact lookups still win when the row exists (backtest path unchanged)
    exact = inputs.opp_idx.get((2019, 10, "KC", "WR"))
    assert exact is not None


def _sched_2019():
    import os
    import pandas as pd
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sched = pd.read_parquet(os.path.join(root, "historical_lines.parquet"))
    return sched[sched["season"] == 2019]

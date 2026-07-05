"""Phase 6.3: garbage-time filter on core rolling stats + measured PROE term
in the deterministic game-script split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflvalue.features import (GARBAGE_Q4_MARGIN, _garbage_mask,
                               build_player_week, build_team_week)
from nflvalue.projection import (PROE_MULT_CAP, PROE_SPLIT_COEF,
                                 game_script_multipliers)


def test_garbage_mask_definition():
    pbp = pd.DataFrame({
        "qtr":                [4, 4, 4, 2, 4, 4],
        "score_differential": [21, -18, 3, 30, 3, np.nan],
        "wp":                 [0.99, 0.02, 0.5, 0.5, 0.03, 0.5],
    })
    m = _garbage_mask(pbp).tolist()
    assert m[0] is True or m[0] == True      # Q4 blowout (margin)
    assert m[1] == True                      # Q4 blowout (both)
    assert m[2] == False                     # Q4 close game
    assert m[3] == False                     # Q2 blowout is NOT garbage (yet)
    assert m[4] == True                      # Q4 wp-extreme, close score
    assert m[5] == False                     # NaN-safe
    assert GARBAGE_Q4_MARGIN == 17


def test_garbage_filter_changes_rates_not_actuals(pbp_tiny):
    on = build_player_week(pbp_tiny, garbage_filter=True)
    off = build_player_week(pbp_tiny, garbage_filter=False)
    key = ["season", "week", "player_id"]
    m = on[key + ["targets", "rec_yards", "roll_ypt", "roll_target_share"]].merge(
        off[key + ["targets", "rec_yards", "roll_ypt", "roll_target_share"]],
        on=key, suffixes=("_on", "_off"))
    # ACTUALS are full-game either way (grading must not change)
    assert (m["targets_on"] == m["targets_off"]).all()
    assert (m["rec_yards_on"] == m["rec_yards_off"]).all()
    # the RATE inputs genuinely differ for some rows
    assert (m["roll_ypt_on"] != m["roll_ypt_off"]).any()
    assert (m["roll_target_share_on"] != m["roll_target_share_off"]).any()
    # ...but the league LEVEL is preserved by the recalibration ratio (<1% drift)
    lvl_on = on.loc[on.role.eq("WR"), "roll_ypt"].mean()
    lvl_off = off.loc[off.role.eq("WR"), "roll_ypt"].mean()
    assert abs(lvl_on - lvl_off) / lvl_off < 0.01


def test_proe_term_tilts_split_and_caps():
    base = game_script_multipliers(0.0)
    assert base == {"pass_mult": 1.0, "rush_mult": 1.0}
    passy = game_script_multipliers(0.0, neutral_proe=8.0, trail_pass_share=0.58)
    runy = game_script_multipliers(0.0, neutral_proe=-8.0, trail_pass_share=0.58)
    assert passy["pass_mult"] > 1.0 > passy["rush_mult"]
    assert runy["pass_mult"] < 1.0 < runy["rush_mult"]
    # magnitude matches the fitted coefficient (t=+3.1, 2019-2023)
    expected = 1.0 + PROE_SPLIT_COEF * (8.0 / 100.0) / 0.58
    assert abs(passy["pass_mult"] - round(expected, 4)) < 1e-3
    # cap engages on absurd PROE, and the spread tilt still composes
    capped = game_script_multipliers(-10.0, neutral_proe=50.0, trail_pass_share=0.58)
    spread_only = game_script_multipliers(-10.0)["pass_mult"]
    assert capped["pass_mult"] <= round(spread_only * (1.0 + PROE_MULT_CAP), 4) + 1e-9
    assert capped["pass_mult"] > spread_only            # proe still helped, within cap
    # missing ingredients -> exact pre-6.3 spread-only behavior
    legacy = game_script_multipliers(-6.5)
    also = game_script_multipliers(-6.5, neutral_proe=None, trail_pass_share=0.58)
    assert legacy == also
    assert PROE_MULT_CAP == 0.03


def test_team_week_carries_script_columns(pbp_tiny):
    tw = build_team_week(pbp_tiny)
    assert {"roll_team_plays", "league_plays_prior", "roll_team_neutral_proe"} <= set(tw.columns)
    plays = tw["roll_team_plays"].dropna()
    assert ((plays > 40) & (plays < 90)).all()
    proe = tw["roll_team_neutral_proe"].dropna()
    # pass_oe percentage points; a team's FIRST week rolls a single game, so
    # the tails are wide -- sanity-bound only
    assert proe.abs().max() < 60

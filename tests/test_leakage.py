"""The #1 kill bug test (PHASE1_HANDSOFF_DESIGN.md): a feature attached to
row (season, week) must never change when data from (season, week) or later
is removed from the input. If it does, some rolling/expanding computation is
reading data it shouldn't be able to see yet.

Method: build each feature table on the full fixture, then again after
deleting every play at-or-after a cutoff week. Rows strictly before the
cutoff must be byte-for-byte identical between the two builds.
"""

from __future__ import annotations

import pandas as pd

from nflvalue.features import build_opp_pos_def, build_player_week, build_team_week

CUTOFF_SEASON, CUTOFF_WEEK = 2020, 8


def _before_cutoff(df: pd.DataFrame) -> pd.Series:
    return (df["season"] < CUTOFF_SEASON) | ((df["season"] == CUTOFF_SEASON) & (df["week"] < CUTOFF_WEEK))


def _truncated(pbp: pd.DataFrame) -> pd.DataFrame:
    keep = (pbp["season"] < CUTOFF_SEASON) | ((pbp["season"] == CUTOFF_SEASON) & (pbp["week"] < CUTOFF_WEEK))
    return pbp[keep].copy()


ROLL_COLS_PLAYER = [
    "roll_games", "roll_targets", "roll_target_share", "roll_air_yards", "roll_adot",
    "roll_carries", "roll_carry_share", "roll_pass_attempts", "roll_completions",
    "roll_ypt", "roll_catch_rate", "roll_ypc", "roll_ypa",
    "roll_pass_td_rate", "roll_rush_td_rate", "roll_rec_td_rate",
    # Phase 6.1: depth/location profiles + the archetype label itself (it
    # feeds the shrinkage prior, so a leaked label would leak the prior)
    "roll_short_tgt_share", "roll_mid_tgt_share", "roll_short_pass_share",
    "archetype",
    # Phase 6.2 red-zone shares + 6.5 durability
    "roll_rz_tgt_share", "roll_rz_carry_share", "roll_gl_carry_share",
    "roll_early_exit_rate",
]
ROLL_COLS_OPP = [
    "roll_games", "roll_ypt_allowed_factor", "roll_ypc_allowed_factor",
    "roll_ypa_allowed_factor", "roll_epa_allowed_factor",
    # Phase 6.1: depth/location shapes + red-zone defense
    "roll_shape_short", "roll_shape_deep", "roll_shape_mid", "roll_shape_out",
    "league_short_share", "league_mid_share", "roll_rz_td_factor",
]
ROLL_COLS_TEAM = ["roll_team_pass_att", "roll_team_rush_att"]


def test_player_week_features_do_not_leak_future_weeks(pbp_fast):
    """Also covers role assignment (checked alongside the roll_* columns
    below) so this only has to build player_week -- the slow step -- once
    per fixture instead of twice."""
    full = build_player_week(pbp_fast)
    trunc = build_player_week(_truncated(pbp_fast))

    full_before = full[_before_cutoff(full)].sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    trunc_before = trunc.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    assert len(full_before) == len(trunc_before), "row count mismatch before the cutoff"
    key_cols = ["season", "week", "player_id"]
    check_cols = ROLL_COLS_PLAYER + ["role"]
    merged = full_before[key_cols + check_cols].merge(
        trunc_before[key_cols + check_cols], on=key_cols, suffixes=("_full", "_trunc"))
    assert len(merged) == len(full_before)

    for col in check_cols:
        a, b = merged[f"{col}_full"], merged[f"{col}_trunc"]
        mismatched = ~(a.eq(b) | (a.isna() & b.isna()))
        # allow tiny float noise, but nothing structural
        if mismatched.any() and a.dtype.kind in "fc":
            close = (a - b).abs() < 1e-9
            mismatched = mismatched & ~close.fillna(False)
        assert not mismatched.any(), (
            f"player_week leakage in '{col}': {mismatched.sum()} row(s) before the cutoff "
            f"changed when future weeks were removed -- future data is leaking into a "
            f"past feature.\n{merged.loc[mismatched, ['season','week','player_id', f'{col}_full', f'{col}_trunc']].head()}"
        )


def test_opp_pos_def_features_do_not_leak_future_weeks(pbp_fast):
    full = build_opp_pos_def(pbp_fast)
    trunc = build_opp_pos_def(_truncated(pbp_fast))

    full_before = full[_before_cutoff(full)].sort_values(["defteam", "role", "season", "week"]).reset_index(drop=True)
    trunc_before = trunc.sort_values(["defteam", "role", "season", "week"]).reset_index(drop=True)
    assert len(full_before) == len(trunc_before)

    key_cols = ["season", "week", "defteam", "role"]
    merged = full_before[key_cols + ROLL_COLS_OPP].merge(
        trunc_before[key_cols + ROLL_COLS_OPP], on=key_cols, suffixes=("_full", "_trunc"))
    assert len(merged) == len(full_before)

    for col in ROLL_COLS_OPP:
        a, b = merged[f"{col}_full"], merged[f"{col}_trunc"]
        mismatched = ~(a.eq(b) | (a.isna() & b.isna()))
        if mismatched.any():
            close = (a - b).abs() < 1e-9
            mismatched = mismatched & ~close.fillna(False)
        assert not mismatched.any(), f"opp_pos_def leakage in '{col}': {mismatched.sum()} row(s) changed"


def test_team_week_features_do_not_leak_future_weeks(pbp_fast):
    full = build_team_week(pbp_fast)
    trunc = build_team_week(_truncated(pbp_fast))

    full_before = full[_before_cutoff(full)].sort_values(["team", "season", "week"]).reset_index(drop=True)
    trunc_before = trunc.sort_values(["team", "season", "week"]).reset_index(drop=True)
    assert len(full_before) == len(trunc_before)

    key_cols = ["season", "week", "team"]
    merged = full_before[key_cols + ROLL_COLS_TEAM].merge(
        trunc_before[key_cols + ROLL_COLS_TEAM], on=key_cols, suffixes=("_full", "_trunc"))
    assert len(merged) == len(full_before)

    for col in ROLL_COLS_TEAM:
        a, b = merged[f"{col}_full"], merged[f"{col}_trunc"]
        mismatched = ~(a.eq(b) | (a.isna() & b.isna()))
        if mismatched.any():
            close = (a - b).abs() < 1e-9
            mismatched = mismatched & ~close.fillna(False)
        assert not mismatched.any(), f"team_week leakage in '{col}': {mismatched.sum()} row(s) changed"



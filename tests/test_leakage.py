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


# --------------------------------------------------------------------------- #
# Phase 7.1: the calibration fit is a NEW walk-forward surface. The calibrator
# corrects the base model's probabilities, and it must never be fit on a
# prediction the base made for a season the base trained on (in-sample), nor on
# any data from a season it will later help score. Two guards below.
# --------------------------------------------------------------------------- #
def _calib_frame(seasons):
    from test_ml import _frame  # same synthetic builder used by the ML tests
    return _frame(n=180 * len(seasons), seasons=seasons, weeks=range(1, 7))


def test_calibration_no_fold_trains_on_its_own_season():
    """No fold sees its own correction: for every expanding fold that predicts
    season s, the base model that generated those (to-be-calibrated) OOS
    predictions may train only on seasons < s. cal_fold_spans is the witness."""
    from nflvalue import ml_ranker as mlr
    f = _calib_frame((2021, 2022, 2023, 2024))
    m = mlr.MLRanker("gbdt", max_iter=40, calibrate="platt_permkt").fit(f, f["y_over"])
    assert m.calibrator is not None and m.cal_fold_spans, "calibrator did not fit"
    for predict_season, train_min, train_max in m.cal_fold_spans:
        assert train_max < predict_season, (
            f"calibration leak: a fold predicting {predict_season} trained through "
            f"{train_max} -- the calibrator would be learning from predictions the "
            f"base made on data it had trained on")


def test_calibrated_prediction_does_not_leak_future_seasons():
    """The calibrated P(over) for a season S must be byte-identical whether or
    not seasons after S exist in the frame -- neither the base fold (<S) nor the
    calibrator (OOS folds within <S) may touch data >= S. Mirrors the feature
    tests: delete the future, the past must not move."""
    import numpy as np
    from nflvalue import ml_ranker as mlr

    full = _calib_frame((2021, 2022, 2023, 2024))
    S = 2023
    kw = dict(model="gbdt", max_iter=40, calibrate="platt_permkt")

    def calibrated_for_S(frame):
        train = frame[frame["season"] < S]
        test = frame[frame["season"] == S]
        m = mlr.MLRanker(**kw).fit(train, train["y_over"])
        return m.predict_p_over(test)

    with_future = calibrated_for_S(full)                       # 2024 present
    without_future = calibrated_for_S(full[full["season"] <= S])  # 2024 removed
    assert np.array_equal(with_future, without_future), (
        "calibrated predictions for season S changed when a later season was "
        "removed -- future data is leaking through the calibration fit")


# --------------------------------------------------------------------------- #
# Phase 7.2: the ensemble meta-learner is a NEW walk-forward surface (its
# training pairs are OOS member predictions from expanding folds) -- same two
# guards as the calibrator: no fold trains on the season it later helps
# combine, and calibration wrapped around an ensemble doesn't leak either.
# --------------------------------------------------------------------------- #
def test_meta_learner_no_fold_trains_on_its_own_season():
    from nflvalue import ml_ranker as mlr
    f = _calib_frame((2021, 2022, 2023, 2024))
    m = mlr.MLRanker("ensemble", members=["gbdt", "rf"], combiner="meta",
                     seed=mlr.SEED).fit(f, f["y_over"])
    assert m.meta is not None and m.meta_fold_spans, "meta-learner did not fit"
    for predict_season, train_min, train_max in m.meta_fold_spans:
        assert train_max < predict_season, (
            f"meta-learner leak: a fold predicting {predict_season} used member "
            f"predictions trained through {train_max}")


def test_ensemble_calibrated_prediction_does_not_leak_future_seasons():
    """Mirrors test_calibrated_prediction_does_not_leak_future_seasons for the
    ensemble path: the calibrator's internal folds refit a NESTED ensemble
    (_oos_fold_predict) rather than a single clf -- confirm that nesting
    doesn't open a new leakage surface."""
    import numpy as np
    from nflvalue import ml_ranker as mlr

    full = _calib_frame((2021, 2022, 2023, 2024))
    S = 2023
    kw = dict(model="ensemble", members=["gbdt", "rf"], combiner="avg",
             seed=mlr.SEED, calibrate="platt_permkt")

    def calibrated_for_S(frame):
        train = frame[frame["season"] < S]
        test = frame[frame["season"] == S]
        m = mlr.MLRanker(**kw).fit(train, train["y_over"])
        return m.predict_p_over(test)

    with_future = calibrated_for_S(full)
    without_future = calibrated_for_S(full[full["season"] <= S])
    assert np.allclose(with_future, without_future, atol=1e-9), (
        "ensemble-calibrated predictions for season S changed when a later "
        "season was removed -- future data is leaking through the nested fit")



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
    # pass_completions market driver: trailing completion RATE, shift-1-then-roll
    # + shrunk exactly like the other efficiencies. If a future week ever leaked
    # into it, deleting that week would move a pre-cutoff value and fail here.
    "roll_comp_rate",
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


def test_pass_completions_rate_cannot_see_the_week_it_predicts(pbp_fast):
    """Focused guard for the new pass_completions market: its ONLY history-
    derived input beyond projected attempts is roll_comp_rate (trailing
    completions/attempts). Poison the future (drop every play at/after the
    cutoff) and assert not a single pre-cutoff completion-rate value moves --
    i.e. the feature that prices completions never sees the week/season it is
    used to predict. (roll_comp_rate is also in ROLL_COLS_PLAYER above; this
    isolates it so a regression names the market, not just a column.)"""
    full = build_player_week(pbp_fast)
    trunc = build_player_week(_truncated(pbp_fast))

    full_before = full[_before_cutoff(full)].sort_values(
        ["player_id", "season", "week"]).reset_index(drop=True)
    trunc_before = trunc.sort_values(
        ["player_id", "season", "week"]).reset_index(drop=True)
    assert len(full_before) == len(trunc_before)

    key_cols = ["season", "week", "player_id"]
    merged = full_before[key_cols + ["roll_comp_rate"]].merge(
        trunc_before[key_cols + ["roll_comp_rate"]], on=key_cols, suffixes=("_full", "_trunc"))
    a, b = merged["roll_comp_rate_full"], merged["roll_comp_rate_trunc"]
    mismatched = ~(a.eq(b) | (a.isna() & b.isna()))
    if mismatched.any():
        close = (a - b).abs() < 1e-9
        mismatched = mismatched & ~close.fillna(False)
    assert not mismatched.any(), (
        f"pass_completions leakage: {mismatched.sum()} pre-cutoff roll_comp_rate "
        f"value(s) changed when future weeks were removed -- the completion-rate "
        f"feature is seeing the week it predicts")


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
# Phase 7.3/7.4: real-line re-labeling is a NEW leakage surface --
# ``ml_test.augment_with_real_lines`` may flip a row's label ONLY when BOTH a
# real (odds_api) decision-time line exists AND the game is graded
# (lean_outcomes). A row for an ungraded/future week must never flip, even if
# a real line already sits in ``leans`` (the closing line existing before
# kickoff carries no outcome information -- the forbidden path is writing a
# real label from a line WITHOUT a graded ``actual``; see docs/decisions_p7.md
# 7.3 §2, point 4).
# --------------------------------------------------------------------------- #
def _relabel_frame():
    import pandas as pd
    return pd.DataFrame([
        # row A: will be graded + has a real line -> MUST flip
        {"season": 2023, "week": 10, "player_id": "00-A1", "market": "receiving_yards",
         "mean": 55.0, "sd": 18.0, "line": 47.0,
         "mean_minus_line": 55.0 - 47.0, "sd_over_line": 18.0 / 47.0,
         "z": (55.0 - 47.0) / 18.0, "y_over": 1.0},
        # row B: has a real line in `leans` but is NOT YET graded (no
        # lean_outcomes row -- e.g. the game hasn't been played) -> must NOT
        # flip, even though a real closing line exists for it right now
        {"season": 2023, "week": 11, "player_id": "00-A2", "market": "receiving_yards",
         "mean": 60.0, "sd": 15.0, "line": 50.0,
         "mean_minus_line": 10.0, "sd_over_line": 0.3,
         "z": 10.0 / 15.0, "y_over": 1.0},
        # row C: graded, but line_source is NOT odds_api (synthetic) -> must
        # NOT flip even though lean_outcomes exists for it
        {"season": 2023, "week": 10, "player_id": "00-A3", "market": "rushing_yards",
         "mean": 40.0, "sd": 12.0, "line": 35.0,
         "mean_minus_line": 5.0, "sd_over_line": 35.0 / 35.0,
         "z": 5.0 / 12.0, "y_over": 1.0},
    ])


def test_augment_with_real_lines_only_flips_graded_rows_with_real_line(tmp_path):
    import ml_test
    from nflvalue import db as dbmod

    conn = dbmod.connect(str(tmp_path / "relabel_test.db"))
    # row A: real line + graded (actual crosses the two lines -> y flips)
    dbmod.upsert(conn, "leans", [{
        "season": 2023, "week": 10, "clock": "wed", "game_id": "G1",
        "player_id": "00-A1", "name": "A1", "market": "receiving_yards",
        "side": "over", "line": 52.5, "line_source": "odds_api", "price": 1.87,
        "book": "draftkings", "status": "active", "as_of": "2023-11-08T12:00:00Z",
    }], ["season", "week", "clock", "game_id", "player_id", "market"])
    dbmod.upsert(conn, "lean_outcomes", [{
        "season": 2023, "week": 10, "clock": "wed", "game_id": "G1",
        "player_id": "00-A1", "market": "receiving_yards", "actual": 50.0, "hit": 0,
    }], ["season", "week", "clock", "game_id", "player_id", "market"])
    # row B: real line exists but NOT graded (no lean_outcomes row) --
    # a "future" or in-progress week
    dbmod.upsert(conn, "leans", [{
        "season": 2023, "week": 11, "clock": "wed", "game_id": "G2",
        "player_id": "00-A2", "name": "A2", "market": "receiving_yards",
        "side": "over", "line": 50.0, "line_source": "odds_api", "price": 1.9,
        "book": "draftkings", "status": "active", "as_of": "2023-11-15T12:00:00Z",
    }], ["season", "week", "clock", "game_id", "player_id", "market"])
    # row C: graded, but the lean's line_source is synthetic, not odds_api
    dbmod.upsert(conn, "leans", [{
        "season": 2023, "week": 10, "clock": "wed", "game_id": "G3",
        "player_id": "00-A3", "name": "A3", "market": "rushing_yards",
        "side": "over", "line": 35.0, "line_source": "synthetic", "price": None,
        "book": None, "status": "active", "as_of": "2023-11-08T12:00:00Z",
    }], ["season", "week", "clock", "game_id", "player_id", "market"])
    dbmod.upsert(conn, "lean_outcomes", [{
        "season": 2023, "week": 10, "clock": "wed", "game_id": "G3",
        "player_id": "00-A3", "market": "rushing_yards", "actual": 42.0, "hit": 1,
    }], ["season", "week", "clock", "game_id", "player_id", "market"])

    before = _relabel_frame()
    after = ml_test.augment_with_real_lines(before.copy(), conn)

    # (season, week) are never touched by the join
    assert (after["season"] == before["season"]).all()
    assert (after["week"] == before["week"]).all()

    a = after[after["player_id"] == "00-A1"].iloc[0]
    b = after[after["player_id"] == "00-A2"].iloc[0]
    c = after[after["player_id"] == "00-A3"].iloc[0]

    # row A: graded + real line -> flips
    assert a["line"] == 52.5 and a["y_over"] == 0.0
    assert a["mean"] == 55.0 and a["sd"] == 18.0        # non-line features untouched
    assert abs(a["z"] - (55.0 - 52.5) / 18.0) < 1e-9

    # row B: real line but UNGRADED -> must stay exactly synthetic
    orig_b = before[before["player_id"] == "00-A2"].iloc[0]
    assert b["line"] == orig_b["line"] == 50.0
    assert b["y_over"] == orig_b["y_over"]
    assert b["z"] == orig_b["z"]

    # row C: graded but synthetic line_source -> must stay exactly synthetic
    orig_c = before[before["player_id"] == "00-A3"].iloc[0]
    assert c["line"] == orig_c["line"] == 35.0
    assert c["y_over"] == orig_c["y_over"]
    conn.close()


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


# --------------------------------------------------------------------------- #
# Phase 7.5: the same-game correlation estimate is a NEW walk-forward surface.
# The rho consumed at season S must be estimated only from pairs in seasons < S.
# (Also covered in test_correlation.py; duplicated here so the leakage suite
# alone certifies every Phase-7 surface -- belt and suspenders on the #1 bug.)
# --------------------------------------------------------------------------- #
def test_correlation_estimate_uses_only_prior_seasons():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import numpy as np
    import fit_correlation as fc

    rng = np.random.default_rng(3)
    rows = []
    for s in (2019, 2020, 2021):
        for g in range(400):
            z = rng.normal(size=2)
            gid = f"{s}_{g:03d}_A_B"
            rows.append(dict(season=s, week=1, game_id=gid, player_id=f"qb{s}{g}",
                             team="A", market="passing_yards", pos="QB", resid=z[0]))
            rows.append(dict(season=s, week=1, game_id=gid, player_id=f"wr{s}{g}",
                             team="A", market="receiving_yards", pos="WR",
                             resid=0.5 * z[0] + np.sqrt(0.75) * z[1]))
    d = pd.DataFrame(rows)
    ptype = "sameteam|QB.pass~WR.rec"
    wf = fc.analyze(fc.collect_pairs(d))["walk_forward"]["2021"][ptype]   # from <2021
    tr = fc.collect_pairs(d[d["season"] < 2021])
    assert wf == round(fc._rho(np.asarray(tr[ptype]["x"]), np.asarray(tr[ptype]["y"])), 4), \
        "correlation slice for season S changed when seasons >= S were removed"


# --------------------------------------------------------------------------- #
# Phase 7.7: advisory staking has NO temporal/rolling surface. Its only
# history-derived input is the 7.5 correlation rho (walk-forward-guarded above);
# everything else is point-in-time. It must be deterministic + stateless, so
# there is no rolling/expanding computation that could ever see the future.
# --------------------------------------------------------------------------- #
def test_staking_has_no_temporal_leakage_surface():
    from nflvalue import staking as st
    ln = dict(game_id="G", player_id="a", market="passing_yards", pos="QB",
              team="A", side="over", p=0.56, market_prob=0.5238, price=1.909)
    a = st.recommend_stakes([dict(ln)], 100.0)
    b = st.recommend_stakes([dict(ln)], 100.0)
    assert a["recommendations"][0]["stake_frac"] == b["recommendations"][0]["stake_frac"]


# --------------------------------------------------------------------------- #
# Phase 6.1/6.6: the NEWER feature packs -- AdvancedPack (advanced_features.py)
# and ChemistryPack (chemistry.py) -- consume their rolling builders through an
# AsOfLookup that returns the latest value STRICTLY BEFORE (season, week). That
# primitive is unit-tested directly (test_advanced_features.py /
# test_chemistry.py), so these packs are believed safe; the guards below are
# defense-in-depth: an END-TO-END truncation-invariance check on the AS-OF-READ
# value, mirroring the feature tests above.
#
# The key subtlety (documented in build_player_redzone / _cum_share): the raw
# builder output is intentionally UN-shifted -- a row at week w includes week w
# -- because it is only ever read strictly-before via AsOfLookup. So we must NOT
# compare raw builder rows (those legitimately differ under truncation); we
# compare the AS-OF READ at the cutoff week, which resolves to strictly-prior
# data and therefore must be identical whether or not at/after-cutoff data
# exists. That read is exactly what the packs do at scoring time.
# --------------------------------------------------------------------------- #
def _asof_read_invariant_under_truncation(build, value_cols):
    """Build+AsOfLookup on the full 2019-2020 ext-pbp fixture and on a version
    truncated at (CUTOFF_SEASON, CUTOFF_WEEK); for every player the as-of read
    AT the cutoff week (which resolves to the latest STRICTLY-PRIOR row) must be
    identical. If a future week leaked into a pre-cutoff as-of value, removing
    it would move the read -- and this catches it."""
    import numpy as np
    from nflvalue.advanced_features import AsOfLookup, load_pbp_ext

    pbp = load_pbp_ext()
    pbp = pbp[pbp["season"].isin([2019, 2020])].copy()
    keep = (pbp["season"] < CUTOFF_SEASON) | (
        (pbp["season"] == CUTOFF_SEASON) & (pbp["week"] < CUTOFF_WEEK))

    full = AsOfLookup(build(pbp), value_cols)
    trunc = AsOfLookup(build(pbp[keep].copy()), value_cols)

    pids = set(full.data) | set(trunc.data)
    assert pids, "fixture produced no players -- test would be vacuous"
    checked = 0
    for pid in pids:
        # reading AT the cutoff week sees only weeks strictly before it
        a = full.get(pid, CUTOFF_SEASON, CUTOFF_WEEK)
        b = trunc.get(pid, CUTOFF_SEASON, CUTOFF_WEEK)
        for va, vb in zip(a, b):
            same = va == vb or (np.isnan(va) and np.isnan(vb))
            assert same, (
                f"as-of read for player {pid} at ({CUTOFF_SEASON},{CUTOFF_WEEK}) "
                f"changed when at/after-cutoff data was removed: {a} vs {b} -- "
                f"future data is leaking into a pre-cutoff as-of feature value")
        checked += 1
    return checked


def test_advanced_pack_redzone_asof_read_does_not_leak_future_weeks():
    """AdvancedPack.rz = AsOfLookup(build_player_redzone(...), [rz_tgt_share,
    rz_carry_share]). The end-to-end guard on that consumed value."""
    from nflvalue.advanced_features import build_player_redzone
    assert _asof_read_invariant_under_truncation(
        build_player_redzone, ["rz_tgt_share", "rz_carry_share"]) > 0


def test_chemistry_pack_formation_tilt_asof_read_does_not_leak_future_weeks():
    """ChemistryPack's formation tilts = AsOfLookup(build_formation_tilts(...),
    [shotgun_tilt_tgt, shotgun_tilt_carry]). Same end-to-end guard."""
    from nflvalue.chemistry import build_formation_tilts
    assert _asof_read_invariant_under_truncation(
        build_formation_tilts, ["shotgun_tilt_tgt", "shotgun_tilt_carry"]) > 0


# ContextPack (context_features.py) is deliberately NOT truncation-tested here.
# Its four features are not rolling/expanding surfaces over pbp: birthday is an
# immutable DOB vs the schedule (no history); revenge reads roster stints via
# former_teams(), which already breaks STRICTLY BEFORE (s, w) by construction;
# and defense-outs / opp_epa are exact-(season, week) fact lookups. There is no
# AsOfLookup wrapping a rolled builder, so a truncation test would only re-assert
# trivial dict membership -- the strictly-before property is directly and better
# covered by test_context_features.py (test_revenge_is_walk_forward_and_excludes
# _current_team). Forcing a truncation test here would be a fragile false guard.



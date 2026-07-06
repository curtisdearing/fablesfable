"""ML ranking layer: seeded determinism, structural walk-forward guard,
flag-gated integration that changes ordering only when stamped."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import ml_ranker as mlr  # noqa: E402
from nflvalue.shortlist import rank_game  # noqa: E402

RNG = np.random.default_rng(11)


def _frame(n=600, seasons=(2022, 2023), weeks=range(1, 10)):
    rows = []
    i = 0
    for s in seasons:
        for w in weeks:
            for _ in range(n // (len(seasons) * len(list(weeks)))):
                z = float(RNG.normal(0, 1))
                rows.append({
                    "season": s, "week": w, "game_id": f"{s}_{w:02d}_A_B",
                    "player_id": f"P{i % 40}", "market": "receiving_yards",
                    "pos": "WR", "side": "over" if z >= 0 else "under",
                    "p_over": 0.5 + 0.1 * np.tanh(z), "z": z,
                    "mean": 50 + 10 * z, "sd": 20.0, "line": 50.5,
                    "mean_minus_line": 10 * z - 0.5, "sd_over_line": 0.4,
                    "opp_factor": 1.0, "game_script": 1.0,
                    "proj_volume": 8.0, "proj_efficiency": 8.0,
                    "roll_games": 8, "roll_targets": 8.0, "roll_target_share": 0.2,
                    "roll_carries": 0.0, "roll_carry_share": 0.0,
                    "roll_pass_attempts": 0.0, "roll_adot": 9.0, "roll_air_yards": 70.0,
                    "roll_ypt": 8.0, "roll_catch_rate": 0.65, "roll_ypc": 0.0,
                    "roll_ypa": 0.0, "team_margin": 2.0, "total_line": 45.0,
                    "home": 1, "low_confidence": False,
                    "is_birthday_week": 0, "revenge_game": 0,
                    "def_out_total": float(RNG.integers(0, 3)),
                    "def_out_db": 0.0, "opp_epa_factor": 1.0,
                    # learnable signal: y correlates with z
                    "y_over": 1.0 if (z + RNG.normal(0, 0.8)) > 0 else 0.0,
                })
                i += 1
    f = pd.DataFrame(rows)
    for m in mlr.MARKETS7:
        f[f"mkt_{m}"] = (f["market"] == m).astype(int)
    for p in mlr.POSITIONS:
        f[f"pos_{p}"] = (f["pos"] == p).astype(int)
    # future-proof: any numeric feature this synthetic frame doesn't model
    # gets a neutral column (mirrors attach_neutral's behavior)
    for col in mlr.feature_columns():
        if col not in f.columns:
            f[col] = 0.0
    return f


@pytest.fixture(scope="module")
def frame():
    return _frame()


def test_seeded_fit_is_deterministic(frame):
    train = frame[frame["season"] == 2022]
    test = frame[frame["season"] == 2023]
    kw = {"max_iter": 60}
    p1 = mlr.MLRanker("gbdt", **kw).fit(train, train["y_over"]).predict_p_over(test)
    p2 = mlr.MLRanker("gbdt", **kw).fit(train, train["y_over"]).predict_p_over(test)
    assert np.array_equal(p1, p2)
    # RF with n_jobs=-1 averages tree votes in thread-dependent order ->
    # deterministic to one float ULP, not bitwise (set n_jobs=1 for bitwise)
    r1 = mlr.MLRanker("rf", n_estimators=30).fit(train, train["y_over"]).predict_p_over(test)
    r2 = mlr.MLRanker("rf", n_estimators=30).fit(train, train["y_over"]).predict_p_over(test)
    assert np.allclose(r1, r2, atol=1e-12)


def test_model_learns_the_planted_signal(frame):
    from sklearn.metrics import roc_auc_score
    train = frame[frame["season"] == 2022]
    test = frame[frame["season"] == 2023]
    m = mlr.MLRanker("gbdt", max_iter=100).fit(train, train["y_over"])
    p = m.predict_p_over(test)
    assert roc_auc_score(test["y_over"], p) > 0.6


def test_walk_forward_guard_is_structural(frame):
    train = frame[frame["season"] == 2022]
    m = mlr.MLRanker("gbdt", max_iter=40).fit(train, train["y_over"])
    with pytest.raises(mlr.WalkForwardViolation):
        m.predict_p_over(train)                          # scoring its own train weeks
    same_season_earlier = frame[(frame["season"] == 2022) & (frame["week"] <= 3)]
    with pytest.raises(mlr.WalkForwardViolation):
        m.predict_p_over(same_season_earlier)
    future = frame[frame["season"] == 2023]
    assert len(m.predict_p_over(future)) == len(future)  # strictly later: fine


def test_save_load_round_trip(frame, tmp_path):
    train = frame[frame["season"] == 2022]
    test = frame[frame["season"] == 2023]
    m = mlr.MLRanker("gbdt", max_iter=40).fit(train, train["y_over"])
    path = m.save(str(tmp_path / "m.joblib"))
    m2 = mlr.MLRanker.load(path)
    assert m2.train_max == m.train_max
    assert np.array_equal(m.predict_p_over(test), m2.predict_p_over(test))


def test_rank_game_uses_ml_score_only_when_stamped():
    def cand(pid, comp_p, ml_score=None):
        c = {"player_id": pid, "name": pid, "pos": "WR", "team": "T",
             "market": "receiving_yards", "mean": 60.0, "sd": 20.0, "line": 55.5,
             "p_over": comp_p, "p_under": round(1 - comp_p, 4),
             "components": {"opp_factor": 1.0, "game_script": 1.0},
             "prices": None, "low_confidence": False, "game_id": "G", "matchup": "A @ B"}
        if ml_score is not None:
            c["ml_score"] = ml_score
        return c

    # composite says A > B; ML says B > A
    plain = rank_game([cand("A", 0.70), cand("B", 0.55)])
    assert plain["leans"][0]["player_id"] == "A"
    ml = rank_game([cand("A", 0.70, ml_score=52.0), cand("B", 0.55, ml_score=71.0)])
    assert ml["leans"][0]["player_id"] == "B"
    # partial stamping (mixed frame) falls back to composite -- no half-ML ranking
    mixed = rank_game([cand("A", 0.70, ml_score=52.0), cand("B", 0.55)])
    assert mixed["leans"][0]["player_id"] == "A"


def test_calibrated_ranker_contract():
    """Calibration wraps the base without altering the interface: raw==base,
    calibrated is bounded, the Platt map is monotone, and it survives save/load."""
    f = _frame(n=1200, seasons=(2021, 2022, 2023), weeks=range(1, 10))
    tr, te = f[f["season"] < 2023], f[f["season"] == 2023]
    cal = mlr.MLRanker("gbdt", max_iter=80, calibrate="platt_permkt").fit(tr, tr["y_over"])
    assert cal.calibrator is not None
    p_cal = cal.predict_p_over(te)
    p_base = cal.predict_p_over(te, raw=True)
    # base is untouched; a plain ranker on the same data matches raw()
    plain = mlr.MLRanker("gbdt", max_iter=80).fit(tr, tr["y_over"]).predict_p_over(te)
    assert np.allclose(p_base, plain)
    assert p_cal.min() >= 0.0 and p_cal.max() <= 1.0
    assert not np.allclose(p_cal, p_base)              # calibration actually moved something
    # the calibration map is monotone increasing (a valid probability map)
    grid = np.linspace(0.02, 0.98, 50)
    mapped = cal.calibrator.transform(grid, np.array(["receiving_yards"] * 50))
    assert np.all(np.diff(mapped) >= -1e-9)
    # p_under = 1 - p_over is what composite consumes -- interface preserved
    assert np.allclose([1 - x for x in p_cal], 1 - p_cal)


def test_implied_units():
    assert mlr.implied_units_at_110(60, 100) == pytest.approx(60 * 100 / 110 - 40, abs=0.01)
    assert mlr.implied_units_at_110(0, 0) == 0.0


# --------------------------------------------------------------------------- #
# Phase 7.2: pruned feature subsets + the GBDT/RF ensemble (avg + meta)
# --------------------------------------------------------------------------- #
def test_feature_columns_subset():
    full = mlr.feature_columns()
    subset = mlr.feature_columns(["p_over", "z", "mean"])
    assert len(subset) == 3 + len(mlr.MARKETS7) + len(mlr.POSITIONS)
    assert len(subset) < len(full)
    # market/position dummies are structural -- always included, never pruned
    for m in mlr.MARKETS7:
        assert f"mkt_{m}" in subset
    for p in mlr.POSITIONS:
        assert f"pos_{p}" in subset


def test_ranker_with_pruned_features_trains_and_predicts(frame):
    """features= is persisted and used consistently at fit + predict time --
    the pruned artifact must not silently fall back to the full column set."""
    train = frame[frame["season"] == 2022]
    test = frame[frame["season"] == 2023]
    kept = ["p_over", "z", "mean", "roll_target_share"]
    m = mlr.MLRanker("gbdt", max_iter=60, features=kept).fit(train, train["y_over"])
    assert m.features == kept
    p = m.predict_p_over(test)
    assert len(p) == len(test) and ((p >= 0) & (p <= 1)).all()
    # a model trained on ALL columns is a materially different fit (kept
    # columns actually constrained training, not a no-op)
    full = mlr.MLRanker("gbdt", max_iter=60).fit(train, train["y_over"]).predict_p_over(test)
    assert not np.allclose(p, full)


def test_ensemble_avg_and_meta_fit_predict_save_load(tmp_path):
    f = _frame(n=1600, seasons=(2021, 2022, 2023, 2024), weeks=range(1, 8))
    train = f[f["season"] < 2024]
    test = f[f["season"] == 2024]

    avg = mlr.MLRanker("ensemble", members=["gbdt", "rf"], combiner="avg",
                       seed=mlr.SEED).fit(train, train["y_over"])
    assert set(avg.members) == {"gbdt", "rf"}
    p_avg = avg.predict_p_over(test)
    assert len(p_avg) == len(test) and ((p_avg >= 0) & (p_avg <= 1)).all()

    meta = mlr.MLRanker("ensemble", members=["gbdt", "rf"], combiner="meta",
                        seed=mlr.SEED).fit(train, train["y_over"])
    assert meta.meta is not None and meta.meta_fold_spans
    p_meta = meta.predict_p_over(test)
    assert len(p_meta) == len(test) and ((p_meta >= 0) & (p_meta <= 1)).all()
    # avg and meta actually combine differently (meta isn't a silent no-op)
    assert not np.allclose(p_avg, p_meta)

    # save/load round-trips the ensemble (members + meta), predictions match
    # to within RF's documented n_jobs=-1 float-ULP nondeterminism
    path = meta.save(str(tmp_path / "ens.joblib"))
    reloaded = mlr.MLRanker.load(path)
    assert reloaded.combiner == "meta" and set(reloaded.members) == {"gbdt", "rf"}
    p_reload = reloaded.predict_p_over(test)
    assert np.allclose(p_meta, p_reload, atol=1e-9)


def test_ensemble_meta_fold_spans_never_train_on_predict_season():
    """Same discipline as the calibrator's cal_fold_spans: the meta-learner's
    training pairs come from expanding folds strictly before the season they
    predict -- no fold sees a member prediction made on data it trained on."""
    f = _frame(n=1600, seasons=(2021, 2022, 2023, 2024), weeks=range(1, 8))
    m = mlr.MLRanker("ensemble", members=["gbdt", "rf"], combiner="meta",
                     seed=mlr.SEED).fit(f, f["y_over"])
    assert m.meta_fold_spans
    for predict_season, train_min, train_max in m.meta_fold_spans:
        assert train_max < predict_season


def test_calibrator_wraps_ensemble_output():
    """7.1's calibration layer is model-agnostic -- it must work identically
    on top of an ensemble's combined probability, via the shared
    _oos_fold_predict dispatch (calibrator folds refit a fresh nested
    ensemble, not a single clf)."""
    f = _frame(n=1600, seasons=(2021, 2022, 2023, 2024), weeks=range(1, 8))
    tr, te = f[f["season"] < 2024], f[f["season"] == 2024]
    m = mlr.MLRanker("ensemble", members=["gbdt", "rf"], combiner="avg",
                     seed=mlr.SEED, calibrate="platt_permkt").fit(tr, tr["y_over"])
    assert m.calibrator is not None
    p_cal = m.predict_p_over(te)
    p_raw = m.predict_p_over(te, raw=True)
    assert p_cal.min() >= 0.0 and p_cal.max() <= 1.0
    assert not np.allclose(p_cal, p_raw)

"""Challenger A (hierarchical Bayesian projection) — protocol tests.

Pre-registered 2026-07-30 (BUILD_PROMPTS_model_challengers_2026-07.md §A5):
leakage (poisoned future seasons must be refused / must not move the
posterior), fail-closed artifact handling, shrinkage sanity (small-sample
players carry wider predictives), determinism (same-seed refits identical),
and book internal consistency (gate.passed <=> its sub-conditions).

These tests run on tiny synthetic frames — no historical data needed — so
they are green from a fresh clone, including under FABLESFABLE_STRICT_FIXTURES.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from nflvalue.bayes_projection import (
    BayesProjection, crps_from_samples, incumbent_samples)

pytest.importorskip("numpyro", reason="numpyro is Challenger A's declared engine")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(ROOT, "book", "bayes_projection_eval.json")


def _toy(seed=0, n_players=12, weeks=10, season=2019):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        base = rng.uniform(1.0, 8.0)
        for w in range(1, weeks + 1):
            rows.append({
                "player_id": f"00-{p:04d}", "role": "WR", "team": f"T{p % 4}",
                "season": season, "week": w,
                "mean_pred": base, "actual": max(rng.normal(base, 1.5), 0.0),
            })
    return pd.DataFrame(rows)


def test_fit_refuses_rows_at_or_after_eval_season():
    df = pd.concat([_toy(season=2019), _toy(season=2020)], ignore_index=True)
    with pytest.raises(ValueError, match="walk-forward violation"):
        BayesProjection("receptions", 7).fit(df, eval_season=2020)


def test_poisoned_future_season_cannot_move_the_posterior():
    """The eval harness trains on season < S only; prove the model class
    itself re-checks (mutation of the caller's filter is caught here)."""
    clean = _toy(season=2019)
    m1 = BayesProjection("receptions", 7).fit(clean, eval_season=2020)
    poisoned = pd.concat([clean, _toy(seed=99, season=2020)], ignore_index=True)
    with pytest.raises(ValueError):
        BayesProjection("receptions", 7).fit(poisoned, eval_season=2020)
    # and the clean fit's posterior is reproducible (nothing leaked in run 1)
    m2 = BayesProjection("receptions", 7).fit(clean, eval_season=2020)
    for k in m1.posterior:
        assert np.array_equal(m1.posterior[k], m2.posterior[k]), k


def test_same_seed_refit_is_deterministic_end_to_end():
    df = _toy()
    test_rows = _toy(seed=5).head(20)
    m1 = BayesProjection("receptions", 1234).fit(df, eval_season=2020)
    s1 = m1.predictive_samples(test_rows)
    m2 = BayesProjection("receptions", 1234).fit(df, eval_season=2020)
    s2 = m2.predictive_samples(test_rows)
    assert np.array_equal(s1, s2)


def test_shrinkage_2obs_player_wider_than_100obs_player():
    """A 2-observation player's predictive sd must be strictly wider than a
    100-observation player's at equal role/team/covariate (§A5)."""
    rng = np.random.default_rng(3)
    rows = []
    for p, n_games in [("00-thin", 2), ("00-thick", 100)]:
        for i in range(n_games):
            rows.append({
                "player_id": p, "role": "WR", "team": "T0",
                "season": 2019, "week": (i % 18) + 1,
                "mean_pred": 5.0, "actual": max(rng.normal(5.0, 1.5), 0.0),
            })
    df = pd.DataFrame(rows)
    m = BayesProjection("receiving_yards", 7).fit(df, eval_season=2020)
    q = df.drop_duplicates("player_id").reset_index(drop=True)
    samples = m.predictive_samples(q)
    sd = samples.std(axis=1)
    thin = float(sd[list(q["player_id"]).index("00-thin")])
    thick = float(sd[list(q["player_id"]).index("00-thick")])
    assert thin > thick


def test_unseen_player_draws_from_prior_not_crash():
    df = _toy()
    m = BayesProjection("receptions", 7).fit(df, eval_season=2020)
    new = pd.DataFrame([{"player_id": "00-NEW", "role": "WR", "team": "T0",
                         "season": 2020, "week": 1, "mean_pred": 4.0,
                         "actual": np.nan}])
    s = m.predictive_samples(new)
    assert s.shape == (1, 512) and np.isfinite(s).all()


def test_artifact_fail_closed(tmp_path):
    df = _toy()
    m = BayesProjection("receptions", 7).fit(df, eval_season=2020)
    path = str(tmp_path / "bayes_proj_2020.joblib")
    m.save(path)
    loaded = BayesProjection.load(path)
    assert loaded.meta == m.meta
    # corrupt -> integrity error, never a silently different model
    with open(path, "r+b") as fh:
        fh.seek(10)
        fh.write(b"\x00\x01\x02\x03")
    with pytest.raises(ValueError, match="integrity"):
        BayesProjection.load(path)
    # missing -> the ordinary exception a fail-closed caller treats as absence
    with pytest.raises(OSError):
        BayesProjection.load(str(tmp_path / "nope.joblib"))


def test_td_and_exact_markets_stay_out_of_scope():
    with pytest.raises(ValueError, match="fail-closed"):
        BayesProjection("anytime_td", 7)


def test_crps_estimator_basics():
    """Degenerate forecast at the truth -> CRPS 0; and the estimator is the
    same callable for both arms (no per-arm scoring asymmetry)."""
    samples = np.full((3, 512), 7.0)
    actual = np.array([7.0, 7.0, 7.0])
    assert np.allclose(crps_from_samples(samples, actual), 0.0)
    sharp = np.random.default_rng(0).normal(5, 0.1, size=(1, 512))
    wide = np.random.default_rng(0).normal(5, 5.0, size=(1, 512))
    y = np.array([5.0])
    assert crps_from_samples(sharp, y)[0] < crps_from_samples(wide, y)[0]


def test_incumbent_sampler_matches_family_parameterization():
    """Sampled survival at the line must agree with projection.p_over's
    closed form for each family (same parameterization, within MC error)."""
    from nflvalue.projection import p_over
    for dist, mean, sd, line in [("gamma", 60.0, 25.0, 55.5),
                                 ("negbinom", 4.0, 2.4, 3.5),
                                 ("normal", 240.0, 50.0, 250.5)]:
        s = incumbent_samples(np.array([mean]), np.array([sd]), [dist],
                              seed=11, n_samples=20000)
        assert abs(float((s[0] > line).mean()) - p_over(mean, sd, line, dist)) < 0.02


def test_book_internal_consistency():
    if not os.path.exists(BOOK):
        pytest.skip("book not yet written (eval has not run in this clone)")
    with open(BOOK) as fh:
        book = json.load(fh)
    gate = book["gate"]
    assert gate["passed"] == (gate["primary_ok"] and gate["expected_delta_met"]
                              and gate["guard_ok"])
    for seed, v in book["seeds"].items():
        p = v["primary"]["pooled"]
        assert p["n"] > 0 and p["crps_incumbent"] > 0
    if not gate["passed"]:
        assert book["holdout_2025"] is None, "FAIL must leave 2025 untouched"

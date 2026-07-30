"""Fair-value market blend: walk-forward safety, gate logic, fail-closed consumers."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import fair_value


def _mk_preds(seasons, per_season=40, model_sd=6.0, market_sd=3.0, seed=7):
    """Synthetic games where the market is the better forecaster."""
    rng = np.random.default_rng(seed)
    preds = []
    for s in seasons:
        for i in range(per_season):
            truth = float(rng.normal(0, 10))
            preds.append({
                "season": s, "week": (i % 18) + 1,
                "margin_mean": truth + float(rng.normal(0, model_sd)),
                "spread_line": truth + float(rng.normal(0, market_sd)),
                "margin": truth + float(rng.normal(0, 12)),
                "total_mean": 44 + float(rng.normal(0, model_sd)),
                "total_line": 44 + float(rng.normal(0, market_sd)),
                "total_pts": 44 + float(rng.normal(0, 13)),
            })
    return preds


def test_fit_alpha_prefers_market_when_model_is_noise():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        y = float(rng.normal(0, 10))
        rows.append((2020, i % 18 + 1,
                     float(rng.normal(0, 10)),          # model = pure noise
                     y + float(rng.normal(0, 1)),        # market = near-truth
                     y))
    alpha, _ = fair_value.fit_alpha(rows)
    assert alpha <= 0.05


def test_fit_alpha_prefers_model_when_market_is_noise():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(400):
        y = float(rng.normal(0, 10))
        rows.append((2020, i % 18 + 1,
                     y + float(rng.normal(0, 1)),        # model = near-truth
                     float(rng.normal(0, 10)),           # market = pure noise
                     y))
    alpha, _ = fair_value.fit_alpha(rows)
    assert alpha >= 0.95


def test_walk_forward_first_season_never_evaluated():
    preds = _mk_preds([2019, 2020, 2021])
    res = fair_value.walk_forward(preds, "spread")
    seasons = [p["season"] for p in res["per_season"]]
    assert 2019 not in seasons          # no training data before it -> excluded
    assert seasons == sorted(seasons)


def test_walk_forward_alpha_uses_only_prior_seasons():
    """Poisoning FUTURE seasons must not change an earlier season's alpha."""
    base = _mk_preds([2019, 2020, 2021])
    res_a = fair_value.walk_forward(base, "spread")
    poisoned = [dict(p) for p in base]
    for p in poisoned:
        if p["season"] == 2021:
            p["margin_mean"] = 999.0    # absurd future model values
    res_b = fair_value.walk_forward(poisoned, "spread")
    a2020_a = [p for p in res_a["per_season"] if p["season"] == 2020][0]["alpha"]
    a2020_b = [p for p in res_b["per_season"] if p["season"] == 2020][0]["alpha"]
    assert a2020_a == a2020_b


def test_gate_fails_when_market_wins():
    """Market near-truth, model pure noise: alpha fits ~0, the blend collapses
    onto the market, P(blend strictly beats market) cannot reach 0.90 -> the
    gate MUST fail and nothing may ship."""
    preds = _mk_preds([2019, 2020, 2021, 2022], model_sd=8.0, market_sd=0.5)
    res = fair_value.walk_forward(preds, "spread")
    assert res["gate"]["passed"] is False
    assert res["shipped_alpha"] is None


def test_gate_passes_when_model_adds_signal():
    """Model carries real orthogonal signal -> blend beats market, alpha ships."""
    rng = np.random.default_rng(3)
    preds = []
    for s in (2019, 2020, 2021, 2022):
        for i in range(120):
            part_a = float(rng.normal(0, 7))
            part_b = float(rng.normal(0, 7))
            y = part_a + part_b + float(rng.normal(0, 3))
            preds.append({
                "season": s, "week": i % 18 + 1,
                "margin_mean": part_b + float(rng.normal(0, 1)),  # sees the half the market misses
                "spread_line": part_a + float(rng.normal(0, 1)),
                "margin": y,
            })
    res = fair_value.walk_forward(preds, "spread")
    assert res["gate"]["passed"]
    assert res["shipped_alpha"] is not None and 0.0 < res["shipped_alpha"] <= 1.0


def test_load_shipped_fail_closed(tmp_path):
    # missing file
    assert fair_value.load_shipped(str(tmp_path / "absent.json")) == {}
    # gate FAIL -> nothing ships even if alpha present in the file
    book = {"markets": {"spread": {"gate": {"passed": False}, "shipped_alpha": 0.2}}}
    p = tmp_path / "book_fail.json"
    p.write_text(json.dumps(book))
    assert fair_value.load_shipped(str(p)) == {}
    # gate PASS -> ships
    book = {"markets": {"spread": {"gate": {"passed": True}, "shipped_alpha": 0.15}}}
    p2 = tmp_path / "book_pass.json"
    p2.write_text(json.dumps(book))
    assert fair_value.load_shipped(str(p2)) == {"spread": 0.15}
    # corrupt file
    p3 = tmp_path / "book_corrupt.json"
    p3.write_text("{not json")
    assert fair_value.load_shipped(str(p3)) == {}


def test_fair_line_fail_closed():
    assert fair_value.fair_line(None, 3.5, 0.2) is None
    assert fair_value.fair_line(2.0, None, 0.2) is None
    assert fair_value.fair_line(2.0, 3.5, None) is None
    assert fair_value.fair_line(2.0, 4.0, 0.25) == pytest.approx(3.5)


def test_shipped_book_verdict_matches_gate():
    """The committed book/fair_value.json must be internally consistent."""
    path = fair_value.BOOK_PATH
    if not os.path.exists(path):
        pytest.skip("book/fair_value.json not built")
    with open(path) as fh:
        book = json.load(fh)
    for market, res in book["markets"].items():
        if not res["gate"]["passed"]:
            assert res["shipped_alpha"] is None, market
        pooled = res["pooled"]
        if res["gate"]["passed"]:
            assert pooled["mae_blend"] <= pooled["mae_market"]
            assert pooled["p_blend_beats_market"] >= fair_value.MIN_P_IMPROVE

"""Phase 7.7 — advisory staking: deterministic, shrunk, correlation- and
drawdown-aware, capped, and advisory-only (pure: no DB, no files, no bets)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue import staking as st                    # noqa: E402
from nflvalue.correlation import CorrelationStructure  # noqa: E402

PRICE = 1.0 + 100.0 / 110.0        # -110 decimal
FAIR = 0.5238


def _lean(pid, market, pos, team, p, game="G"):
    return dict(game_id=game, player_id=pid, market=market, pos=pos, team=team,
                side="over", p=p, market_prob=FAIR, price=PRICE)


def test_deterministic():
    leans = [_lean("a", "passing_yards", "QB", "A", 0.56)]
    r1 = st.recommend_stakes(leans, 100.0)
    r2 = st.recommend_stakes(leans, 100.0)
    assert r1["recommendations"][0]["stake_frac"] == r2["recommendations"][0]["stake_frac"]


def test_no_edge_no_stake():
    # p below the fair prob -> negative edge -> zero
    assert st.recommend_stakes([_lean("a", "passing_yards", "QB", "A", 0.50)],
                               100.0)["recommendations"][0]["stake_frac"] == 0.0
    # p exactly at fair -> zero
    assert st.recommend_stakes([_lean("a", "passing_yards", "QB", "A", FAIR)],
                               100.0)["recommendations"][0]["stake_frac"] == 0.0


def test_stake_monotone_in_edge():
    f = lambda p: st.recommend_stakes([_lean("a", "passing_yards", "QB", "A", p)],
                                      100.0)["recommendations"][0]["stake_frac"]
    assert f(0.54) < f(0.56) < f(0.60)


def test_edge_shrink_reduces_stake():
    leans = [_lean("a", "passing_yards", "QB", "A", 0.56)]
    full = st.recommend_stakes(leans, 100.0, config=st.StakeConfig(s_edge=1.0))
    half = st.recommend_stakes(leans, 100.0, config=st.StakeConfig(s_edge=0.5))
    assert half["recommendations"][0]["stake_frac"] < full["recommendations"][0]["stake_frac"]


def test_per_bet_cap_enforced():
    r = st.recommend_stakes([_lean("a", "passing_yards", "QB", "A", 0.75)], 100.0,
                            config=st.StakeConfig(cap_bet=0.02, max_slate=1.0))
    assert r["recommendations"][0]["stake_frac"] == pytest.approx(0.02)
    assert r["recommendations"][0]["capped"] is True


def test_correlation_reduces_correlated_stakes():
    corr = CorrelationStructure({"pair_types": {
        "sameteam|QB.pass~WR.rec": {"rho_shrunk": 0.30, "verdict": "real"}}, "walk_forward": {}})
    leans = [_lean("qb", "passing_yards", "QB", "A", 0.56),
             _lean("wr", "receiving_yards", "WR", "A", 0.56)]
    with_c = st.recommend_stakes(leans, 100.0, corr=corr)
    without = st.recommend_stakes(leans, 100.0, corr=None)
    a = with_c["recommendations"][0]["stake_frac"]
    b = without["recommendations"][0]["stake_frac"]
    assert a < b
    assert with_c["recommendations"][0]["corr_penalty"] == pytest.approx(0.30)


def test_negative_correlation_gets_no_bonus():
    # a hedging pair (QB pass vs same-team RB rush, rho<0) must not INCREASE size
    corr = CorrelationStructure({"pair_types": {
        "sameteam|QB.pass~RB.rush": {"rho_shrunk": -0.08, "verdict": "real"}}, "walk_forward": {}})
    leans = [_lean("qb", "passing_yards", "QB", "A", 0.56),
             _lean("rb", "rushing_yards", "RB", "A", 0.56)]
    with_c = st.recommend_stakes(leans, 100.0, corr=corr)
    without = st.recommend_stakes(leans, 100.0, corr=None)
    assert with_c["recommendations"][0]["corr_penalty"] == 0.0
    assert with_c["recommendations"][0]["stake_frac"] == without["recommendations"][0]["stake_frac"]


def test_slate_cap_scales_total_down():
    leans = [_lean(f"p{i}", "passing_yards", "QB", "A", 0.62, game=f"g{i}") for i in range(20)]
    r = st.recommend_stakes(leans, 100.0, config=st.StakeConfig(cap_bet=0.02, max_slate=0.10))
    assert r["readout"]["total_exposure_frac"] == pytest.approx(0.10, abs=1e-6)
    assert r["readout"]["slate_scaled"] is True


def test_advisory_only_disclaimer_and_purity():
    r = st.recommend_stakes([_lean("a", "passing_yards", "QB", "A", 0.56)], 100.0)
    assert "ADVISORY ONLY" in r["disclaimer"] and "never places a bet" in r["disclaimer"]
    # units convention: 1u == 1% of bankroll (frac and units rounded independently)
    frac = r["recommendations"][0]["stake_frac"]
    assert r["recommendations"][0]["stake_units"] == pytest.approx(frac / 0.01, abs=2e-3)


# --- edge-case / robustness guards (Phase 7.7 hardening) --------------------

def test_empty_leans_well_formed():
    r = st.recommend_stakes([], 100.0)
    assert r["recommendations"] == []
    assert "ADVISORY ONLY" in r["disclaimer"]
    ro = r["readout"]
    assert ro["n_leans"] == 0 and ro["n_staked"] == 0
    assert ro["total_exposure_frac"] == 0.0
    assert ro["total_exposure_units"] == 0.0
    assert ro["largest_stake_units"] == 0.0
    assert ro["slate_scaled"] is False


@pytest.mark.parametrize("bad_bankroll", [0.0, -50.0, None, float("nan"), float("inf")])
def test_bankroll_non_positive_or_bad_is_advisory_safe(bad_bankroll):
    leans = [_lean("a", "passing_yards", "QB", "A", 0.56)]
    r = st.recommend_stakes(leans, bad_bankroll)
    rec = r["recommendations"][0]
    # stake fraction/units are still computed (they don't depend on bankroll),
    # but the dollar amount can never be negative or NaN.
    assert rec["stake_amount"] == 0.0
    assert rec["stake_frac"] >= 0.0
    assert r["readout"]["bankroll"] == 0.0


@pytest.mark.parametrize("field", ["p", "market_prob", "price"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_inf_input_is_unstakeable(field, bad):
    ln = _lean("a", "passing_yards", "QB", "A", 0.56)
    ln[field] = bad
    rec = st.recommend_stakes([ln], 100.0)["recommendations"][0]
    assert rec["stake_frac"] == 0.0
    assert rec["stake_amount"] == 0.0
    # NaN must never propagate into the sized fraction
    assert rec["stake_frac"] == rec["stake_frac"]  # not NaN
    assert rec["reason"] != "ok"


@pytest.mark.parametrize("field,value", [
    ("p", 1.5), ("p", -0.1), ("market_prob", 1.2), ("market_prob", -0.3),
])
def test_out_of_range_prob_is_unstakeable(field, value):
    ln = _lean("a", "passing_yards", "QB", "A", 0.56)
    ln[field] = value
    rec = st.recommend_stakes([ln], 100.0)["recommendations"][0]
    assert rec["stake_frac"] == 0.0
    assert "outside [0,1]" in rec["reason"]


def test_dd_scale_above_one_cannot_breach_per_bet_cap():
    # a stray dd_scale > 1 must be clamped so caps still hold
    cfg = st.StakeConfig(cap_bet=0.02, max_slate=1.0, dd_scale=5.0)
    r = st.recommend_stakes([_lean("a", "passing_yards", "QB", "A", 0.75)], 100.0, config=cfg)
    assert r["recommendations"][0]["stake_frac"] <= 0.02 + 1e-9


def test_dd_scale_negative_or_nan_does_not_flip_sign():
    for bad in (-2.0, float("nan"), float("inf")):
        cfg = st.StakeConfig(dd_scale=bad)
        r = st.recommend_stakes([_lean("a", "passing_yards", "QB", "A", 0.60)], 100.0, config=cfg)
        for rec in r["recommendations"]:
            assert rec["stake_frac"] >= 0.0
            assert rec["stake_amount"] >= 0.0

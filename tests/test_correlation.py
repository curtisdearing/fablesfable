"""Phase 7.5 — same-game correlation: classification, shrinkage, the accessor,
and the walk-forward leakage guard (a correlation consumed at season S may be
estimated only from seasons < S)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nflvalue import correlation as corr           # noqa: E402
import fit_correlation as fc                        # noqa: E402


def _resid_frame(seasons, per_season, rho=0.5, seed=7):
    """Synthetic residuals: each game has a QB (passing_yards) + WR
    (receiving_yards) on the same team, with a planted within-game correlation.
    -> pooled type ``sameteam|QB.pass~WR.rec``."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for g in range(per_season):
            z = rng.normal(size=2)
            qb = z[0]
            wr = rho * z[0] + np.sqrt(1 - rho ** 2) * z[1]
            gid = f"{s}_{g:03d}_AAA_BBB"
            rows.append({"season": s, "week": 1, "game_id": gid, "player_id": f"qb{s}{g}",
                         "team": "AAA", "market": "passing_yards", "pos": "QB", "resid": qb})
            rows.append({"season": s, "week": 1, "game_id": gid, "player_id": f"wr{s}{g}",
                         "team": "AAA", "market": "receiving_yards", "pos": "WR", "resid": wr})
    return pd.DataFrame(rows)


def test_classify_pair_is_order_independent_and_restricts_cross_volume():
    a = ("QB", "passing_yards", "p1", "AAA")
    b = ("WR", "receiving_yards", "p2", "AAA")
    assert corr.classify_pair(*a, *b) == corr.classify_pair(*b, *a) == "sameteam|QB.pass~WR.rec"
    # cross-player volume market is NOT a measured type (avoids double-counting)
    assert corr.classify_pair("QB", "pass_attempts", "p1", "AAA",
                              "WR", "receiving_yards", "p2", "AAA") is None
    # same player keeps volume markets (yards<->attempts)
    assert corr.classify_pair("QB", "pass_attempts", "p1", "AAA",
                              "QB", "passing_yards", "p1", "AAA") == "sameplayer|QB.pass~QB.pass"
    # opponent vs same-team distinguished
    assert corr.classify_pair("QB", "passing_yards", "p1", "AAA",
                              "QB", "passing_yards", "p2", "BBB") == "opponent|QB.pass~QB.pass"


def test_walk_forward_correlation_uses_only_prior_seasons():
    """walk_forward[S] must be byte-identical whether or not seasons >= S exist
    in the input -- the estimate a consumer uses at S saw nothing from >= S."""
    ptype = "sameteam|QB.pass~WR.rec"
    full = _resid_frame([2019, 2020, 2021], per_season=400, rho=0.5)
    payload = fc.analyze(fc.collect_pairs(full))
    wf_2021 = payload["walk_forward"]["2021"][ptype]      # from {2019, 2020}

    trunc = full[full["season"] < 2021]                   # delete the future
    store_tr = fc.collect_pairs(trunc)
    rho_tr = round(fc._rho(np.asarray(store_tr[ptype]["x"]),
                           np.asarray(store_tr[ptype]["y"])), 4)
    assert wf_2021 == rho_tr, "walk-forward slice changed when future seasons were removed"


def test_shrinkage_pulls_thin_noisy_types_to_zero():
    # a well-measured moderate correlation barely moves; a thin near-zero one collapses
    rhos = {"strong": 0.30, "thin_noise": 0.04}
    ns = {"strong": 12000, "thin_noise": 350}
    shrunk, tau2 = corr.eb_fisher_z_shrink(rhos, ns)
    assert abs(shrunk["strong"] - 0.30) < 0.02
    assert abs(shrunk["thin_noise"]) < abs(0.04)          # shrunk toward zero


def test_accessor_returns_zero_for_noise_and_unknown():
    payload = {"pair_types": {
        "sameteam|QB.pass~WR.rec": {"rho_shrunk": 0.30, "verdict": "real"},
        "sameteam|WR.rec~WR.rec": {"rho_shrunk": 0.03, "verdict": "noise"},
    }, "walk_forward": {"2024": {"sameteam|QB.pass~WR.rec": 0.29}}}
    cs = corr.CorrelationStructure(payload)
    # real type -> shrunk rho; noise -> 0; unknown -> 0
    assert cs.rho("sameteam|QB.pass~WR.rec") == 0.30
    assert cs.rho("sameteam|WR.rec~WR.rec") == 0.0
    assert cs.rho("does|not~exist") == 0.0
    # walk-forward lookup, and 0 when no prior-season slice exists yet
    assert cs.rho("sameteam|QB.pass~WR.rec", as_of_season=2024) == 0.29
    assert cs.rho("sameteam|QB.pass~WR.rec", as_of_season=2099) == 0.0
    # end-to-end classify + lookup
    assert cs.rho_for("QB", "passing_yards", "p1", "AAA",
                      "WR", "receiving_yards", "p2", "AAA") == 0.30


# --------------------------------------------------------------------------- #
# Phase 7.6 -- consumption: redundancy_discount + sgp_joint_prob
# --------------------------------------------------------------------------- #
def test_redundancy_discount_positive_only():
    assert corr.redundancy_discount(0.76) == pytest.approx(0.76)
    assert corr.redundancy_discount(-0.30) == 0.0     # diversifying -- never penalized
    assert corr.redundancy_discount(0.0) == 0.0


def test_redundancy_discount_strength_scaling_and_cap():
    assert corr.redundancy_discount(0.76, strength=0.5) == pytest.approx(0.38)
    assert corr.redundancy_discount(1.0, strength=1.0) == 0.95   # never fully total
    assert corr.redundancy_discount(2.0, strength=1.0) == 0.95   # clipped


def test_sgp_joint_prob_independence_fallback_is_product():
    assert corr.sgp_joint_prob(0.6, "over", 0.55, "over", 0.0) == pytest.approx(0.6 * 0.55, abs=1e-6)


def test_sgp_joint_prob_degenerate_inputs_return_none():
    assert corr.sgp_joint_prob(0.0, "over", 0.5, "over", 0.3) is None
    assert corr.sgp_joint_prob(1.0, "over", 0.5, "over", 0.3) is None
    assert corr.sgp_joint_prob(0.5, "sideways", 0.5, "over", 0.3) is None


def test_sgp_joint_prob_symmetric_under_leg_swap():
    a = corr.sgp_joint_prob(0.6, "over", 0.55, "under", 0.25)
    b = corr.sgp_joint_prob(0.55, "under", 0.6, "over", 0.25)
    assert a == pytest.approx(b, abs=1e-6)


def test_sgp_joint_prob_positive_rho_raises_over_over_joint():
    """Two OVER legs with positive rho should co-occur MORE than independence
    (the whole point of measuring same-game correlation)."""
    indep = 0.6 * 0.55
    joint = corr.sgp_joint_prob(0.6, "over", 0.55, "over", 0.30)
    assert joint > indep


def test_sgp_joint_prob_positive_rho_lowers_over_under_joint():
    """A positive-rho pair (they move together) makes an OVER+UNDER split
    LESS likely than independence -- the hedge direction, not the stack.
    (p_i, p_j are each already the probability of hitting THEIR selected
    side, so the independence baseline is simply their product.)"""
    indep = 0.6 * 0.55
    joint = corr.sgp_joint_prob(0.6, "over", 0.55, "under", 0.30)
    assert joint < indep

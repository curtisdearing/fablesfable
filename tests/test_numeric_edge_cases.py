"""Phase 7.4 -- property tests for the distribution and de-vig math.

These are deliberately adversarial. Every assertion here encodes a
NON-NEGOTIABLE from the hardening contract:

* probabilities are always in [0, 1] and always finite;
* vig removal is order-invariant (no dict-ordering dependence in a numeric
  path -- see the determinism rule);
* MISSING DATA SHOWS AS MISSING. A NaN input must never be rendered as a
  confident probability. This is the rule that `p_over` used to break: the
  clamp ``max(0.0, min(1.0, nan))`` evaluates to 1.0 in CPython, so a player
  with no rolling usage history projected as a 100%-certain OVER and ranked
  first on the board. See `test_nan_inputs_never_become_certainty`.

No network, no randomness, no wall clock.
"""

from __future__ import annotations

import itertools
import math

import pytest

from nflvalue import oddsmath
from nflvalue.projection import MARKETS, p_over, project

NAN = float("nan")
INF = float("inf")

DISTS = ("normal", "gamma", "negbinom", "poisson")

# Values chosen to straddle every guard in the module: zero, negative,
# sub-epsilon, ordinary, and absurd.
MEANS = (0.0, -5.0, 1e-9, 0.5, 8.0, 62.5, 1e6)
SDS = (0.0, 1e-9, 0.75, 12.0, 1e5)
LINES = (-10.5, 0.0, 0.5, 4.5, 62.5, 1e6, INF, -INF)


# --------------------------------------------------------------------------- #
# projection.p_over
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dist", DISTS)
def test_p_over_always_a_valid_probability(dist):
    """Across the whole cross product of pathological inputs, p_over stays a
    finite number in [0, 1]. Zero/negative means, sd collapsing to 0, and
    infinite lines all included."""
    for mean, sd, line in itertools.product(MEANS, SDS, LINES):
        p = p_over(mean, sd, line, dist)
        assert isinstance(p, float)
        assert math.isfinite(p), f"non-finite p for {dist} mean={mean} sd={sd} line={line}"
        assert 0.0 <= p <= 1.0, f"p={p} out of range for {dist} mean={mean} sd={sd} line={line}"


@pytest.mark.parametrize("dist", DISTS)
def test_p_over_is_monotone_non_increasing_in_the_line(dist):
    """A higher line can never be easier to clear. This is the one shape
    property every survival function must have; a violation means the
    distribution parameterisation is wrong, not merely imprecise."""
    mean, sd = 62.5, 18.0
    lines = [0.5, 10.5, 25.5, 50.5, 62.5, 75.5, 120.5, 500.5]
    probs = [p_over(mean, sd, x, dist) for x in lines]
    for lo, hi, p_lo, p_hi in zip(lines, lines[1:], probs, probs[1:]):
        assert p_hi <= p_lo + 1e-12, (
            f"{dist}: P(>{hi})={p_hi} exceeded P(>{lo})={p_lo}")


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("bad", ["mean", "sd", "line"])
def test_nan_inputs_never_become_certainty(dist, bad):
    """FAIL CLOSED. A NaN in any numeric slot must not yield a usable
    probability -- and above all must not yield 1.0.

    Regression pin: ``max(0.0, min(1.0, nan))`` returns 1.0 in CPython
    because every NaN comparison is False, so the old clamp turned 'we have
    no history for this player' into 'this is a lock'.
    """
    args = {"mean": 62.5, "sd": 18.0, "line": 50.5}
    args[bad] = NAN
    p = p_over(args["mean"], args["sd"], args["line"], dist)
    assert p is None or math.isnan(p), (
        f"{dist}: NaN {bad} produced p={p!r}; missing data must stay missing")
    assert p != 1.0


def test_sd_collapsing_to_zero_is_a_step_function_not_a_nan():
    """sd -> 0 means the projection is a point mass at the mean. Below the
    mean the over is certain, above it impossible -- but it must be computed,
    not NaN, and it must not blow up on the division by sd."""
    mean = 62.5
    for dist in DISTS:
        below = p_over(mean, 0.0, mean - 20.0, dist)
        above = p_over(mean, 0.0, mean + 20.0, dist)
        assert math.isfinite(below) and math.isfinite(above)
        assert below >= above


def test_unknown_distribution_is_rejected_not_silently_normal():
    """`_SF.get(dist, _norm_sf)` used to substitute a normal for any
    unrecognised distribution name -- a default where a real value is
    absent. A typo'd market spec must fail loudly."""
    with pytest.raises((KeyError, ValueError)):
        p_over(62.5, 18.0, 50.5, "not_a_real_distribution")


# --------------------------------------------------------------------------- #
# projection.project -- the contract the ranker consumes
# --------------------------------------------------------------------------- #
def _player_row(**over):
    row = {"player_id": "00-TEST", "player_name": "Test Player", "role": "WR",
           "roll_games": 8.0, "roll_target_share": 0.22, "roll_targets": 7.5,
           "roll_ypt": 9.0, "roll_catch_rate": 0.65, "roll_rush_td_rate": 0.0,
           "roll_rec_td_rate": 0.05, "roll_carries": 0.0}
    row.update(over)
    return row


_TEAM_ROW = {"roll_team_targets": 35.0, "roll_team_carries": 25.0,
             "roll_team_pass_att": 35.0}


@pytest.mark.parametrize("market", ["receiving_yards", "receptions"])
def test_missing_usage_history_yields_no_probability_and_no_eligibility(market):
    """The live bug this file was written for.

    A NaN rolling-usage column (the codebase's own documented encoding for
    'no prior history' -- see AsOfLookup) flowed through expected_volume into
    a NaN mean, survived ``max(mean_, 0.0)``, and came out of p_over as
    1.0000 with eligible_for_shortlist=True. That is the single most
    dangerous shape a bug can take in this system: it does not error, it
    ranks first.

    Both usage columns are NaN'd here because expected_volume can legitimately
    recover from either one alone -- that fallback is correct behaviour and is
    covered by ``test_partial_usage_history_still_projects``. The failure mode
    under test is the one where NO volume basis exists at all.
    """
    row = _player_row(roll_target_share=NAN, roll_targets=NAN)
    out = project(row, market, team_row=_TEAM_ROW, line=50.5)

    assert out["p_over"] is None, f"expected missing, got {out['p_over']!r}"
    assert out["p_under"] is None
    assert out["eligible_for_shortlist"] is False
    assert out["low_confidence"] is True


@pytest.mark.parametrize("missing", ["roll_target_share", "roll_targets"])
def test_partial_usage_history_still_projects(missing):
    """Paired control for the guard above: losing ONE of the two volume
    bases is recoverable, and must stay recoverable. A fail-closed rule that
    also refuses projectable rows would be a silent availability outage."""
    out = project(_player_row(**{missing: NAN}), "receiving_yards",
                  team_row=_TEAM_ROW, line=50.5)
    assert out["p_over"] is not None
    assert 0.0 <= out["p_over"] <= 1.0


def test_healthy_row_still_produces_a_normal_probability():
    """The guard above must not fire on ordinary data -- this is the paired
    control that proves the NaN check did not just disable projections."""
    out = project(_player_row(), "receiving_yards", team_row=_TEAM_ROW, line=50.5)
    assert out["p_over"] is not None
    assert 0.0 < out["p_over"] < 1.0
    assert math.isclose(out["p_over"] + out["p_under"], 1.0, abs_tol=1e-9)
    assert out["eligible_for_shortlist"] is True


def test_projection_probabilities_stay_in_range_across_all_markets():
    for market in sorted(MARKETS):
        out = project(_player_row(), market, team_row=_TEAM_ROW, line=0.5)
        if out["p_over"] is None:
            continue
        assert 0.0 <= out["p_over"] <= 1.0
        assert 0.0 <= out["p_under"] <= 1.0


# --------------------------------------------------------------------------- #
# oddsmath -- de-vigging
# --------------------------------------------------------------------------- #
def test_devig_sums_to_one_and_stays_in_range():
    cases = [
        [1.91, 1.91],            # standard -110/-110
        [1.05, 15.0],            # heavy favourite / longshot
        [2.0, 2.0],              # fair coin, no vig
        [1.001, 1000.0],         # extreme
        [1.5, 2.2, 4.0, 9.0],    # N-way
    ]
    for decimals in cases:
        ps = oddsmath.devig_multiplicative(decimals)
        assert all(math.isfinite(p) for p in ps)
        assert all(0.0 <= p <= 1.0 for p in ps)
        assert math.isclose(sum(ps), 1.0, abs_tol=1e-9)


def test_devig_is_order_invariant():
    """Permuting the input prices must permute the output identically --
    no accumulation-order dependence."""
    decimals = [1.5, 2.2, 4.0, 9.0]
    base = dict(zip(decimals, oddsmath.devig_multiplicative(decimals)))
    for perm in itertools.permutations(decimals):
        got = dict(zip(perm, oddsmath.devig_multiplicative(list(perm))))
        for d in decimals:
            assert math.isclose(base[d], got[d], rel_tol=0, abs_tol=1e-12)


def test_consensus_is_invariant_to_book_insertion_order():
    """DETERMINISM. `consensus_two_way` iterates a dict and accumulates
    floats; float addition is not associative, so a different insertion
    order could shift the last ULP of a shipped probability. Pin exact
    equality -- not approximate -- because the contract is byte-identical
    outputs for identical inputs."""
    books = {"draftkings": (1.91, 1.91), "betmgm": (1.87, 1.95),
             "hardrockbet": (2.05, 1.80), "pinnacle": (1.95, 1.93)}
    reference = oddsmath.consensus_two_way(books)
    for perm in itertools.permutations(books):
        shuffled = {k: books[k] for k in perm}
        got = oddsmath.consensus_two_way(shuffled)
        assert got["p_a"] == reference["p_a"], (
            f"book order changed p_a: {got['p_a']!r} vs {reference['p_a']!r}")
        assert got["p_b"] == reference["p_b"]
        assert got["n_books"] == reference["n_books"]


def test_consensus_n_books_counts_only_contributing_books():
    """TRACEABILITY. Books whose prices are unusable (<= 1.0 decimal, i.e. a
    missing or malformed side) are skipped by the consensus loop, but
    `n_books` used to report ``len(book_prices)`` -- advertising more market
    support than actually existed. n_books is surfaced on the published
    lean, so it has to mean what it says."""
    books = {"draftkings": (1.91, 1.91),
             "betmgm": (1.87, 1.95),
             "deadbook": (0.0, 0.0)}       # no usable price
    out = oddsmath.consensus_two_way(books)
    assert out["n_books"] == 2, f"counted {out['n_books']} books, only 2 priced"


def test_consensus_with_one_sided_market_returns_empty_not_a_guess():
    """A market with no two-sided price anywhere cannot be de-vigged. The
    honest answer is 'no consensus', which the caller renders as
    `no_market` -- never an imputed 50/50."""
    assert oddsmath.consensus_two_way({"draftkings": (1.91, 0.0)}) == {}
    assert oddsmath.consensus_two_way({}) == {}


def test_devig_with_a_missing_side_does_not_invent_a_uniform_prior():
    """All-unusable prices must not silently become a flat 1/n prior, which
    is a fabricated probability with no market behind it."""
    ps = oddsmath.devig_multiplicative([0.0, 0.0])
    assert all(p is None or math.isnan(p) for p in ps), (
        f"fabricated a prior from unusable prices: {ps}")


def test_overround_is_non_negative_for_real_books_and_finite():
    for decimals in ([1.91, 1.91], [1.05, 15.0], [2.0, 2.0]):
        o = oddsmath.overround(decimals)
        assert math.isfinite(o)
        assert o >= -1e-12


# --------------------------------------------------------------------------- #
# oddsmath -- conversions and staking
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("american", [-1000, -200, -110, 100, 150, 5000])
def test_american_decimal_roundtrip(american):
    dec = oddsmath.american_to_decimal(american)
    assert dec > 1.0
    assert oddsmath.decimal_to_american(dec) == pytest.approx(american, abs=1)


def test_kelly_never_recommends_a_negative_or_insane_stake():
    for prob in (0.0, 0.01, 0.5, 0.9, 1.0):
        for dec in (1.01, 1.91, 5.0, 100.0):
            f = oddsmath.kelly_fraction(prob, dec)
            assert math.isfinite(f)
            assert 0.0 <= f <= 1.0


def test_logit_sigmoid_roundtrip_is_stable_at_the_boundaries():
    for p in (1e-9, 0.001, 0.5, 0.999, 1 - 1e-9):
        x = oddsmath.logit(p)
        assert math.isfinite(x)
        back = oddsmath.sigmoid(x)
        assert 0.0 <= back <= 1.0

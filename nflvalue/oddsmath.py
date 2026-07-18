"""The math of value betting: odds conversions, de-vigging, EV, and Kelly.

Everything here is pure and easily testable. No network, no state.

Key ideas
---------
* A sportsbook price implies a probability, but it is inflated by the book's
  margin (the "vig" / "juice"). Summing both sides gives > 100%.
* "De-vigging" removes that margin to recover the book's fair probability.
* Averaging the de-vigged probabilities across many books (weighting sharp
  books like Pinnacle more) gives a robust "market consensus" fair price.
* Expected value (EV) compares that fair probability against the BEST price
  available anywhere. A bet is +EV when fair_prob * best_decimal_odds > 1.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Odds format conversions
# --------------------------------------------------------------------------- #
def american_to_decimal(american: float) -> float:
    """+150 -> 2.5, -200 -> 1.5."""
    american = float(american)
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> int:
    """2.5 -> +150, 1.5 -> -200."""
    if decimal <= 1.0:
        return 0
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    return int(round(-100.0 / (decimal - 1.0)))


def american_str(american: float) -> str:
    a = int(round(american))
    return f"+{a}" if a > 0 else str(a)


def implied_prob(decimal: float) -> float:
    """Raw implied probability of a decimal price (still contains vig)."""
    return 1.0 / decimal if decimal > 0 else 0.0


# --------------------------------------------------------------------------- #
# De-vigging (removing the bookmaker margin)
# --------------------------------------------------------------------------- #
def devig_multiplicative(decimals: Sequence[float]) -> List[float]:
    """Normalise a set of prices so the implied probabilities sum to 1.

    Works for 2-way (moneyline, spread, total over/under) and N-way markets.
    Returns fair probabilities in the same order as ``decimals``.
    """
    qs = [implied_prob(d) for d in decimals]
    total = sum(qs)
    if total <= 0:
        # FAIL CLOSED (Phase 7.4). This used to return a flat ``1/n`` prior,
        # which is a fabricated probability with no market behind it: a
        # malformed payload carrying 0.0 prices became a confident 50/50 and
        # was stored as if it were a real de-vigged price. Missing data has to
        # stay missing; callers test for finiteness and skip.
        return [float("nan")] * len(decimals)
    return [q / total for q in qs]


def overround(decimals: Sequence[float]) -> float:
    """The book's margin, e.g. 0.045 means the book is holding ~4.5%."""
    return sum(implied_prob(d) for d in decimals) - 1.0


# --------------------------------------------------------------------------- #
# Market consensus across multiple books
# --------------------------------------------------------------------------- #
def consensus_two_way(
    book_prices: Dict[str, Tuple[float, float]],
    sharp_books: Sequence[str] = ("pinnacle",),
    sharp_weight: float = 2.0,
) -> Dict[str, float]:
    """Combine several books' two-way prices into one fair estimate.

    Parameters
    ----------
    book_prices : {book_key: (decimal_side_a, decimal_side_b)}
    sharp_books : books whose de-vigged opinion is trusted more
    sharp_weight: how much extra weight a sharp book gets (2.0 = double)

    Returns a dict with:
        p_a, p_b      -> consensus fair probabilities (sum to 1)
        best_a, best_b-> best (highest) decimal price for each side
        best_a_book, best_b_book
        n_books
    """
    weighted_pa = 0.0
    weight_total = 0.0
    best_a = best_b = 0.0
    best_a_book = best_b_book = None
    n_priced = 0

    # DETERMINISM (Phase 7.4). Float addition is not associative, so
    # accumulating in dict-insertion order made ``p_a`` depend on the order
    # the books happened to be fetched in -- 24 orderings of 4 books produced
    # two different last-ULP values. Sorting the keys makes the sum a pure
    # function of the prices. See test_consensus_is_invariant_to_book_insertion_order.
    for book in sorted(book_prices):
        da, db = book_prices[book]
        if da <= 1.0 or db <= 1.0:
            continue
        pa, _pb = devig_multiplicative([da, db])
        if not math.isfinite(pa):
            continue
        w = sharp_weight if book.lower() in sharp_books else 1.0
        weighted_pa += w * pa
        weight_total += w
        n_priced += 1
        if da > best_a:
            best_a, best_a_book = da, book
        if db > best_b:
            best_b, best_b_book = db, book

    if weight_total == 0:
        return {}

    p_a = weighted_pa / weight_total
    p_a = min(max(p_a, 1e-4), 1 - 1e-4)
    return {
        "p_a": p_a,
        "p_b": 1.0 - p_a,
        "best_a": best_a,
        "best_b": best_b,
        "best_a_book": best_a_book,
        "best_b_book": best_b_book,
        # TRACEABILITY (Phase 7.4): count the books that actually contributed a
        # usable two-sided price, not every key handed in. n_books is published
        # on the lean as evidence of market support, so an unpriced or
        # malformed book must not inflate it.
        "n_books": n_priced,
    }


# --------------------------------------------------------------------------- #
# Expected value, edge, and staking
# --------------------------------------------------------------------------- #
def ev_pct(prob: float, decimal: float) -> float:
    """Expected profit per 1 unit staked. 0.05 == +5% EV."""
    return prob * decimal - 1.0


def prob_edge(prob: float, decimal: float) -> float:
    """How much our probability beats the price's implied probability."""
    return prob - implied_prob(decimal)


def kelly_fraction(prob: float, decimal: float) -> float:
    """Full-Kelly stake as a fraction of bankroll (>=0; 0 if not +EV)."""
    b = decimal - 1.0
    if b <= 0:
        return 0.0
    f = (prob * decimal - 1.0) / b
    return max(0.0, f)


# --------------------------------------------------------------------------- #
# Small numerical helpers used by the factor model
# --------------------------------------------------------------------------- #
def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

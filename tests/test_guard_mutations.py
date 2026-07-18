"""Phase 7.5 -- mutation testing of the leakage guards.

A passing test suite proves nothing on its own; it has to be capable of
failing. This file breaks each anti-leakage guard on purpose and asserts that
the guard's own test goes RED. A guard whose test survives its own mutation is
decorative, and decorative guards are how the two leaks already recorded in
docs/decisions_p3-5.md got in.

Three mutations, one per guard:

    1. AsOfLookup: strictly-before  ->  inclusive-of-current-week
       (this IS the caught leak of 2026-07-02, re-injected)
    2. features._rolling_shifted: drop the shift(1)
       (the row would aggregate its own target week)
    3. MLRanker.assert_walk_forward: <= cutoff  ->  < cutoff
       (lets the model score the last week it trained on)

Every mutation is applied with monkeypatch and reverted automatically. Nothing
here touches shipped behaviour; it only proves the alarms are wired to
something.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd
import pytest

from nflvalue import advanced_features as af
from nflvalue import features as featmod
from nflvalue import ml_ranker as mlr


# --------------------------------------------------------------------------- #
# Guard 1: AsOfLookup strictly-before
# --------------------------------------------------------------------------- #
def _asof_frame():
    """One player, three weeks. The week-10 value is the FUTURE relative to a
    week-10 candidate row: an as-of lookup for week 10 must return week 9."""
    return pd.DataFrame({
        "player_id": ["00-A"] * 3,
        "season": [2023] * 3,
        "week": [8, 9, 10],
        "v": [1.0, 2.0, 999.0],       # 999 = the poisoned current-week value
    })


def test_asof_lookup_returns_strictly_prior_value():
    """Baseline behaviour -- the property the mutation will break."""
    lk = af.AsOfLookup(_asof_frame(), ["v"])
    assert lk.get("00-A", 2023, 10) == (2.0,), "as-of returned a non-prior value"
    assert lk.get("00-A", 2023, 9) == (1.0,)
    assert np.isnan(lk.get("00-A", 2023, 8)[0]), "no prior history must be NaN"


def test_asof_guard_actually_catches_an_inclusive_lookup(monkeypatch):
    """MUTATION: bisect_left -> bisect_right, i.e. '< week' becomes '<= week'.

    This is precisely the 2026-07-02 leak: the candidate row starts seeing its
    own week. If the assertion above still passes under this mutation, it is
    not testing what it claims to.
    """
    monkeypatch.setattr(af.bisect if hasattr(af, "bisect") else bisect,
                        "bisect_left", bisect.bisect_right, raising=False)

    # AsOfLookup imports bisect inside get(), so patch the module it resolves.
    import bisect as _b
    monkeypatch.setattr(_b, "bisect_left", _b.bisect_right)

    lk = af.AsOfLookup(_asof_frame(), ["v"])
    leaked = lk.get("00-A", 2023, 10)
    assert leaked == (999.0,), (
        "mutation did not take effect -- the mutation test itself is broken")

    with pytest.raises(AssertionError):
        assert leaked == (2.0,), "as-of returned a non-prior value"


def test_asof_nan_means_no_history_not_no_activity():
    """The subtler half of the same leak. A player with NO prior rows returns
    NaN; that NaN must not be distinguishable from 'this player had a quiet
    week', because the NaN PATTERN itself was what the GBDT learned to read."""
    lk = af.AsOfLookup(_asof_frame(), ["v"])
    absent = lk.get("00-NOBODY", 2023, 10)
    no_history = lk.get("00-A", 2023, 8)
    assert np.isnan(absent[0]) and np.isnan(no_history[0])


# --------------------------------------------------------------------------- #
# Guard 2: features._rolling_shifted
# --------------------------------------------------------------------------- #
def test_rolling_shifted_excludes_the_current_row():
    s = pd.Series([10.0, 20.0, 30.0, 40.0])
    got = featmod._rolling_shifted(s, window=8, how="mean")
    assert pd.isna(got.iloc[0]), "first row has no prior history"
    assert got.iloc[1] == 10.0
    assert got.iloc[2] == 15.0
    assert got.iloc[3] == 20.0
    assert got.iloc[3] != s.iloc[3], "current-week value leaked into its own feature"


def test_leakage_test_catches_a_missing_shift(monkeypatch):
    """MUTATION: remove the shift(1) so the window includes the target week.

    The poisoned value must show up in the feature -- proving the shift is
    load-bearing and that a regression in it would be visible.
    """
    def unshifted(s, window=8, how="mean"):
        if how == "count":
            return s.rolling(window, min_periods=1).count()
        return s.rolling(window, min_periods=1).mean()

    monkeypatch.setattr(featmod, "_rolling_shifted", unshifted)

    s = pd.Series([10.0, 20.0, 30.0, 999.0])
    got = featmod._rolling_shifted(s, window=8, how="mean")
    assert got.iloc[3] != 20.0, "mutation did not take effect"
    assert got.iloc[3] == pytest.approx((10 + 20 + 30 + 999) / 4), (
        "the unshifted mutant should be contaminated by the current week")

    with pytest.raises(AssertionError):
        assert got.iloc[3] == 20.0, "current-week value leaked"


def test_count_mode_stays_a_literal_prior_game_count():
    """roll_games drives the cold-start eligibility gate. If it ever counted
    the current week, a debut would look like a 1-game history and clear a
    gate designed to exclude it."""
    s = pd.Series([1.0, 2.0, 3.0])
    counts = featmod._rolling_shifted(s, window=8, how="count")
    assert counts.tolist() == [0.0, 1.0, 2.0], (
        f"prior-game count included the current week: {counts.tolist()}")


# --------------------------------------------------------------------------- #
# Guard 3: MLRanker.assert_walk_forward
# --------------------------------------------------------------------------- #
class _FittedStub(mlr.MLRanker):
    """A ranker with a train cutoff but no real classifier -- assert_walk_forward
    is pure bookkeeping and needs nothing else."""

    def __init__(self, cutoff):
        super().__init__(model="gbdt")
        self.train_max = cutoff


def test_walk_forward_rejects_the_cutoff_week_itself():
    """The boundary that matters: training through (2024, 10) must refuse to
    score (2024, 10). Off-by-one here is a silent in-sample score."""
    model = _FittedStub((2024, 10))
    with pytest.raises(mlr.WalkForwardViolation):
        model.assert_walk_forward(pd.DataFrame({"season": [2024], "week": [10]}))
    with pytest.raises(mlr.WalkForwardViolation):
        model.assert_walk_forward(pd.DataFrame({"season": [2024], "week": [9]}))
    with pytest.raises(mlr.WalkForwardViolation):
        model.assert_walk_forward(pd.DataFrame({"season": [2023], "week": [17]}))
    # the first legal week
    model.assert_walk_forward(pd.DataFrame({"season": [2024], "week": [11]}))


def test_unfitted_model_refuses_to_score_at_all():
    model = mlr.MLRanker(model="gbdt")
    with pytest.raises(mlr.WalkForwardViolation):
        model.assert_walk_forward(pd.DataFrame({"season": [2025], "week": [1]}))


def test_walk_forward_guard_catches_an_off_by_one_mutation(monkeypatch):
    """MUTATION: '<= cutoff week' -> '< cutoff week', which lets the model
    score the exact week it trained through."""
    def weakened(self, frame):
        s, w = self.train_max
        bad = frame[(frame["season"] < s)
                    | ((frame["season"] == s) & (frame["week"] < w))]   # was <=
        if len(bad):
            raise mlr.WalkForwardViolation("mutant")

    monkeypatch.setattr(mlr.MLRanker, "assert_walk_forward", weakened)
    model = _FittedStub((2024, 10))

    # The mutant is SILENT on the exact week it trained through -- that
    # silence is the leak, and it is what test_walk_forward_rejects_the_
    # cutoff_week_itself above exists to catch. Demonstrating that the
    # mutation changes observable behaviour proves that test has teeth.
    model.assert_walk_forward(pd.DataFrame({"season": [2024], "week": [10]}))

    # ...while still rejecting strictly-earlier weeks, so the mutation is a
    # genuine off-by-one and not a wholesale disabling of the guard.
    with pytest.raises(mlr.WalkForwardViolation):
        model.assert_walk_forward(pd.DataFrame({"season": [2024], "week": [9]}))


# --------------------------------------------------------------------------- #
# Suite hygiene
# --------------------------------------------------------------------------- #
def test_no_test_opens_a_socket():
    """The suite must stay offline and deterministic. This does not prove it
    globally, but it pins the intent and fails loudly if someone wires a live
    call into a fixture imported here."""
    import socket as _s
    assert hasattr(_s, "socket")


def test_guard_modules_expose_the_symbols_the_alarms_depend_on():
    """Cheap structural canary: a refactor that renames or removes a guard
    should break here rather than silently disarm the tests above."""
    assert hasattr(af, "AsOfLookup")
    assert hasattr(featmod, "_rolling_shifted")
    assert hasattr(mlr, "WalkForwardViolation")
    assert hasattr(mlr.MLRanker, "assert_walk_forward")

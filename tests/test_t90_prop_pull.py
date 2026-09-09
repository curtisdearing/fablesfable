"""T-90 must price the game it is processing.

RED against f6a0ff7: ``run_t90`` never touched ``lines`` in either direction
-- no ``pull_week_props``, no read. It re-enumerated candidates, re-ran the
roster/inactives gate, voided OUT players and re-shortlisted. So a game the
Wednesday rotation had skipped stayed ``NO_MARKET`` through kickoff and no
run in the week ever priced it. T-90 is in fact the BEST moment to spend the
credit: the line is closest to its close and the inactives are known.
"""

from __future__ import annotations

import inspect

import pipeline_weekly as pw


def test_run_t90_accepts_injected_odds_callables():
    """Injectability is the contract the offline suite needs; without it the
    T-90 pull could only ever be exercised against the live provider."""
    params = inspect.signature(pw.run_t90).parameters
    assert "odds_fetch" in params
    assert "list_events_fn" in params


def test_run_t90_pulls_props_for_its_own_game():
    src = inspect.getsource(pw.run_t90)
    assert "pull_week_props" in src, (
        "T-90 must spend its one budgeted event-pull on the game it is "
        "processing; before this it never called the odds API at all")
    assert "load_recent_lines" in src, (
        "T-90 must also read quotes already stored for the game, so a line "
        "pulled earlier in the week still prices the board")


def test_t90_pull_happens_before_feature_and_ml_stamping():
    """run_week documents the ordering catch: re-enumerating AFTER stamping
    silently drops the ML/learning/context layers exactly when real lines
    exist. T-90 must not reintroduce it."""
    src = inspect.getsource(pw.run_t90)
    assert src.index("pull_week_props") < src.index("_maybe_stamp_ml"), (
        "the prop pull and its re-enumeration must precede ML stamping")


def test_t90_reports_its_line_activity():
    assert "line_note" in inspect.getsource(pw.run_t90), (
        "a run that spends a metered credit must say so in its result")


def test_t90_degrades_instead_of_aborting_on_a_dead_provider():
    """One flaky HTTP call must cost that call only -- the inactives pass is
    the whole point of T-90 and must still happen."""
    src = inspect.getsource(pw.run_t90)
    pull_at = src.index("pull_week_props")
    assert "except Exception" in src[pull_at:pull_at + 1200]
    assert "BudgetExceeded" in src[pull_at - 400:pull_at + 1200], (
        "overspending a metered free tier stays a hard stop, not a degradation")

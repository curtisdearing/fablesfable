"""The T-90 inactives feed: fetched at all, and never trusted when empty.

RED against f6a0ff7. Two defects, one latent behind the other:

1. ``gather_live_feeds(game_event_ids=...)`` had NO caller. Both call sites
   omitted it, so at T-90 ``for eid in game_event_ids or []`` never ran, the
   feed arrived empty and unstamped, and the gate refused the board with
   "inactives: no/unparseable timestamp". The T-90 inactives pass -- the
   entire reason the clock exists -- had never executed.

2. Passing an event id would have been worse. Measured live on the 2026 Week
   1 opener (event 401872656) 43 minutes before kickoff: ``period: 0`` and
   ``active: false`` on all 56 entries per side, the starting quarterback
   included. ``parse_event_roster`` reads that as not-playing and
   ``resolve_statuses`` turns it into OUT -- so wiring the id naively marks
   BOTH ENTIRE ROSTERS inactive and voids every lean in the game.
"""

from __future__ import annotations

import inspect

import pipeline_weekly as pw
from nflvalue.sources import availability as av

#: The real shape ESPN served before kickoff, trimmed to three entries.
UNPOPULATED = {"period": 0, "entries": [
    {"playerId": 2975863, "displayName": "Saubert", "active": False,
     "starter": True, "didNotPlay": False, "jersey": "81"},
    {"playerId": 3912547, "displayName": "Darnold", "active": False,
     "starter": True, "didNotPlay": False, "jersey": "14"},
    {"playerId": 4361307, "displayName": "Reserve Guy", "active": False,
     "starter": False, "didNotPlay": True, "jersey": "77"},
]}

POPULATED = {"period": 1, "entries": [
    {"playerId": 1, "displayName": "Plays", "active": True,
     "starter": True, "didNotPlay": False, "jersey": "1"},
    {"playerId": 2, "displayName": "Inactive Guy", "active": False,
     "starter": False, "didNotPlay": True, "jersey": "2"},
]}


# --------------------------------------------------------------------------- #
# The unpopulated shape is recognised
# --------------------------------------------------------------------------- #
def test_pre_kickoff_scaffolding_is_not_a_populated_roster():
    ok, reason = av.event_roster_populated(UNPOPULATED)
    assert ok is False
    assert "not populated" in reason or "no entry is marked active" in reason


def test_a_real_event_roster_is_populated():
    ok, reason = av.event_roster_populated(POPULATED)
    assert ok is True and reason == ""


def test_empty_entries_is_not_populated():
    assert av.event_roster_populated({"period": 0, "entries": []})[0] is False


def test_the_starting_quarterback_would_have_parsed_as_out():
    """Documents the blast radius of consuming the unpopulated payload."""
    rows = av.parse_event_roster(UNPOPULATED)
    assert all(r["active"] is False for r in rows)
    assert any(r["name"] == "Darnold" for r in rows), (
        "the starter parses as not-active, which resolve_statuses reads as OUT")


# --------------------------------------------------------------------------- #
# The pipeline refuses to derive OUT from it
# --------------------------------------------------------------------------- #
def test_unpopulated_rows_never_reach_the_resolver():
    src = inspect.getsource(pw.gather_live_feeds)
    assert "inactives_state" in src, "the three states must be distinguished"
    assert 'inactives_state != "populated"' in src and "inactive_rows = []" in src, (
        "an unpopulated payload must be emptied before resolve_statuses sees it")


def test_t90_active_names_is_none_when_unpopulated():
    """An empty set of active names is 'we do not know', not 'nobody was
    elevated' -- the eligibility check must not read the two the same way."""
    src = inspect.getsource(pw.gather_live_feeds)
    assert 'inactives_state == "populated" and inactive_rows is not None' in src


# --------------------------------------------------------------------------- #
# The event id is actually resolved and passed
# --------------------------------------------------------------------------- #
def test_run_t90_passes_game_event_ids():
    src = inspect.getsource(pw.run_t90)
    assert "game_event_ids=" in src, (
        "gather_live_feeds's game_event_ids parameter had no caller; T-90 "
        "never fetched inactives at all")
    assert "find_event_ids" in src


def test_find_event_ids_matches_on_the_scoreboard():
    board = {"events": [{"id": "401872656", "competitions": [{"competitors": [
        {"homeAway": "home", "team": {"abbreviation": "SEA"}},
        {"homeAway": "away", "team": {"abbreviation": "NE"}}]}]}]}
    captured = {}

    def fake_get_json(url, params=None):
        captured["dates"] = (params or {}).get("dates")
        return board

    orig = av.get_json
    av.get_json = fake_get_json
    try:
        got = av.find_event_ids([{"game_id": "2026_01_NE_SEA", "gameday": "2026-09-09",
                                  "home_team": "SEA", "away_team": "NE"}])
    finally:
        av.get_json = orig
    assert got == {"2026_01_NE_SEA": "401872656"}
    assert captured["dates"] == "20260909", "scoreboard is queried by the game's date"


def test_an_unmatched_game_yields_no_id_rather_than_a_guess():
    orig = av.get_json
    av.get_json = lambda url, params=None: {"events": []}
    try:
        assert av.find_event_ids([{"game_id": "g", "gameday": "2026-09-09",
                                   "home_team": "SEA", "away_team": "NE"}]) == {}
    finally:
        av.get_json = orig


# --------------------------------------------------------------------------- #
# The reported reason is the true one
# --------------------------------------------------------------------------- #
def test_the_gate_reason_names_the_real_cause():
    src = inspect.getsource(pw.run_t90)
    assert "source has not published yet" in src, (
        '"no/unparseable timestamp" reads as a parsing bug and sends the next '
        "reader hunting for a defect that is not there")

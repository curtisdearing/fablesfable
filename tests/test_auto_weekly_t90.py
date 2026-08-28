"""What the T-90 job pulls before it re-ranks.

T-90 is the read closest to kickoff, so it is where a real sportsbook price
is worth the most -- and it is the last chance to capture one. Two distinct
spends, both hard-stopped by the shared monthly credit budget:

* RESNAP -- games that already carry an entry line get a second, pre-kickoff
  snapshot. This is what makes CLV resolvable (entry = Wednesday, close =
  here) and what lets the re-rank price against a line that has since moved.
* FIRST PULL -- games with NO line at all. Wednesday's rotation and per-run
  cap skip several games every week and they publish `no_market`. Pulling one
  at T-90 is the most informative credit this pipeline can spend, but it is a
  real change to the monthly credit profile, so it is OPT-IN and never
  bypasses the budget stop.

The distinction matters because getting it wrong is expensive in the two
directions that matter: silently spending credits nobody authorised, or
silently publishing a synthetic line 90 minutes before kickoff when a real
one was one call away.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import auto_weekly  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402

SEASON, WEEK = 2025, 10
HAS_LINES = f"{SEASON}_10_AAA_BBB"
NO_LINES = f"{SEASON}_10_CCC_DDD"


def _soon() -> pd.DataFrame:
    return pd.DataFrame([
        dict(game_id=HAS_LINES, season=SEASON, week=WEEK,
             home_team="BBB", away_team="AAA"),
        dict(game_id=NO_LINES, season=SEASON, week=WEEK,
             home_team="DDD", away_team="CCC"),
    ])


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t90.db"))
    dbmod.upsert(c, "lines", [{
        "ts": "2025-11-05T17:00:00Z", "game_id": HAS_LINES, "book": "bookx",
        "market": "receiving_yards", "player_id": None, "player_name": "A Wideout",
        "side": "over", "point": 61.5, "price": 1.87,
    }], ["ts", "game_id", "book", "market", "player_name", "side"])
    yield c
    c.close()


class _Spy:
    """Stands in for a real Odds API call: records, never fetches."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def __call__(self, cfg, event_map, conn=None, **kw):
        self.calls.append(sorted(event_map))
        return {"pulled": sorted(event_map), "skipped_budget": [], "skipped_cap": [],
                "rows_written": 0, "budget_remaining": 400.0, "ts": "now"}


def _emap(cfg, slate, list_events_fn=None):
    """Offline stand-in for build_event_map (which lists events over HTTP)."""
    return {g.game_id: f"evt_{g.game_id}" for g in slate.itertuples(index=False)}


def _cfg(**kw):
    base = {"odds_api_key": "k", "odds_budget": {"monthly_credits": 500, "reserve": 50}}
    base.update(kw)
    return base


# =========================================================================== #
# The resnap half: unchanged CLV behaviour.
# =========================================================================== #
def test_resnap_covers_only_games_that_already_have_an_entry_line(conn):
    resnap, pull = _Spy("resnap"), _Spy("pull")
    out = auto_weekly.t90_line_snapshot(_cfg(), conn, _soon(),
                                        resnap=resnap, pull=pull, event_map_fn=_emap)
    assert resnap.calls == [[HAS_LINES]], \
        "the closing snapshot must target exactly the games with entry lines"
    assert out["resnapped"] == [HAS_LINES]


def test_no_api_key_spends_nothing_and_says_so(conn):
    resnap, pull = _Spy("resnap"), _Spy("pull")
    out = auto_weekly.t90_line_snapshot({"odds_budget": {}}, conn, _soon(),
                                        resnap=resnap, pull=pull, event_map_fn=_emap)
    assert resnap.calls == [] and pull.calls == []
    assert out["resnapped"] == [] and out["first_pulled"] == []
    assert "no odds_api_key" in out["note"]


# =========================================================================== #
# The first-pull half: opt-in, and visible either way.
# =========================================================================== #
def test_lineless_games_are_reported_even_when_first_pull_is_off(conn):
    """Off by default -- but never silently. A game about to be published
    `no_market` 90 minutes before kickoff is a fact the operator should see."""
    resnap, pull = _Spy("resnap"), _Spy("pull")
    out = auto_weekly.t90_line_snapshot(_cfg(), conn, _soon(),
                                        resnap=resnap, pull=pull, event_map_fn=_emap)
    assert pull.calls == [], "a first pull must not happen unless it is enabled"
    assert out["first_pull_enabled"] is False
    assert out["without_lines"] == [NO_LINES]
    assert NO_LINES in out["note"]


def test_first_pull_is_opt_in_and_covers_the_lineless_games(conn):
    resnap, pull = _Spy("resnap"), _Spy("pull")
    cfg = _cfg(odds_budget={"monthly_credits": 500, "reserve": 50,
                            "t90_first_pull": True})
    out = auto_weekly.t90_line_snapshot(cfg, conn, _soon(),
                                        resnap=resnap, pull=pull, event_map_fn=_emap)
    assert pull.calls == [[NO_LINES]], \
        "the first pull must cover exactly the games with no line yet"
    assert resnap.calls == [[HAS_LINES]], "the resnap half is unchanged"
    assert out["first_pull_enabled"] is True
    assert out["first_pulled"] == [NO_LINES]


def test_first_pull_does_not_re_pull_a_game_that_already_has_a_line(conn):
    """The two halves must never both spend on the same game."""
    resnap, pull = _Spy("resnap"), _Spy("pull")
    cfg = _cfg(odds_budget={"t90_first_pull": True})
    auto_weekly.t90_line_snapshot(cfg, conn, _soon(),
                                  resnap=resnap, pull=pull, event_map_fn=_emap)
    assert HAS_LINES not in (pull.calls[0] if pull.calls else [])


def test_a_budget_stop_is_surfaced_not_swallowed(conn):
    """Credits are metered. A skipped game must reach the log, because the
    consequence is a published `no_market` the reader will otherwise read as
    'the model had nothing to say'."""
    class Broke(_Spy):
        def __call__(self, cfg, event_map, conn=None, **kw):
            self.calls.append(sorted(event_map))
            return {"pulled": [], "skipped_budget": sorted(event_map),
                    "skipped_cap": [], "rows_written": 0,
                    "budget_remaining": 0.0, "ts": "now"}

    out = auto_weekly.t90_line_snapshot(
        _cfg(odds_budget={"t90_first_pull": True}), conn, _soon(),
        resnap=Broke("resnap"), pull=Broke("pull"), event_map_fn=_emap)
    assert out["skipped_budget"] == sorted({HAS_LINES, NO_LINES})
    assert "budget" in out["note"]


def test_a_failing_snapshot_never_takes_the_t90_run_down_with_it(conn):
    """The re-rank is the product; a dead odds call degrades it to synthetic
    lines, it does not cancel it."""
    def boom(cfg, event_map, conn=None, **kw):
        raise RuntimeError("odds api exploded")

    out = auto_weekly.t90_line_snapshot(_cfg(), conn, _soon(),
                                        resnap=boom, pull=boom, event_map_fn=_emap)
    assert out["resnapped"] == []
    assert "odds api exploded" in out["note"]

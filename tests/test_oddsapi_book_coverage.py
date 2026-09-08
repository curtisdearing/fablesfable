"""Why did only DraftKings return quotes on 2026-09-02?

The pull sends ``bookmakers=<config books>``; every bookmaker present in the
provider's response is parsed and stored (no per-book filtering exists in
the client).  So a stored table that carries one book means the provider
returned one book.  These tests pin that: the client keeps every returned
book, and it now reports which requested books the provider did not return
so the next run's log says so instead of leaving it to forensics.
"""
from __future__ import annotations

import pandas as pd

from nflvalue import db as dbmod
from nflvalue.sources import oddsapi_props as oap

CFG = {"odds_api_key": "k", "books": ["draftkings", "betmgm", "hardrockbet"],
       "prop_markets_internal": ["receiving_yards"], "max_prop_games_per_run": 4}


def _payload(books):
    return {"bookmakers": [
        {"key": b, "markets": [{"key": "player_reception_yds", "outcomes": [
            {"name": "Over", "description": "A Player", "point": 50.5, "price": 1.9},
            {"name": "Under", "description": "A Player", "point": 50.5, "price": 1.9}]}]}
        for b in books]}


def _run(tmp_path, books):
    conn = dbmod.connect(str(tmp_path / "o.db"))
    calls = []

    def fetch(url, params):
        calls.append(params)
        return _payload(books)

    res = oap.pull_week_props(CFG, {"2026_01_AAA_BBB": "evt1"}, conn=conn, fetch=fetch, ts="2026-09-02T01:13:15Z")
    stored = pd.read_sql("SELECT * FROM lines", conn)
    return res, stored, calls


def test_every_returned_book_is_stored_and_absent_requested_books_are_named(tmp_path):
    res, stored, calls = _run(tmp_path, ["draftkings"])
    assert calls[0]["bookmakers"] == "draftkings,betmgm,hardrockbet"
    assert sorted(stored["book"].unique()) == ["draftkings"]
    cov = res["book_coverage"]
    assert cov["requested"] == ["draftkings", "betmgm", "hardrockbet"]
    assert cov["returned"] == ["draftkings"]
    assert cov["absent_from_provider_response"] == ["betmgm", "hardrockbet"]
    assert cov["by_game"]["2026_01_AAA_BBB"] == ["draftkings"]


def test_a_second_returned_book_is_kept_not_filtered(tmp_path):
    res, stored, _ = _run(tmp_path, ["draftkings", "betmgm"])
    assert sorted(stored["book"].unique()) == ["betmgm", "draftkings"]
    assert res["book_coverage"]["absent_from_provider_response"] == ["hardrockbet"]
    # two books quoting both sides at the same point -> a two-book market
    rows = oap.match_player_ids(stored.to_dict("records"),
                                pd.DataFrame([{"player_id": "P1", "name": "A.Player"}]))
    frame = oap.to_prop_lines_frame(rows)
    assert int(frame.iloc[0]["n_books"]) == 2


def test_resnap_reports_coverage_too(tmp_path):
    conn = dbmod.connect(str(tmp_path / "r.db"))
    res = oap.resnap_lines(CFG, {"2026_01_AAA_BBB": "evt1"}, conn=conn,
                           fetch=lambda url, params: _payload(["draftkings"]), ts="2026-09-09T23:15:00Z")
    assert res["book_coverage"]["absent_from_provider_response"] == ["betmgm", "hardrockbet"]

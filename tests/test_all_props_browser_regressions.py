import re

import pytest

from nflvalue import all_props


def _payload():
    visible = {
        "player": "Sample Receiver",
        "player_id": "opaque-player-id",
        "team": "ABC",
        "market": "receiving_yards",
        "side": "over",
        "line": 54.5,
        "book": "ExampleBook",
        "odds": 105,
        "quote_updated_at": "2026-10-10T17:00:00Z",
        "retrieved_at": "2026-10-10T17:01:00Z",
        "status": "analyst_lean",
        "display_tier": "preferred",
        "display_why": "The line is below his recent workload.",
        "display_risk": "A slower game could limit opportunities.",
        "rationale": {"must_not": "render"},
        "counterargument": {"must_not": "render"},
        "sources": [{"title": "Example source", "url": "https://example.test/source"}],
    }
    hidden = {
        "player": "Early Placeholder",
        "market": "receiving_yards",
        "side": "under",
        "line": 54.5,
        "book": "ExampleBook",
        "odds": -110,
        "status": "research",
        "display_status": "pass",
        "rationale": "Do not display this.",
    }
    return {
        "state": "ready",
        "cards": [
            {"event_id": "opaque-game-id", "game_id": "2026_05_ABC_DEF", "away": "ABC", "home": "DEF",
             "kickoff": "2026-10-11T20:05:00Z", "source_as_of": "2026-10-10T17:01:00Z",
             "context": {"opaque": {"raw": "dict"}}, "picks": [visible, hidden]},
            {"event_id": "no-pick-game", "game_id": "2026_05_GHI_JKL", "away": "GHI", "home": "JKL",
             "kickoff": "2026-10-12T20:05:00Z", "no_pick_reason": "No reliable current player price.",
             "picks": [{"player": "No named player prop", "status": "pass"}]},
        ],
        "rows": [],
        "counts": all_props._counts([], []),
    }


def test_picks_board_is_plain_language_and_hides_research_inventory():
    page = all_props.render_page(_payload())

    assert "Sample Receiver OVER 54.5 Receiving yards" in page
    assert "ExampleBook +105" in page
    assert "The line is below his recent workload." in page
    assert "Risk:" in page
    assert "Preferred picks" in page
    assert "Early Placeholder" not in page
    assert "No named player prop" not in page
    assert "opaque-player-id" not in page
    assert "2026_05_ABC_DEF" not in page
    assert "{'must_not': 'render'}" not in page
    assert "Complete captured/rejected market rows" not in page
    assert "<table" not in page
    assert page.count("Sample Receiver OVER 54.5 Receiving yards") == 1
    assert "No pick." in page
    assert "No reliable current player price." in page
    assert "Download research data (JSON)" in page
    assert "api/all-props.json" in page


def test_sections_are_not_invented_without_explicit_tiers():
    payload = _payload()
    payload["cards"][0]["picks"][0].pop("display_tier")
    page = all_props.render_page(payload)

    assert "<h2>Preferred picks</h2>" not in page
    assert "<h2>Other leans</h2>" not in page


def test_categorical_td_offer_and_placeholder_count():
    row = {"player": "Real Player", "side": "yes", "line": None, "book": "Book", "odds": 120,
           "captured_at": "2026-10-10T17:00:00Z"}
    assert all_props._has_offer(row)
    gap = {**row, "player": "No named player", "raw_market_row_type": "coverage_gap"}
    assert not all_props._has_offer(gap)
    assert all_props._counts([], [all_props._normalise_row(row), all_props._normalise_row(gap)])["unique_athletes"] == 1


def test_et_clock_and_source_scheme():
    assert "4:05 PM ET" in all_props._kickoff("2026-10-11T20:05:00Z")
    assert not all_props._source_url("javascript:alert(1)")
    assert all_props._source_url("https://www.espn.com/")


def browser_smoke_optional():
    playwright = pytest.importorskip("playwright.sync_api")
    browser = None
    try:
        browser = playwright.sync_playwright().start().chromium.launch(channel="chrome")
    except Exception as exc:
        pytest.skip(f"Chrome unavailable: {exc}")
    try:
        assert browser is not None
        page = browser.new_page(viewport={"width": 320, "height": 700})
        page.set_content("<style>" + (
            "*{box-sizing:border-box}body{margin:0;padding:12px;overflow-wrap:anywhere}"
            ".picks-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:14px}"
            ".pick-card,.ap-game{min-width:0;padding:14px}"
        ) + "</style>" + all_props.render_page(_payload()))
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    finally:
        if browser:
            browser.close()

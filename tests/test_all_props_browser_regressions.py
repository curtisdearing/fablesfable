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


def test_native_model_projection_compares_pick_to_line_and_keeps_full_forecast_rows_honest():
    payload = _payload()
    card = payload["cards"][0]
    pick = card["picks"][0]
    pick["model_projection"] = 68.2
    pick["model_run_as_of"] = "2026-10-10T22:35:00Z"
    card["player_forecasts"] = [
        {"name": "Sample Receiver", "position": "WR", "means": {"receptions": 6.4,
                                                             "receiving_yards": 68.2,
                                                             "anytime_td": 0.61},
         "anytime_td_p_ge_1": 0.46, "model_run_as_of": "2026-10-10T22:35:00Z", "role": "primary receiver"},
        {"player": "Unmatched Back", "pos": "RB", "means": {"rush_attempts": 10.5,
                                                                    "rushing_yards": 43.7},
         "role": "committee", "supported": False},
    ]

    page = all_props.render_page(payload)
    visible_text = re.sub(r"<[^>]+>", "", page)

    assert "Model: 68.2 yards · Line: 54.5" in visible_text
    assert "Uncalibrated model lean" in page
    assert "Model run:" in visible_text and "2026-10-10T22:35:00Z" in visible_text
    assert "<summary>Player projections</summary>" in page
    assert "Pass yds</th><th>Carries</th><th>Rush yds</th><th>Catches</th><th>Rec yds</th>" in page
    assert "TD probability" in page and "46%" in page
    assert "Unmatched Back" in page and "Not model-supported" in page
    assert "0.61" not in page  # A TD mean is not a touchdown probability.
    assert "No current line" not in page and "No line" not in page


def test_uncalibrated_native_disclosure_appears_once_for_multiple_model_leans():
    payload = _payload()
    first = payload["cards"][0]["picks"][0]
    first["model_projection"] = 68.2
    second = dict(first, player="Another Receiver", line=44.5, model_projection=51.1)
    payload["cards"][0]["picks"].append(second)

    page = all_props.render_page(payload)

    assert page.count("Uncalibrated model leans") == 1


def test_completed_game_is_not_promoted_as_a_current_native_forecast():
    payload = _payload()
    payload["cards"][0]["status"] = "completed"
    payload["cards"][0]["player_forecasts"] = [{"name": "Sample Receiver", "means": {"receiving_yards": 68.2}}]

    page = all_props.render_page(payload)

    assert "Sample Receiver OVER 54.5 Receiving yards" not in page
    assert "Player projections" not in page
    assert "No pick." in page


def test_native_model_differences_sort_per_game_by_absolute_percentage_gap_and_show_signed_gap():
    payload = _payload()
    first, second = payload["cards"]
    first["native_model"] = True
    first["picks"] = [
        {"player": "Yard Receiver", "team": "ABC", "market": "receiving_yards", "side": "over",
         "line": 80.0, "model_projection": 100.0, "book": "ExampleBook", "odds": -110,
         "status": "analyst_lean"},
        {"player": "Catch Receiver", "team": "ABC", "market": "receptions", "side": "under",
         "line": 5.0, "model_projection": 3.0, "book": "ExampleBook", "odds": -110,
         "status": "analyst_lean"},
        {"player": "Small Yard Gap", "team": "ABC", "market": "rushing_yards", "side": "over",
         "line": 40.0, "model_projection": 45.0, "book": "ExampleBook", "odds": -110,
         "status": "analyst_lean"},
    ]
    second["native_model"] = True
    second["picks"] = [
        {"player": "Other Game First", "team": "GHI", "market": "receiving_yards", "side": "over",
         "line": 50.0, "model_projection": 55.0, "book": "ExampleBook", "odds": -110,
         "status": "analyst_lean"},
        {"player": "Other Game Largest", "team": "GHI", "market": "receptions", "side": "under",
         "line": 4.0, "model_projection": 2.0, "book": "ExampleBook", "odds": -110,
         "status": "analyst_lean"},
    ]

    page = all_props.render_page(payload)
    first_game = page.split("id='game-opaque-game-id'", 1)[1].split("id='game-no-pick-game'", 1)[0]
    second_game = page.split("id='game-no-pick-game'", 1)[1]

    # 40% receptions gap outranks a 25% yardage gap despite unlike raw units.
    assert first_game.index("Catch Receiver") < first_game.index("Yard Receiver") < first_game.index("Small Yard Gap")
    # Sorting restarts for each game rather than globally.
    assert second_game.index("Other Game Largest") < second_game.index("Other Game First")
    text = re.sub(r"<[^>]+>", "", first_game)
    assert "Model-vs-line gap: -2 catches (40%)" in text
    assert "Model-vs-line gap: +20 yards (25%)" in text
    assert "discrepancy, not confidence" in page


def test_native_model_differences_leave_noncomparable_zero_missing_and_td_rows_after_sorted_rows_with_stable_ties():
    payload = _payload()
    card = payload["cards"][0]
    card["native_model"] = True
    card["picks"] = [
        {"player": "Zulu", "team": "ABC", "market": "receiving_yards", "side": "over", "line": 20.0,
         "model_projection": 30.0, "book": "ExampleBook", "odds": -110, "status": "analyst_lean"},
        {"player": "Alpha", "team": "ABC", "market": "receptions", "side": "under", "line": 4.0,
         "model_projection": 2.0, "book": "ExampleBook", "odds": -110, "status": "analyst_lean"},
        {"player": "TD Player", "team": "ABC", "market": "anytime_td", "side": "yes", "line": None,
         "model_probability": 0.75, "book": "ExampleBook", "odds": 120, "status": "analyst_lean"},
        {"player": "Zero Line", "team": "ABC", "market": "receptions", "side": "over", "line": 0.0,
         "model_projection": 1.0, "book": "ExampleBook", "odds": -110, "status": "analyst_lean"},
        {"player": "Missing Projection", "team": "ABC", "market": "rushing_yards", "side": "under", "line": 30.0,
         "book": "ExampleBook", "odds": -110, "status": "analyst_lean"},
    ]

    page = all_props.render_page(payload)
    game = page.split("id='game-opaque-game-id'", 1)[1].split("id='game-no-pick-game'", 1)[0]
    visible_text = re.sub(r"<[^>]+>", "", game)

    # Alpha and Zulu have the same 50% gap; player then market resolves ties deterministically.
    assert game.index("Alpha") < game.index("Zulu") < game.index("Missing Projection") < game.index("TD Player") < game.index("Zero Line")
    assert visible_text.count("Model-vs-line gap: Not comparable") == 3
    assert "Model-vs-line gap: -2 catches (50%)" in visible_text
    assert "Model-vs-line gap: +10 yards (50%)" in visible_text


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

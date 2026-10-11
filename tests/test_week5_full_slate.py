import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("week5_board", ROOT / "scripts" / "build_week5_full_slate.py")
board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(board)


def _event(event_id, away, home, date, final=False):
    return {
        "id": event_id,
        "date": date,
        "shortName": f"{away} @ {home}",
        "status": {"type": {"completed": final, "name": "STATUS_FINAL" if final else "STATUS_SCHEDULED"}},
        "competitions": [{"venue": {"fullName": "Test Stadium", "indoor": True}, "competitors": [
            {"homeAway": "away", "team": {"abbreviation": away}, "score": "24" if final else "0"},
            {"homeAway": "home", "team": {"abbreviation": home}, "score": "16" if final else "0"},
        ]}],
    }


def test_integrate_full_schedule_preserves_late_raw_rows_and_marks_actual_coverage_gaps(tmp_path):
    events = [
        _event("1", "TB", "DAL", "2026-10-09T00:15Z", final=True),
        _event("2", "CHI", "GB", "2026-10-11T17:00Z"),
        _event("3", "DEN", "LAC", "2026-10-11T20:05Z"),
    ]
    events.extend(_event(str(i), f"A{i}", f"H{i}", f"2026-10-12T{i:02}:00Z") for i in range(4, 16))
    scoreboard = {"season": {"year": 2026, "type": 2}, "week": {"number": 5}, "events": events}
    early = {"cards": [{"game_id": "2026_05_CHI_GB", "player": "Test Receiver", "team": "GB",
                         "market": "receiving_yards", "side": "over", "line": 50.5,
                         "quote": {"book": "draftkings", "price_american": "+100", "captured_at": "2026-10-09T10:00:00Z"},
                         "run_as_of": "2026-10-09T10:01:00Z", "rationale": "Saved source card",
                         "countercase": "Role change", "invalidation": ["inactive"], "status": "research"}]}
    late = [{"game": "DEN_LAC", "player_label": "Test Back RB DEN", "Prop": "Receiving yards", "Line": "27.5",
             "Best over": "-114 FanDuel", "Best under": "-110 BetMGM", "Books": "2", "source": "https://example.test/late",
             "retrieved_at": "2026-10-10T16:13:21+00:00", "observed_history": "[]", "history_n": 0,
             "model_probability": "", "calibrated_probability": "", "quote_verification": "Secondary published listing",
             "rationale": "Observed usage", "disposition": "primary_conditional_analyst_lean", "selected_side": "OVER", "selected_price": "-114 FanDuel"}]
    for name, value in (("scoreboard.json", scoreboard), ("early.json", early), ("late.json", late)):
        (tmp_path / name).write_text(json.dumps(value))

    cards, rows, manifest = board.integrate(str(tmp_path / "scoreboard.json"), str(tmp_path / "early.json"),
                                            str(tmp_path / "late.json"))

    assert [c["event_id"] for c in cards][:3] == ["1", "2", "3"] and len(cards) == 15
    assert cards[0]["status"] == "completed" and cards[0]["outcome"]["score"] == "TB 24, DAL 16"
    assert cards[1]["coverage"]["state"] == "coverage_gap"
    assert cards[2]["picks"][0]["status"] == "analyst_lean"
    assert {r["side"] for r in rows if r["event_id"] == "3"} == {"over", "under"}
    assert rows[0].get("raw_source_row") is not None
    assert manifest["raw_market_rows"] == 1 and manifest["outcome_rows"] == 2


def test_rendered_board_shows_clean_pick_card_without_research_inventory():
    payload = {"schema": "fablesfable.all_props.v1", "state": "ready", "counts": {"games": 1, "raw_market_rows": 1,
               "outcome_rows": 1, "quote_rows": 1, "unique_athletes": 1, "model_priced": 0,
               "qualitatively_reviewed": 1, "unavailable_or_unsupported": 0}, "errors": [],
               "cards": [{"event_id": "1", "game_id": "2026_05_DEN_LAC", "away": "DEN", "home": "LAC",
                          "kickoff": "2026-10-11T20:05:00Z", "source_as_of": "2026-10-10T16:13:21Z", "status": "upcoming",
                          "preview": "Readable analysis", "winner_lean": "No winner lean issued.",
                          "context": {"venue": "Test Stadium"}, "coverage": {"state": "captured"},
                          "sources": [{"title": "Source", "url": "https://example.test/source"}], "picks": [{
                              "player": "Test Receiver", "market": "receiving_yards", "side": "over", "line": 27.5,
                              "book": "FanDuel", "odds": -114, "status": "analyst_lean",
                              "display_why": "The line is low for his role.", "display_risk": "Volume could fall.",
                              "sources": [{"title": "Source", "url": "https://example.test/source"}],
                          }]}],
               "rows": [{"event_id": "1", "game_id": "2026_05_DEN_LAC", "player": "Test", "team": "DEN",
                         "market": "receiving_yards", "side": "over", "period": "full_game", "status": "analyst_lean",
                         "disposition": "reviewed", "line": 27.5, "odds": -114, "book": "FanDuel",
                         "captured_at": "2026-10-10T16:13:21Z", "provider_updated_at": None, "reason": "Why",
                         "offer_label": "Published secondary listing — not sportsbook-verified", "sources": []}]}
    from nflvalue import all_props
    page = all_props.render_page(payload)
    assert "Test Receiver OVER 27.5 Receiving yards" in page
    assert "FanDuel -114" in page
    assert "The line is low for his role." in page and "Risk:" in page
    assert 'href="https://example.test/source"' in page
    assert "No winner lean issued" not in page and "Test Stadium" not in page
    assert "<table" not in page


def test_rendered_game_header_leads_with_three_market_decisions_and_both_team_injuries():
    from nflvalue import all_props
    payload = {"schema": "fablesfable.all_props.v1", "state": "ready",
               "counts": {"games": 1, "raw_market_rows": 0, "outcome_rows": 0, "quote_rows": 0,
                          "unique_athletes": 0, "model_priced": 0, "qualitatively_reviewed": 0,
                          "unavailable_or_unsupported": 0}, "errors": [],
               "cards": [{"event_id": "1", "game_id": "2026_05_AWY_HOME", "away": "AWY", "home": "HOME",
                          "kickoff": "2026-10-11T20:05:00Z", "source_as_of": "2026-10-10T23:02:49Z",
                          "status": "upcoming", "preview": "No player pick.", "winner_lean": None,
                          "picks": [],
                          "game_markets": [
                              {"market": "moneyline", "decision": "No supported pick", "line": None,
                               "price": None, "source_state": "missing"},
                              {"market": "spread", "decision": "Prior analyst lean: AWY +3 (not current)", "line": "+3",
                               "price": "+105", "source_state": "stale", "captured_at": "2026-10-10T17:00:00Z"},
                              {"market": "total", "decision": "No supported pick", "line": "44.5",
                               "price": None, "source_state": "line_only"},
                          ],
                          "injuries": {"source": "ESPN event summary", "url": "https://example.test/summary?event=1", "retrieved_at": "2026-10-10T23:02:49Z",
                                       "teams": [{"team": "AWY", "items": [{"name": "Away QB", "position": "QB",
                                                                            "status": "Questionable", "impact": "Starting-QB availability unresolved."}]},
                                                 {"team": "HOME", "items": [{"name": "Home LT", "position": "OT",
                                                                             "status": "Out", "impact": "Starting tackle unavailable."}]}]}}],
               "rows": []}
    page = all_props.render_page(payload)
    section = page[page.index("<section class='ap-game'"):]
    assert "Moneyline" in section and "Spread" in section and "Total" in section
    assert "No supported pick" in section
    assert "AWY injuries" in section and "HOME injuries" in section
    assert "Away QB" in section and "Home LT" in section
    assert "href='https://example.test/summary?event=1'" in section
    assert section.index("Moneyline") < section.index("Away QB") < section.index("No pick.")


def test_frontmatter_overlay_keeps_markets_unsupported_when_fresh_espn_has_no_odds_and_marks_archive_unknown(tmp_path):
    front_spec = importlib.util.spec_from_file_location("week5_frontmatter", ROOT / "scripts" / "build_week5_frontmatter.py")
    assert front_spec and front_spec.loader
    frontmatter = importlib.util.module_from_spec(front_spec)
    front_spec.loader.exec_module(frontmatter)
    summary = {"odds": [], "injuries": [{"team": {"abbreviation": "AWY"}, "injuries": [{
        "status": "Questionable", "athlete": {"displayName": "Away Quarterback", "position": {"abbreviation": "QB"}}}]},
        {"team": {"abbreviation": "HOME"}, "injuries": [{"status": "Out", "athlete": {
            "displayName": "Home Tackle", "position": {"abbreviation": "OT"}}}]}]}
    (tmp_path / "1.json").write_text(json.dumps(summary))
    cards = [{"event_id": "1", "away": "AWY", "home": "HOME", "context": {"game_market": {
        "details": "AWY +3", "total": 44.5, "provider": "Captured source"}}},
             {"event_id": "archive", "away": "OLD", "home": "DONE", "context": {}}]
    result = frontmatter.overlay(cards, [{"event_id": "1", "url": "https://example.test/1",
                                          "retrieved_at": "2026-10-10T23:02:49Z"}], tmp_path)
    current, archived = result
    assert [market["market"] for market in current["game_markets"]] == ["moneyline", "spread", "total"]
    assert all(market["decision"] == "No supported pick" for market in current["game_markets"])
    assert current["game_markets"][1]["line"] is None and current["game_markets"][1]["price"] is None
    assert current["game_markets"][1]["source_state"] == "ESPN pickcenter close unavailable"
    assert "Quarterback availability" in current["injuries"]["teams"][0]["items"][0]["impact"]
    assert archived["injuries"]["teams"] == [{"team": "OLD", "items": []}, {"team": "DONE", "items": []}]


def test_frontmatter_normalizes_pickcenter_close_fields_without_fabricating_juice_or_signs():
    front_spec = importlib.util.spec_from_file_location("week5_frontmatter", ROOT / "scripts" / "build_week5_frontmatter.py")
    assert front_spec and front_spec.loader
    frontmatter = importlib.util.module_from_spec(front_spec)
    front_spec.loader.exec_module(frontmatter)
    summary = {"pickcenter": [{"provider": {"name": "DraftKings"},
               "moneyline": {"away": {"close": {"odds": "+310"}}, "home": {"close": {"odds": "-395"}}},
               "pointSpread": {"away": {"close": {"line": "+7.5", "odds": "-112"}},
                               "home": {"close": {"line": "-7.5", "odds": "-108"}}},
               "total": {"over": {"close": {"line": "o41.5", "odds": "-112"}},
                         "under": {"close": {"line": "u41.5", "odds": "-108"}}}}]}
    markets = frontmatter._markets({"away": "PHI", "home": "JAX"}, summary)
    by_market = {m["market"]: m for m in markets}
    assert by_market["moneyline"]["line"] == "PHI +310 · JAX -395"
    assert by_market["spread"]["line"] == "PHI +7.5 · JAX -7.5"
    assert by_market["spread"]["price"] == "PHI -112 · JAX -108"
    assert by_market["total"]["line"] == "Over 41.5 · Under 41.5"
    assert by_market["total"]["price"] == "Over -112 · Under -108"
    assert all(m["decision"] == "No supported pick" for m in markets)
    assert all(m["source_state"] == "ESPN listed close; not executable verified" for m in markets)


def test_frontmatter_preserves_zero_prices_and_omits_absent_sides_without_default_juice():
    front_spec = importlib.util.spec_from_file_location("week5_frontmatter", ROOT / "scripts" / "build_week5_frontmatter.py")
    assert front_spec and front_spec.loader
    frontmatter = importlib.util.module_from_spec(front_spec)
    front_spec.loader.exec_module(frontmatter)
    summary = {"pickcenter": [{"provider": {"name": "DraftKings"},
               "moneyline": {"away": {"close": {"odds": 0}}},
               "pointSpread": {"away": {"close": {"line": "-3", "odds": 0}}},
               "total": {"over": {"close": {"line": "o42.5"}}}}]}
    by_market = {m["market"]: m for m in frontmatter._markets({"away": "WAS", "home": "LA"}, summary)}
    assert by_market["moneyline"]["line"] == "WAS 0"
    assert by_market["spread"]["line"] == "WAS -3"
    assert by_market["spread"]["price"] == "WAS 0"
    assert by_market["total"]["line"] == "Over 42.5"
    assert by_market["total"]["price"] is None
    assert "-110" not in repr(by_market)


def test_frontmatter_injury_rows_include_report_date_and_latest_description_sorted_by_priority():
    front_spec = importlib.util.spec_from_file_location("week5_frontmatter", ROOT / "scripts" / "build_week5_frontmatter.py")
    assert front_spec and front_spec.loader
    frontmatter = importlib.util.module_from_spec(front_spec)
    front_spec.loader.exec_module(frontmatter)
    summary = {"injuries": [{"team": {"abbreviation": "WAS"}, "injuries": [
        {"status": "Questionable", "date": "2026-10-10T01:11Z", "athlete": {"displayName": "Wideout", "position": {"abbreviation": "WR"}},
         "details": {"type": "Ankle", "detail": "Limited in practice"}},
        {"status": "Out", "date": "2026-10-10T02:24Z", "athlete": {"displayName": "Quarterback", "position": {"abbreviation": "QB"}},
         "details": {"type": "Knee", "detail": "Will not play"}},
    ]}]}
    items = frontmatter._injury_team(summary, "WSH")["items"]
    assert [item["name"] for item in items] == ["Quarterback", "Wideout"]
    assert items[0]["report_date"] == "2026-10-10T02:24Z"
    assert items[0]["body_part"] == "Knee"
    assert items[0]["description"] == "Will not play"
    assert "Quarterback availability" in items[0]["impact"]


def test_frontmatter_overlay_accepts_versioned_receipt_index(tmp_path):
    front_spec = importlib.util.spec_from_file_location("week5_frontmatter", ROOT / "scripts" / "build_week5_frontmatter.py")
    assert front_spec and front_spec.loader
    frontmatter = importlib.util.module_from_spec(front_spec)
    front_spec.loader.exec_module(frontmatter)
    (tmp_path / "1.json").write_text(json.dumps({"pickcenter": [], "injuries": []}))
    cards = [{"event_id": "1", "away": "AWY", "home": "HOME"}]
    receipt = {"schema": "fablesfable.espn-event-summary-receipts.v1", "events": [{
        "event_id": "1", "url": "https://example.test/1", "retrieved_at": "2026-10-11T02:00:23Z"}]}
    result = frontmatter.overlay(cards, receipt, tmp_path)
    assert result[0]["injuries"]["retrieved_at"] == "2026-10-11T02:00:23Z"


def test_rendered_completed_game_is_labeled_archived_not_a_current_no_pick():
    from nflvalue import all_props
    payload = {"schema": "fablesfable.all_props.v1", "state": "ready",
               "counts": {"games": 1, "raw_market_rows": 0, "outcome_rows": 0, "quote_rows": 0,
                          "unique_athletes": 0, "model_priced": 0, "qualitatively_reviewed": 0,
                          "unavailable_or_unsupported": 0}, "errors": [],
               "cards": [{"event_id": "1", "game_id": "2026_05_TB_DAL", "away": "TB", "home": "DAL",
                          "kickoff": "2026-10-09T00:15:00Z", "source_as_of": "2026-10-09T00:00:00Z",
                          "status": "completed", "picks": [], "game_markets": [],
                          "injuries": {"teams": []}, "no_pick_reason": "Completed game — see archive."}], "rows": []}
    page = all_props.render_page(payload)
    assert "Archived — not a prospective pick." in page
    assert "No pick. Completed game" not in page


def test_integrate_merges_typed_card_and_row_extras_without_saved_native_picks(tmp_path):
    events = [_event("1", "DEN", "LAC", "2026-10-11T20:05Z")]
    events.extend(_event(str(i), f"A{i}", f"H{i}", f"2026-10-12T{i:02}:00Z") for i in range(2, 16))
    scoreboard = {"events": events}
    saved = {"cards": [{"game_id": "2026_05_DEN_LAC", "player": "Old Native", "team": "DEN",
                        "market": "receiving_yards", "side": "under", "line": 1.5,
                        "quote": {"book": "old", "price_american": "-110"}, "run_as_of": "2026-10-07T20:00:00Z"}]}
    late = [{"game": "DEN_LAC", "player_label": "RJ Harvey RB DEN", "Prop": "Receiving yards", "Line": "27.5",
             "Best over": "-114 FanDuel", "Best under": "-110 BetMGM", "source": "https://example.test/late",
             "retrieved_at": "2026-10-10T16:13:21Z", "rationale": "Observed usage",
             "disposition": "primary_conditional_analyst_lean", "selected_side": "OVER"}]
    cards_extra = [{"event_id": "1", "game_id": "2026_05_DEN_LAC", "away": "DEN", "home": "LAC",
                    "kickoff": "2026-10-11T20:05Z", "source_as_of": "2026-10-10T16:13:21Z", "status": "upcoming",
                    "preview": "Captured late preview.", "winner_lean": "No game-winner lean was issued.",
                    "context": {"countercase": "Role uncertainty."}, "coverage": {"state": "captured"},
                    "sources": [{"title": "Late source", "url": "https://example.test/late"}], "picks": []}]
    coverage_extra = [{"event_id": "1", "game_id": "2026_05_DEN_LAC", "player": "UNNORMALIZED_PUBLIC_PAGE_FAMILY",
                       "team": "DEN/LAC", "market": "passing_yards", "side": None, "period": "full_game",
                       "status": "pending", "disposition": "pending", "line": None, "odds": None, "book": None,
                       "captured_at": "2026-10-10T16:13:21Z", "raw_market_row_type": "family_availability", "sources": []}]
    for name, value in (("scoreboard.json", scoreboard), ("saved.json", saved), ("late.json", late),
                        ("cards-extra.json", cards_extra), ("rows-extra.json", coverage_extra)):
        (tmp_path / name).write_text(json.dumps(value))

    cards, rows, manifest = board.integrate(str(tmp_path / "scoreboard.json"), str(tmp_path / "saved.json"),
                                            str(tmp_path / "late.json"),
                                            [str(tmp_path / "cards-extra.json"), str(tmp_path / "rows-extra.json")])

    den = next(card for card in cards if card["event_id"] == "1")
    assert den["preview"] == "Captured late preview."
    assert den["source_as_of"] == "2026-10-10T16:13:21Z"
    assert {pick["player"] for pick in den["picks"]} == {"RJ Harvey"}
    assert "Missing Week 2 is not assumed zero" in den["picks"][0]["counterargument"]
    assert any(row.get("raw_market_row_type") == "family_availability" for row in rows)
    assert manifest["raw_market_rows"] == 1
    assert manifest["outcome_rows"] == 2
    assert manifest["coverage_family_rows"] == 1 and manifest["coverage_gap_rows"] == 0


def test_all_props_accepts_unpriced_coverage_records_without_counting_them_as_outcomes(tmp_path):
    from nflvalue import all_props
    card = {"event_id": "1", "game_id": "2026_05_DEN_LAC", "away": "DEN", "home": "LAC",
            "kickoff": "2026-10-11T20:05Z", "source_as_of": "2026-10-10T16:13:21Z", "picks": []}
    outcome = {"event_id": "1", "game_id": "2026_05_DEN_LAC", "player": "RJ Harvey", "team": "DEN",
               "market": "receiving_yards", "side": "over", "period": "full_game", "status": "research",
               "disposition": "pending", "line": 27.5, "odds": -114, "book": "FanDuel", "captured_at": "2026-10-10T16:13:21Z"}
    family = {"event_id": "1", "game_id": "2026_05_DEN_LAC", "player": "UNNORMALIZED_PUBLIC_PAGE_FAMILY",
              "team": "DEN/LAC", "market": "passing_yards", "side": None, "period": "full_game", "status": "pending",
              "disposition": "pending", "line": None, "odds": None, "book": None,
              "raw_market_row_type": "family_availability"}
    (tmp_path / "cards.json").write_text(json.dumps([card]))
    (tmp_path / "rows.json").write_text(json.dumps([outcome, family]))
    payload = all_props.load(str(tmp_path / "cards.json"), str(tmp_path / "rows.json"), expected_event_ids={"1"})
    assert payload["state"] == "ready"
    assert payload["counts"]["outcome_rows"] == 1

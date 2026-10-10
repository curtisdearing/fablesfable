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


def test_rendered_board_labels_secondary_listings_and_renders_winner_context_and_sources():
    payload = {"schema": "fablesfable.all_props.v1", "state": "ready", "counts": {"games": 1, "raw_market_rows": 1,
               "outcome_rows": 1, "quote_rows": 1, "unique_athletes": 1, "model_priced": 0,
               "qualitatively_reviewed": 1, "unavailable_or_unsupported": 0}, "errors": [],
               "cards": [{"event_id": "1", "game_id": "2026_05_DEN_LAC", "away": "DEN", "home": "LAC",
                          "kickoff": "2026-10-11T20:05:00Z", "source_as_of": "2026-10-10T16:13:21Z", "status": "upcoming",
                          "preview": "Readable analysis", "winner_lean": "No winner lean issued.",
                          "context": {"venue": "Test Stadium"}, "coverage": {"state": "captured"},
                          "sources": [{"title": "Source", "url": "https://example.test/source"}], "picks": []}],
               "rows": [{"event_id": "1", "game_id": "2026_05_DEN_LAC", "player": "Test", "team": "DEN",
                         "market": "receiving_yards", "side": "over", "period": "full_game", "status": "analyst_lean",
                         "disposition": "reviewed", "line": 27.5, "odds": -114, "book": "FanDuel",
                         "captured_at": "2026-10-10T16:13:21Z", "provider_updated_at": None, "reason": "Why",
                         "offer_label": "Published secondary listing — not sportsbook-verified", "sources": []}]}
    from nflvalue import all_props
    page = all_props.render_page(payload)
    assert "Published secondary listing — not sportsbook-verified" in page
    assert "No winner lean issued" in page and "Test Stadium" in page
    assert 'href="https://example.test/source"' in page


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

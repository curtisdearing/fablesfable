"""Evidence cards: fail-closed gates, no actionable row without a validated market."""

import datetime as dt

from nflvalue import pick_cards as pc

NOW = dt.datetime(2026, 9, 24, 20, 0, tzinfo=dt.timezone.utc)


def _row(**kw):
    r = {"name": "K.Pitts", "player_id": "p1", "game_id": "2026_03_ATL_GB", "market": "receiving_yards",
         "side": "OVER", "line": 45.5, "line_source": "odds_api", "price": 1.87, "book": "draftkings",
         "quote_ts": "2026-09-24T18:30:00Z", "as_of": "2026-09-24T18:35:00Z", "mean": 52.0,
         "sd": 26.0, "p_side": 0.58, "composite": 0.7, "roll_games": 6, "status": "active"}
    r.update(kw)
    return r


def test_no_card_is_actionable_while_no_market_is_validated():
    assert pc.VALIDATED_MARKETS == frozenset()
    c = pc.build_card(_row(), NOW, "v")
    assert c["status"] == "watch" and c["model_p_status"] == "unvalidated_at_offered_lines"
    assert c["price_american"] == "-115" and c["breakeven"] == round(1 / 1.87, 4)
    assert "not a probability" in c["ordering_score_note"]


def test_gates_fail_closed():
    assert pc.build_card(_row(line_source="synthetic_trailing_mean", price=None), NOW, "v")["status"] == "research"
    syn = pc.build_card(_row(line_source="synthetic_trailing_mean"), NOW, "v")
    assert syn["status"] == "research" and syn["book"] is None and syn["price_decimal"] is None
    assert pc.build_card(_row(quote_ts=None), NOW, "v")["status"] == "pass"
    assert pc.build_card(_row(quote_ts="2026-09-24T08:00:00Z"), NOW, "v")["status"] == "pass"
    assert pc.build_card(_row(status="voided", void_reason="OUT"), NOW, "v")["status"] == "pass"
    assert pc.build_card(_row(mean=None), NOW, "v")["status"] == "pass"
    assert pc.build_card(_row(roll_games=1), NOW, "v")["status"] == "research"


def test_card_states_evidence_and_invalidation_and_html_escapes():
    c = pc.build_card(_row(name="<b>x</b>"), NOW, "ff-football-only-v1")
    assert c["quote_clock"] == "2026-09-24T18:30:00Z" and c["forecast_version"] == "ff-football-only-v1"
    assert any("NOT verified health" in i for i in c["invalidation"])
    html = pc.render_html([c], "t", "g")
    assert "<b>x</b>" not in html and "&lt;b&gt;" in html and "No card is a recommended wager" in html

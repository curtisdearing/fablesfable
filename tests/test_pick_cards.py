"""Evidence cards: one exact quote identity, recorded provenance, fail-closed gates."""

import datetime as dt
import sqlite3

import pytest

from nflvalue import pick_cards as pc

NOW = dt.datetime(2026, 9, 24, 20, 0, tzinfo=dt.timezone.utc)


def _row(**kw):
    r = {"name": "K.Pitts", "player_id": "p1", "game_id": "2026_03_ATL_GB", "market": "receiving_yards",
         "side": "over", "line": 45.5, "line_source": "odds_api", "price": 1.87, "book": "draftkings/fanduel",
         "quote_book": "draftkings", "quote_ts": "2026-09-24T18:30:00Z", "as_of": "2026-09-24T18:35:00Z",
         "mean": 52.0, "sd": 26.0, "p_side": 0.58, "composite": 70.0, "status": "active",
         "run_id": "gha:1-1", "code_sha": "abc", "forecast_version": "ff-football-only-v1",
         "selection_source": "ml_gbdt", "_quote_verified": True}
    r.update(kw)
    return r


def test_verified_fresh_quote_is_watch_never_actionable():
    assert pc.VALIDATED_MARKETS == frozenset()
    c = pc.build_card(_row(), NOW)
    assert c["status"] == "watch"
    assert c["quote"] == {"book": "draftkings", "price_decimal": 1.87, "price_american": "-115",
                          "captured_at": "2026-09-24T18:30:00Z"}
    assert c["provenance"]["run_id"] == "gha:1-1" and c["rationale"].startswith("Football-only projection")
    assert c["breakeven"] == round(1 / 1.87, 4) and "never selects" in c["value_composite_note"]


@pytest.mark.parametrize("kw,reason", [
    ({"quote_book": None}, "no single executable quote"),
    ({"quote_book": "draftkings/fanduel"}, "no single executable quote"),
    ({"quote_ts": None}, "quote clock not recorded"),
    ({"quote_ts": "2026-09-25T18:30:00Z"}, "in the future"),
    ({"quote_ts": "2026-09-24T08:00:00Z"}, "h old"),
    ({"_quote_verified": False}, "not found in captured lines"),
    ({"price": 1.0}, "not a valid decimal price"),
    ({"price": None}, "not a valid decimal price"),
    ({"p_side": 1.3}, "outside [0, 1]"),
    ({"p_side": float("nan")}, "outside [0, 1]"),
    ({"sd": 0.0}, "non-positive"),
    ({"sd": -3}, "non-positive"),
    ({"side": "yes"}, "invalid side"),
    ({"line": float("inf")}, "line missing"),
    ({"mean": None}, "mean missing"),
    ({"status": "voided", "void_reason": "OUT"}, "voided"),
])
def test_every_invalid_input_fails_safe(kw, reason):
    c = pc.build_card(_row(**kw), NOW)
    assert c["status"] == "pass" and c["quote"] is None and c["ev_per_unit_unvalidated"] is None
    assert any(reason in r for r in c["status_reasons"]), c["status_reasons"]


def test_synthetic_line_is_research_without_a_quote():
    c = pc.build_card(_row(line_source="synthetic_trailing_mean"), NOW)
    assert c["status"] == "research" and c["quote"] is None


def test_legacy_row_provenance_stays_unknown_and_is_not_called_football_only():
    c = pc.build_card(_row(run_id=None, code_sha=None, forecast_version=None, selection_source=None), NOW)
    assert c["provenance"]["forecast_version"].startswith("unknown")
    assert c["rationale"].startswith("Projection (version not recorded)")


def test_no_role_claim_from_trailing_sample():
    assert not hasattr(pc, "MIN_ROLE_GAMES")
    c = pc.build_card(_row(roll_games=1), NOW)
    assert c["status"] == "watch" and not any("role" in r for r in c["status_reasons"])


def test_quote_verification_needs_the_exact_row_and_rejects_multibook_labels():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE lines (ts, game_id, book, market, player_id, player_name, side, point, price)")
    conn.execute("INSERT INTO lines VALUES ('2026-09-24T18:30:00Z','2026_03_ATL_GB','draftkings',"
                 "'receiving_yards','','Kyle Pitts','Over',45.5,1.87)")
    rows = [_row(), _row(price=1.95), _row(quote_ts="2026-09-24T18:31:00Z"), _row(line=46.5),
            _row(quote_book="draftkings/fanduel"), _row(side="under")]
    pc.verify_quotes(conn, rows)
    assert [r["_quote_verified"] for r in rows] == [True, False, False, False, False, False]


def test_html_escapes_and_disclaims():
    html = pc.render_cards_html([pc.build_card(_row(name="<b>x</b>"), NOW)])
    assert "<b>x</b>" not in html and "&lt;b&gt;" in html

"""Regression tests for the full-setup evaluation catches: the live-odds
ordering bug, t90 writeups, TD one-sided edge, ranking-score display, the
weekly document drop, and mismatch highlights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import config as cfgmod, db as dbmod  # noqa: E402
from nflvalue.composite import score_candidate  # noqa: E402
from nflvalue.document import render_drop  # noqa: E402
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402

GAME_ID = f"{SEASON}_09_AAA_BBB"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    real_connect = dbmod.connect
    db_path = str(tmp_path / "pipe.db")
    monkeypatch.setattr(dbmod, "connect", lambda p=None: real_connect(db_path))
    from nflvalue import report as rptmod
    monkeypatch.setattr(rptmod, "REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(rptmod, "WEEKLY_PROPS_JSON", str(tmp_path / "weekly_props.json"))
    monkeypatch.setattr(cfgmod, "LATEST_PATH", str(tmp_path / "latest.json"))
    monkeypatch.setattr(cfgmod, "DASHBOARD_PATH", str(tmp_path / "dashboard.html"))
    from nflvalue import document as docmod
    monkeypatch.setattr(docmod, "DROPS_DIR", str(tmp_path / "drops"))
    return tmp_path


def _feeds(now):
    from tests.test_pipeline_weekly import _roster
    return {"injury_rows": [], "injuries_fetched_at": now,
            "sleeper_df": None, "sleeper_fetched_at": now,
            "news_items": [], "news_fetched_at": now,
            "active_roster": _roster(now)}


def _fake_odds(payload):
    def fetch(url, params=None):
        return json.loads(json.dumps(payload))
    return fetch


def test_live_odds_path_keeps_context_and_notes(env, monkeypatch):
    """THE ordering-bug regression: with a real odds pull, leans must still
    carry the context features and game notes (previously the re-enumeration
    happened after stamping and silently dropped everything)."""
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    # synthetic odds payload quoting Alpha Wideout both sides
    payload = {"bookmakers": [{"key": "bookx", "markets": [{
        "key": "player_reception_yds",
        "outcomes": [
            {"name": "Over", "description": "Alpha Wideout", "price": 1.87, "point": 61.5},
            {"name": "Under", "description": "Alpha Wideout", "price": 1.95, "point": 61.5}]}]}]}
    monkeypatch.setitem(cfgmod.DEFAULT_CONFIG, "odds_api_key", "test")
    monkeypatch.setenv("ODDS_API_KEY", "test")
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_feeds(now), live_odds=True,
                      odds_fetch=_fake_odds(payload),
                      list_events_fn=lambda cfg: [
                          {"id": "e1", "home_team": "Bravo Bees", "away_team": "Alpha Ants"}])
    g = res["games"][0]
    # game notes exist (records line at minimum)
    assert g.get("notes"), "game notes missing on the live-odds path"
    # leans still carry context feature fields
    lean = g["leans"][0]
    assert "is_birthday_week" in lean and "def_out_total" in lean
    # the priced lean exists with a real edge if it made the top-5
    priced = [l for g_ in res["games"] for l in g_["leans"]
              if l.get("line_source") == "odds_api"]
    for l in priced:
        assert l["edge"] is not None


def test_t90_leans_carry_notes_and_features(env):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                inject_feeds=_feeds(now))
    feeds = _feeds(now)
    feeds["inactive_rows"] = []
    feeds["inactives_fetched_at"] = now
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=feeds)
    g = res["games"][0]
    assert g.get("notes")
    assert "is_birthday_week" in g["leans"][0]
    assert res.get("drop_path") and Path(res["drop_path"]).exists()


def test_anytime_td_one_sided_edge():
    from nflvalue.projection import p_over as _po
    p = round(_po(0.9, 0.95, 0.5, "poisson"), 4)        # P(>=1 TD) ~.593
    c = {"player_id": "P", "name": "N", "pos": "RB", "team": "T", "market": "anytime_td",
         "mean": 0.9, "sd": 0.95, "line": 0.5, "dist": "poisson",
         "p_over": p, "p_under": round(1 - p, 4),
         "components": {"opp_factor": 1.0, "game_script": 1.0},
         # yes-only quote from two books; no consensus carried -> raw implied
         "prices": {"over": 2.30, "under": None, "book": "bookx", "n_books": 2},
         "low_confidence": True}
    s = score_candidate(c)
    assert s["no_market"] is False
    # raw implied of 2.30 = .4348; model ~.593 -> edge ~ .158 (conservative, vig included)
    assert s["edge"] == pytest.approx(p - 1 / 2.30, abs=1e-3)


def test_drop_document_renders():
    payload = {"season": 2026, "week": 3, "clock": "wed", "as_of": "t", "publish": True,
               "games": [{"game_id": "G", "matchup": "A @ B", "screened_n": 40,
                          "notes": ["Records: A 2-0 @ B 1-1", "Birthday week: X (A)"],
                          "leans": [{"name": "X", "pos": "WR", "team": "A",
                                     "market": "receiving_yards", "side": "over",
                                     "line": 61.5, "line_source": "odds_api", "mean": 70.0,
                                     "edge": 0.06, "composite": 60.0, "ml_score": 71.2,
                                     "reason": "why"}]}],
               "contexts": {"G": {"entries": [{"name": "X", "items": ["contract year"]}]}}}
    html_doc = render_drop(payload)
    assert "NFL Prop Leans — 2026 Week 3" in html_doc
    assert "Birthday week" in html_doc
    assert "1-800-GAMBLER" in html_doc
    assert "71.2" in html_doc                        # ranking score shown, not composite
    assert "no_market" not in html_doc               # priced lean shows real edge


def test_mismatch_highlight_in_notes():
    from nflvalue.game_notes import build_game_notes
    cands = [{"name": "X", "team": "AAA", "defteam": "BBB", "pos": "WR",
              "market": "receiving_yards", "proj_components": {"opp_factor": 1.30},
              "is_birthday_week": 0, "revenge_game": 0, "is_contract_year": 0,
              "def_out_total": 0, "def_out_db": 0, "oline_outs": 0,
              "qb_continuity": 0.9, "wind": 0.0, "temp": 70.0}]
    notes = build_game_notes("G", cands, {}, 2026, 3, "BBB", "AAA")
    assert any("Mismatch" in n and "+30%" in n and "soft" in n for n in notes)

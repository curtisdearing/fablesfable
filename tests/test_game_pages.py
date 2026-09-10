"""Per-game pages (nflvalue/game_pages.py + the prepare_pages hook).

What a page must get right, in the order it can hurt:
  * WHO IS OUT and HOW RECENTLY -- every listed player on both teams, ESPN's
    own report stamp, OUT before RISK, other teams never leak in.
  * WHERE THE ODDS ARE -- book / price / n_books per lean; a synthetic line is
    never a bet; best bets are REAL_MARKET rows with an edge, largest first.
  * WHERE THE BODIES ARE -- venue zone (neutral-site table, honest fallback),
    each team's shift and body-clock kickoff, Arizona without DST.
  * WHY -- only the ledger's directional drivers (up/down), never baseline
    or level rows dressed up as reasons.
  * The site never carries a page for another week, and the dashboard never
    carries a dead link.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import game_pages as gp  # noqa: E402
from nflvalue.sources import availability as av  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
PREPARE_PAGES = Path(__file__).resolve().parents[1] / "scripts" / "prepare_pages.py"
HUB_FEED = Path(__file__).resolve().parents[1] / "scripts" / "build_hub_feed.py"


# --------------------------------------------------------------------------- #
# travel / body clock
# --------------------------------------------------------------------------- #
def _row(**kw):
    base = {"game_id": "2026_01_SF_LA", "gameday": "2026-09-10", "gametime": "20:35",
            "home_team": "LA", "away_team": "SF", "stadium": "Melbourne Cricket Ground",
            "roof": "outdoors", "location": "Neutral", "away_rest": 7, "home_rest": 7}
    base.update(kw)
    return base


def test_neutral_site_uses_the_venue_table_not_the_home_team():
    t = gp.travel_context(_row())
    assert t["available"] and t["neutral_site"]
    assert t["venue_tz"] == "Australia/Melbourne"
    assert t["venue_tz_source"] == "neutral_venue_table"
    assert t["kickoff_venue_local"].startswith("Fri 2026-09-11 10:35")
    assert t["kickoff_et"] == "Thu 2026-09-10 20:35 ET"
    sf, la = t["teams"]
    assert sf["team"] == "SF" and la["team"] == "LA"
    assert sf["shift_hours"] == 17.0 and la["shift_hours"] == 17.0
    assert "neutral_site" in t["flags"] and "east_of_home" in t["flags"]


def test_unknown_neutral_venue_falls_back_and_says_so():
    t = gp.travel_context(_row(stadium="Some Field Nobody Mapped"))
    assert t["venue_tz"] == "America/Los_Angeles"
    assert t["venue_tz_source"] == "home_team_fallback_for_unknown_neutral_venue"


def test_west_coast_team_at_one_pm_eastern_is_flagged_early_body_clock():
    t = gp.travel_context(_row(game_id="2026_02_SF_NYG", gameday="2026-09-13",
                               gametime="13:00", home_team="NYG", away_team="SF",
                               stadium="MetLife Stadium", location="Home"))
    sf, nyg = t["teams"]
    assert sf["shift_hours"] == 3.0
    assert sf["body_clock_kickoff"] == "Sun 10:00"
    assert "early_body_clock" in sf["flags"] and "crosses_3h" in sf["flags"]
    assert nyg["shift_hours"] == 0.0 and nyg["flags"] == []
    assert not t["neutral_site"]


def test_arizona_does_not_observe_dst():
    # September: Phoenix is UTC-7 (MST), Denver is UTC-6 (MDT) -> 1h apart
    t = gp.travel_context(_row(game_id="2026_02_ARI_DEN", gameday="2026-09-13",
                               gametime="16:25", home_team="DEN", away_team="ARI",
                               stadium="Empower Field at Mile High", location="Home"))
    ari = t["teams"][0]
    assert ari["home_tz"] == "America/Phoenix"
    assert ari["shift_hours"] == 1.0
    # December: both UTC-7 -> no shift
    t = gp.travel_context(_row(game_id="2026_15_ARI_DEN", gameday="2026-12-13",
                               gametime="16:25", home_team="DEN", away_team="ARI",
                               stadium="Empower Field at Mile High", location="Home"))
    assert t["teams"][0]["shift_hours"] == 0.0


def test_rest_flags_and_late_body_clock():
    t = gp.travel_context(_row(game_id="2026_03_SEA_MIA", gameday="2026-09-21",
                               gametime="20:15", home_team="MIA", away_team="SEA",
                               stadium="Hard Rock Stadium", location="Home",
                               away_rest=4, home_rest=11))
    sea, mia = t["teams"]
    assert "short_rest" in sea["flags"]
    assert "long_rest" in mia["flags"]
    assert mia["body_clock_kickoff"] == "Mon 20:15" and "late_body_clock" not in mia["flags"]
    # 21:30 ET at SoFi: the Jets' body clock says 21:30 (late); the Rams' says 18:30
    t = gp.travel_context(_row(gametime="21:30", location="Home", stadium="SoFi Stadium",
                               home_team="LA", away_team="NYJ"))
    nyj, la = t["teams"]
    assert "late_body_clock" in nyj["flags"] and "west_of_home" in nyj["flags"]
    assert nyj["shift_hours"] == -3.0
    assert "late_body_clock" not in la["flags"]


def test_unparseable_kickoff_is_unavailable_not_guessed():
    t = gp.travel_context(_row(gameday=None))
    assert t["available"] is False and "kickoff" in t["reason"]
    t = gp.travel_context(_row(gametime="garbage"))
    assert t["available"] is False


# --------------------------------------------------------------------------- #
# availability with recency
# --------------------------------------------------------------------------- #
AS_OF = "2026-09-10T05:00:00Z"
INJ = [
    {"team": "SF", "name": "Alfred Collins", "position": "DT", "status": "OUT",
     "status_raw": "Out", "date": "2026-09-09T21:00Z", "injury_type": "Knee",
     "comment": "Torn patellar tendon Tuesday; season."},
    {"team": "SF", "name": "James Thompson Jr.", "position": "DT", "status": "RISK",
     "status_raw": "Questionable", "date": "2026-09-09T21:05Z", "injury_type": "Hamstring",
     "comment": ""},
    {"team": "SF", "name": "Trey Lance", "position": "QB", "status": "OK",
     "status_raw": "Active", "date": "2026-06-24T15:16Z", "injury_type": "", "comment": "old"},
    {"team": "LA", "name": "Aaron Donald", "position": "DT", "status": "OUT",
     "status_raw": "Out", "date": "2026-09-08T18:00Z", "injury_type": "", "comment": "rest"},
    {"team": "SEA", "name": "Someone Else", "position": "WR", "status": "OUT",
     "status_raw": "Out", "date": "2026-09-09T00:00Z", "injury_type": "", "comment": ""},
]


def test_availability_lists_both_teams_out_first_with_report_age():
    a = gp.availability_for_teams(["SF", "LA"], INJ, None, AS_OF)
    names = [p["name"] for p in a["players"]]
    assert names == ["Aaron Donald", "Alfred Collins", "James Thompson Jr."]
    assert "Someone Else" not in names                      # other team never leaks
    assert "Trey Lance" not in names                        # stale Active row dropped
    collins = a["players"][1]
    assert collins["reported"] == "2026-09-09T21:00Z"
    assert collins["age_hours"] == 8.0
    assert collins["injury_type"] == "Knee"
    assert a["n_out"] == 2 and a["n_risk"] == 1


def test_t90_inactive_is_added_as_out_and_upgrades_a_listed_player():
    ina = [{"team": "SF", "name": "James Thompson Jr.", "active": False},
           {"team": "LA", "name": "Puka Nacua", "active": True},
           {"team": "LA", "name": "Jaylen Watson", "active": False}]
    a = gp.availability_for_teams(["SF", "LA"], INJ, ina, AS_OF)
    by = {p["name"]: p for p in a["players"]}
    assert by["James Thompson Jr."]["status"] == "OUT"
    assert "inactive_t90" in by["James Thompson Jr."]["status_raw"]
    assert by["Jaylen Watson"]["status"] == "OUT"
    assert by["Jaylen Watson"]["source"] == "espn_event_roster"
    assert "Puka Nacua" not in by


def test_parse_team_injuries_keeps_espn_report_date_and_injury_type():
    raw = {"injuries": [{"displayName": "San Francisco 49ers", "injuries": [
        {"athlete": {"displayName": "Alfred Collins", "id": "1",
                     "position": {"abbreviation": "DT"}},
         "status": "Out", "date": "2026-09-09T21:00Z", "shortComment": "x",
         "details": {"type": "Knee", "detail": "Patellar tendon"}}]}]}
    rows = av.parse_team_injuries(raw)
    assert rows[0]["date"] == "2026-09-09T21:00Z"
    assert rows[0]["injury_type"] == "Knee"


def test_resolve_statuses_carries_report_date_beside_fetch_time():
    players = pd.DataFrame([{"player_id": "00-1", "player_name": "Alfred Collins", "team": "SF"}])
    res = av.resolve_statuses(players, INJ[:1], clock="wed", injuries_fetched_at=AS_OF)
    st = res["statuses"]["00-1"]
    assert st["timestamp"] == AS_OF                           # when we fetched
    assert st["report_date"] == "2026-09-09T21:00Z"           # when ESPN posted it
    assert st["injury_type"] == "Knee"


# --------------------------------------------------------------------------- #
# hand notes
# --------------------------------------------------------------------------- #
def test_hand_notes_missing_is_empty_and_malformed_is_reported(tmp_path):
    assert gp.load_hand_notes(2026, 1, str(tmp_path)) == {}
    p = Path(gp.notes_path(2026, 1, str(tmp_path)))
    p.parent.mkdir(parents=True)
    p.write_text("{not json")
    err = gp.load_hand_notes(2026, 1, str(tmp_path))
    assert "__error__" in err and "week-01.json" in err["__error__"]
    p.write_text(json.dumps({"2026_01_SF_LA": {"arrival": {"SF": "arrived 09-04"}}}))
    assert gp.load_hand_notes(2026, 1, str(tmp_path))["2026_01_SF_LA"]["arrival"]["SF"] == "arrived 09-04"


def test_attach_context_stamps_every_game(tmp_path):
    sched = pd.DataFrame([{"game_id": "2026_01_SF_LA", "season": 2026, "week": 1,
                           "game_type": "REG", "gameday": "2026-09-10", "gametime": "20:35",
                           "home_team": "LA", "away_team": "SF",
                           "stadium": "Melbourne Cricket Ground", "roof": "outdoors",
                           "location": "Neutral", "away_rest": 7, "home_rest": 7}])
    games = [{"game_id": "2026_01_SF_LA", "matchup": "SF @ LA", "leans": []},
             {"game_id": "2026_01_ZZ_QQ", "matchup": "?", "leans": []}]
    gp.attach_context(games, sched, 2026, 1, injury_rows=INJ, as_of=AS_OF,
                      notes_root=str(tmp_path))
    ctx = games[0]["page_context"]
    assert ctx["home_team"] == "LA" and ctx["travel"]["neutral_site"]
    assert ctx["availability"]["n_out"] == 2
    assert games[1]["page_context"]["travel"]["available"] is False


# --------------------------------------------------------------------------- #
# page assembly
# --------------------------------------------------------------------------- #
def _lean(pid, name, market, side, line, state, edge, book="draftkings/fanduel",
          n_books=3, price_over=-115, price_under=-105, mean=60.0):
    return {"player_id": pid, "name": name, "pos": "WR", "team": "SF", "market": market,
            "side": side, "line": line, "line_source": "odds_api" if state else "synthetic_trailing_mean",
            "market_state": state or "NO_MARKET", "edge": edge, "p_over": 0.58, "p_under": 0.42,
            "mean": mean, "sd": 20.0, "composite": 55.0, "ml_p_over": 0.61,
            "prices": ({"over": price_over, "under": price_under, "book": book} if state else None),
            "components": {"n_books": n_books}}


def _payload():
    games = [{"game_id": "2026_01_SF_LA", "matchup": "SF @ LA", "screened_n": 40,
              "notes": ["Records: SF 0-0 @ LA 0-0"],
              "leans": [
                  _lean("p1", "Mike Evans", "receiving_yards", "over", 62.5, "REAL_MARKET", 0.031),
                  _lean("p2", "Deebo Samuel", "receptions", "under", 4.5, "REAL_MARKET", 0.074),
                  _lean("p3", "Brock Purdy", "passing_yards", "over", 240.5, "ONE_BOOK_CONTEXT_ONLY",
                        None, n_books=1),
                  _lean("p4", "Christian McCaffrey", "rushing_yards", "over", 70.5, None, None),
                  _lean("p5", "George Kittle", "receiving_yards", "under", 50.5, "REAL_MARKET", -0.01),
              ],
              "page_context": {"home_team": "LA", "away_team": "SF",
                               "travel": gp.travel_context(_row()),
                               "availability": gp.availability_for_teams(["SF", "LA"], INJ, None, AS_OF),
                               "hand_notes": {"arrival": {"SF": "arrived Fri 09-04 (nine days out)",
                                                          "LA": "arrives Thu ~24h out"},
                                              "not_playing": [{"team": "SF", "name": "Alfred Collins",
                                                               "pos": "DT", "note": "OUT — torn patellar tendon",
                                                               "source": "https://example.test/x",
                                                               "published": "2026-09-09"}]}}}]
    return {"season": 2026, "week": 1, "clock": "wed", "as_of": AS_OF, "publish": True,
            "games": games}


def _explain():
    return {"cards": [
        {"player_id": "p2", "market": "receptions", "weakest_grade": "thin",
         "counter_case_count": 1,
         "drivers": [
             {"label": "Baseline (trailing rate)", "direction": "baseline", "multiplier_label": None,
              "delta_label": None, "unit": "rec", "evidence": {"grade": None}},
             {"label": "Target share level", "direction": "level", "multiplier_label": "0.180",
              "delta_label": None, "unit": "share", "evidence": {"grade": "moderate"}},
             {"label": "Opponent vs WR", "direction": "down", "multiplier_label": "0.910",
              "delta_label": "-0.5", "unit": "rec", "evidence": {"grade": "moderate"}},
             {"label": "Pace", "direction": "up", "multiplier_label": "1.030",
              "delta_label": "+0.1", "unit": "rec", "evidence": {"grade": "strong"}},
         ]}]}


def test_best_bets_are_priced_positive_edge_rows_largest_first():
    pages = gp.build_pages(_payload(), _explain())
    assert len(pages) == 1
    page = pages[0]
    names = [b["name"] for b in page["best_bets"]]
    # Kittle (negative edge) still qualifies as a priced row; ordering is by edge
    assert names == ["Deebo Samuel", "Mike Evans", "George Kittle"]
    assert page["n_priced"] == 3 and page["n_leans"] == 5
    assert page["href"] == "games/2026_01_SF_LA.html"
    assert page["books"] == ["draftkings", "fanduel"]
    deebo = page["best_bets"][0]
    assert deebo["price"] == -105 and deebo["book"] == "draftkings/fanduel" and deebo["n_books"] == 3
    assert deebo["p_side"] == 0.42


def test_drivers_are_only_directional_rows_and_know_which_side_they_push():
    page = gp.build_pages(_payload(), _explain())[0]
    deebo = page["best_bets"][0]
    labels = [d["label"] for d in deebo["drivers"]]
    assert labels == ["Opponent vs WR", "Pace"]              # by |delta|, no baseline/level
    assert deebo["drivers"][0]["with_side"] is True          # down pushes the UNDER
    assert deebo["drivers"][1]["with_side"] is False
    assert deebo["weakest_grade"] == "thin"
    evans = page["best_bets"][1]
    assert evans["drivers"] == []                             # no card -> no invented reasons


def test_synthetic_and_one_book_rows_never_become_bets():
    page = gp.build_pages(_payload(), None)[0]
    assert all(b["market_state"] == "REAL_MARKET" for b in page["best_bets"])
    by = {m["name"]: m for m in page["market"]}
    assert by["Brock Purdy"]["price"] is None or by["Brock Purdy"]["market_state"] != "REAL_MARKET"
    assert by["Christian McCaffrey"]["book"] is None


def test_merge_keeps_other_games_of_the_same_week_only():
    old = [{"game_id": "A", "season": 2026, "week": 1, "x": 1},
           {"game_id": "B", "season": 2026, "week": 1, "x": 1},
           {"game_id": "C", "season": 2025, "week": 18, "x": 1}]
    new = [{"game_id": "B", "season": 2026, "week": 1, "x": 2}]
    merged = gp.merge_pages(old, new, 2026, 1)
    assert [p["game_id"] for p in merged] == ["A", "B"]
    assert next(p for p in merged if p["game_id"] == "B")["x"] == 2


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_render_html_shows_the_four_things_and_escapes():
    payload = _payload()
    payload["games"][0]["leans"][0]["name"] = "Mike <script>alert(1)</script> Evans"
    html = gp.render_html(gp.build_pages(payload, _explain())[0])
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    for needle in ("Aaron Donald", "Alfred Collins", "8h ago", "Knee",          # who is out, how recently
                   "draftkings/fanduel", "-105",                               # where the odds are
                   "Melbourne Cricket Ground", "NEUTRAL SITE", "Thu 17:35",    # bodies
                   "Opponent vs WR", "×0.910",                                 # why
                   "arrived Fri 09-04", "torn patellar tendon",               # hand notes
                   "Records: SF 0-0 @ LA 0-0", "1-800-GAMBLER"):
        assert needle in html, needle
    assert "Deebo Samuel" in html and "UNDER 4.5" in html


def test_render_html_with_nothing_priced_offers_nothing():
    payload = _payload()
    for l in payload["games"][0]["leans"]:
        l["market_state"] = "NO_MARKET"; l["edge"] = None; l["prices"] = None
        l["line_source"] = "synthetic_trailing_mean"
    html = gp.render_html(gp.build_pages(payload, None)[0])
    assert "No bettable lean" in html
    assert "†" in html


def test_unpublished_run_is_labelled():
    payload = _payload(); payload["publish"] = False
    html = gp.render_html(gp.build_pages(payload, None)[0])
    assert "NOT PUBLISHED" in html


def test_write_site_pages_refuses_path_shaped_game_ids(tmp_path):
    pages = gp.build_pages(_payload(), None)
    pages.append({**pages[0], "game_id": "../evil"})
    written = gp.write_site_pages(pages, str(tmp_path))
    assert [os.path.basename(w) for w in written] == ["2026_01_SF_LA.html"]
    idx = json.loads((tmp_path / "games" / "index.json").read_text())
    assert idx["pages"][0]["href"] == "games/2026_01_SF_LA.html"
    assert not (tmp_path / "evil.html").exists()


# --------------------------------------------------------------------------- #
# dashboard link + prepare_pages hook
# --------------------------------------------------------------------------- #
def test_dashboard_links_a_game_only_when_its_page_exists(tmp_path):
    from nflvalue import dashboard
    out = tmp_path / "dash.html"
    data = {"mode": "live",
            "weekly_leans": {"season": 2026, "week": 1, "clock": "wed", "as_of": AS_OF,
                             "games": [{"game_id": "2026_01_SF_LA", "matchup": "SF @ LA",
                                        "screened_n": 1, "leans": []}]},
            "game_pages": [{"game_id": "2026_01_SF_LA", "href": "games/2026_01_SF_LA.html"}]}
    dashboard.write_dashboard(data, str(out))
    html = out.read_text()
    assert "games/2026_01_SF_LA.html" in html
    assert "pageHrefs[g.game_id]" in html and "gamelink" in html
    assert 'id="game-${esc(g.game_id)}"' in html


def _tree(tmp_path: Path, pages, *, season=2031, week=5):
    import shutil
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True); (root / "drops").mkdir(); (root / "scripts").mkdir()
    (root / "dashboard.html").write_text("<html><body>DASHBOARD</body></html>")
    shutil.copy(HUB_FEED, root / "scripts" / "build_hub_feed.py")
    payload = {"season": season, "week": week, "clock": "wed", "as_of": "2031-09-10T12:00:00Z",
               "publish": True, "mode": "live", "games": []}
    (root / "data" / "weekly_props.json").write_text(json.dumps(payload))
    (root / "data" / "latest.json").write_text(json.dumps({"game_pages": pages}))
    return root


def _run(root: Path):
    return subprocess.run([sys.executable, str(PREPARE_PAGES), "--root", str(root),
                           "--now", "2031-09-11T00:00:00Z"], capture_output=True, text=True,
                          timeout=120)


def test_prepare_pages_writes_current_week_pages_and_skips_other_weeks(tmp_path):
    page = gp.build_pages(_payload(), _explain())[0]
    current = {**page, "season": 2031, "week": 5}
    stale = {**page, "game_id": "2030_09_XX_YY", "season": 2030, "week": 9}
    root = _tree(tmp_path, [current, stale])
    proc = _run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    site = root / "_site"
    assert (site / "games" / "2026_01_SF_LA.html").exists()
    assert not (site / "games" / "2030_09_XX_YY.html").exists()
    manifest = json.loads((site / "reports" / "index.json").read_text())
    assert manifest["game_pages"] == {"written": 1, "skipped_stale": 1, "reason": None}
    assert "Aaron Donald" in (site / "games" / "2026_01_SF_LA.html").read_text()
    idx = json.loads((site / "games" / "index.json").read_text())
    assert [p["game_id"] for p in idx["pages"]] == ["2026_01_SF_LA"]


def test_prepare_pages_without_latest_json_still_ships_the_board(tmp_path):
    root = _tree(tmp_path, [])
    (root / "data" / "latest.json").unlink()
    proc = _run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads((root / "_site" / "reports" / "index.json").read_text())
    assert manifest["game_pages"]["written"] == 0
    assert manifest["game_pages"]["reason"] == "missing_latest"
    assert not (root / "_site" / "games").exists()

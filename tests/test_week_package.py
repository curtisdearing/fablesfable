"""The canonical weekly product contract (player-prop research leans).

Wednesday initializes a FULL-WEEK package. A T-90 run is a PATCH for one
game: it replaces that game and must leave every other game -- Wednesday's
and any earlier T-90's -- exactly where it was. Two sequential patches for
different games must ACCUMULATE, never clobber each other.

These are contract tests on the shared payload, not on any one renderer:
Markdown, HTML, the dashboard input and the Discord input all read the same
object, so a game that survives here survives everywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import config as cfgmod  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue.freshness import stamp_now  # noqa: E402
from tests.test_report_phase2 import SEASON, WEEK, _pw_row  # noqa: E402
from nflvalue.candidates import WeekInputs  # noqa: E402

MATCHUPS = [("AAA", "BBB"), ("CCC", "DDD"), ("EEE", "FFF")]
GAME_IDS = [f"{SEASON}_09_{a}_{h}" for a, h in MATCHUPS]
GAME_1, GAME_2, GAME_3 = GAME_IDS


# --------------------------------------------------------------------------- #
# A three-game synthetic slate (same shape as tests/test_report_phase2.py,
# widened so "the rest of the slate" is a thing that can be erased).
# --------------------------------------------------------------------------- #
def synthetic_inputs_multi() -> WeekInputs:
    rows, opd_rows, tw_rows, sched_rows = [], [], [], []
    for away, home in MATCHUPS:
        for wk in range(1, 10):
            rows.append(_pw_row(wk, f"WR_{away}", f"{away} Wideout", away, home, "WR",
                                targets=8.0, receptions=5.0, rec_yards=68.0,
                                roll_targets=8.0, roll_target_share=0.25, roll_ypt=8.4,
                                roll_catch_rate=0.64))
            rows.append(_pw_row(wk, f"RB_{away}", f"{away} Back", away, home, "RB",
                                carries=16.0, rush_yards=72.0,
                                roll_carries=16.0, roll_carry_share=0.62, roll_ypc=4.4))
            rows.append(_pw_row(wk, f"QB_{home}", f"{home} Quarterback", home, away, "QB",
                                pass_attempts=34.0, completions=22.0, pass_yards=245.0,
                                roll_pass_attempts=34.0, roll_completions=22.0, roll_ypa=7.2))
            for defteam in (away, home):
                for role, col in (("QB", "roll_ypa_allowed_factor"),
                                  ("WR", "roll_ypt_allowed_factor"),
                                  ("TE", "roll_ypt_allowed_factor"),
                                  ("RB", "roll_ypc_allowed_factor")):
                    r = dict(season=SEASON, week=wk, defteam=defteam, role=role,
                             targets_allowed=0.0, rec_yards_allowed=0.0, carries_allowed=0.0,
                             rush_yards_allowed=0.0, pass_yards_allowed=0.0,
                             epa_allowed_sum=0.0, plays_faced=30.0, roll_games=wk - 1,
                             roll_ypt_allowed_factor=None, roll_ypc_allowed_factor=None,
                             roll_ypa_allowed_factor=None, roll_epa_allowed_factor=1.0)
                    r[col] = 1.10 if defteam == home else 0.95
                    opd_rows.append(r)
            for t in (away, home):
                tw_rows.append(dict(season=SEASON, week=wk, team=t,
                                    roll_team_pass_att=32.0, roll_team_rush_att=26.0))
        sched_rows.append(dict(
            game_id=f"{SEASON}_09_{away}_{home}", season=SEASON, week=WEEK, game_type="REG",
            gameday="2023-11-05", gametime="13:00", home_team=home, away_team=away,
            spread_line=3.0, total_line=44.5))
    return WeekInputs(pw=pd.DataFrame(rows), opd=pd.DataFrame(opd_rows),
                      tw=pd.DataFrame(tw_rows), schedules=pd.DataFrame(sched_rows))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolate every write target: DB, reports, drops, dashboard, latest.json."""
    real_connect = dbmod.connect
    db_path = str(tmp_path / "pkg.db")
    monkeypatch.setattr(dbmod, "connect", lambda p=None: real_connect(db_path))
    from nflvalue import report as rptmod
    from nflvalue import document as docmod
    monkeypatch.setattr(rptmod, "REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(rptmod, "WEEKLY_PROPS_JSON", str(tmp_path / "weekly_props.json"))
    monkeypatch.setattr(docmod, "DROPS_DIR", str(tmp_path / "drops"))
    monkeypatch.setattr(cfgmod, "LATEST_PATH", str(tmp_path / "latest.json"))
    monkeypatch.setattr(cfgmod, "DASHBOARD_PATH", str(tmp_path / "dashboard.html"))
    return {"tmp": tmp_path, "db_path": db_path}


def _wed_feeds(now: str):
    return {"injury_rows": [{"team": a, "name": f"{a} Wideout", "status_raw": "Active",
                             "status": "OK", "comment": ""} for a, _h in MATCHUPS],
            "injuries_fetched_at": now,
            "sleeper_df": None, "sleeper_fetched_at": now}


def _t90_feeds(now: str, out_team: str | None = None):
    """T-90 feeds for one game. ``out_team``'s wideout is declared inactive."""
    feeds = dict(_wed_feeds(now))
    rows = [{"espn_id": "9", "name": "Filler Body", "active": True,
             "did_not_play": False, "starter": False, "team": "ZZZ"}]
    if out_team:
        rows.append({"espn_id": "1", "name": f"{out_team} Wideout", "active": False,
                     "did_not_play": True, "starter": True, "team": out_team})
    feeds["inactive_rows"] = rows
    feeds["inactives_fetched_at"] = now
    return feeds


def _ids(payload) -> list:
    return [g["game_id"] for g in payload["games"]]


# =========================================================================== #
# A. Two sequential T-90 patches accumulate; nothing else is erased.
# =========================================================================== #
def test_second_t90_patch_preserves_the_first_and_every_untouched_game(env):
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    wed = pw.run_week(SEASON, WEEK, mode="live", inputs=inputs,
                      inject_feeds=_wed_feeds(now))
    assert _ids(wed) == sorted(GAME_IDS), "premise: Wednesday is a full slate"

    first = pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
                       inject_feeds=_t90_feeds(now, out_team="AAA"))
    assert _ids(first) == sorted(GAME_IDS), \
        "a T-90 patch dropped the rest of the Wednesday slate"

    second = pw.run_t90(SEASON, WEEK, GAME_2, mode="live", inputs=inputs,
                        inject_feeds=_t90_feeds(now, out_team="CCC"))
    assert _ids(second) == sorted(GAME_IDS), \
        "the second T-90 patch erased the first patch or an untouched game"

    by_id = {g["game_id"]: g for g in second["games"]}
    assert by_id[GAME_1]["clock"] == "t90"      # first patch retained
    assert by_id[GAME_2]["clock"] == "t90"      # second patch applied
    assert by_id[GAME_3]["clock"] == "wed"      # untouched game keeps Wednesday
    # each patched game kept its OWN latest data, not the other's
    assert not any(l["player_id"] == "WR_AAA" for l in by_id[GAME_1]["leans"])
    assert not any(l["player_id"] == "WR_CCC" for l in by_id[GAME_2]["leans"])
    assert any(l["player_id"] == "WR_EEE" for l in by_id[GAME_3]["leans"])


def test_canonical_payload_on_disk_is_the_full_slate_after_a_patch(env):
    """One canonical finalized payload feeds Markdown, HTML, dashboard and
    Discord -- so the file they all read must be the whole week."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=_wed_feeds(now))
    pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
               inject_feeds=_t90_feeds(now, out_team="AAA"))
    pw.run_t90(SEASON, WEEK, GAME_2, mode="live", inputs=inputs,
               inject_feeds=_t90_feeds(now, out_team="CCC"))

    canonical = json.loads((env["tmp"] / "weekly_props.json").read_text())
    assert [g["game_id"] for g in canonical["games"]] == sorted(GAME_IDS)
    assert set(canonical["contexts"]) == set(GAME_IDS)

    latest = json.loads((env["tmp"] / "latest.json").read_text())
    assert [g["game_id"] for g in latest["weekly_leans"]["games"]] == sorted(GAME_IDS), \
        "the dashboard input lost games to a one-game T-90 patch"


def test_t90_drop_is_one_complete_weekly_html_document(env):
    """A one-game file must never stand in for the weekly document."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=_wed_feeds(now))
    res = pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
                     inject_feeds=_t90_feeds(now, out_team="AAA"))
    html = Path(res["drop_path"]).read_text()
    for away, home in MATCHUPS:
        assert f"{away} @ {home}" in html, f"{away} @ {home} missing from the T-90 drop"
    # the honesty furniture survives the patch path
    assert "1-800-GAMBLER" in html
    assert "Leans, not locks" in html
    assert "screened" in html


# =========================================================================== #
# C. T-90 persistence is GAME-SCOPED.
# =========================================================================== #
def test_t90_persistence_does_not_delete_another_games_rows(env):
    """The forward log is what CLV and the kill-check read. Patching one game
    must not wipe the whole (season, week, 't90') slice -- that would silently
    delete an already-published game's leans from the record."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=_wed_feeds(now))

    pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
               inject_feeds=_t90_feeds(now, out_team="AAA"))
    conn = dbmod.connect()
    first = dbmod.query_df(conn, """
        SELECT player_id, market, line, composite FROM leans
        WHERE season=? AND week=? AND clock='t90' AND game_id=?
        ORDER BY player_id, market""", (SEASON, WEEK, GAME_1))
    conn.close()
    assert not first.empty, "premise: the first T-90 patch persisted leans"

    pw.run_t90(SEASON, WEEK, GAME_2, mode="live", inputs=inputs,
               inject_feeds=_t90_feeds(now, out_team="CCC"))
    conn = dbmod.connect()
    still = dbmod.query_df(conn, """
        SELECT player_id, market, line, composite FROM leans
        WHERE season=? AND week=? AND clock='t90' AND game_id=?
        ORDER BY player_id, market""", (SEASON, WEEK, GAME_1))
    games = set(dbmod.query_df(conn, """
        SELECT DISTINCT game_id FROM leans
        WHERE season=? AND week=? AND clock='t90'""", (SEASON, WEEK))["game_id"])
    conn.close()

    assert not still.empty, \
        "the second T-90 patch deleted the first game's T-90 rows"
    pd.testing.assert_frame_equal(first, still)
    assert games == {GAME_1, GAME_2}


def test_t90_rerun_of_the_same_game_still_replaces_its_own_rows(env):
    """Game-scoped must not become append-only: re-patching one game replaces
    that game's rows (no duplicates, no orphans from a prior ranking)."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=_wed_feeds(now))
    pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
               inject_feeds=_t90_feeds(now, out_team="AAA"))
    conn = dbmod.connect()
    dbmod.upsert(conn, "leans", [{
        "season": SEASON, "week": WEEK, "clock": "t90", "game_id": GAME_1,
        "player_id": "GHOST", "name": "Old Ranking Ghost", "market": "anytime_td",
        "side": "under", "line": 0.5, "line_source": "synthetic_trailing_mean",
        "price": None, "book": None, "mean": 0.03, "sd": 0.35, "p_side": 0.97,
        "composite": 59.0, "edge": None, "confidence_comp": 0.66, "matchup_comp": 0.5,
        "screened_n": 44, "reason": "stale", "status": "active", "void_reason": None,
        "as_of": "old", "created_at": "old",
    }], ["season", "week", "clock", "game_id", "player_id", "market"])
    conn.close()

    pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
               inject_feeds=_t90_feeds(now, out_team="AAA"))
    conn = dbmod.connect()
    left = dbmod.query_df(conn, """
        SELECT player_id FROM leans WHERE season=? AND week=? AND clock='t90'
        AND game_id=?""", (SEASON, WEEK, GAME_1))
    dupes = dbmod.query_df(conn, """
        SELECT COUNT(*) AS n FROM (SELECT season, week, clock, game_id, player_id,
        market, COUNT(*) c FROM leans GROUP BY 1,2,3,4,5,6 HAVING c > 1)""").iloc[0]["n"]
    conn.close()
    assert "GHOST" not in set(left["player_id"])
    assert dupes == 0


# =========================================================================== #
# E. T-90 consumes the latest REAL prop lines already resnapped into the DB.
# =========================================================================== #
REAL_POINT = 44.5


def _seed_resnapped_lines(db_path: str, game_id: str, player_name: str,
                          ts: str, point: float = REAL_POINT) -> None:
    """What ``oddsapi_props.resnap_lines`` leaves behind: a two-sided,
    two-book snapshot for one game in the ``lines`` table."""
    real_connect = dbmod.connect.__wrapped__ if hasattr(dbmod.connect, "__wrapped__") \
        else dbmod.connect
    conn = real_connect(db_path)
    rows = []
    for book, over, under in (("bookx", 1.87, 1.95), ("booky", 1.91, 1.91)):
        for side, price in (("over", over), ("under", under)):
            rows.append({"ts": ts, "game_id": game_id, "book": book,
                         "market": "receiving_yards", "player_id": None,
                         "player_name": player_name, "side": side,
                         "point": point, "price": price})
    dbmod.upsert(conn, "lines", rows,
                 ["ts", "game_id", "book", "market", "player_name", "side"])
    conn.close()


def test_t90_uses_the_resnapped_real_line_not_a_synthetic_one(env):
    """T-90 is the clock closest to kickoff, so it is exactly where a real
    price matters most. If a resnap already put one in the DB, the re-rank
    must price against it -- otherwise the final read silently regresses to a
    synthetic reference line."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=_wed_feeds(now))
    _seed_resnapped_lines(env["db_path"], GAME_1, "AAA Wideout", ts=now)

    res = pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
                     inject_feeds=_t90_feeds(now))
    by_id = {g["game_id"]: g for g in res["games"]}
    hit = [l for l in by_id[GAME_1]["leans"]
           if l["player_id"] == "WR_AAA" and l["market"] == "receiving_yards"]
    assert hit, "premise: the wideout's receiving-yards prop is a T-90 lean"
    lean = hit[0]
    assert lean["line_source"] == "odds_api", \
        "T-90 ignored a real sportsbook line already resnapped into the DB"
    assert lean["line"] == REAL_POINT
    assert lean.get("edge") is not None, "a real line must produce a real edge"


def test_t90_without_a_real_line_stays_explicitly_synthetic(env):
    """Graceful degradation, still visibly labelled: no resnap for this game
    means synthetic reference lines and no_market, never a quiet blend."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs, inject_feeds=_wed_feeds(now))
    _seed_resnapped_lines(env["db_path"], GAME_1, "AAA Wideout", ts=now)

    res = pw.run_t90(SEASON, WEEK, GAME_2, mode="live", inputs=inputs,
                     inject_feeds=_t90_feeds(now))
    by_id = {g["game_id"]: g for g in res["games"]}
    for lean in by_id[GAME_2]["leans"]:
        assert lean["line_source"] == "synthetic_trailing_mean"
        assert lean.get("edge") is None
    html = Path(res["drop_path"]).read_text()
    assert "†" in html and "synthetic reference line" in html


# =========================================================================== #
# 9. One complete weekly HTML drop -- no one-game file stands in for the week.
# =========================================================================== #
def test_a_patch_never_overwrites_the_wednesday_document(env):
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    wed = pw.run_week(SEASON, WEEK, mode="live", inputs=inputs,
                      inject_feeds=_wed_feeds(now))
    wed_html = Path(wed["drop_path"]).read_text()

    res = pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
                     inject_feeds=_t90_feeds(now, out_team="AAA"))
    assert Path(res["drop_path"]) != Path(wed["drop_path"])
    assert Path(wed["drop_path"]).read_text() == wed_html, \
        "the T-90 run overwrote the Wednesday document"
    # ... and the patch's own drop is still the WHOLE week
    patched = Path(res["drop_path"]).read_text()
    assert all(f"{a} @ {h}" in patched for a, h in MATCHUPS)


def test_merged_screen_denominator_is_the_whole_weeks(env):
    """"5 of N" is a published honesty number; a one-game patch must not
    shrink the week's total to that one game's count."""
    now = stamp_now()
    inputs = synthetic_inputs_multi()
    pw.run_week(SEASON, WEEK, mode="live", inputs=inputs,
                inject_feeds=_wed_feeds(now))
    res = pw.run_t90(SEASON, WEEK, GAME_1, mode="live", inputs=inputs,
                     inject_feeds=_t90_feeds(now, out_team="AAA"))
    assert res["n_candidates"] == sum(g["screened_n"] for g in res["games"])
    assert res["n_candidates"] > max(g["screened_n"] for g in res["games"])
    for g in res["games"]:
        assert g["screened"] == f"{len(g['leans'])} of {g['screened_n']}"

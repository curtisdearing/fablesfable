"""Block B: two-clock pipeline. The T-90 refresh must auto-void a lean whose
player went inactive; the freshness gate must halt publishing on stale feeds;
reruns must be idempotent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import config as cfgmod  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402

GAME_ID = f"{SEASON}_09_AAA_BBB"
FRESH_TS = None  # filled per-test with stamp_now


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolate every write target: DB, reports, dashboard, latest.json."""
    real_connect = dbmod.connect
    db_path = str(tmp_path / "pipe.db")
    monkeypatch.setattr(dbmod, "connect", lambda p=None: real_connect(db_path))
    from nflvalue import report as rptmod
    from nflvalue import document as docmod
    monkeypatch.setattr(rptmod, "REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(rptmod, "WEEKLY_PROPS_JSON", str(tmp_path / "weekly_props.json"))
    monkeypatch.setattr(docmod, "DROPS_DIR", str(tmp_path / "drops"))
    monkeypatch.setattr(cfgmod, "LATEST_PATH", str(tmp_path / "latest.json"))
    monkeypatch.setattr(cfgmod, "DASHBOARD_PATH", str(tmp_path / "dashboard.html"))
    return {"tmp": tmp_path, "db_path": db_path}


def _fresh_feeds(now: str, wr_a_status: str = "Active"):
    return {
        "injury_rows": [
            {"team": "AAA", "name": "Alpha Wideout", "status_raw": wr_a_status,
             "status": {"Active": "OK", "Questionable": "RISK", "Out": "OUT"}[wr_a_status],
             "comment": ""},
        ],
        "injuries_fetched_at": now,
        "sleeper_df": None,
        "sleeper_fetched_at": now,
    }


def test_wed_run_live_publishes_and_persists(env):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_fresh_feeds(now))
    assert res["publish"] is True
    assert res["games"] and res["games"][0]["game_id"] == GAME_ID
    conn = dbmod.connect()
    leans = dbmod.query_df(conn, "SELECT * FROM leans WHERE clock='wed'")
    assert len(leans) == len(res["games"][0]["leans"])
    assert (leans["status"] == "active").all()
    # dashboard written with the leans payload embedded
    html = (env["tmp"] / "dashboard.html").read_text()
    assert "Weekly Leans" in html and "weekly_leans" in html
    conn.close()


def test_wed_rerun_is_idempotent(env):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                inject_feeds=_fresh_feeds(now))
    pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                inject_feeds=_fresh_feeds(now))
    conn = dbmod.connect()
    n = dbmod.query_df(conn, """
        SELECT COUNT(*) AS n FROM (SELECT season, week, clock, game_id, player_id, market,
        COUNT(*) c FROM leans GROUP BY 1,2,3,4,5,6 HAVING c > 1)""").iloc[0]["n"]
    assert n == 0
    conn.close()


def test_freshness_halt_blocks_publish(env):
    """Stale injuries (7 days old) -> publish=false, NOT PUBLISHED banner."""
    stale = _fresh_feeds("2020-01-01T00:00:00Z")
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=stale)
    assert res["publish"] is False
    assert any("staleness" in r or "stale" in r for r in res["publish_reasons"])
    md = Path(res["md_path"]).read_text()
    assert "NOT PUBLISHED" in md


def test_out_player_never_reaches_the_ranker(env):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_fresh_feeds(now, wr_a_status="Out"))
    names = {l["player_id"] for g in res["games"] for l in g["leans"]}
    assert "WR_A" not in names


def test_t90_voids_inactive_and_reranks(env):
    """The core two-clock promise: Wednesday lean on a player who goes
    inactive is auto-voided at T-90 and the game re-ranks without him."""
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    wed = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_fresh_feeds(now))
    assert any(l["player_id"] == "WR_A" for g in wed["games"] for l in g["leans"]), \
        "test premise: WR_A must be a Wednesday lean"

    t90_feeds = dict(_fresh_feeds(now))
    t90_feeds["inactive_rows"] = [
        {"espn_id": "1", "name": "Alpha Wideout", "active": False, "did_not_play": True,
         "starter": True, "team": "AAA"}]
    t90_feeds["inactives_fetched_at"] = now
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=t90_feeds)

    assert any(v["player_id"] == "WR_A" for v in res["voided"])
    conn = dbmod.connect()
    voided = dbmod.query_df(conn, """
        SELECT status, void_reason FROM leans
        WHERE clock='wed' AND player_id='WR_A'""")
    assert (voided["status"] == "voided").all()
    assert voided["void_reason"].str.contains("t90").all()
    t90_leans = dbmod.query_df(conn, "SELECT player_id FROM leans WHERE clock='t90'")
    assert "WR_A" not in set(t90_leans["player_id"])
    conn.close()
    md = Path(res["md_path"]).read_text()
    assert "auto-voided" in md and "Alpha Wideout" in md


def test_t90_requires_inactives_feed(env):
    """No inactives injected and no live fetch -> the availability resolver's
    hard requirement surfaces (fail loud, not a silent Wednesday fallback)."""
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    feeds = _fresh_feeds(now)
    feeds["inactive_rows"] = None
    with pytest.raises(ValueError, match="t90"):
        pw.run_t90(SEASON, WEEK, GAME_ID, mode="live",
                   inputs=synthetic_inputs(), inject_feeds=feeds)

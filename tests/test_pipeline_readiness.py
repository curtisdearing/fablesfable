"""Readiness through the REAL consumers (run_week / run_t90 -> DB -> cards).

Availability evidence per player, starting-QB context with issuing-run clocks, observed
snaps as context, and later context never rewriting what an earlier pick knew.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import factor_integration as fimod  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from nflvalue import qb_readiness as qr  # noqa: E402
from nflvalue.freshness import stamp_now  # noqa: E402
from tests.test_pipeline_weekly import GAME_ID, _fresh_feeds, _roster, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402
from tests.test_t90_factor_receipts import _t90_feeds  # noqa: E402

NAMED_ROSTER = [{"player_id": "QB_B", "name": "Bravo Quarterback", "team": "BBB",
                 "position": "QB", "status": "ACT", "week": WEEK}]


def _feeds(now, **kw):
    f = dict(_fresh_feeds(now))
    roster = _roster(now)
    roster["rows"] = [r for r in roster["rows"] if r["player_id"] != "QB_B"] + NAMED_ROSTER
    f["active_roster"] = roster
    f.update(kw)
    return f


def _doc(claims=()):
    return {"schema": "factor_context/1", "season": SEASON, "week": WEEK, "news": list(claims),
            "records": []}


def _claim(fetched_at, published_at="2023-11-03T15:00:00Z", kind="confirmed"):
    return {"story_id": f"bbb_starter_{fetched_at}", "entity_id": "BBB", "entity_type": "team",
            "team": "BBB", "game_id": GAME_ID, "category": "qb_news", "claim_key": "starter",
            "claim_value": "Bravo Quarterback", "claim": "Bravo Quarterback will start.",
            "attribution": "BBB (team site)", "source_tier": "team_official", "claim_kind": kind,
            "source_url": "https://www.bbb.example/news/starter", "published_at": published_at,
            "fetched_at": fetched_at}


def _stamps(conn, clock):
    return {(r[0], r[1]): json.loads(r[2]) for r in conn.execute(
        "SELECT player_id, market, stage_json FROM leans WHERE clock=?", (clock,)).fetchall()}


def _labels(panel):
    return {it["factor_id"]: it for g in panel["groups"] for it in g["items"]}


def test_wed_run_persists_per_player_availability_and_holds_unknown_cards(env):
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_feeds(stamp_now(), factor_context_doc=_doc()))
    rc = res["factor_receipt"]
    assert rc["availability"]["report_state"] == "received"
    # BBB has no rows in the received feed: its players are unknown, not healthy
    assert rc["availability"]["summary"]["availability_state"].get("team_not_in_report", 0) >= 1
    conn = dbmod.connect()
    st = _stamps(conn, "wed")
    qb = [s for (p, _m), s in st.items() if p == "QB_B"]
    wr = [s for (p, _m), s in st.items() if p == "WR_A"]
    assert qb and all(s["availability"]["eligibility"] == "degraded" for s in qb)
    assert wr and all(s["availability"]["eligibility"] == "eligible" for s in wr)
    cards = pc.week_cards(conn, SEASON, WEEK)["cards"]
    for c in cards:
        lab = _labels(c["factor_panel"])
        a = lab[f"availability:{c['player_id']}"]
        if c["player_id"] == "QB_B":
            assert a["status"] == "unavailable_unverified"
            assert c["status"] in ("research", "pass")
        if c["player_id"] == "WR_A":
            assert a["status"] == "considered_no_change"


def test_missing_injury_report_is_unknown_everywhere_not_neutral(env):
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_feeds(stamp_now(), injury_rows=[], injuries_fetched_at=None,
                                          factor_context_doc=_doc()))
    rc = res["factor_receipt"]
    assert rc["availability"]["report_state"] == "missing"
    assert "realloc_volume" not in rc["stages_executed"]
    assert "injury report missing" in rc["stages_not_executed"]["realloc_volume"]
    conn = dbmod.connect()
    for s in _stamps(conn, "wed").values():
        assert s["availability"]["status"] == "UNKNOWN"
        assert s["stages"]["realloc_volume"]["state"] == "not_evaluated"


def test_qb_context_uses_only_claims_this_run_captured_before_its_clock(env):
    early = _claim("2023-11-04T12:00:00Z")
    late = {**_claim("2099-01-01T00:00:00Z"), "story_id": "late"}
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_feeds(stamp_now(), factor_context_doc=_doc([late])))
    q = res["factor_receipt"]["qb_context"]["BBB"]
    assert q["state"] != qr.NO_PRIOR and qr.REJ_CAPTURED_LATE in q["rejected"]
    res2 = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                       inject_feeds=_feeds(stamp_now(), factor_context_doc=_doc([early])))
    q2 = res2["factor_receipt"]["qb_context"]["BBB"]
    assert q2["qb_id"] == "QB_B" and q2["rejected"] == []
    conn = dbmod.connect()
    card = [c for c in pc.week_cards(conn, SEASON, WEEK)["cards"] if c["player_id"] == "QB_B"][0]
    rec = _labels(card["factor_panel"])["qb_starter_readiness:BBB"]
    assert rec["status"] == "context_only"          # never numeric: the backup-QB rule is blocked
    assert "Bravo Quarterback" in rec["observation"]


def test_observed_snaps_are_context_records_never_zero_filled(env):
    snaps = pd.DataFrame([
        {"game_id": f"{SEASON}_0{w}_AAA_BBB", "season": SEASON, "game_type": "REG", "week": w,
         "player": "Alpha Wideout", "pfr_player_id": "AlphWi00", "position": "WR", "team": "AAA",
         "opponent": "BBB", "offense_snaps": 50 + w, "offense_pct": 0.8} for w in (7, 8)])
    feeds = _feeds(stamp_now(), factor_context_doc=_doc(), snap_counts=snaps,
                   snap_counts_source={"url": "https://example/snaps.parquet",
                                       "fetched_at": "2023-11-04T00:00:00Z"})
    feeds["active_roster"]["rows"] = [dict(r, pfr_id="AlphWi00") if r["player_id"] == "WR_A" else r
                                      for r in feeds["active_roster"]["rows"]]
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(), inject_feeds=feeds)
    part = res["factor_receipt"]["participation"]
    assert part["status"].startswith("ok") and part["identity"]["linked"] == 2
    conn = dbmod.connect()
    for c in pc.week_cards(conn, SEASON, WEEK)["cards"]:
        lab = _labels(c["factor_panel"])
        snap = [v for k, v in lab.items() if k.startswith(f"observed_snaps:{c['player_id']}")]
        assert snap, c["player_id"]
        if c["player_id"] == "WR_A":
            assert snap[0]["status"] == "context_only" and "58 snaps" in snap[0]["observation"]
        else:
            assert snap[0]["status"] == "unavailable_unverified"   # no rows: missing, not zero
        assert lab["participation:routes"]["status"] == "unavailable_unverified"


def test_later_t90_context_never_rewrites_what_the_wed_pick_knew(env):
    wed = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_feeds(stamp_now(), factor_context_doc=_doc()))
    conn = dbmod.connect()
    wed_before = _stamps(conn, "wed")
    wed_quotes = conn.execute("SELECT player_id, market, quote_ts, run_id FROM leans "
                              "WHERE clock='wed' ORDER BY 1,2").fetchall()
    conn.close()
    t90_feeds = dict(_t90_feeds(stamp_now()))
    t90_feeds.update(active_roster=_feeds(stamp_now())["active_roster"],
                     factor_context_doc=_doc([_claim(stamp_now())]))   # captured after Wednesday
    t90 = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=t90_feeds)
    conn = dbmod.connect()
    assert _stamps(conn, "wed") == wed_before
    assert conn.execute("SELECT player_id, market, quote_ts, run_id FROM leans WHERE clock='wed' "
                        "ORDER BY 1,2").fetchall() == wed_quotes
    assert wed["factor_receipt"]["qb_context"]["BBB"]["qb_id"] is None
    assert t90["factor_receipt"]["qb_context"]["BBB"]["qb_id"] == "QB_B"
    assert t90["factor_receipt"]["run_id"] != wed["factor_receipt"]["run_id"]
    receipts = fimod.load_receipts(conn, SEASON, WEEK)
    ctx = fimod.load_context_records(conn, SEASON, WEEK)
    for r in dbmod.query_df(conn, "SELECT * FROM leans WHERE clock='wed'").to_dict("records"):
        lab = _labels(fimod.card_panel(r, receipts, ctx))
        assert "Bravo Quarterback will start" not in json.dumps(lab)
        team = json.loads(r["stage_json"])["team"]
        q = lab[f"qb_starter_readiness:{team}"]
        assert q["status"] != "numeric_applied"
        assert "Confirmed starter" not in q["observation"]

"""Confirmed-starter eligibility for QB markets (pass_attempts, passing_yards).

Reproduced defect (2026 Week 3, ATL@GB): the team source confirmed Michael Penix Jr.
(00-0039917) before the run, the run's own qb_context_records resolved it, yet the board
priced C.Rush (00-0033662) pass_attempts -- the resolver was context-only and nothing tied it
to executability.  The gate blocks execution of a non-starter QB's rows; it changes no number
and never asserts a backup when the starter is not confirmed.
"""
from __future__ import annotations

import copy
import json

import pytest

import pipeline_weekly as pw
from nflvalue import candidates as cand
from nflvalue import db as dbmod
from nflvalue import factor_integration as fimod
from nflvalue import pick_cards as pc
from nflvalue import qb_readiness as qr

tfi = pytest.importorskip("test_factor_integration")   # native run_week harness (same dir)
env = tfi.env

PENIX, RUSH, STRAND = "00-0039917", "00-0033662", "00-0041194"
ROSTER = [{"player_id": PENIX, "name": "Michael Penix Jr.", "team": "ATL", "position": "QB"},
          {"player_id": RUSH, "name": "Cooper Rush", "team": "ATL", "position": "QB"},
          {"player_id": STRAND, "name": "Jack Strand", "team": "ATL", "position": "QB"}]
KICK = {"ATL": "2026-09-25T00:15:00Z"}
AFTER = "2026-09-23T04:06:06Z"   # the saved replay run's decision clock
ROWS = [{"player_id": p, "team": "ATL", "market": m} for p in (PENIX, RUSH, STRAND)
        for m in ("pass_attempts", "passing_yards")] + [
    {"player_id": RUSH, "team": "ATL", "market": "rush_attempts"}]


def _doc():
    return json.load(open(fimod.context_path(2026, 3)))   # committed file: real Penix claim


def _qb(doc, as_of=AFTER, roster=ROSTER):
    return fimod.qb_context_records(["ATL"], doc=doc, pbp=None, roster_rows=roster, season=2026,
                                    week=3, as_of=as_of, kickoffs=KICK)


def _claim(doc):
    return next(i for i in doc["news"] if i.get("claim_key") == "starter" and i.get("team") == "ATL")


def test_confirmed_starter_blocks_only_other_qbs_qb_markets():
    qb = _qb(_doc())
    assert qb["ATL"]["qb_id"] == PENIX and qb["ATL"]["state"] in (qr.SOURCED_CHANGED, qr.NO_PRIOR)
    gate = cand.confirmed_starter_gate(ROWS, qb)
    assert {k for k, g in gate.items() if g["blocks_execution"]} == {
        (RUSH, "pass_attempts"), (RUSH, "passing_yards"),
        (STRAND, "pass_attempts"), (STRAND, "passing_yards")}
    assert gate[(PENIX, "pass_attempts")]["state"] == "confirmed_starter"
    assert (RUSH, "rush_attempts") not in gate, "non-QB markets are not gated"
    g = gate[(RUSH, "pass_attempts")]
    assert g["starter_qb_id"] == PENIX and g["published_at"] and g["source_tier"] == "team_official"


def test_same_confirmed_starter_is_not_blocked():
    qb = {"ATL": {"state": qr.SOURCED_SAME, "qb_id": RUSH, "qb_name": "Cooper Rush"}}
    gate = cand.confirmed_starter_gate(ROWS, qb)
    assert gate[(RUSH, "passing_yards")] ["state"] == "confirmed_starter"
    assert gate[(PENIX, "passing_yards")]["blocks_execution"] is True


def test_claim_captured_after_the_decision_clock_confirms_nothing():
    doc = _doc()
    c = _claim(doc)
    c["fetched_at"] = "2026-09-24T12:00:00Z"
    c["observed_at"] = "2026-09-24T12:00:00Z"
    qb = _qb(doc, as_of="2026-09-24T00:00:00Z")
    gate = cand.confirmed_starter_gate(ROWS, qb)
    assert qb["ATL"]["qb_id"] is None
    assert not any(g["blocks_execution"] for g in gate.values())
    assert gate[(RUSH, "pass_attempts")]["state"] == "starter_not_confirmed"
    assert "not asserted to be the starter or a backup" in gate[(RUSH, "pass_attempts")]["reason"]


def test_conflicting_claims_disclose_and_do_not_block():
    doc = _doc()
    other = copy.deepcopy(_claim(doc))
    other.update(story_id="atl_rush_named_starter_test", claim_value="Cooper Rush")
    doc["news"].append(other)
    qb = _qb(doc)
    assert qb["ATL"]["state"] == qr.CONFLICT
    gate = cand.confirmed_starter_gate(ROWS, qb)
    assert not any(g["blocks_execution"] for g in gate.values())
    assert {g["team_state"] for g in gate.values()} == {qr.CONFLICT}


def test_unlinked_claim_identity_confirms_nothing():
    qb = _qb(_doc(), roster=[r for r in ROSTER if r["player_id"] != PENIX])
    assert qb["ATL"]["qb_id"] is None
    assert not any(g["blocks_execution"] for g in cand.confirmed_starter_gate(ROWS, qb).values())


def test_gate_changes_no_number_and_leaves_old_qb_rules_alone():
    qb = _qb(_doc())
    before = copy.deepcopy(qb)
    rows = [dict(r, mean=20.25, backup_qb_adj=1.0) for r in ROWS]
    snapshot = copy.deepcopy(rows)
    cand.confirmed_starter_gate(rows, qb)
    assert rows == snapshot and qb == before
    assert qr.NUMERIC_BLOCK["backup_qb_adj"], "numeric backup-QB rule stays blocked"


def _card_row(stamp):
    return {"player": "C.Rush", "player_id": RUSH, "game_id": "2026_03_ATL_GB", "market": "pass_attempts",
            "side": "under", "line": 20.5, "price": 1.9, "mean": 20.25, "sd": 13.05, "p_side": 0.51,
            "quote_book": "dk", "quote_ts": "2026-09-23T04:00:00Z", "line_source": "odds_api",
            "_quote_verified": True, "stage_json": json.dumps(stamp)}


def test_blocked_row_renders_research_and_rerender_preserves_each_runs_decision():
    import datetime as dt
    now = dt.datetime(2026, 9, 23, 5, tzinfo=dt.timezone.utc)
    ok = {"availability": {"status": "OK", "eligibility": "eligible", "availability_state": "not_listed"}}
    earlier = pc.build_card(_card_row(ok), now)                   # persisted by an earlier run
    stamps = {(RUSH, "pass_attempts"): copy.deepcopy(ok)}
    pw._apply_starter_gate(stamps, cand.confirmed_starter_gate(ROWS, _qb(_doc())))
    blocked = pc.build_card(_card_row(stamps[(RUSH, "pass_attempts")]), now)
    assert earlier["status"] == "watch"
    assert blocked["status"] == "research" and "not_confirmed_starter" in " ".join(blocked["status_reasons"])
    assert blocked["mean"] == earlier["mean"]
    # re-rendering the earlier run's persisted row later still gives that run's decision
    assert pc.build_card(_card_row(ok), now)["status"] == "watch"
    a = stamps[(RUSH, "pass_attempts")]["availability"]
    assert a["status"] == "OK" and a["eligibility_before_starter_gate"] == "eligible"


def test_native_wed_run_persists_the_gate_and_exports_diagnostics(env, monkeypatch):
    from nflvalue.freshness import stamp_now

    def run(starter):
        real = fimod.qb_context_records

        def fake(teams, **kw):
            out = real(teams, **kw)
            if starter:
                for t in out:
                    out[t] = {**out[t], "state": qr.SOURCED_CHANGED, "qb_id": starter,
                              "qb_name": "Test Starter", "source_tier": "team_official",
                              "published_at": "2026-09-01T00:00:00Z"}
            return out
        monkeypatch.setattr(fimod, "qb_context_records", fake)
        return pw.run_week(tfi.SEASON, tfi.WEEK, mode="live", inputs=tfi.synthetic_inputs(),
                           inject_feeds=tfi._fresh_feeds(stamp_now()))

    base = run(None)
    gated = run("00-NOT-ON-BOARD")
    means = lambda r: {(l["player_id"], l["market"]): l.get("mean") for g in r["games"] for l in g["leans"]}
    assert means(base) == means(gated), "the gate never changes a number or the selection"
    conn = dbmod.connect()
    rid = gated["factor_receipt"]["run_id"]
    diag = gated["factor_receipt"]["qb_starter_gate"]
    assert diag and all(d["confirmed"] for d in diag.values())
    assert all(d["starter_no_card_reason"].startswith("no candidate row") for d in diag.values())
    leans = dbmod.query_df(conn, "SELECT market, stage_json FROM leans WHERE clock='wed' AND run_id=?", (rid,)) \
        if "run_id" in dbmod.query_df(conn, "SELECT * FROM leans LIMIT 1").columns else \
        dbmod.query_df(conn, "SELECT market, stage_json FROM leans WHERE clock='wed'")
    qbm = leans[leans.market.isin(cand.STARTER_GATED_MARKETS)]
    assert len(qbm), "fixture board must contain QB-market leans"
    for s in qbm.stage_json:
        st = json.loads(s)
        if st.get("qb_eligibility"):
            assert st["availability"]["availability_state"] == "not_confirmed_starter"
    cards = [c for c in pc.week_cards(conn, tfi.SEASON, tfi.WEEK)["cards"]
             if c["market"] in cand.STARTER_GATED_MARKETS]
    assert cards and all(c["status"] not in ("watch", "actionable") for c in cards)

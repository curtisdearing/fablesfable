"""T-90 refresh through the real consumer: run_t90 -> DB -> pick_cards.

RED against b10e40c: ``run_t90`` (the function ``scripts/auto_weekly.job_t90`` calls per due
game) stamped no stages, ran no shadow and persisted no receipt or context. Its leans carried
the process run id, so a T-90 card silently borrowed the Wednesday receipt and context --
reloaded data shown as a new execution, at the wrong clock. The refresh is now its own issuing
run per game, with only the stages it actually executed.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import candidates as candmod  # noqa: E402
from nflvalue import config as cfgmod  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import factor_integration as fimod  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from nflvalue.freshness import stamp_now  # noqa: E402
from tests.test_pipeline_weekly import GAME_ID, _fresh_feeds, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402

T90_RUN_SUFFIX = f":t90:{GAME_ID}"


def _t90_feeds(now):
    f = dict(_fresh_feeds(now))
    f["inactive_rows"] = [{"espn_id": "9", "name": "Nobody Inactive", "active": False,
                           "did_not_play": True, "starter": False, "team": "BBB"}]
    f["inactives_fetched_at"] = now
    return f


def _wed_then_t90(**t90_kw):
    now = stamp_now()
    wed = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_fresh_feeds(now))
    t90 = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=_t90_feeds(stamp_now()), **t90_kw)
    return wed, t90


def _receipts(conn):
    return {r[0]: (r[1], json.loads(r[2])) for r in
            conn.execute("SELECT run_id, clock, receipt_json FROM run_receipts").fetchall()}


def _labels(panel):
    return {it["factor_id"]: it for g in panel["groups"] for it in g["items"]}


def test_t90_persists_its_own_stamps_shadow_receipt_and_context(env):
    wed, t90 = _wed_then_t90()
    conn = dbmod.connect()
    rec = _receipts(conn)
    wed_id = wed["factor_receipt"]["run_id"]
    t90_id = [k for k in rec if k.endswith(T90_RUN_SUFFIX)]
    assert len(rec) == 2 and len(t90_id) == 1 and t90_id[0] != wed_id
    clock, r = rec[t90_id[0]]
    assert clock == "t90" and r["clock"] == "t90" and r["game_ids"] == [GAME_ID]
    assert r["as_of"] == t90["as_of"] and r["as_of"] >= wed["factor_receipt"]["as_of"]
    # only what the refresh executed; the Wednesday stages are not assumed
    assert "backup_qb" in r["stages_executed"]
    for s in ("realloc_volume", "realloc_efficiency", "absence_qb"):
        assert s not in r["stages_executed"]
        assert r["stages_not_executed"][s] == "not executed by the T-90 refresh"
    assert r["shadow"]["status"] and r["inactives_state"] == "populated"
    assert r["context"]["status"] and r["base_run_id"] == wed_id  # same process, own issuing id
    leans = dbmod.query_df(conn, "SELECT * FROM leans WHERE clock='t90'")
    assert len(leans) and leans["stage_json"].notna().all()
    assert (leans["run_id"] == t90_id[0]).all()
    for j in leans["stage_json"]:
        st = json.loads(j)["stages"]
        assert st["realloc_volume"]["state"] == "not_evaluated"
    ctx_runs = {row[0] for row in conn.execute("SELECT run_id FROM factor_context").fetchall()}
    assert {wed_id, t90_id[0]} <= ctx_runs
    cards = pc.week_cards(conn, SEASON, WEEK)["cards"]
    t90_cards = [c for c in cards if c.get("clock") == "t90"] or cards
    for c in t90_cards:
        lab = _labels(c["factor_panel"])
        assert "model_stages" not in lab and f"context_not_collected:{GAME_ID}" in lab


def test_wed_card_and_other_runs_context_unchanged_by_t90(env):
    now = stamp_now()
    pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(), inject_feeds=_fresh_feeds(now))
    conn = dbmod.connect()
    base = conn.execute("SELECT run_id FROM run_receipts").fetchone()[0]
    # another game's T-90 from the SAME scheduled process, and an older run's context
    other = fe_ctx("2026_03_OTH_ERS", now)
    fimod.persist_run(conn, SEASON, WEEK, "t90", {"run_id": f"{base}:t90:2026_03_OTH_ERS", "as_of": now},
                      [other])
    fimod.persist_run(conn, SEASON, WEEK, "wed", {"run_id": "older-run", "as_of": now}, [other])
    snap = sorted(conn.execute("SELECT run_id, game_id, factor_id, record_json FROM factor_context").fetchall())
    wed = dbmod.query_df(conn, "SELECT * FROM leans WHERE clock='wed'")
    panels = [fimod.card_panel(r, fimod.load_receipts(conn, SEASON, WEEK),
                               fimod.load_context_records(conn, SEASON, WEEK)) for r in wed.to_dict("records")]
    conn.close()
    pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
               inject_feeds=_t90_feeds(stamp_now()))
    conn = dbmod.connect()
    after = sorted(conn.execute("SELECT run_id, game_id, factor_id, record_json FROM factor_context").fetchall())
    assert set(snap) <= set(after)                                     # nothing rewritten or deleted
    assert any(r[0].endswith(T90_RUN_SUFFIX) for r in after)
    wed2 = dbmod.query_df(conn, "SELECT * FROM leans WHERE clock='wed'")
    pd.testing.assert_frame_equal(wed.drop(columns=["status", "void_reason"]),
                                  wed2.drop(columns=["status", "void_reason"]))
    panels2 = [fimod.card_panel(r, fimod.load_receipts(conn, SEASON, WEEK),
                                fimod.load_context_records(conn, SEASON, WEEK)) for r in wed2.to_dict("records")]
    assert panels == panels2


def fe_ctx(game, clock):
    from nflvalue import factor_evidence as fe
    return fe.normalize_record(dict(
        factor_id="test:other", category="team_news", entity_type="game", entity_id=game,
        game_id=game, as_of=clock, published_at=clock, fetched_at=clock, observed_at=clock,
        source_url="https://www.packers.com/test-synthetic-fixture", verified=True,
        measurement_kind="observed", observation="other game context"))


def test_unknown_future_stage_and_missing_qb_input_are_missing_not_neutral(env, monkeypatch):
    monkeypatch.setitem(fimod.STAGES, "future_stage", ("future_mult", None))
    real_bqb = candmod.apply_backup_qb_adjustment

    def no_qb_input(c):
        out = real_bqb(c)
        out["qb_continuity"] = float("nan")          # the QB-continuity input is missing
        return out
    monkeypatch.setattr(candmod, "apply_backup_qb_adjustment", no_qb_input)
    seen = {}
    real_build = fimod.build_stamps
    monkeypatch.setattr(fimod, "build_stamps",
                        lambda c, ran, why, **k: seen.setdefault("s", real_build(c, ran, why, **k)))
    pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
               inject_feeds=_t90_feeds(stamp_now()))
    stamps = seen["s"]
    assert stamps
    for (_pid, market), s in stamps.items():
        assert s["stages"]["future_stage"]["state"] == "not_evaluated"
        if market in fimod.STAGES["backup_qb"][1]:
            assert s["stages"]["backup_qb"]["state"] == "not_evaluated"
            assert "qb_continuity missing" in s["stages"]["backup_qb"]["reason"]
    assert any(m in fimod.STAGES["backup_qb"][1] for _p, m in stamps)


def test_context_source_failure_is_reported_not_healthy(env, monkeypatch, tmp_path):
    d = tmp_path / "ctx"
    d.mkdir()
    (d / f"{SEASON}-w{WEEK:02d}.json").write_text("{not json")
    monkeypatch.setattr(fimod, "CONTEXT_DIR", str(d))
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=_t90_feeds(stamp_now()))
    assert res["factor_receipt"]["context"]["status"].startswith("error")
    conn = dbmod.connect()
    fids = {r[0] for r in conn.execute("SELECT factor_id FROM factor_context").fetchall()}
    assert f"context_source_failed:{GAME_ID}" in fids
    for c in pc.week_cards(conn, SEASON, WEEK)["cards"]:
        it = _labels(c["factor_panel"])[f"context_source_failed:{GAME_ID}"]
        assert it["status"] == "unavailable_unverified"


def _stored_quote_run(monkeypatch, over_price, quote_ts):
    real = cfgmod.load_config
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {**real(*a, **k),
                                                                 "odds_api_key": "TEST-DUMMY"})
    conn = dbmod.connect()
    conn.execute("DELETE FROM lines")
    for book in ("draftkings", "fanduel"):
        for side, price in (("over", over_price), ("under", 1.9)):
            conn.execute("INSERT INTO lines VALUES (?,?,?,?,?,?,?,?,?)",
                         (quote_ts, GAME_ID, book, "receiving_yards", None, "Alpha Wideout",
                          side, 55.5, price))
    conn.commit()
    conn.close()

    def refuse(*a, **k):
        raise AssertionError("no odds acquisition in this test")
    return pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_t90_feeds(stamp_now()), odds_fetch=refuse,
                      list_events_fn=lambda cfg: [])


def test_reused_quotes_keep_their_capture_clock_and_stamps_ignore_prices(env, monkeypatch):
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    a = _stored_quote_run(monkeypatch, 1.91, old)
    lines = a["factor_receipt"]["lines"]
    assert lines["rows"] == 4 and lines["games_pulled_this_run"] == []
    assert lines["quote_clock_min"] == lines["quote_clock_max"] == old
    assert lines["source"].startswith("stored quotes")
    conn = dbmod.connect()
    ql = dbmod.query_df(conn, "SELECT quote_ts, stage_json, shadow_json FROM leans "
                              "WHERE clock='t90' AND player_id='WR_A' AND market='receiving_yards'")
    conn.close()
    assert len(ql) == 1 and ql["quote_ts"].iloc[0] == old          # the capture clock, not the run's
    stamps_a = dbmod.query_df(dbmod.connect(), "SELECT player_id, market, stage_json, shadow_json "
                                               "FROM leans WHERE clock='t90' ORDER BY player_id, market")
    _stored_quote_run(monkeypatch, 3.40, old)                        # the market moved a lot
    stamps_b = dbmod.query_df(dbmod.connect(), "SELECT player_id, market, stage_json, shadow_json "
                                               "FROM leans WHERE clock='t90' ORDER BY player_id, market")
    common = stamps_a.merge(stamps_b, on=["player_id", "market"])
    assert len(common)
    assert (common["stage_json_x"] == common["stage_json_y"]).all()
    assert (common["shadow_json_x"].fillna("") == common["shadow_json_y"].fillna("")).all()


def test_t90_shadow_cannot_change_the_published_pick(env, monkeypatch):
    def published(v):
        monkeypatch.setattr(fimod, "shadow_opportunity", lambda pw_, cands, **k: {
            "status": "ok", "component": "x", "players": {
                pid: {"expected": {"expected_targets": v}, "share": {}} for pid in cands["player_id"]}})
        pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                   inject_feeds=_t90_feeds(stamp_now()))
        conn = dbmod.connect()
        df = dbmod.query_df(conn, "SELECT player_id, market, side, line, mean, sd, p_side, composite "
                                  "FROM leans WHERE clock='t90' ORDER BY rowid")
        conn.close()
        return df
    a, b = published(0.0), published(999.0)
    assert len(a)
    pd.testing.assert_frame_equal(a, b)

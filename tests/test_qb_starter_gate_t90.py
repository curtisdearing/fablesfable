"""Native T-90 verification of the confirmed-starter gate (run_t90 -> DB -> pick_cards).

Nothing under test is mocked: the context document is injected through the run's own
``factor_context_doc`` feed, the REAL resolver (``qb_context_records``) links the claim to the
roster by team + QB + exact name, and stored offered quotes make the non-starter's passing
rows executable in the control run.  RED on b0808c6 (no gate): the blocked QB's cards are
``watch``.  Synthetic software fixture -- not statistical evidence.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import config as cfgmod  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from nflvalue.freshness import stamp_now  # noqa: E402
from tests.test_pipeline_weekly import GAME_ID, _roster, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402
from tests.test_t90_factor_receipts import _t90_feeds  # noqa: E402

QB_MARKETS = ("pass_attempts", "passing_yards")
LINES = {"pass_attempts": 33.5, "passing_yards": 244.5}
# claim clocks: published and captured before the (2023-11-05 18:00Z) kickoff and the run
PUB, GOT = "2023-11-03T20:00:00Z", "2023-11-04T12:00:00Z"


def _doc(starter_name):
    news = [] if starter_name is None else [{
        "story_id": "bbb_starter_test", "entity_id": "BBB", "entity_type": "team", "team": "BBB",
        "game_id": GAME_ID, "category": "qb_news", "claim_key": "starter", "claim_kind": "confirmed",
        "claim_value": starter_name, "claim": f"{starter_name} named BBB's starting quarterback.",
        "attribution": "BBB team site (test fixture)", "source_tier": "team_official",
        "source_url": "https://example.test/bbb-starter", "published_at": PUB,
        "observed_at": PUB, "fetched_at": GOT}]
    return {"schema": "factor_context/1", "season": SEASON, "week": WEEK, "news": news, "records": []}


def _feeds(starter_name):
    now = stamp_now()
    f = _t90_feeds(now)
    ros = _roster(now, extra=[{"player_id": "QB_B2", "team": "BBB", "status": "ACT", "week": WEEK}])
    names = {"QB_B": ("Bravo Quarterback", "QB"), "QB_B2": ("Charlie Starter", "QB"),
             "WR_A": ("Alpha Wideout", "WR"), "RB_A": ("Alpha Back", "RB")}
    for r in ros["rows"]:
        r["name"], r["position"] = names[r["player_id"]]
    f["active_roster"] = ros
    # BBB's report is received (QB_B not listed -> eligible, not "healthy"); without it the
    # availability hold, an independent block, would already make QB_B research
    f["injury_rows"] = list(f["injury_rows"]) + [
        {"team": "BBB", "name": "Bravo Lineman", "status_raw": "Out", "status": "OUT", "comment": ""}]
    f["factor_context_doc"] = _doc(starter_name)
    return f


def _run(monkeypatch, starter_name):
    real = cfgmod.load_config
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {**real(*a, **k),
                                                                 "odds_api_key": "TEST-DUMMY"})
    quote_ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = dbmod.connect()
    conn.execute("DELETE FROM lines")
    for m, pt in LINES.items():
        for side in ("over", "under"):
            conn.execute("INSERT INTO lines VALUES (?,?,?,?,?,?,?,?,?)",
                         (quote_ts, GAME_ID, "draftkings", m, None, "Bravo Quarterback", side, pt, 1.91))
    conn.commit()
    conn.close()

    def refuse(*a, **k):
        raise AssertionError("no odds acquisition in this test")
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=_feeds(starter_name), odds_fetch=refuse, list_events_fn=lambda cfg: [])
    conn = dbmod.connect()
    cards = [c for c in pc.week_cards(conn, SEASON, WEEK)["cards"]
             if c["player_id"] == "QB_B" and c["market"] in QB_MARKETS]
    leans = dbmod.query_df(conn, "SELECT player_id, market, mean, sd, p_side, line, run_id, stage_json "
                                 "FROM leans WHERE clock='t90' AND player_id='QB_B'")
    conn.close()
    return res, cards, leans


def test_t90_control_without_a_claim_makes_the_qb_executable(env, monkeypatch):
    _, cards, _ = _run(monkeypatch, None)
    assert {c["market"] for c in cards} == set(QB_MARKETS), "fixture must publish both QB cards"
    assert all(c["status"] == "watch" for c in cards), [(c["market"], c["status_reasons"]) for c in cards]


def test_t90_confirmed_other_starter_blocks_the_qb_after_persistence_and_rerender(env, monkeypatch):
    _, control, control_leans = _run(monkeypatch, None)
    res, cards, leans = _run(monkeypatch, "Charlie Starter")
    assert len(control) == len(cards) == 2 and len(leans)
    assert all(c["status"] == "watch" for c in control)
    # the real resolver linked the sourced claim to QB_B2 under this run's own clock
    rc = res["factor_receipt"]
    assert rc["qb_context"]["BBB"]["state"] == "no_prior_realized_starter"
    assert rc["qb_context"]["BBB"]["qb_id"] == "QB_B2"
    # persisted, then rendered: never watch/actionable
    for c in cards:
        assert c["status"] == "research" and c["quote"] is None
        assert "not_confirmed_starter" in " ".join(c["status_reasons"])
    for s in leans["stage_json"]:
        st = json.loads(s)
        assert st["availability"]["availability_state"] == "not_confirmed_starter"
    # a second render from the DB gives the same decision
    conn = dbmod.connect()
    again = {c["market"]: c["status"] for c in pc.week_cards(conn, SEASON, WEEK)["cards"]
             if c["player_id"] == "QB_B" and c["market"] in QB_MARKETS}
    conn.close()
    assert again == {m: "research" for m in QB_MARKETS}
    # numbers unchanged by the gate
    key = ["market", "mean", "sd", "p_side", "line"]
    a = control_leans[key].sort_values("market").reset_index(drop=True)
    b = leans[key].sort_values("market").reset_index(drop=True)
    assert len(a) == len(b) == 2 and a.equals(b)
    assert {c["market"]: (c["mean"], c["line"]) for c in control} == \
           {c["market"]: (c["mean"], c["line"]) for c in cards}
    # exported diagnostics: the starter has no card, with the exact reason
    diag = rc["qb_starter_gate"]["BBB"]
    assert diag["starter_qb_id"] == "QB_B2" and diag["starter_no_card_reason"]
    assert sorted(map(tuple, diag["blocked_rows"])) == [("QB_B", m) for m in QB_MARKETS]


def test_t90_same_confirmed_starter_stays_executable(env, monkeypatch):
    res, cards, _ = _run(monkeypatch, "Bravo Quarterback")
    assert res["factor_receipt"]["qb_context"]["BBB"]["qb_id"] == "QB_B"
    assert len(cards) == 2 and all(c["status"] == "watch" for c in cards)

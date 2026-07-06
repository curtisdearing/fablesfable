#!/usr/bin/env python3
"""Phase 7.3/7.4 — worked example of real-line re-labeling + CLV, on fixtures.

No live Odds API key exists (and July is the offseason), so this proves the
math the 7.3 design specifies using the SYNTHETIC v4 fixture
(``tests/fixtures/oddsapi_event_props_synthetic.json``) and a throwaway DB. It
drives the REAL production code paths — ``oddsapi_props.parse_event_props`` /
``match_player_ids``, ``ml_test.augment_with_real_lines``,
``clv.log_close_for_week``, ``killcheck.report`` — not a re-implementation, so
what passes here is what 7.4 will run live.

It demonstrates, end to end:
  1. capture   two snapshots (Wed entry, pre-kickoff close) into ``lines``;
  2. RE-LABEL  a training row flips synthetic -> real line, y and the
               line-dependent features recompute, non-line features untouched;
  3. CLV       de-vigged entry vs close consensus -> clv_prob per lean;
  4. MONITOR   killcheck reads the accrued sample and returns its verdict.
  5. GAP #1    a snapshot outside the close window must NOT resolve as a close
               (proves CLV can't be faked ~0 against a stale entry-era line).
  6. GAP #3    a wed+t90 lean for the same key resolves to exactly ONE clv row
               (the ``clv`` table's PK omits ``clock``).
  7. GAP #2    the entry-event budget reservation formula in action.

Run: python3 scripts/clv_worked_example.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import db as dbmod                        # noqa: E402
from nflvalue import clv as clvmod                      # noqa: E402
from nflvalue import killcheck                          # noqa: E402
from nflvalue.sources import oddsapi_props as op        # noqa: E402
import ml_test                                          # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "oddsapi_event_props_synthetic.json")

SEASON, WEEK, CLOCK = 2023, 10, "wed"
GAME_ID = "2023_10_CLE_BAL"
ANDREWS = "00-0033288"          # Mark Andrews (synthetic gsis id for the demo)
KICKOFF = "2023-11-12T18:00:00Z"
ENTRY_TS = "2023-11-08T17:00:00Z"    # Wednesday snapshot
CLOSE_TS = "2023-11-12T16:30:00Z"    # ~90 min pre-kickoff snapshot


def _candidates() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": ANDREWS, "name": "Mark Andrews"},
        {"player_id": "00-0036355", "name": "Amari Cooper"},
        {"player_id": "00-0034796", "name": "Lamar Jackson"},
    ])


def _snapshot_rows(payload: dict, ts: str, price_shift: float = 0.0) -> list:
    """Parse the fixture payload into line rows at ``ts``; ``price_shift``
    fattens the OVER (shorter decimal price = higher implied prob) to simulate
    the market moving toward our side by the close."""
    rows = op.parse_event_props(payload, ts)
    for r in rows:
        r["game_id"] = GAME_ID
        if price_shift and r["market"] == "receiving_yards" and r["side"] == "over":
            r["price"] = round(r["price"] - price_shift, 3)   # over gets more expensive
    return op.match_player_ids(rows, _candidates())


def main() -> None:
    payload = json.load(open(FIXTURE))["payload"]
    dbpath = os.path.join(tempfile.mkdtemp(), "clv_demo.db")
    conn = dbmod.connect(dbpath)

    # -- 1. capture two snapshots ------------------------------------------- #
    entry_rows = _snapshot_rows(payload, ENTRY_TS, price_shift=0.0)
    close_rows = _snapshot_rows(payload, CLOSE_TS, price_shift=0.17)  # over shortens
    dbmod.upsert(conn, "lines", entry_rows + close_rows,
                 ["ts", "game_id", "book", "market", "player_name", "side"])
    matched = sum(1 for r in entry_rows if r["player_id"] is not None)
    print(f"[1] capture: {len(entry_rows)} entry + {len(close_rows)} close line rows "
          f"({matched}/{len(entry_rows)} matched to gsis ids; "
          f"unmatched stay player_id=NULL and can never mint an edge)")

    # the published lean (decision-time real line = the point we can transact at)
    dbmod.upsert(conn, "leans", [{
        "season": SEASON, "week": WEEK, "clock": CLOCK, "game_id": GAME_ID,
        "player_id": ANDREWS, "name": "Mark Andrews", "market": "receiving_yards",
        "side": "over", "line": 52.5, "line_source": "odds_api", "price": 1.87,
        "book": "draftkings", "as_of": ENTRY_TS, "status": "active",
    }], ["season", "week", "clock", "game_id", "player_id", "market"])
    # the graded outcome (post-game actual). Actual chosen to CROSS the two
    # lines: over the synthetic 47.0, under the real 52.5 -> the label flips.
    ACTUAL = 50.0
    dbmod.upsert(conn, "lean_outcomes", [{
        "season": SEASON, "week": WEEK, "clock": CLOCK, "game_id": GAME_ID,
        "player_id": ANDREWS, "name": "Mark Andrews", "market": "receiving_yards",
        "side": "over", "line": 52.5, "actual": ACTUAL, "hit": 0,
    }], ["season", "week", "clock", "game_id", "player_id", "market"])

    # -- 2. RE-LABEL: synthetic -> real ------------------------------------- #
    frame = pd.DataFrame([{
        "season": SEASON, "week": WEEK, "player_id": ANDREWS,
        "market": "receiving_yards", "mean": 55.0, "sd": 18.0,
        "line": 47.0,                                   # synthetic trailing-mean line
        "mean_minus_line": 55.0 - 47.0, "sd_over_line": 18.0 / 47.0,
        "z": (55.0 - 47.0) / 18.0,
        "y_over": 1.0,                                  # actual 50 > synthetic 47
    }])
    before = frame.iloc[0]
    relabeled = ml_test.augment_with_real_lines(frame.copy(), conn)
    after = relabeled.iloc[0]
    print(f"\n[2] RE-LABEL row (season={SEASON} wk{WEEK} Mark Andrews receiving_yards, "
          f"actual={ACTUAL}):")
    print(f"      line          {before['line']:.1f}  ->  {after['line']:.1f}   "
          "(synthetic trailing-mean -> real decision-time line)")
    print(f"      y_over        {before['y_over']:.0f}    ->  {after['y_over']:.0f}     "
          f"(50>47 was OVER; 50<52.5 is UNDER — the label flips)")
    print(f"      z             {before['z']:+.3f} ->  {after['z']:+.3f}  (recomputed vs real line)")
    print(f"      mean_minus_line {before['mean_minus_line']:+.2f} ->  {after['mean_minus_line']:+.2f}")
    print(f"      mean/sd        {before['mean']:.1f}/{before['sd']:.1f} -> "
          f"{after['mean']:.1f}/{after['sd']:.1f}   (non-line features UNCHANGED)")
    assert after["line"] == 52.5 and after["y_over"] == 0.0, "re-label failed"
    assert after["mean"] == before["mean"] and after["sd"] == before["sd"], "corrupted non-line feature"
    assert abs(after["z"] - (55.0 - 52.5) / 18.0) < 1e-9, "z not recomputed vs real line"

    # -- 3. CLV: de-vigged entry vs close ----------------------------------- #
    resolved = clvmod.log_close_for_week(conn, SEASON, WEEK, {GAME_ID: KICKOFF})
    r = resolved.iloc[0]
    print(f"\n[3] CLV (Mark Andrews receiving_yards OVER, de-vigged consensus, "
          f"n_books≥2):")
    print(f"      entry_prob   {r['entry_prob']:.4f}  (Wed snapshot {r['entry_ts'][:10]})")
    print(f"      close_prob   {r['close_prob']:.4f}  (pre-kickoff {r['close_ts'][:10]})")
    print(f"      clv_prob     {r['clv_prob']:+.4f}   (close − entry, de-vigged prob-points)")
    print(f"      point_moved  {r['point_moved']:+.2f}")
    # entry/close/clv are each rounded independently (5dp) inside clv.py, so
    # compare at that resolution, not to machine epsilon.
    assert abs(r["clv_prob"] - (r["close_prob"] - r["entry_prob"])) < 2e-5
    assert r["clv_prob"] > 0, "expected positive CLV (market moved toward our over)"

    # -- 4. MONITOR: kill-check reads the accrued sample -------------------- #
    verdict = killcheck.report(conn, min_sample=150)
    print(f"\n[4] MONITOR (killcheck, pre-committed n≥150 / avgCLV>0 / ≥52% positive):")
    print(f"      resolved n = {verdict['n']}  |  verdict = {verdict['verdict']}")
    print(f"      {verdict['detail']}")
    assert verdict["verdict"] == "INSUFFICIENT_SAMPLE"   # one lean is not a referendum
    print("\nSteps 1-4 passed — re-label flips correctly, CLV computes correctly, "
          "monitor is honest about sample size.")

    # -- 5. GAP #1: close-window floor --------------------------------------- #
    conn2 = dbmod.connect(os.path.join(tempfile.mkdtemp(), "clv_gap1.db"))
    stale_rows = _snapshot_rows(payload, "2023-11-08T17:00:00Z")   # entry only
    stale_rows2 = _snapshot_rows(payload, "2023-11-09T09:00:00Z")  # >6h pre-kick, still stale
    dbmod.upsert(conn2, "lines", stale_rows + stale_rows2,
                 ["ts", "game_id", "book", "market", "player_name", "side"])
    dbmod.upsert(conn2, "leans", [{
        "season": SEASON, "week": WEEK, "clock": CLOCK, "game_id": GAME_ID,
        "player_id": ANDREWS, "name": "Mark Andrews", "market": "receiving_yards",
        "side": "over", "line": 52.5, "line_source": "odds_api", "price": 1.87,
        "book": "draftkings", "as_of": ENTRY_TS, "status": "active",
    }], ["season", "week", "clock", "game_id", "player_id", "market"])
    stale_resolved = clvmod.log_close_for_week(conn2, SEASON, WEEK, {GAME_ID: KICKOFF},
                                               close_window_hours=6.0)
    print(f"\n[5] GAP #1 close-window floor: only a stale pre-Nov-9 snapshot exists "
          f"(>6h before the {KICKOFF} kickoff) -> resolved rows = {len(stale_resolved)} "
          f"(must be 0; a real close snapshot inside [kickoff-6h, kickoff] never arrived)")
    assert stale_resolved.empty, "a stale snapshot outside the close window must not resolve"
    has_close = clvmod.has_close_snapshot(conn2, GAME_ID, KICKOFF, close_window_hours=6.0)
    print(f"      has_close_snapshot(...) = {has_close}  "
          "(the T-90 scheduling guard agrees: nothing in-window yet, so a real resnap is still owed)")
    assert has_close is False

    # -- 6. GAP #3: clock dedup ------------------------------------------------ #
    conn3 = dbmod.connect(os.path.join(tempfile.mkdtemp(), "clv_gap3.db"))
    entry_rows3 = _snapshot_rows(payload, ENTRY_TS, price_shift=0.0)
    close_rows3 = _snapshot_rows(payload, CLOSE_TS, price_shift=0.17)
    dbmod.upsert(conn3, "lines", entry_rows3 + close_rows3,
                 ["ts", "game_id", "book", "market", "player_name", "side"])
    dbmod.upsert(conn3, "leans", [{
        "season": SEASON, "week": WEEK, "clock": "wed", "game_id": GAME_ID,
        "player_id": ANDREWS, "name": "Mark Andrews", "market": "receiving_yards",
        "side": "over", "line": 52.5, "line_source": "odds_api", "price": 1.87,
        "book": "draftkings", "as_of": ENTRY_TS, "status": "active",
    }, {
        # a t90 refresh of the SAME (game, player, market, side) with a LATER
        # as_of — the clv table's PK omits clock, so both would collide
        "season": SEASON, "week": WEEK, "clock": "t90", "game_id": GAME_ID,
        "player_id": ANDREWS, "name": "Mark Andrews", "market": "receiving_yards",
        "side": "over", "line": 52.5, "line_source": "odds_api", "price": 1.75,
        "book": "draftkings", "as_of": "2023-11-12T15:00:00Z", "status": "active",
    }], ["season", "week", "clock", "game_id", "player_id", "market"])
    dedup_resolved = clvmod.log_close_for_week(conn3, SEASON, WEEK, {GAME_ID: KICKOFF})
    print(f"\n[6] GAP #3 clock dedup: wed (as_of {ENTRY_TS[:10]}) + t90 "
          f"(as_of 2023-11-12) leans for the SAME key -> resolved rows = "
          f"{len(dedup_resolved)} (must be exactly 1, against the EARLIEST as_of)")
    assert len(dedup_resolved) == 1, "clock dedup must collapse to exactly one row"
    assert dedup_resolved.iloc[0]["entry_ts"][:10] == ENTRY_TS[:10], \
        "must resolve against the earliest (wed) as_of, not the later t90 one"

    # -- 7. GAP #2: budget reservation ---------------------------------------- #
    class _FakeBudget:
        def __init__(self, remaining):
            self.remaining = remaining
    cost_per_event = 7.0   # 7 markets x 1 region
    cap_full_month = op.entry_event_cap(_FakeBudget(450.0), cost_per_event)
    cap_half_month = op.entry_event_cap(_FakeBudget(225.0), cost_per_event)
    cap_exhausted = op.entry_event_cap(_FakeBudget(0.0), cost_per_event)
    print(f"\n[7] GAP #2 budget reservation (cost/event={cost_per_event:.0f}, "
          f"formula floor(weekly_budget / (2*cost_per_event)), ~4.3 weeks/month):")
    print(f"      remaining=450.0 credits -> entry cap/week = {cap_full_month}")
    print(f"      remaining=225.0 credits -> entry cap/week = {cap_half_month}")
    print(f"      remaining=0.0   credits -> entry cap/week = {cap_exhausted}")
    assert cap_exhausted == 0
    assert cap_half_month <= cap_full_month
    assert cap_full_month * cost_per_event * 2 <= 450.0 + cost_per_event * 2, \
        "the reservation must leave roughly half the remaining budget for closes"

    print("\nAll assertions passed (steps 1-7) — capture, re-label, CLV, monitor, "
          "and all three Phase 7.3 gaps (close-window floor, clock dedup, budget "
          "reservation) behave exactly as the design spec requires, on fixtures. "
          "No number above came from a live sportsbook -- this is fixture/synthetic "
          "evidence only, per the Checkpoint B requirement.")


if __name__ == "__main__":
    main()

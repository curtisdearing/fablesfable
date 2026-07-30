#!/usr/bin/env python3
"""Real-line reliability + CLV backtest — the report the whole roadmap waits on.

Every historical prop hit-rate in this repo is graded at synthetic reference
lines and says so.  THIS report reads only real-market tables that accrue
during live weeks — ``line_open_close`` (every snapshotted market's open and
close, PR #7), ``leans``/``lean_outcomes`` (published picks and their grades),
and ``clv`` (entry vs same-side consensus close) — and produces the honest
scoreboard: market movement structure, reliability of published probabilities
against real outcomes, and the pre-committed CLV/kill verdict.

Fail-closed by construction: with thin or empty tables every section reports
``insufficient_data`` with the exact n it has and the n it needs — it never
extrapolates, never substitutes synthetic numbers, and never flips the
kill-check's verdict logic (that stays in ``nflvalue/killcheck.py``).

Run (any time; it is safe offseason):
    python3 analysis/real_line_backtest.py [--db data/nfl_props.db]

Writes book/real_line_backtest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import db as dbmod                    # noqa: E402
from nflvalue import killcheck                      # noqa: E402
from nflvalue.calibration import binary_calibration  # noqa: E402
from nflvalue.clv import rolling_clv                 # noqa: E402

BOOK_PATH = os.path.join(ROOT, "book", "real_line_backtest.json")

MIN_MOVEMENT_ROWS = 100     # per-market movement stats below this: report n only
MIN_RELIABILITY_N = 100     # protocol matched-control floor reused for bins
MIN_CLV_RESOLVED = 150      # accuracy_protocol.forward_clv.minimum_resolved


def coverage(conn) -> dict:
    df = dbmod.query_df(conn, "SELECT season, week, market, open_ts, close_ts "
                              "FROM line_open_close")
    if df.empty:
        return {"status": "insufficient_data", "n_rows": 0,
                "note": "line_open_close is empty — accrues on live weekly runs"}
    both = df["open_ts"].notna() & df["close_ts"].notna()
    return {
        "status": "ok",
        "n_rows": int(len(df)),
        "seasons_weeks": sorted({(int(s), int(w)) for s, w in
                                 zip(df["season"], df["week"])}),
        "markets": sorted(df["market"].dropna().unique().tolist()),
        "share_with_open_and_close": round(float(both.mean()), 4),
    }


def movement(conn) -> dict:
    df = dbmod.query_df(conn, "SELECT market, open_prob, close_prob, prob_moved, "
                              "point_moved FROM line_open_close "
                              "WHERE open_prob IS NOT NULL AND close_prob IS NOT NULL")
    if df.empty:
        return {"status": "insufficient_data", "n_rows": 0}
    out = {"status": "ok", "n_rows": int(len(df)), "per_market": {}}
    for market, g in df.groupby("market"):
        entry = {"n": int(len(g))}
        if len(g) >= MIN_MOVEMENT_ROWS:
            entry.update({
                "mean_abs_prob_move": round(float(g["prob_moved"].abs().mean()), 4),
                "p90_abs_prob_move": round(float(g["prob_moved"].abs().quantile(0.9)), 4),
                "mean_abs_point_move": (round(float(g["point_moved"].abs().mean()), 3)
                                        if g["point_moved"].notna().any() else None),
                "share_unmoved": round(float((g["prob_moved"].abs() < 1e-9).mean()), 4),
            })
        else:
            entry["note"] = f"below {MIN_MOVEMENT_ROWS}-row floor; stats withheld"
        out["per_market"][str(market)] = entry
    return out


def reliability(conn) -> dict:
    """Published lean probabilities vs real graded outcomes (REAL lines only)."""
    df = dbmod.query_df(conn, """
        SELECT l.p_side AS p, o.hit AS y, l.market
        FROM leans l
        JOIN lean_outcomes o
          ON l.season = o.season AND l.week = o.week AND l.clock = o.clock
         AND l.game_id = o.game_id AND l.player_id = o.player_id
         AND l.market = o.market AND l.side = o.side
        WHERE l.line_source != 'synthetic' AND o.hit IN (0, 1)
              AND l.p_side IS NOT NULL""")
    if len(df) < MIN_RELIABILITY_N:
        return {"status": "insufficient_data", "n_resolved_real_line_leans": int(len(df)),
                "needs": MIN_RELIABILITY_N,
                "note": "reliability on real lines waits for graded real-line leans"}
    cal = binary_calibration(df["y"].tolist(), df["p"].tolist(), bins=10)
    return {"status": "ok", "n": int(len(df)), "calibration": cal}


def clv_and_kill(conn) -> dict:
    clv = rolling_clv(conn)
    kill = killcheck.report(conn)
    resolved = clv.get("n") or 0
    verdict = {"resolved": int(resolved), "needs": MIN_CLV_RESOLVED}
    if resolved < MIN_CLV_RESOLVED:
        verdict["status"] = "insufficient_data"
        verdict["note"] = (f"{resolved}/{MIN_CLV_RESOLVED} resolved leans — "
                          "the only edge test that matters is still accruing")
    else:
        verdict["status"] = "ok"
    return {"clv": clv, "killcheck": kill, "gate_state": verdict}


def build_report(db_path=None) -> dict:
    conn = dbmod.connect(db_path) if db_path else dbmod.connect()
    try:
        report = {
            "note": ("REAL lines only. Synthetic-line results live elsewhere and "
                     "are labeled as such; nothing in this file is graded at a "
                     "synthetic number."),
            "coverage": coverage(conn),
            "movement": movement(conn),
            "reliability": reliability(conn),
            "clv": clv_and_kill(conn),
        }
    finally:
        conn.close()
    sections = [report["coverage"], report["movement"], report["reliability"],
                report["clv"]["gate_state"]]
    n_ready = sum(1 for s in sections if s.get("status") == "ok")
    report["summary"] = {
        "sections_ready": f"{n_ready}/4",
        "headline": ("REAL-LINE RECORD ACCRUING — no honest real-line verdict yet"
                     if n_ready < 4 else
                     "real-line record populated — see killcheck verdict"),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    report = build_report(args.db)
    os.makedirs(os.path.dirname(BOOK_PATH), exist_ok=True)
    with open(BOOK_PATH, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps(report["summary"], indent=1))
    for name in ("coverage", "movement", "reliability"):
        s = report[name]
        print(f"{name}: {s.get('status')} (n={s.get('n_rows', s.get('n', s.get('n_resolved_real_line_leans', 0)))})")
    print(f"clv: {report['clv']['gate_state']['status']} "
          f"({report['clv']['gate_state']['resolved']}/{MIN_CLV_RESOLVED} resolved)")
    print(f"wrote {BOOK_PATH}")


if __name__ == "__main__":
    main()

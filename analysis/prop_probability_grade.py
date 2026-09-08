#!/usr/bin/env python3
"""Uncertainty grade for the prop decision probability layer.

Run: python3 analysis/prop_probability_grade.py [--db data/nfl_props.db]

What this is: a calibration/Brier audit of the DISTRIBUTION probability that
was attached to every graded lean (``leans.p_side``) against the graded
outcome (``lean_outcomes.hit``), with a cluster bootstrap for uncertainty.

What this is NOT: a performance, profit, ROI, or market-edge claim. Under
``analysis/accuracy_protocol.json`` only leans priced at a REAL line whose
market-quality gate was open (``market_state='REAL_MARKET'``) can count as
prospective real-line evidence, and only once ``MIN_PROSPECTIVE_N`` of them
have resolved. Synthetic/reference-line rows are research-only regression
material; an empty or thin real-line record reports insufficiency.

Row hygiene (every rejection is counted and reported, never silently dropped):
  * one decision per (season, week, game, player, market, side): the T-90
    re-rank of a Wednesday lean is the same decision, not a second sample --
    the latest clock wins;
  * voided leans are excluded (they were never a live decision);
  * a real-line row whose stored market_state is not REAL_MARKET is
    ``real_line_context_only`` (one-book / killcheck rows), not evidence;
  * a NULL market_state on a real-line row (written before the gate existed)
    is ``real_line_ungated`` and is reported but not promoted;
  * p_side outside [0, 1], non-finite, or a non-binary hit is ``invalid``.

Uncertainty: the protocol's paired resampling unit is season-week, and
decisions within a week share games, players, injuries and weather. The
bootstrap therefore resamples WEEKS (clusters), not rows. With fewer than
``MIN_CLUSTERS`` weeks the interval is reported as unavailable rather than
computed from an i.i.d. assumption that does not hold.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nflvalue import db as dbmod  # noqa: E402
from nflvalue.calibration import binary_calibration  # noqa: E402

MIN_PROSPECTIVE_N = 100
MIN_CLUSTERS = 5
CLOCK_ORDER = {"wed": 0, "t90": 1}


def _cluster_interval(values: np.ndarray, clusters: np.ndarray, bootstrap_n: int,
                      seed: int) -> Dict:
    """95% interval for the mean of ``values`` by resampling clusters."""
    if not len(values):
        return {"low": None, "high": None, "n_clusters": 0, "method": "unavailable"}
    labels, inv = np.unique(clusters, return_inverse=True)
    k = len(labels)
    if k < MIN_CLUSTERS:
        return {"low": None, "high": None, "n_clusters": int(k),
                "method": f"unavailable (<{MIN_CLUSTERS} season-week clusters)"}
    rng = np.random.default_rng(seed)
    sums = np.bincount(inv, weights=values, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    means = np.empty(bootstrap_n)
    for i in range(bootstrap_n):
        pick = rng.integers(0, k, k)
        means[i] = sums[pick].sum() / counts[pick].sum()
    return {"low": round(float(np.quantile(means, 0.025)), 6),
            "high": round(float(np.quantile(means, 0.975)), 6),
            "n_clusters": int(k), "method": "cluster bootstrap by season-week"}


def _finite01(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def classify_rows(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """Bucket raw rows -> {'real_line', 'real_line_context_only',
    'real_line_ungated', 'synthetic', 'invalid', 'voided', 'superseded'}."""
    buckets: Dict[str, List[Dict]] = {k: [] for k in (
        "real_line", "real_line_context_only", "real_line_ungated", "synthetic",
        "invalid", "voided", "superseded")}
    # one decision per key; the later clock supersedes
    best: Dict[tuple, Dict] = {}
    for row in rows:
        key = (row.get("season"), row.get("week"), row.get("game_id"),
               row.get("player_id"), row.get("market"), row.get("side"))
        if any(k is None for k in key[:5]):
            buckets["invalid"].append({**row, "reject_reason": "missing key field"})
            continue
        if str(row.get("status") or "active") != "active":
            buckets["voided"].append(row)
            continue
        cur = best.get(key)
        if cur is None or CLOCK_ORDER.get(row.get("clock"), -1) > CLOCK_ORDER.get(cur.get("clock"), -1):
            if cur is not None:
                buckets["superseded"].append(cur)
            best[key] = row
        else:
            buckets["superseded"].append(row)
    for row in best.values():
        p = _finite01(row.get("p_side"))
        try:
            y = int(row.get("hit"))
        except (TypeError, ValueError):
            y = None
        if p is None or y not in (0, 1):
            buckets["invalid"].append({**row, "reject_reason": "p_side not in [0,1] or hit not binary"})
            continue
        clean = {**row, "p_side": p, "hit": y}
        if row.get("line_source") != "odds_api":
            buckets["synthetic"].append(clean)
        elif row.get("market_state") is None:
            buckets["real_line_ungated"].append(clean)
        elif row.get("market_state") == "REAL_MARKET":
            buckets["real_line"].append(clean)
        else:
            buckets["real_line_context_only"].append(clean)
    return buckets


def _score(sample: List[Dict], bootstrap_n: int, seed: int) -> Dict:
    p = np.asarray([r["p_side"] for r in sample], dtype=float)
    y = np.asarray([r["hit"] for r in sample], dtype=float)
    clusters = np.asarray([f"{r.get('season')}-{r.get('week')}" for r in sample])
    losses = (p - y) ** 2
    return {
        "n": int(len(sample)),
        "calibration": binary_calibration(y, p, bins=10),
        "brier_95_interval": _cluster_interval(losses, clusters, bootstrap_n, seed),
        "hit_rate_95_interval": _cluster_interval(y, clusters, bootstrap_n, seed + 1),
    }


def grade_rows(rows, *, bootstrap_n: int = 4000, seed: int = 20260907) -> dict:
    """Grade graded-lean rows [{p_side, hit, line_source, market_state, season,
    week, game_id, player_id, market, side, clock, status}] -> report."""
    b = classify_rows(list(rows))
    real, synthetic = b["real_line"], b["synthetic"]
    if len(real) >= MIN_PROSPECTIVE_N:
        evidence_kind = "prospective_real_line"
        claim = ("Prospective real-line calibration of the distribution probability "
                 "(no ROI/profit/edge claim; CLV is the market judge)")
    elif real:
        evidence_kind = "real_line_insufficient"
        claim = (f"Real-line record too thin ({len(real)} < {MIN_PROSPECTIVE_N}); "
                 "no performance claim")
    elif synthetic:
        evidence_kind = "synthetic_research_only"
        claim = "No real-line record; synthetic/reference-line rows are research-only"
    else:
        evidence_kind = "no_evidence"
        claim = "No graded decisions; nothing to grade"
    denominators = {
        "rows_received": int(len(rows)),
        "graded_real_line": len(real),
        "real_line_context_only": len(b["real_line_context_only"]),
        "real_line_ungated_pre_gate_rows": len(b["real_line_ungated"]),
        "synthetic_or_reference": len(synthetic),
        "voided": len(b["voided"]),
        "superseded_duplicate_clock": len(b["superseded"]),
        "invalid": len(b["invalid"]),
        "prospective_floor": MIN_PROSPECTIVE_N,
    }
    report = {
        "evidence_kind": evidence_kind,
        "claim": claim,
        "probability_source": "distribution (mean/sd/line/dist); not an ML score; "
                              "agreement with the distribution is not calibration",
        "coverage": denominators,
        "invalid_rows": [{k: r.get(k) for k in ("season", "week", "game_id", "player_id",
                                                "market", "side", "clock", "reject_reason")}
                         for r in b["invalid"][:50]],
        "uncertainty": {"method": "cluster bootstrap by season-week (protocol paired "
                                  "resampling unit); unavailable below "
                                  f"{MIN_CLUSTERS} clusters",
                        "bootstrap_n": bootstrap_n, "seed": seed},
        "real_line": _score(real, bootstrap_n, seed) if real else None,
        "synthetic_research_only": _score(synthetic, bootstrap_n, seed + 100) if synthetic else None,
    }
    # the headline block is whatever the evidence_kind names; nothing else
    # is promoted into it
    head = report["real_line"] if real else (report["synthetic_research_only"] if synthetic else None)
    report["calibration"] = head["calibration"] if head else binary_calibration([], [], bins=10)
    report["uncertainty"]["brier_95_interval"] = (
        [head["brier_95_interval"]["low"], head["brier_95_interval"]["high"]] if head else [None, None])
    report["uncertainty"]["hit_rate_95_interval"] = (
        [head["hit_rate_95_interval"]["low"], head["hit_rate_95_interval"]["high"]] if head else [None, None])
    report["uncertainty"]["n_clusters"] = head["brier_95_interval"]["n_clusters"] if head else 0
    return report


def load_rows(db_path=None):
    conn = dbmod.connect(db_path) if db_path else dbmod.connect()
    try:
        df = dbmod.query_df(conn, """SELECT l.season, l.week, l.clock, l.game_id, l.player_id,
            l.market, l.side, l.status, l.p_side, l.line_source, l.market_state, o.hit
            FROM leans l JOIN lean_outcomes o
            ON l.season=o.season AND l.week=o.week AND l.clock=o.clock
            AND l.game_id=o.game_id AND l.player_id=o.player_id AND l.market=o.market
            AND l.side=o.side WHERE l.p_side IS NOT NULL""")
        return df.to_dict("records")
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None)
    ap.add_argument("--output", default=os.path.join(ROOT, "book", "prop_probability_grade.json"))
    ap.add_argument("--bootstrap-n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()
    report = grade_rows(load_rows(args.db), bootstrap_n=args.bootstrap_n, seed=args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

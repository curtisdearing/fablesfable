#!/usr/bin/env python3
"""Uncertainty grade for the prop decision probability layer.

Run: python3 analysis/prop_probability_grade.py [--db data/nfl_props.db]
     [--kickoffs historical/lines_extra.parquet]

What this is: a calibration/Brier audit of the DISTRIBUTION probability that
was attached to every graded lean (``leans.p_side``) against the settled
outcome (``lean_outcomes.hit`` under the settlement contract), with a cluster
bootstrap for uncertainty.

What this is NOT: a performance, profit, ROI, or market-edge claim.

Two labels that must never be confused:

* ``market_state == 'REAL_MARKET'`` is a MARKET-QUALITY label (two books,
  valid prices, coherent probability).  It says nothing about WHEN the
  decision was captured.
* provenance ``prospective`` means the decision's own creation stamp
  (``leans.created_at``, else ``as_of``) is an explicit-offset timestamp
  strictly BEFORE the game's kickoff.  ``retrospective`` means at/after
  kickoff; ``unknown`` means no kickoff or no parseable stamp.  Only
  prospective real-line rows count toward the prospective floor.  Row count
  never upgrades unknown or retrospective rows.

Population accounting (every recommendation row is counted exactly once):
  rows_received; missing_probability (p_side NULL/NaN); invalid (bad key,
  non-binary hit, probability outside [0,1]); void; push; unresolved
  (settlement unresolved, or no outcome row yet); superseded (dedup);
  eligible (graded real or synthetic); graded_real_line /
  graded_real_line_prospective; real_line_context_only; real_line_ungated;
  synthetic_or_reference.

Deduplication policy (declared, tested): one decision per
(season, week, game_id, player_id, market).  A T-90 re-decision supersedes
the Wednesday decision for the same key WHATEVER its side or line (a flipped
side or moved line at T-90 is the same decision, re-made); among rows on the
same clock the latest ``created_at`` wins (repeated invocation).  Superseded
rows are counted with their reason so the policy cannot silently improve the
record: a superseded Wednesday row that would have lost is still visible.

Outcome parsing rejects anything that is not exactly binary BEFORE numeric
coercion: 0.7, NaN, inf, "yes", 2, None are invalid, never truncated.

Uncertainty: the protocol's paired resampling unit is season-week; the
bootstrap resamples weeks.  Below ``MIN_CLUSTERS`` weeks the interval is
reported as unavailable.  ``bootstrap_n`` must be a positive int and
``seed`` an int; anything else is rejected.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
DEDUP_POLICY = ("one decision per (season, week, game_id, player_id, market): the later clock "
                "(t90 > wed) supersedes regardless of side or line; same clock -> latest created_at "
                "wins; superseded rows are counted with a reason, never dropped")
SETTLED = ("win", "loss")
NON_GRADED_SETTLEMENTS = ("push", "void", "unresolved")


# ------------------------------------------------------------------ parsing --
def parse_binary_outcome(value) -> Optional[int]:
    """Exactly 0 or 1 (int, bool, 0.0/1.0, or the strings '0'/'1'); else None.

    Deliberately NOT ``int(value)``: int(0.7) == 0 would turn a corrupt row
    into a loss, int(True) hides a type confusion, int(float('inf')) raises.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value) if value in (0, 1) else None
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return None
        return int(value) if value in (0.0, 1.0) else None
    if isinstance(value, str):
        text = value.strip()
        return int(text) if text in ("0", "1") else None
    return None


def _finite01(x) -> Optional[float]:
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def parse_aware(value) -> Optional[dt.datetime]:
    """ISO-8601 with an explicit offset -> aware UTC datetime; naive/unparseable -> None."""
    if value is None or isinstance(value, (int, float, bool)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.timezone.utc)


def provenance(row: Dict) -> str:
    kickoff = parse_aware(row.get("kickoff"))
    created = parse_aware(row.get("created_at")) or parse_aware(row.get("as_of"))
    if kickoff is None or created is None:
        return "unknown"
    return "prospective" if created < kickoff else "retrospective"


# --------------------------------------------------------------- classify --
def classify_rows(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """Bucket raw rows; see the module docstring for the population contract."""
    buckets: Dict[str, List[Dict]] = {k: [] for k in (
        "real_line", "real_line_context_only", "real_line_ungated", "synthetic",
        "invalid", "voided", "superseded", "missing_probability", "push", "void", "unresolved")}
    best: Dict[tuple, Dict] = {}
    for row in rows:
        key = (row.get("season"), row.get("week"), row.get("game_id"), row.get("player_id"), row.get("market"))
        if any(k is None for k in key):
            buckets["invalid"].append({**row, "reject_reason": "missing key field"})
            continue
        cur = best.get(key)
        if cur is None:
            best[key] = row
            continue
        new_clock, cur_clock = CLOCK_ORDER.get(row.get("clock"), -1), CLOCK_ORDER.get(cur.get("clock"), -1)
        if new_clock != cur_clock:
            winner, loser = (row, cur) if new_clock > cur_clock else (cur, row)
            reason = "clock_supersession"
            if winner.get("side") != loser.get("side"):
                reason += "+side_change"
            if winner.get("line") != loser.get("line"):
                reason += "+line_change"
        else:
            new_c, cur_c = str(row.get("created_at") or ""), str(cur.get("created_at") or "")
            winner, loser = (row, cur) if new_c >= cur_c else (cur, row)
            reason = "repeat_invocation"
        best[key] = winner
        buckets["superseded"].append({**loser, "superseded_reason": reason})
    for row in best.values():
        status = str(row.get("status") or "active")
        settlement = row.get("settlement")
        if status != "active" or settlement == "void":
            buckets["void" if settlement == "void" else "voided"].append(row)
            continue
        if settlement == "push":
            buckets["push"].append(row)
            continue
        if settlement == "unresolved" or (settlement is None and row.get("hit") is None):
            buckets["unresolved"].append(row)
            continue
        p = _finite01(row.get("p_side"))
        if row.get("p_side") is None or (isinstance(row.get("p_side"), float) and math.isnan(row["p_side"])):
            buckets["missing_probability"].append(row)
            continue
        y = parse_binary_outcome(row.get("hit"))
        if p is None or y is None:
            buckets["invalid"].append({**row, "reject_reason": "p_side not in [0,1] or hit not exactly binary"})
            continue
        clean = {**row, "p_side": p, "hit": y, "provenance": provenance(row),
                 "settlement_basis": settlement if settlement in SETTLED else "legacy_binary"}
        if row.get("line_source") != "odds_api":
            buckets["synthetic"].append(clean)
        elif row.get("market_state") is None:
            buckets["real_line_ungated"].append(clean)
        elif row.get("market_state") == "REAL_MARKET":
            buckets["real_line"].append(clean)
        else:
            buckets["real_line_context_only"].append(clean)
    return buckets


# ---------------------------------------------------------------- scoring --
def _cluster_interval(values: np.ndarray, clusters: np.ndarray, bootstrap_n: int, seed: int) -> Dict:
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
    pick = rng.integers(0, k, size=(bootstrap_n, k))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return {"low": round(float(np.quantile(means, 0.025)), 6),
            "high": round(float(np.quantile(means, 0.975)), 6),
            "n_clusters": int(k), "method": "cluster bootstrap by season-week"}


def _score(sample: List[Dict], bootstrap_n: int, seed: int) -> Dict:
    p = np.asarray([r["p_side"] for r in sample], dtype=float)
    y = np.asarray([r["hit"] for r in sample], dtype=float)
    clusters = np.asarray([f"{r.get('season')}-{r.get('week')}" for r in sample])
    losses = (p - y) ** 2
    return {"n": int(len(sample)), "calibration": binary_calibration(y, p, bins=10),
            "brier_95_interval": _cluster_interval(losses, clusters, bootstrap_n, seed),
            "hit_rate_95_interval": _cluster_interval(y, clusters, bootstrap_n, seed + 1),
            "provenance": {k: sum(1 for r in sample if r.get("provenance") == k)
                           for k in ("prospective", "retrospective", "unknown")}}


def _validate_args(bootstrap_n, seed) -> None:
    if isinstance(bootstrap_n, bool) or not isinstance(bootstrap_n, (int, np.integer)) or bootstrap_n < 1:
        raise ValueError(f"bootstrap_n must be a positive int, got {bootstrap_n!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an int, got {seed!r}")


def grade_rows(rows, *, bootstrap_n: int = 4000, seed: int = 20260907) -> dict:
    """Grade recommendation rows -> report (see module docstring)."""
    _validate_args(bootstrap_n, seed)
    rows = list(rows)
    b = classify_rows(rows)
    real, synthetic = b["real_line"], b["synthetic"]
    real_prospective = [r for r in real if r["provenance"] == "prospective"]
    if len(real_prospective) >= MIN_PROSPECTIVE_N:
        evidence_kind = "prospective_real_line"
        claim = ("Prospective real-line calibration of the distribution probability "
                 "(no ROI/profit/edge claim; CLV is the market judge)")
    elif real:
        evidence_kind = "real_line_insufficient"
        claim = (f"Real-line record too thin or not provably prospective "
                 f"({len(real_prospective)} prospective of {len(real)} graded < {MIN_PROSPECTIVE_N}); "
                 "no performance claim")
    elif synthetic:
        evidence_kind = "synthetic_research_only"
        claim = "No real-line record; synthetic/reference-line rows are research-only"
    else:
        evidence_kind = "no_evidence"
        claim = "No graded decisions; nothing to grade"
    prov_counts = {k: sum(1 for r in real + synthetic + b["real_line_ungated"] + b["real_line_context_only"]
                          if r.get("provenance") == k) for k in ("prospective", "retrospective", "unknown")}
    denominators = {
        "rows_received": int(len(rows)),
        "eligible": len(real) + len(synthetic),
        "graded_real_line": len(real),
        "graded_real_line_prospective": len(real_prospective),
        "real_line_context_only": len(b["real_line_context_only"]),
        "real_line_ungated_pre_gate_rows": len(b["real_line_ungated"]),
        "synthetic_or_reference": len(synthetic),
        "missing_probability": len(b["missing_probability"]),
        "push": len(b["push"]),
        "void": len(b["void"]) + len(b["voided"]),
        "unresolved": len(b["unresolved"]),
        "voided": len(b["voided"]),
        "superseded_duplicate_decision": len(b["superseded"]),
        "superseded_reasons": _counts([r["superseded_reason"] for r in b["superseded"]]),
        "invalid": len(b["invalid"]),
        "provenance_prospective": prov_counts["prospective"],
        "provenance_retrospective": prov_counts["retrospective"],
        "provenance_unknown": prov_counts["unknown"],
        "prospective_floor": MIN_PROSPECTIVE_N,
    }
    report = {
        "evidence_kind": evidence_kind,
        "claim": claim,
        "probability_source": "distribution (mean/sd/line/dist); not an ML score; "
                              "agreement with the distribution is not calibration",
        "market_state_is_not_provenance": ("REAL_MARKET is a market-quality label; only a pre-kickoff "
                                           "creation stamp makes a row prospective"),
        "dedup_policy": DEDUP_POLICY,
        "settlement_policy": "graded rows are settled non-push outcomes (win/loss) or legacy binary rows; "
                             "push/void/unresolved are counted, never graded",
        "coverage": denominators,
        "invalid_rows": [{k: r.get(k) for k in ("season", "week", "game_id", "player_id",
                                                "market", "side", "clock", "reject_reason")}
                         for r in b["invalid"][:50]],
        "uncertainty": {"method": "cluster bootstrap by season-week (protocol paired resampling unit); "
                                  f"unavailable below {MIN_CLUSTERS} clusters",
                        "bootstrap_n": int(bootstrap_n), "seed": int(seed)},
        "real_line": _score(real, bootstrap_n, seed) if real else None,
        "real_line_prospective": _score(real_prospective, bootstrap_n, seed + 50) if real_prospective else None,
        "synthetic_research_only": _score(synthetic, bootstrap_n, seed + 100) if synthetic else None,
    }
    head = (report["real_line_prospective"] if evidence_kind == "prospective_real_line"
            else report["real_line"] if real else (report["synthetic_research_only"] if synthetic else None))
    report["calibration"] = head["calibration"] if head else binary_calibration([], [], bins=10)
    report["uncertainty"]["brier_95_interval"] = (
        [head["brier_95_interval"]["low"], head["brier_95_interval"]["high"]] if head else [None, None])
    report["uncertainty"]["hit_rate_95_interval"] = (
        [head["hit_rate_95_interval"]["low"], head["hit_rate_95_interval"]["high"]] if head else [None, None])
    report["uncertainty"]["n_clusters"] = head["brier_95_interval"]["n_clusters"] if head else 0
    return report


def _counts(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ loading --
def load_kickoffs(path: Optional[str]) -> Dict[str, str]:
    """{game_id: kickoff ISO-UTC} from a schedule parquet (gameday + Eastern gametime)."""
    if not path or not os.path.exists(path):
        return {}
    import pandas as pd
    from zoneinfo import ZoneInfo
    s = pd.read_parquet(path)
    out: Dict[str, str] = {}
    for r in s.itertuples(index=False):
        gameday, gametime = getattr(r, "gameday", None), getattr(r, "gametime", None)
        if not gameday or not gametime:
            continue
        try:
            local = dt.datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M")
            local = local.replace(tzinfo=ZoneInfo("America/New_York"))
        except ValueError:
            continue
        out[str(r.game_id)] = local.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def load_rows(db_path=None, kickoffs: Optional[Dict[str, str]] = None):
    """Every recommendation row (LEFT JOIN outcomes) with provenance fields attached."""
    conn = dbmod.connect(db_path) if db_path else dbmod.connect()
    try:
        df = dbmod.query_df(conn, """SELECT l.season, l.week, l.clock, l.game_id, l.player_id,
            l.market, l.side, l.line, l.status, l.p_side, l.line_source, l.market_state,
            l.as_of, l.created_at, o.hit, o.settlement, o.graded_at
            FROM leans l LEFT JOIN lean_outcomes o
            ON l.season=o.season AND l.week=o.week AND l.clock=o.clock
            AND l.game_id=o.game_id AND l.player_id=o.player_id AND l.market=o.market""")
        rows = df.to_dict("records")
        kickoffs = kickoffs or {}
        for r in rows:
            r["kickoff"] = kickoffs.get(str(r.get("game_id")))
            for k in ("hit", "settlement", "p_side"):
                if isinstance(r.get(k), float) and math.isnan(r[k]):
                    r[k] = None
        return rows
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None)
    ap.add_argument("--kickoffs", default=os.path.join(ROOT, "historical", "lines_extra.parquet"))
    ap.add_argument("--output", default=os.path.join(ROOT, "book", "prop_probability_grade.json"))
    ap.add_argument("--bootstrap-n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()
    report = grade_rows(load_rows(args.db, load_kickoffs(args.kickoffs)),
                        bootstrap_n=args.bootstrap_n, seed=args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

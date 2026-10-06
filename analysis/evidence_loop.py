#!/usr/bin/env python3
"""Cumulative result-to-research evidence loop (workstream D).

    python -m analysis.evidence_loop freeze  [--registry analysis/research_registry.json]
    python -m analysis.evidence_loop run --historical DIR [--real-lines CSV] [--issued-grade JSON]
                                         [--registry ...] --output OUT.json

Reads immutable issued/offered-line records and native final outcomes, scores
them, and screens at most two registered mechanism challengers under criteria
frozen in ``research_registry.json`` before any challenger number existed.
Nothing here writes a production weight, calibration, default, or line.

Four quantities that are never pooled:

* forecast calibration -- Brier, log loss, ECE/reliability, per-market n,
  game-clustered bootstrap intervals;
* performance (point) error -- reported by the settlement grader, not here;
* hypothetical ROI -- flat 1u at the ACTUAL offered decimal price; ``None``
  with a reason when the source carries no price receipt;
* actual account P/L -- ``None`` unless a ticket ledger is supplied (none is).

Chronology (``window_of``): a row counts toward prospective confirmation only
if it is 2026 week >= 5, its decision clock is an explicit-offset timestamp
strictly before kickoff and it carries an immutable capture sha256.  2026
weeks 1-4 were already read and are retrospective whatever their clocks say.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue.calibration import binary_calibration  # noqa: E402

REGISTRY_PATH = os.path.join(ROOT, "analysis", "research_registry.json")
PROSPECTIVE_START = (2026, 5)
MAX_ACTIVE = 2
ACTIVE = "active_prospective"
MIN_CLUSTERS = 5
P_CLIP = (0.02, 0.98)
WINDOWS = ("prospective_confirmation", "retrospective_exploratory", "excluded_late", "unknown_clock")


# ------------------------------------------------------------- chronology --
def parse_aware(value) -> Optional[dt.datetime]:
    """Explicit-offset timestamps only; a naive clock is unknown, never assumed UTC."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TRUSTED_CAPTURE = "ledger_pre_kickoff_event"
EVIDENCE_SECTIONS = ("recommendations_given", "delivered_historical_import", "watch_published", "retrospective")


def trusted_capture(row: Dict, kick: Optional[dt.datetime] = None) -> bool:
    """Is there trusted evidence the row was captured before kickoff?

    Only the issued ledger's own clock counts: the receipt digest (64-hex sha256) of the first
    pre-kickoff stage event, whose ledger ``capture_recorded_at`` precedes kickoff, on a record
    that is not a historical/postgame import. A content identifier, a source's claim that it
    was pregame or a backdated ``decision_ts`` never qualifies on its own."""
    kick = kick or parse_aware(row.get("kickoff"))
    seen = parse_aware(row.get("capture_recorded_at"))
    return (row.get("capture_basis") == TRUSTED_CAPTURE and not row.get("historical_import")
            and bool(SHA256_RE.match(str(row.get("capture_sha256") or "")))
            and seen is not None and kick is not None and seen < kick)


def window_of(row: Dict) -> str:
    decided, kick = parse_aware(row.get("decision_ts")), parse_aware(row.get("kickoff"))
    if decided is None or kick is None:
        return "unknown_clock"
    if decided >= kick:
        return "excluded_late"
    if (int(row["season"]), int(row["week"])) >= PROSPECTIVE_START and trusted_capture(row, kick):
        return "prospective_confirmation"
    # includes historically pregame decisions imported after the fact: graded honestly, never confirmation
    return "retrospective_exploratory"


def unused_confirmation(row: Dict, frozen_at: Optional[str]) -> bool:
    """Untouched evidence for a challenger frozen at ``frozen_at``: a prospective row whose trusted
    capture postdates the freeze. Earlier issued forecasts stay gradable but were already seen."""
    frozen = parse_aware(frozen_at)
    seen = parse_aware(row.get("capture_recorded_at"))
    return (window_of(row) == "prospective_confirmation" and frozen is not None and seen is not None
            and seen > frozen)


# --------------------------------------------------------------- adapters --
def american_to_decimal(price) -> Optional[float]:
    try:
        a = float(price)
    except (TypeError, ValueError):
        return None
    if math.isnan(a) or -100 < a < 100:
        return None
    return 1 + (a / 100 if a > 0 else 100 / -a)


def _outcome(settlement) -> str:
    s = str(settlement or "").strip().lower()
    return {"win": "win", "loss": "loss", "push": "push", "void": "void"}.get(s, "pending")


def _issued_row(g: Dict, section: str) -> Dict:
    price = g.get("quote_price")
    pre = section in ("recommendations_given", "watch_published")
    receipt = g.get("capture_receipt_sha256") if pre else None
    return {
        "evidence_id": g["record_id"] if section == "recommendations_given" else f"{g['record_id']}:{section}",
        "source": "issued_ledger", "record_id": g["record_id"], "section": section,
        "season": g["season"], "week": g["week"],
        "game_id": g.get("game_id"), "market": g["market"], "side": g.get("side"), "line": g.get("line"),
        "issued_class": "recommendation" if g.get("pick_class") == "recommendation" else str(g.get("pick_class")),
        "tier": g.get("tier"), "policy_class": g.get("policy_class"),
        "decision_ts": g.get("decision_ts"), "kickoff": g.get("kickoff"),
        # trusted capture = receipt of the first pre-kickoff ledger event + its ledger clock;
        # record_id is a content id and proves nothing about timing
        "capture_sha256": receipt, "capture_recorded_at": g.get("first_seen_in_ledger") if pre else None,
        "capture_basis": (TRUSTED_CAPTURE if receipt else
                          "historical_import" if g.get("historical_import") else "no_pre_kickoff_ledger_event"),
        "historical_import": bool(g.get("historical_import")), "original_issue_ts": g.get("original_issue_ts"),
        "delivery_evidence_kind": g.get("delivery_evidence_kind"),
        "outcome": _outcome(g.get("settlement")), "actual": g.get("actual"),
        "actuals_sha256": g.get("actuals_sha256"),
        "price_decimal": float(price) if price else None, "book": g.get("quote_book"),
        "p": {"issued": g.get("model_p_side")}}


def from_issued_grading(grade_output: Dict, section: str = "recommendations_given") -> List[Dict]:
    """Adapter for ``nflvalue.issued_grading.grade()`` output (settlement -> analysis contract)."""
    return [_issued_row(g, section) for g in grade_output["sections"].get(section, {}).get("rows", [])]


def from_results_rows(rows: Iterable[Dict]) -> List[Dict]:
    """Adapter for ``nflvalue.issued_results.current()``: the latest persisted grade per record and
    section (earlier revisions stay in the results table). Only verified final-box grades exist there."""
    return [_issued_row(r, r["section"]) for r in rows
            if r.get("section") in EVIDENCE_SECTIONS and r.get("actuals_sha256")]


def from_card_grade(grade_json: Dict, season: int, week: int, game_id: str, kickoff: Optional[str]) -> List[Dict]:
    """Adapter for a hand-graded delivered card (rows with result WIN/LOSS/PUSH/PASS).

    Such cards carry no machine decision clock, so every row is ``unknown_clock``:
    a chat record is never relabelled as prospectively auto-ingested."""
    out = []
    for r in grade_json["rows"]:
        is_rec = bool(r.get("count_in_recommendation_record"))
        result = str(r.get("result")).upper()
        outcome = _outcome(r.get("threshold_outcome") if result == "PASS" else result)
        out.append({
            "evidence_id": r["pick_id"], "source": "delivered_card_grade", "season": season, "week": week,
            "game_id": game_id, "market": r.get("category"), "side": r.get("side"), "line": r.get("line"),
            "issued_class": "recommendation" if is_rec else "pass", "tier": r.get("issued_status"),
            "decision_ts": None, "kickoff": kickoff, "capture_sha256": None, "outcome": outcome,
            "actual": r.get("actual"), "price_decimal": american_to_decimal(r.get("price")), "book": r.get("book"),
            "p": {"issued": r.get("raw_probability")}})
    return out


REAL_LINE_ARMS = {"A1": "p_A1", "C1m": "p_C1m", "C2": "p_C2", "market": "consensus_p_over"}


def from_real_line_events(df, kickoffs: Dict[str, str]) -> List[Dict]:
    """Adapter for offered-line event rows (one latest-clock quote per player-market-game).

    Probabilities are P(over) for every arm, so the scored side is always Over."""
    rows = []
    for r in df.to_dict("records"):
        y_over = r.get("y_over")
        if y_over is None or (isinstance(y_over, float) and math.isnan(y_over)):
            outcome = "pending"  # never inferred from another column's side convention
        else:
            outcome = "win" if int(y_over) == 1 else "loss"
        rows.append({
            "evidence_id": f"{r['game_id']}|{r['player_id']}|{r['market']}|{r['point']}", "source": "offered_line_event",
            "season": int(r["season"]), "week": int(r["week"]), "game_id": r["game_id"], "market": r["market"],
            "side": "over", "line": float(r["point"]), "issued_class": "candidate_event", "tier": None,
            "decision_ts": r.get("ts"), "kickoff": kickoffs.get(r["game_id"]), "capture_sha256": None,
            "outcome": outcome,
            "price_decimal": float(r["over_price"]), "price_decimal_other": float(r["under_price"]),
            "p": {arm: r.get(col) for arm, col in REAL_LINE_ARMS.items()}})
    return rows


# --------------------------------------------------------- append-only log --
def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def append_evidence(path: str, rows: Iterable[Dict], recorded_at: Optional[str] = None) -> Dict:
    """Append rows to a sha256-chained JSONL ledger.  Same id + same content -> skipped;
    same id + changed content -> appended as a new revision (the old line is never edited)."""
    import fcntl
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # one writer; a second fails loudly
        return _append_locked(path, rows, recorded_at)


def _append_locked(path: str, rows: Iterable[Dict], recorded_at: Optional[str]) -> Dict:
    existing = read_ledger(path)
    latest = {}
    for e in existing:
        latest[e["row"]["evidence_id"]] = e["content_sha256"]
    prev = existing[-1]["entry_sha256"] if existing else "0" * 64
    added = skipped = revised = 0
    stamp = recorded_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "a") as fh:
        for row in rows:
            content = hashlib.sha256(_canon(row).encode()).hexdigest()
            eid = row["evidence_id"]
            if latest.get(eid) == content:
                skipped += 1
                continue
            revised += eid in latest
            entry = {"row": row, "content_sha256": content, "prev_sha256": prev, "recorded_at": stamp,
                     "revision_of": latest.get(eid)}
            entry["entry_sha256"] = hashlib.sha256(_canon({k: entry[k] for k in sorted(entry)}).encode()).hexdigest()
            fh.write(_canon(entry) + "\n")
            prev, latest[eid] = entry["entry_sha256"], content
            added += 1
    return {"appended": added, "skipped_identical": skipped, "revisions": revised}


def read_ledger(path: str, verify: bool = True) -> List[Dict]:
    if not os.path.exists(path):
        return []
    entries, prev = [], "0" * 64
    with open(path) as fh:
        for i, line in enumerate(fh):
            e = json.loads(line)
            if verify:
                body = {k: e[k] for k in sorted(e) if k != "entry_sha256"}
                if e["prev_sha256"] != prev or hashlib.sha256(_canon(body).encode()).hexdigest() != e["entry_sha256"] \
                        or hashlib.sha256(_canon(e["row"]).encode()).hexdigest() != e["content_sha256"]:
                    raise ValueError(f"evidence ledger chain broken at line {i + 1}")
            prev = e["entry_sha256"]
            entries.append(e)
    return entries


def latest_rows(entries: List[Dict]) -> List[Dict]:
    out = {}
    for e in entries:
        out[e["row"]["evidence_id"]] = e["row"]
    return list(out.values())


# ---------------------------------------------------------------- scoring --
def _p(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def coverage(rows: Sequence[Dict], arms: Sequence[str], scope: str = "recommendation") -> Dict:
    """Every received row is counted exactly once in ``by_status``; only ``scored`` rows are graded."""
    by = {"scored": 0, "push": 0, "void": 0, "pending": 0, "missing_probability": 0, "not_in_scope": 0}
    for r in rows:
        if scope != "all" and r.get("issued_class") != scope:
            by["not_in_scope"] += 1
        elif r["outcome"] in ("push", "void", "pending"):
            by[r["outcome"]] += 1
        elif any(_p(r["p"].get(a)) is None for a in arms):
            by["missing_probability"] += 1
        else:
            by["scored"] += 1
    by_window = {w: sum(1 for r in rows if window_of(r) == w) for w in WINDOWS}
    return {"rows_received": len(rows), "by_status": by, "by_window": by_window,
            "missing_price": sum(1 for r in rows if not r.get("price_decimal"))}


def scored_rows(rows: Sequence[Dict], arms: Sequence[str], scope: str = "recommendation") -> List[Dict]:
    return [r for r in rows if (scope == "all" or r.get("issued_class") == scope)
            and r["outcome"] in ("win", "loss") and all(_p(r["p"].get(a)) is not None for a in arms)]


def _losses(p: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": (p - y) ** 2, "logloss": -(y * np.log(pc) + (1 - y) * np.log(1 - pc))}


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> Optional[float]:
    if not len(p):
        return None
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def cluster_boot(values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int) -> Dict:
    """Percentile interval of the row mean, resampling whole clusters; one-sided p = P(boot mean >= 0)."""
    labels, inv = np.unique(clusters, return_inverse=True)
    k = len(labels)
    point = float(values.mean()) if len(values) else None
    if k < MIN_CLUSTERS:
        return {"point": point, "low": None, "high": None, "p_ge_0": None, "n_clusters": int(k),
                "method": f"unavailable (<{MIN_CLUSTERS} clusters)"}
    rng = np.random.default_rng(seed)
    sums, counts = np.bincount(inv, weights=values, minlength=k), np.bincount(inv, minlength=k).astype(float)
    pick = rng.integers(0, k, size=(n_boot, k))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return {"point": point, "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975)),
            "p_ge_0": float((np.sum(means >= 0) + 1) / (n_boot + 1)), "n_clusters": int(k),
            "method": "cluster bootstrap"}


def flat_roi(rows: Sequence[Dict], arm: str) -> Dict:
    """Flat 1u at the actual offered decimal price.  An issued recommendation is always a bet on its
    issued side (a below-breakeven analyst pick was still issued, so its loss counts); a candidate
    event is a bet only on the side whose probability beats that side's breakeven."""
    if any(not r.get("price_decimal") for r in rows):
        return {"status": "unavailable: no offered-price receipt in source", "roi": None}
    bets = won = 0
    pnl = 0.0
    for r in rows:
        p_side, price = _p(r["p"][arm]), r["price_decimal"]
        other = r.get("price_decimal_other")
        y = 1 if r["outcome"] == "win" else 0
        if r.get("issued_class") == "recommendation" or p_side > 1 / price:
            bets, won, pnl = bets + 1, won + y, pnl + (price - 1 if y else -1.0)
        elif other and (1 - p_side) > 1 / other:
            bets, won, pnl = bets + 1, won + (1 - y), pnl + (other - 1 if not y else -1.0)
    return {"status": "hypothetical flat 1u at offered prices; not account P/L", "bets": bets, "won": won,
            "lost": bets - won, "units": round(pnl, 4), "roi": round(pnl / bets, 4) if bets else None}


def score_arms(rows: Sequence[Dict], arms: Sequence[str], baseline: Optional[str] = None,
               cluster_key: str = "game_id", n_boot: int = 2000, seed: int = 20261006) -> Dict:
    y = np.asarray([1.0 if r["outcome"] == "win" else 0.0 for r in rows])
    clusters = np.asarray([str(r.get(cluster_key)) for r in rows])
    weeks = np.asarray([f"{r['season']}-{r['week']}" for r in rows])
    markets = sorted({str(r["market"]) for r in rows})
    out = {"n": len(rows), "n_games": int(len(set(clusters))), "n_season_weeks": int(len(set(weeks))),
           "per_market_n": {m: sum(1 for r in rows if str(r["market"]) == m) for m in markets}, "arms": {}}
    probs = {a: np.asarray([_p(r["p"][a]) for r in rows], dtype=float) for a in arms}
    for i, a in enumerate(arms):
        loss = _losses(probs[a], y)
        out["arms"][a] = {
            "brier": cluster_boot(loss["brier"], clusters, n_boot, seed + i),
            "logloss": float(loss["logloss"].mean()) if len(y) else None,
            "ece": ece(probs[a], y), "reliability": binary_calibration(y, probs[a], bins=10),
            "per_market_brier": {m: float(loss["brier"][[str(r["market"]) == m for r in rows]].mean())
                                 for m in markets},
            "hypothetical_roi": flat_roi(rows, a), "account_pl": None}
        if baseline and a != baseline and len(y):
            base = _losses(probs[baseline], y)
            d_b, d_l = loss["brier"] - base["brier"], loss["logloss"] - base["logloss"]
            out["arms"][a]["vs_" + baseline] = {
                "d_brier_game_cluster": cluster_boot(d_b, clusters, n_boot, seed + 100 + i),
                "d_brier_week_cluster": cluster_boot(d_b, weeks, n_boot, seed + 200 + i),
                "d_logloss": float(d_l.mean())}
    return out


# --------------------------------------------------------------- registry --
def criteria_sha256(registry: Dict) -> str:
    frozen = {"protocol": registry["protocol"],
              "criteria": {c["candidate_id"]: c["criteria"] for c in registry["candidates"] if c.get("criteria")}}
    return hashlib.sha256(_canon(frozen).encode()).hexdigest()


def validate_registry(registry: Dict) -> List[str]:
    errors = []
    active = [c["candidate_id"] for c in registry["candidates"] if c["status"] == ACTIVE]
    if len(active) > MAX_ACTIVE:
        errors.append(f"{len(active)} active candidates {active} > max {MAX_ACTIVE}")
    if registry.get("freeze", {}).get("criteria_sha256") != criteria_sha256(registry):
        errors.append("frozen criteria changed after freeze (criteria_sha256 mismatch): write a new protocol version")
    for c in registry["candidates"]:
        if c["status"].startswith("rejected") and not c.get("evidence"):
            errors.append(f"{c['candidate_id']}: rejection without retained evidence")
    return errors


def holm(pvalues: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    items = sorted((p, k) for k, p in pvalues.items() if p is not None)
    m, adj, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return {k: adj.get(k) for k in pvalues}


def screen(results: Dict[str, Dict], criteria: Dict[str, Dict]) -> Dict[str, Dict]:
    """Apply frozen screening gates.  A pass only makes a candidate eligible for prospective
    confirmation; it never promotes it.  Activation respects MAX_ACTIVE."""
    adj = holm({k: (v.get("eval") or {}).get("d_brier_game_cluster", {}).get("p_ge_0") for k, v in results.items()})
    out = {}
    for cid, res in results.items():
        g, ev = criteria[cid]["screening_gate"], res.get("eval") or {}
        checks = {
            "S1_sample": ev.get("n", 0) >= g["min_eval_graded"] and ev.get("n_exposed", 0) >= g["min_eval_exposed"],
            "S2_holm_game_cluster": adj.get(cid) is not None and adj[cid] < g["alpha_familywise"],
            "S2b_week_cluster_upper_lt_0": (ev.get("d_brier_week_cluster", {}).get("high") or 1.0) < 0,
            "S3_logloss_point_lt_0": (ev.get("d_logloss") if ev.get("d_logloss") is not None else 1.0) < 0,
            "S4_mechanism_sign": res.get("beta_dev") is not None and
            math.copysign(1, res["beta_dev"]) == g["expected_beta_sign"] and res["beta_dev"] != 0}
        verdict = ("insufficient_data" if not checks["S1_sample"] else
                   "eligible_for_prospective" if all(checks.values()) else "retired_retrospective_fail")
        out[cid] = {"checks": checks, "holm_adjusted_p": adj.get(cid), "verdict": verdict}
    return out


# ------------------------------------------------- game-total challengers --
def _kickoff_utc(gameday, gametime) -> Optional[str]:
    from zoneinfo import ZoneInfo
    try:
        local = dt.datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
    except (TypeError, ValueError):
        return None
    return local.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_kickoffs(hist_dir: str) -> Dict[str, str]:
    import pandas as pd
    s = pd.read_parquet(os.path.join(hist_dir, "lines_extra.parquet"), columns=["game_id", "gameday", "gametime"])
    return {r.game_id: _kickoff_utc(r.gameday, r.gametime) for r in s.itertuples(index=False)}


def build_totals_frame(hist_dir: str):
    """One row per game with a real public closing total (nflverse schedules) and two signals,
    each computed from games strictly before the target game."""
    import pandas as pd
    g = pd.read_parquet(os.path.join(hist_dir, "lines_extra.parquet"))
    g = g.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    g["kickoff"] = [_kickoff_utc(a, b) for a, b in zip(g.gameday, g.gametime)]
    # ---- M2 signal: as-of offence/defence scoring form, last 8 completed games, shrunk n/(n+4)
    long = pd.concat([
        g[["game_id", "season", "week", "kickoff", "home_team", "home_score", "away_score"]].rename(
            columns={"home_team": "team", "home_score": "pf", "away_score": "pa"}),
        g[["game_id", "season", "week", "kickoff", "away_team", "away_score", "home_score"]].rename(
            columns={"away_team": "team", "away_score": "pf", "home_score": "pa"})]).dropna(subset=["pf", "pa"])
    long = long.sort_values("kickoff")
    form = {}
    for r in g.itertuples(index=False):
        prior = long[long.kickoff < r.kickoff]
        league = float(np.concatenate([prior.pf.values, prior.pa.values]).mean()) if len(prior) else None
        parts = {}
        for side in ("home", "away"):
            t = prior[prior.team == getattr(r, f"{side}_team")].tail(8)
            n = len(t)
            if league is None or n == 0:
                parts = None
                break
            parts[side] = ((t.pf.sum() + 4 * league) / (n + 4), (t.pa.sum() + 4 * league) / (n + 4))
        form[r.game_id] = (None if parts is None else
                           (parts["home"][0] + parts["away"][1]) / 2 + (parts["away"][0] + parts["home"][1]) / 2)
    g["asof_expected_total"] = g.game_id.map(form)
    g["m2_signal"] = g.asof_expected_total - g.total_line
    # ---- M1 signal: current QB-role change state, count of teams (0-2)
    qb = _qb_change(hist_dir, g)
    g["home_qb_change"] = [qb.get((r.game_id, r.home_team)) for r in g.itertuples(index=False)]
    g["away_qb_change"] = [qb.get((r.game_id, r.away_team)) for r in g.itertuples(index=False)]
    g["m1_signal"] = g.home_qb_change.fillna(0).astype(float) + g.away_qb_change.fillna(0).astype(float)
    g["qb_history_missing"] = g.home_qb_change.isna() | g.away_qb_change.isna()
    return g


def _qb_change(hist_dir: str, games) -> Dict:
    """(game_id, team) -> 1 if the team enters the game in a QB-role change state, 0 if not, absent if no history.

    primary = most pass attempts in the team's strictly-prior games this season (week 1: prior season);
    change = previous game's leading passer != primary, OR primary listed Out/Doubtful on the official
    injury report for that week (published pregame)."""
    import pandas as pd
    cols = ["season", "week", "game_id", "posteam", "passer_player_id", "pass_attempt"]
    frames = []
    for f in ("historical_pbp.parquet", "pbp_2024.parquet", "pbp_2025.parquet", "pbp_2026.parquet"):
        p = os.path.join(hist_dir, f)
        if os.path.exists(p):
            d = pd.read_parquet(p, columns=cols)
            frames.append(d[d.season >= 2023])
    pbp = pd.concat(frames)
    pbp = pbp[(pbp.pass_attempt == 1) & pbp.passer_player_id.notna()]
    att = pbp.groupby(["season", "week", "game_id", "posteam", "passer_player_id"]).size().rename("att").reset_index()
    inj = pd.read_parquet(os.path.join(hist_dir, "injuries.parquet"))
    inj_out = inj[inj.report_status.isin(["Out", "Doubtful"]) & inj.season.notna() & inj.week.notna()]
    out_set = set(zip(inj_out.season.astype(int), inj_out.week.astype(int), inj_out.team, inj_out.gsis_id))
    res = {}
    for r in games.itertuples(index=False):
        for team in (r.home_team, r.away_team):
            mine = att[(att.posteam == team)]
            cur = mine[(mine.season == r.season) & (mine.week < r.week)]
            hist = cur if len(cur) else mine[mine.season == r.season - 1]
            if not len(hist):
                continue
            primary = hist.groupby("passer_player_id").att.sum().idxmax()
            last_key = hist[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).iloc[-1]
            last = hist[(hist.season == last_key.season) & (hist.week == last_key.week)]
            leader = last.sort_values("att").passer_player_id.iloc[-1]
            injured = (int(r.season), int(r.week), team, primary) in out_set
            res[(r.game_id, team)] = int(leader != primary or injured)
    return res


def run_totals_challengers(frame, criteria: Dict[str, Dict], n_boot: int, seed: int) -> Dict:
    reg = frame[(frame.game_type == "REG") & frame.total.notna() & frame.total_line.notna()].copy()
    reg["outcome"] = np.where(reg.total > reg.total_line, "win", np.where(reg.total < reg.total_line, "loss", "push"))
    results = {}
    for i, (cid, col) in enumerate((("QBROLE-TOTAL-v1", "m1_signal"), ("OPPFORM-TOTAL-v1", "m2_signal"))):
        c = criteria[cid]
        dev = reg[reg.season.isin(c["splits"]["fit"]) & (reg.outcome != "push") & reg[col].notna()]
        x, yc = dev[col].astype(float).values, (dev.outcome == "win").astype(float).values - 0.5
        beta = float((x * yc).sum() / (x * x).sum()) if (x * x).sum() > 0 else None
        res = {"signal": col, "beta_dev": beta, "fit_n": int(len(dev)),
               "price_receipt": "none: source carries closing total lines without prices; ROI not computed"}
        for split, seasons, weeks in (("eval", c["splits"]["evaluate"], None),
                                      ("descriptive_2026", [2026], c["splits"]["descriptive_only_2026_weeks"])):
            part = reg[reg.season.isin(seasons)]
            if weeks:
                part = part[part.week.isin(weeks)]
            counts = {"games": int(len(part)), "push": int((part.outcome == "push").sum()),
                      "signal_missing": int(part[col].isna().sum())}
            part = part[(part.outcome != "push") & part[col].notna()]
            rows = [{"evidence_id": r.game_id, "season": int(r.season), "week": int(r.week), "game_id": r.game_id,
                     "market": "game_total", "outcome": r.outcome, "issued_class": "candidate_event",
                     "p": {"market": 0.5, cid: float(np.clip(0.5 + (beta or 0.0) * getattr(r, col), *P_CLIP))}}
                    for r in part.itertuples(index=False)]
            if not rows:
                res[split] = {"n": 0, "counts": counts}
                continue
            s = score_arms(rows, ["market", cid], baseline="market", n_boot=n_boot, seed=seed + 10 * i)
            vs = s["arms"][cid]["vs_market"]
            res[split] = {"n": s["n"], "n_games": s["n_games"], "counts": counts,
                          "n_exposed": int((part[col].astype(float) != 0).sum()) if col == "m1_signal" else s["n"],
                          "brier_market": s["arms"]["market"]["brier"]["point"],
                          "brier_challenger": s["arms"][cid]["brier"]["point"],
                          "d_brier_game_cluster": vs["d_brier_game_cluster"],
                          "d_brier_week_cluster": vs["d_brier_week_cluster"], "d_logloss": vs["d_logloss"],
                          "residual_mean_by_signal": _residual_table(part, col)}
        results[cid] = res
    return results


def _residual_table(part, col) -> Dict:
    resid = (part.total - part.total_line).astype(float)
    if col == "m1_signal":
        keys = part[col].astype(int)
    else:
        keys = np.sign(part[col]).astype(int)
    return {str(k): {"n": int((keys == k).sum()), "mean_actual_minus_line": round(float(resid[keys == k].mean()), 3),
                     "over_rate": round(float((resid[keys == k] > 0).mean()), 4)} for k in sorted(set(keys))}


# -------------------------------------------------------------------- CLI --
def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def research_status(ledger_path: str, registry_path: str = REGISTRY_PATH, checked_at: Optional[str] = None,
                    last_append: Optional[Dict] = None) -> Dict:
    """Public research status from the persisted evidence ledger and the frozen registry.

    Descriptive counts and the exact missing-data gates only. Nothing here promotes a candidate,
    changes a production weight/default, refits on recent results or blends the market into the
    football forecast; a passing test or one winning slate is not evidence."""
    entries = read_ledger(ledger_path)
    rows = [r for r in latest_rows(entries) if r.get("source") == "issued_ledger"]
    reg = json.load(open(registry_path))
    frozen_at = (reg.get("freeze") or {}).get("frozen_at_utc")
    given = [r for r in rows if r.get("section") == "recommendations_given"]
    pro = [r for r in given if window_of(r) == "prospective_confirmation"]
    scored = [r for r in pro if r.get("outcome") in ("win", "loss") and r.get("price_decimal")
              and _p((r.get("p") or {}).get("issued")) is not None]
    unused = [r for r in scored if unused_confirmation(r, frozen_at)]
    games = sorted({r.get("game_id") for r in scored if r.get("game_id")})
    gates = []
    if not scored:
        gates.append("prospective confirmation: 0 settled 2026 week>=5 given-before-kickoff recommendations "
                     "with a trusted pre-kickoff ledger capture, an offered price and an issued probability")
    if len(games) < MIN_CLUSTERS:
        gates.append(f"game clusters: {len(games)} of the {MIN_CLUSTERS} needed for any clustered interval")
    if not unused:
        gates.append(f"untouched confirmation for challengers frozen at {frozen_at}: 0 rows captured after the freeze")
    gates += ["account P/L: unknown (no ticket/stake ledger exists; hypothetical flat-stake ROI is not P/L)",
              "totals CLV/ROI: no entry/close over-under price receipts for the retrospective totals screens",
              "prop opponent-factor ablation: capture rows do not record opp_factor/opp_source"]

    def tally(pred):
        c = Counter(str(r.get("outcome")) for r in rows if pred(r))
        return {k: c.get(k, 0) for k in ("win", "loss", "push", "void", "pending")}

    sections = {s: {"n": sum(r.get("section") == s for r in rows), "outcomes": tally(lambda r, s=s: r.get("section") == s),
                    "by_policy_class": {pc: tally(lambda r, s=s, pc=pc: r.get("section") == s
                                                  and str(r.get("policy_class")) == pc)
                                        for pc in sorted({str(r.get("policy_class")) for r in rows
                                                          if r.get("section") == s})}}
                for s in EVIDENCE_SECTIONS}
    return {
        "schema": "fablesfable.research_status.v1", "checked_at": checked_at, "protocol": reg["protocol"]["protocol_id"],
        "registry_criteria_sha256": (reg.get("freeze") or {}).get("criteria_sha256"), "registry_frozen_at": frozen_at,
        "ledger": {"entries": len(entries), "records": len(rows),
                   "revisions": len(entries) - len({e["row"]["evidence_id"] for e in entries}),
                   "head_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
                   "last_append": last_append},
        "windows": dict(sorted(Counter(f"{r.get('section')}|{window_of(r)}" for r in rows).items())),
        "historical_imports": sum(bool(r.get("historical_import")) for r in rows),
        "sections": sections,
        "prospective": {"settled_scored": len(scored), "unused_since_freeze": len(unused), "game_clusters": len(games)},
        "candidates": [{"candidate_id": c["candidate_id"], "status": c["status"]} for c in reg["candidates"]],
        "promotion": {"passed_predeclared_gate": [], "production_weights_or_defaults_changed": False,
                      "market_blend": False, "refit_on_latest_results": False,
                      "statement": ("No candidate has passed a predeclared prospective gate, so nothing was "
                                    "promoted. Sections are graded separately and never pooled; losses are kept. "
                                    "Model probabilities are confidence diagnostics, not a demonstrated edge.")},
        "missing_data_gates": gates}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--registry", default=REGISTRY_PATH)
    rn = sub.add_parser("run")
    rn.add_argument("--registry", default=REGISTRY_PATH)
    rn.add_argument("--historical", required=True)
    rn.add_argument("--real-lines", default=None)
    rn.add_argument("--issued-grade", default=None, help="hand-graded delivered card JSON (rows with result)")
    rn.add_argument("--card-game-id", default=None, help="nflverse game_id of --issued-grade, e.g. 2026_04_ATL_NO")
    rn.add_argument("--settlement-grade", default=None,
                    help="JSON written from nflvalue.issued_grading.grade() (settlement -> analysis contract)")
    rn.add_argument("--ledger", default=None, help="append-only evidence JSONL (cumulative store)")
    rn.add_argument("--output", required=True)
    rn.add_argument("--bootstrap-n", type=int, default=4000)
    rn.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args(argv)
    with open(args.registry) as fh:
        registry = json.load(fh)
    if args.cmd == "freeze":
        print(json.dumps({"criteria_sha256": criteria_sha256(registry), "errors": validate_registry(registry)}))
        return 0
    errors = validate_registry(registry)
    if errors:
        print(json.dumps({"refused": errors}))
        return 2
    import pandas as pd
    kick = load_kickoffs(args.historical)
    inputs = {f: _sha_file(os.path.join(args.historical, f)) for f in
              ("lines_extra.parquet", "injuries.parquet", "historical_pbp.parquet", "pbp_2024.parquet",
               "pbp_2025.parquet", "pbp_2026.parquet")}
    report = {"schema": "evidence-loop-report-v1", "criteria_sha256": criteria_sha256(registry),
              "inputs_sha256": inputs, "bootstrap_n": args.bootstrap_n, "seed": args.seed,
              "never": "no production weight/calibration/default change; no odds/API call; no synthetic line"}
    all_rows: List[Dict] = []
    if args.real_lines:
        inputs["real_lines"] = _sha_file(args.real_lines)
        ev = from_real_line_events(pd.read_csv(args.real_lines), kick)
        all_rows += ev
        arms = list(REAL_LINE_ARMS)
        report["offered_line_props"] = {
            "label": "REAL offered lines, 2026 W1-2, retrospective/exploratory (already read)",
            "coverage": coverage(ev, arms, scope="all"),
            "scores": score_arms(scored_rows(ev, arms, scope="all"), arms, baseline="A1",
                                 n_boot=args.bootstrap_n, seed=args.seed)}
    if args.settlement_grade:
        inputs["settlement_grade"] = _sha_file(args.settlement_grade)
        with open(args.settlement_grade) as fh:
            issued = from_issued_grading(json.load(fh))
        all_rows += issued
        report["issued_ledger"] = {
            "label": "issued-pick ledger settlement rows (recommendations_given section)",
            "coverage": coverage(issued, ["issued"]),
            "scores": score_arms(scored_rows(issued, ["issued"]), ["issued"], n_boot=args.bootstrap_n, seed=args.seed),
            "prospective": score_arms([r for r in scored_rows(issued, ["issued"])
                                       if window_of(r) == "prospective_confirmation"], ["issued"],
                                      n_boot=args.bootstrap_n, seed=args.seed)}
    if args.issued_grade:
        if not args.card_game_id or args.card_game_id not in kick:
            ap.error("--issued-grade needs --card-game-id naming a game in lines_extra.parquet")
        inputs["issued_grade"] = _sha_file(args.issued_grade)
        with open(args.issued_grade) as fh:
            card = json.load(fh)
        season, week = (int(x) for x in args.card_game_id.split("_")[:2])
        card_rows = from_card_grade(card, season, week, args.card_game_id, kick[args.card_game_id])
        all_rows += card_rows
        rec = scored_rows(card_rows, ["issued"])
        report["issued_card"] = {
            "label": "delivered card graded from ESPN final box; no machine decision clock -> unknown_clock",
            "coverage": coverage(card_rows, ["issued"]), "recommendation_record": {
                "won": sum(r["outcome"] == "win" for r in rec), "lost": sum(r["outcome"] == "loss" for r in rec)},
            "scores": score_arms(rec, ["issued"], n_boot=args.bootstrap_n, seed=args.seed),
            "pass_rows_counted_not_scored": sum(r["issued_class"] == "pass" for r in card_rows)}
    if args.ledger:
        report["ledger_append"] = append_evidence(args.ledger, all_rows)
        report["ledger_rows"] = len(latest_rows(read_ledger(args.ledger)))
    frame = build_totals_frame(args.historical)
    # frozen criteria are re-screened on every run whatever the logged status, so a retirement stays reproducible
    crit = {c["candidate_id"]: c["criteria"] for c in registry["candidates"] if c.get("criteria")}
    results = run_totals_challengers(frame, crit, args.bootstrap_n, args.seed)
    report["challengers"] = {"label": "REAL public closing game totals (nflverse schedules) + native final scores; "
                                      "retrospective screening; W5+ prospective confirmation reserved",
                             "results": results, "screening": screen(results, crit)}
    prospective = [r for r in all_rows if window_of(r) == "prospective_confirmation"]
    report["prospective_confirmation"] = {"rows": len(prospective), "status": "no qualifying rows yet"
                                          if not prospective else "accruing"}
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps({"output": args.output, "screening": report["challengers"]["screening"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Context hypothesis ledger: measure the birthday, PROVE it before it moves a bet.

The locked spec decision (PROP_SHORTLISTER_SPEC.md §0/§4, PREMORTEM.md) is
that personal context — birthdays, revenge spots, bereavement, contract
years — is displayed and never scored, because small-n narrative factors are
exactly where overfitting lives. This module is the honest middle path:

  RECORD   every context tag attached to a lean at publish time
           (``context_ledger``: who, market, tag, source).
  GRADE    join tags to ``lean_outcomes`` once the week is graded.
  TEST     per tag: n, hit rate vs the untagged baseline, exact binomial
           p-value, Benjamini–Hochberg correction across all tags tested.
  PROMOTE  only when a tag clears ALL pre-committed bars —
           n >= MIN_N (100), BH-adjusted q < 0.05 — the report marks it
           PROMOTABLE. Even then, nothing changes until a human lists the
           tag in config.json ``context_learning.enabled_tags``; only then
           does the pipeline apply a bounded composite multiplier
           (±MAX_EFFECT, from the tag's own measured effect, shrunk).

So the system "uses" birthdays from day one — by counting them. It bets on
them only after they've earned it, and the evidence trail is a queryable
table, not a vibe.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import pandas as pd

from . import db as dbmod

MIN_N = 100                 # occurrences before a tag may even be judged
ALPHA = 0.05                # BH-adjusted significance bar
MAX_EFFECT = 0.10           # promoted tags move composite at most +/-10%
SHRINK_K = 150.0            # pseudo-observations pulling effect toward 0

# canonical tag vocabulary (keyword -> tag), applied to note text
TAG_KEYWORDS = {
    "birthday": "birthday", "revenge": "revenge", "former team": "revenge",
    "bereavement": "bereavement", "funeral": "bereavement",
    "contract": "contract_year", "extension": "contract_year", "holdout": "contract_year",
    "baby": "new_baby", "born": "new_baby", "wedding": "wedding",
    "homecoming": "homecoming", "milestone": "milestone", "record": "milestone",
    "incentive": "contract_incentive", "bonus": "contract_incentive",
    "escalator": "contract_incentive",
    # Phase 6.6 situational families (all measured NOT-significant on
    # 2019-2025 history -- data/situation_study.json -- accruing live
    # evidence here; the closest was revenge_trade, q=.36)
    "primetime": "primetime", "division game": "division_game",
    "short week": "short_week", "timezones": "long_travel_2tz",
    "early eastern kickoff": "west_east_early",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tags_from_text(text: str) -> List[str]:
    t = str(text or "").lower()
    return sorted({tag for kw, tag in TAG_KEYWORDS.items() if kw in t})


def record_tags(conn, season: int, week: int, games: List[Dict],
                contexts: Dict[str, Dict]) -> int:
    """Persist every tag visible on a published lean's context entries."""
    rows = []
    for g in games:
        ctx = contexts.get(g["game_id"]) or {}
        items_by_pid = {e["player_id"]: e.get("items", []) for e in ctx.get("entries", [])}
        for l in g.get("leans", []):
            for item in items_by_pid.get(l["player_id"], []):
                for tag in tags_from_text(item):
                    rows.append({"season": season, "week": week,
                                 "player_id": l["player_id"], "market": l["market"],
                                 "tag": tag, "source": "context_panel",
                                 "note": str(item)[:300], "created_at": _now()})
    if rows:
        dbmod.upsert(conn, "context_ledger", rows,
                     ["season", "week", "player_id", "market", "tag"])
    return len(rows)


# --------------------------------------------------------------------------- #
# The study
# --------------------------------------------------------------------------- #
def _binom_p(hits: int, n: int, p0: float) -> Optional[float]:
    try:
        from scipy import stats
        return float(stats.binomtest(hits, n, p0, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001 - scipy always present per requirements
        return None


def _evaluate_tags(joined: pd.DataFrame, baseline: float) -> Dict:
    """The pre-committed verdict machinery, shared by the live-ledger study
    and the Phase 6.6 historical study: exact binomial vs the pooled
    baseline, Benjamini-Hochberg across every testable tag, PROMOTABLE only
    past n>=MIN_N and q<ALPHA."""
    results = []
    for tag, grp in joined.groupby("tag"):
        n, hits = int(len(grp)), int(grp["hit"].sum())
        p = _binom_p(hits, n, baseline) if n >= MIN_N else None
        results.append({"tag": tag, "n": n, "hits": hits,
                        "hit_rate": round(hits / n, 4),
                        "baseline": round(baseline, 4), "p_value": p})
    # Benjamini–Hochberg across every tag that reached testable size
    testable = sorted([r for r in results if r["p_value"] is not None],
                      key=lambda r: r["p_value"])
    m = len(testable)
    q_prev = 1.0
    for rank in range(m, 0, -1):
        r = testable[rank - 1]
        q = min(q_prev, r["p_value"] * m / rank)
        r["q_value"] = round(q, 4)
        q_prev = q
    for r in results:
        n_ok = r["n"] >= MIN_N
        q_ok = r.get("q_value") is not None and r["q_value"] < ALPHA
        r["verdict"] = ("PROMOTABLE" if (n_ok and q_ok)
                        else ("insufficient_n" if not n_ok else "not_significant"))
        if r["verdict"] == "PROMOTABLE":
            raw_effect = r["hit_rate"] - r["baseline"]
            shrunk = raw_effect * (r["n"] / (r["n"] + SHRINK_K))
            r["proposed_mult"] = round(float(max(1 - MAX_EFFECT,
                                                 min(1 + MAX_EFFECT, 1 + 2 * shrunk))), 4)
    return {"n_tagged": int(len(joined)), "baseline_hit_rate": round(baseline, 4),
            "bars": {"min_n": MIN_N, "alpha_bh": ALPHA, "max_effect": MAX_EFFECT},
            "tags": {r["tag"]: r for r in results}}


def study(conn) -> Dict:
    """Per-tag evidence report. PROMOTABLE only past every pre-committed bar."""
    ledger = dbmod.query_df(conn, "SELECT * FROM context_ledger")
    outcomes = dbmod.query_df(conn, "SELECT * FROM lean_outcomes")
    if ledger.empty or outcomes.empty:
        return {"n_tagged": 0, "tags": {},
                "note": "no graded, tagged leans yet — the ledger fills as live weeks run"}

    joined = ledger.merge(outcomes[["season", "week", "player_id", "market", "hit"]],
                          on=["season", "week", "player_id", "market"], how="inner")
    if joined.empty:
        return {"n_tagged": int(len(ledger)), "tags": {},
                "note": "tags recorded but none graded yet"}
    return _evaluate_tags(joined, float(outcomes["hit"].mean()))


def study_frame(tag_table: pd.DataFrame, graded: pd.DataFrame) -> Dict:
    """Phase 6.6 historical mode: same bars, but the population is a graded
    HISTORICAL candidate frame (e.g. data/ml_frame.parquet, the population
    the original birthday/revenge verdicts were measured on) instead of the
    live ledger. ``tag_table``: [season, week, player_id, market, tag];
    ``graded`` must carry [season, week, player_id, market, hit]."""
    keys = ["season", "week", "player_id", "market"]
    joined = tag_table.merge(graded[keys + ["hit"]].drop_duplicates(subset=keys),
                             on=keys, how="inner")
    if joined.empty:
        return {"n_tagged": 0, "tags": {}, "note": "no tag matched a graded row"}
    return _evaluate_tags(joined, float(graded["hit"].mean()))


def enabled_multipliers(cfg: Dict, conn) -> Dict[str, float]:
    """{tag: multiplier} for tags a HUMAN promoted in config AND that still
    hold up in the current study. Empty by default — and therefore a no-op."""
    enabled = ((cfg.get("context_learning") or {}).get("enabled_tags")) or []
    if not enabled:
        return {}
    s = study(conn)
    out = {}
    for tag in enabled:
        r = (s.get("tags") or {}).get(tag)
        if r and r.get("verdict") == "PROMOTABLE" and r.get("proposed_mult"):
            out[tag] = r["proposed_mult"]
    return out


def apply_context_multipliers(cands: pd.DataFrame, conn, season: int, week: int,
                              multipliers: Dict[str, float]) -> pd.DataFrame:
    """Stamp context_mult on candidates whose (player, market) carries a
    promoted tag this week. No multipliers -> untouched frame."""
    if not multipliers or cands.empty:
        return cands
    ledger = dbmod.query_df(conn, """
        SELECT player_id, market, tag FROM context_ledger WHERE season=? AND week=?
        """, (season, week))
    if ledger.empty:
        return cands
    mult_by_key: Dict[tuple, float] = {}
    for r in ledger.itertuples(index=False):
        if r.tag in multipliers:
            key = (r.player_id, r.market)
            mult_by_key[key] = mult_by_key.get(key, 1.0) * multipliers[r.tag]
    if not mult_by_key:
        return cands
    cands = cands.copy()
    # Untagged rows get 1.0 (a no-op multiplier) rather than None, so the
    # column stays float-dtype. The composite multiplies by context_mult, so
    # 1.0 means "no adjustment" -- behavior-preserving for the scored result.
    cands["context_mult"] = [mult_by_key.get((p, m), 1.0)
                             for p, m in zip(cands["player_id"], cands["market"])]
    return cands

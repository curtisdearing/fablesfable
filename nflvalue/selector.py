"""Post-projection best-picks selector: the LAST layer, run only after every
candidate has been fully evaluated.

Position in the flow (this module never front-runs it):

    1. enumerate the FULL candidate pool for every game
    2. projections, simulations, injuries, weather, reallocation, ML
       probability, real market prices, composite scoring
    3. THEN this module reads the fully-scored pool and translates it into
       ranked, tiered picks per game -- overs or unders, several per game.

What it adds on top of the existing top-5 shortlist (which stays untouched
as the learning population):

  * BET LOGIC -- for candidates with a real sportsbook line, the side is the
    one with the stronger model-vs-market edge (already computed two-sided in
    ``composite.score_candidate``: model P(over) vs de-vigged fair P(over),
    same for under, best side wins; anytime_td is YES-only and never gets an
    artificial under).
  * A CONFIDENCE SCALE, market-specific and configurable (config "selector"):
        PASS      real line, but no meaningful positive edge
        LEAN      small positive edge -- tracking/learning material
        PLAYABLE  meaningful edge with acceptable confidence
        STRONG    best-tier edge for that market
        RESEARCH  synthetic/no-market line -- NEVER labeled as a best bet
    Downgrades (one tier each, recorded in ``tier_notes``): availability
    RISK, stale line snapshot, heavy positive correlation with a
    higher-ranked selected pick, and a model probability below the tier's
    calibration floor.
  * SENTENCE WRITEUPS -- factual, no hype: pick, why this side, model vs
    market probability, edge/EV, supporting projection, risks.

LINE MOVEMENT IS NOT AN INPUT. This selector sees only the CURRENTLY
available line and price. Favorable movement since entry is after-the-fact
feedback -- it belongs to CLV grading (``clv.py``) and threshold tuning,
never to a live pick's value. A test pins this (selector output is invariant
to any entry-history/movement fields on the candidate).

Picks persist to their own ``picks`` table (status-aware: runs blocked by
the freshness gate persist as ``blocked`` and are excluded from grading) and
are graded by ``grade_picks`` on the Tuesday pass, so the system learns from
several picks per game -- not just the top-5 leans.

Honesty framing unchanged: leans, not locks; advisory/research only; no bet
placement. 1-800-GAMBLER.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import db as dbmod

TIERS = ("RESEARCH", "PASS", "LEAN", "PLAYABLE", "STRONG")
_TIER_ORDER = {t: i for i, t in enumerate(TIERS)}

# Market-specific tier thresholds on the PROBABILITY edge (model minus fair,
# in probability points), plus an EV floor at the best available price for
# PLAYABLE+ and model-probability calibration floors. Config "selector"
# overrides any of this; these are the shipped defaults:
#   yardage props   lower bars (two-sided, liquid, model's strongest markets)
#   receptions/att  moderate bars (count markets, chunkier distributions)
#   anytime_td      highest bars (one-sided pricing, raw-implied vig, variance)
DEFAULT_SELECTOR_CFG: Dict = {
    "enabled": True,
    "max_picks_per_game": 5,
    "max_per_player": 2,
    "stale_line_hours": 24.0,     # line snapshot older than this -> one tier down
    "corr_downgrade_rho": 0.50,   # positive rho vs a selected pick above this -> one tier down
    "min_model_prob": {"playable": 0.52, "strong": 0.55},
    "thresholds": {
        "default":         {"lean": 0.020, "playable": 0.045, "strong": 0.080, "ev_min": 0.0},
        "receiving_yards": {"lean": 0.020, "playable": 0.040, "strong": 0.070, "ev_min": 0.0},
        "rushing_yards":   {"lean": 0.020, "playable": 0.040, "strong": 0.070, "ev_min": 0.0},
        "passing_yards":   {"lean": 0.020, "playable": 0.040, "strong": 0.070, "ev_min": 0.0},
        "receptions":      {"lean": 0.025, "playable": 0.050, "strong": 0.085, "ev_min": 0.0},
        "rush_attempts":   {"lean": 0.025, "playable": 0.050, "strong": 0.085, "ev_min": 0.0},
        "pass_attempts":   {"lean": 0.025, "playable": 0.050, "strong": 0.085, "ev_min": 0.0},
        "anytime_td":      {"lean": 0.040, "playable": 0.080, "strong": 0.130, "ev_min": 0.02},
    },
    "tuning_note": ("starting defaults -- tune from forward CLV + historical replay "
                    "(scripts/tune_selector_thresholds.py prints replay hit rates per bar)"),
}


def selector_config(cfg: Optional[Dict] = None) -> Dict:
    """Config 'selector' merged over the shipped defaults (deep for thresholds)."""
    user = dict((cfg or {}).get("selector") or {})
    out = {**DEFAULT_SELECTOR_CFG, **user}
    thr = {**DEFAULT_SELECTOR_CFG["thresholds"], **(user.get("thresholds") or {})}
    out["thresholds"] = thr
    out["min_model_prob"] = {**DEFAULT_SELECTOR_CFG["min_model_prob"],
                             **(user.get("min_model_prob") or {})}
    return out


def _tier_down(tier: str) -> str:
    """One tier down, floored at LEAN (a downgrade never re-labels a real-edge
    pick as PASS -- it stays visible as a tracked lean with its reason)."""
    order = ["LEAN", "PLAYABLE", "STRONG"]
    if tier not in order:
        return tier
    i = order.index(tier)
    return order[max(i - 1, 0)]


def classify(cand: Dict, sel_cfg: Dict,
             availability: Optional[Dict[str, Dict]] = None,
             line_age_hours: Optional[float] = None,
             max_corr_rho: float = 0.0) -> Tuple[str, List[str]]:
    """Tier one FULLY-SCORED candidate. Returns (tier, tier_notes).

    Inputs are the candidate's CURRENT market data only -- no movement, no
    entry history (see module docstring)."""
    notes: List[str] = []
    real_line = cand.get("line_source") == "odds_api"
    edge = cand.get("edge")
    if not real_line or cand.get("no_market") or edge is None:
        return "RESEARCH", ["no real market line — synthetic reference; research only"]

    thr = sel_cfg["thresholds"].get(cand.get("market"),
                                    sel_cfg["thresholds"]["default"])
    ev = (cand.get("components") or {}).get("ev_best_price")
    model_p = (cand.get("components") or {}).get("model_prob")
    mmp = sel_cfg["min_model_prob"]

    edge = float(edge)
    if edge < float(thr["lean"]):
        return "PASS", [f"edge {edge:+.3f} under the {cand.get('market')} lean bar {thr['lean']:.3f}"]
    tier = "LEAN"
    ev_ok = ev is not None and float(ev) >= float(thr.get("ev_min", 0.0))
    if edge >= float(thr["playable"]) and ev_ok \
            and (model_p is not None and float(model_p) >= float(mmp["playable"])):
        tier = "PLAYABLE"
    if tier == "PLAYABLE" and edge >= float(thr["strong"]) \
            and (model_p is not None and float(model_p) >= float(mmp["strong"])):
        tier = "STRONG"
    if tier in ("PLAYABLE", "STRONG") and not ev_ok:
        tier = "LEAN"
        notes.append("EV at best price under the floor")

    # -- downgrades (each one tier, reasons kept) ---------------------------- #
    status = ((availability or {}).get(cand.get("player_id")) or {}).get("status")
    if status == "RISK":
        notes.append("availability RISK (Questionable) — one tier down")
        tier = _tier_down(tier)
    if line_age_hours is not None and line_age_hours > float(sel_cfg["stale_line_hours"]):
        notes.append(f"line snapshot {line_age_hours:.0f}h old (> {sel_cfg['stale_line_hours']:.0f}h) — one tier down")
        tier = _tier_down(tier)
    if max_corr_rho > float(sel_cfg["corr_downgrade_rho"]):
        notes.append(f"correlates (ρ≈{max_corr_rho:+.2f}) with a higher-ranked pick — one tier down")
        tier = _tier_down(tier)
    return tier, notes


# --------------------------------------------------------------------------- #
# Per-game selection
# --------------------------------------------------------------------------- #
def _rank_key(c: Dict) -> tuple:
    """Real-edge picks rank by edge desc, EV desc; deterministic tie-break."""
    ev = (c.get("components") or {}).get("ev_best_price") or 0.0
    return (-float(c.get("edge") or 0.0), -float(ev), str(c.get("player_id")), str(c.get("market")))


def select_game_picks(scored_pool: List[Dict], sel_cfg: Dict,
                      availability: Optional[Dict[str, Dict]] = None,
                      line_age_hours: Optional[float] = None,
                      corr=None, as_of_season: Optional[int] = None) -> Dict:
    """All of one game's FULLY-SCORED candidates -> ranked, tiered picks.

    Returns {"picks": [...], "research": [...]} where picks are real-line
    candidates ranked by model-vs-market edge (multiple per game, overs and
    unders alike), and research is the best few no-market leans, clearly
    separated so they can never read as best bets."""
    real = [c for c in scored_pool
            if c.get("line_source") == "odds_api" and not c.get("no_market")
            and c.get("edge") is not None]
    synth = [c for c in scored_pool if c not in real]

    picks: List[Dict] = []
    per_player: Dict[str, int] = {}
    for c in sorted(real, key=_rank_key):
        if len(picks) >= int(sel_cfg["max_picks_per_game"]):
            break
        pid = str(c.get("player_id"))
        if per_player.get(pid, 0) >= int(sel_cfg["max_per_player"]):
            continue
        # correlation vs ALREADY-SELECTED (higher-ranked) picks
        max_rho = 0.0
        if corr is not None:
            for s in picks:
                rho = corr.rho_for(c.get("pos"), c.get("market"), c.get("player_id"), c.get("team"),
                                   s.get("pos"), s.get("market"), s.get("player_id"), s.get("team"),
                                   as_of_season=as_of_season)
                max_rho = max(max_rho, rho)
        tier, notes = classify(c, sel_cfg, availability=availability,
                               line_age_hours=line_age_hours, max_corr_rho=max_rho)
        if tier == "PASS":
            continue          # labeled no-edge -- never occupies a pick slot
        pick = dict(c)
        pick["tier"] = tier
        pick["tier_notes"] = notes
        pick["writeup"] = pick_writeup(pick)
        picks.append(pick)
        per_player[pid] = per_player.get(pid, 0) + 1

    research = []
    for c in sorted(synth, key=lambda r: (-(r.get("composite") or 0.0),
                                          str(r.get("player_id")), str(r.get("market"))))[:3]:
        r = dict(c)
        r["tier"] = "RESEARCH"
        r["tier_notes"] = ["no real market line — synthetic reference; research only"]
        r["writeup"] = pick_writeup(r)
        research.append(r)
    return {"picks": picks, "research": research}


def picks_for_games(games: List[Dict], cfg: Optional[Dict] = None,
                    availability: Optional[Dict[str, Dict]] = None,
                    line_age_hours: Optional[float] = None,
                    corr=None, as_of_season: Optional[int] = None) -> None:
    """Stamp ``g["picks"]`` / ``g["research_leans"]`` onto each game dict,
    consuming the game's FULL scored pool (``g["scored_pool"]``, produced by
    ``shortlist.rank_game`` AFTER every candidate was evaluated). The pool
    key is removed afterwards (it exists only to guarantee the selector sees
    everything the ranker saw)."""
    sel_cfg = selector_config(cfg)
    for g in games:
        pool = g.pop("scored_pool", None) or []
        if not sel_cfg.get("enabled", True):
            g["picks"], g["research_leans"] = [], []
            continue
        out = select_game_picks(pool, sel_cfg, availability=availability,
                                line_age_hours=line_age_hours,
                                corr=corr, as_of_season=as_of_season)
        g["picks"] = out["picks"]
        g["research_leans"] = out["research"]


# --------------------------------------------------------------------------- #
# Writeups: factual sentences, no hype
# --------------------------------------------------------------------------- #
def _side_phrase(c: Dict) -> str:
    market = str(c.get("market", "")).replace("_", " ")
    if c.get("market") == "anytime_td":
        return f"{c.get('name')} anytime TD (YES)"
    return f"{c.get('name')} {str(c.get('side', '')).lower()} {c.get('line')} {market}"


def _risks(c: Dict) -> List[str]:
    risks: List[str] = []
    if c.get("market") == "anytime_td":
        risks.append("touchdowns are high-variance and the market is one-sided (vig not fully removable)")
    if c.get("tier_notes"):
        risks.extend(n for n in c["tier_notes"] if "tier down" in n or "research" in n)
    cw = c.get("corr_with")
    if cw:
        risks.append(f"correlated with {cw.get('name')} {str(cw.get('market', '')).replace('_', ' ')} (ρ≈{cw.get('rho')})")
    if c.get("sd_source") == "default_fraction":
        risks.append("dispersion is a default (not enough walk-forward residual history)")
    wx = c.get("wx_pass_mult")
    if wx is not None and not pd.isna(wx) and abs(float(wx) - 1.0) > 0.02:
        risks.append(f"weather-adjusted (×{float(wx):.2f})")
    prices = c.get("prices") or {}
    if prices.get("n_books") is not None and int(prices.get("n_books") or 0) <= 1:
        risks.append("single-book price (thin consensus)")
    risks.append("late injury/news can change availability")
    return risks


def pick_writeup(c: Dict) -> str:
    """One factual paragraph: what, why, model vs market, EV, projection, risks."""
    comps = c.get("components") or {}
    proj = c.get("proj_components") or {}
    parts: List[str] = []

    if c.get("tier") == "RESEARCH" or c.get("no_market") or c.get("edge") is None:
        parts.append(
            f"{_side_phrase(c)} is a NO-MARKET research lean: the {c.get('line')} reference is the "
            f"player's own trailing mean (synthetic), not a sportsbook price, so no edge or EV can be "
            f"computed. The model projects {c.get('mean')} and this is tracked for learning only.")
    else:
        mp, kp = comps.get("model_prob"), comps.get("market_prob")
        edge = float(c.get("edge") or 0.0)
        sent = (f"{_side_phrase(c)} is selected because the model projects {c.get('mean')} "
                f"and gives this side a {float(mp):.0%} chance" if mp is not None else
                f"{_side_phrase(c)} is selected on a positive model-vs-market edge")
        if kp is not None:
            sent += f", against a {float(kp):.0%} market-implied fair probability"
        sent += f" — a {edge * 100:+.1f}-point probability edge"
        ev = comps.get("ev_best_price")
        prices = c.get("prices") or {}
        price = prices.get("over") if c.get("side") == "over" else prices.get("under")
        if ev is not None and price:
            sent += f" and {float(ev):+.1%} expected value at the best available price ({price}, {prices.get('book')})"
        sent += "."
        parts.append(sent)
        vol, eff, opp = proj.get("volume"), proj.get("efficiency"), proj.get("opp_factor")
        if vol is not None and eff is not None:
            basis = f"Projection basis: {vol} expected opportunities × {eff} efficiency"
            if opp not in (None, 1.0):
                basis += f", opponent-vs-position factor {opp}"
            parts.append(basis + ".")
        if kp is not None and comps.get("n_books"):
            parts.append(f"The market probability is the de-vigged consensus across {comps['n_books']} book(s).")

    risks = _risks(c)
    if risks:
        parts.append("Main risks: " + "; ".join(risks[:3]) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Persistence + grading (learning covers every pick, not just top-5 leans)
# --------------------------------------------------------------------------- #
def persist_picks(conn, season: int, week: int, clock: str, games: List[Dict],
                  as_of: str, status: str = "active") -> int:
    """Replace-the-run semantics, like leans. ``status='blocked'`` (freshness
    gate failed) rows persist for the audit trail but are EXCLUDED from
    grading/CLV/kill-check-adjacent evidence by the status filter."""
    conn.execute("DELETE FROM picks WHERE season=? AND week=? AND clock=?",
                 (season, week, clock))
    conn.commit()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for g in games:
        for rank, p in enumerate(g.get("picks", []) or [], start=1):
            prices = p.get("prices") or {}
            comps = p.get("components") or {}
            rows.append({
                "season": season, "week": week, "clock": clock,
                "game_id": g.get("game_id"), "rank": rank,
                "player_id": p.get("player_id"), "name": p.get("name"),
                "market": p.get("market"), "side": p.get("side"),
                "line": p.get("line"), "line_source": p.get("line_source"),
                "price": prices.get("over") if p.get("side") == "over" else prices.get("under"),
                "book": prices.get("book"), "tier": p.get("tier"),
                "edge": p.get("edge"), "ev": comps.get("ev_best_price"),
                "model_prob": comps.get("model_prob"), "market_prob": comps.get("market_prob"),
                "mean": p.get("mean"), "writeup": p.get("writeup"),
                "status": status, "hit": None, "actual": None,
                "as_of": as_of, "created_at": now,
            })
    if not rows:
        return 0
    return dbmod.upsert(conn, "picks", rows,
                        ["season", "week", "clock", "game_id", "player_id", "market"])


def grade_picks(conn, season: int, week: int, pw: pd.DataFrame,
                clock: str = "wed") -> Dict:
    """Grade ACTIVE picks of a completed week against actual stats (same
    grading convention as lean grading; blocked/voided rows excluded)."""
    from .candidates import ACTUAL_COL
    picks = dbmod.query_df(conn, """
        SELECT * FROM picks WHERE season=? AND week=? AND clock=? AND status='active'
        """, (season, week, clock))
    if picks.empty:
        return {"graded": 0}
    wk = pw[(pw["season"] == season) & (pw["week"] == week)]
    actuals: Dict[tuple, float] = {}
    for r in wk.itertuples(index=False):
        for market, col in ACTUAL_COL.items():
            actuals[(r.player_id, market)] = float(getattr(r, col))
        actuals[(r.player_id, "anytime_td")] = float(r.rush_tds + r.rec_tds)
    graded = 0
    for p in picks.itertuples(index=False):
        a = actuals.get((p.player_id, p.market))
        if a is None:
            continue
        if p.market == "anytime_td":
            hit = int(a >= 1.0)
        else:
            hit = int(a > float(p.line)) if p.side == "over" else int(a < float(p.line))
        conn.execute("""UPDATE picks SET hit=?, actual=? WHERE season=? AND week=?
                        AND clock=? AND game_id=? AND player_id=? AND market=?""",
                     (hit, a, season, week, clock, p.game_id, p.player_id, p.market))
        graded += 1
    conn.commit()
    return {"graded": graded}


def picks_record(conn) -> Dict:
    """Hit rates by tier over all graded ACTIVE picks -- the feedback loop the
    thresholds are tuned from (alongside forward CLV)."""
    df = dbmod.query_df(conn, "SELECT tier, hit FROM picks WHERE status='active' AND hit IS NOT NULL")
    if df.empty:
        return {"n": 0, "by_tier": {}}
    out = {}
    for tier, grp in df.groupby("tier"):
        out[tier] = {"n": int(len(grp)), "hit_rate": round(float(grp["hit"].mean()), 4)}
    return {"n": int(len(df)), "by_tier": out}

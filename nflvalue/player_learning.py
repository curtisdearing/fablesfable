"""Player-level, context-aware sequential learning — ready to fill from live weeks.

The market-level loop (``prop_learning.py``) corrects each MARKET's average
projection bias week to week. This adds the next layer down: **what a specific
player did, WHERE (home/away, opponent), and HOW (volume vs efficiency)** — and
folds it forward into the next week's projection for that player.

It has two halves:

  1. THE LEDGER (``record_player_residuals``) — after every completed week, for
     the FULL graded candidate pool (selection-bias-safe, never picks-only), one
     row per (player, market): projected mean vs actual, the log residual, its
     volume/efficiency split (HOW), and home/away + opponent (WHERE). Pure
     recording; it changes no projection. This is what makes the system "ready
     to take data" — the evidence accrues from week one, queryable and
     segmentable, long before it's trusted enough to act on.

  2. THE PLAYER BIAS (``compute_player_adjustments`` / ``apply_player_bias``) — a
     heavily-SHRUNK, player-SPECIFIC mean multiplier, applied to next week's
     projection. Per-player samples are the #1 noise trap (a player plays ~once
     a week), so this is deliberately timid:
       * it isolates the part of a player's actual/pred ratio NOT already
         explained by the market's average ratio — so a player who tracks the
         market average gets a ~1.0 (no-op) multiplier and only genuinely
         mis-projected players move;
       * empirical-Bayes shrinkage toward that market ratio with ``shrink_k``
         pseudo-games, so a thin sample barely moves;
       * a ``min_games`` floor before any bias applies at all;
       * a tight ``bias_clip``; and
       * the whole layer is **OFF by default** (config ``player_learning.enabled``)
         until live weeks earn it.

Walk-forward + reproducible, exactly like the market loop: adjustments are
persisted effective-at week+1 and computed only from weeks strictly before.
Synthetic-line caveat until real CLV — nothing here is a profit claim.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import db as dbmod

DEFAULTS = {
    "enabled": False,       # OFF until live weeks accrue enough to earn it
    "min_games": 4,         # a player needs this many graded games before any bias
    "shrink_k": 8.0,        # pseudo-games of market-prior weight (aggressive shrink)
    "lr": 0.5,              # learning rate on the isolated player-specific deviation
    "bias_clip": 0.12,      # |player mean multiplier - 1| capped here
    "window_weeks": 40,     # trailing games per player considered (covers >2 seasons)
    "min_market_n": 100,    # market ratio needs this many pooled rows to be trusted
}

# market -> the pbp usage column that is that market's "opportunity" (for the
# volume/efficiency split); markets whose opportunity is not a simple count
# (yards, td) fall back to a pure log-residual with no split.
_OPP_COL = {"receiving_yards": "targets", "receptions": "targets",
            "rushing_yards": "carries", "rush_attempts": "carries",
            "passing_yards": "pass_attempts", "pass_attempts": "pass_attempts"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def player_config(cfg: Optional[Dict] = None) -> Dict:
    return {**DEFAULTS, **((cfg or {}).get("player_learning") or {})}


def _actual_for(arow: Dict, market: str) -> Optional[float]:
    from .candidates import ACTUAL_COL
    if arow is None:
        return None
    if market == "anytime_td":
        return float(arow.get("rush_tds", 0.0) + arow.get("rec_tds", 0.0))
    col = ACTUAL_COL.get(market)
    return float(arow.get(col)) if col and arow.get(col) is not None else None


# --------------------------------------------------------------------------- #
# 1. The ledger: record every graded (player, market) with WHERE + HOW
# --------------------------------------------------------------------------- #
def record_player_residuals(conn, season: int, week: int, cands: pd.DataFrame,
                            pw: pd.DataFrame) -> int:
    """One row per (player, market) of the FULL graded pool -> player_week_residuals.

    Records projected mean vs actual, the log residual, its volume/efficiency
    split (HOW), and home/away + opponent (WHERE). Selection-bias-safe: the whole
    screened pool, not just published picks. Records nothing it can't grade."""
    if cands is None or cands.empty:
        return 0
    dbmod.ensure_columns(conn, "player_week_residuals",
                         {"early_exit": "INTEGER DEFAULT 0",
                          "game_meaningless": "INTEGER DEFAULT 0"})
    wk = pw[(pw["season"] == season) & (pw["week"] == week)]
    arows = {r["player_id"]: r for r in wk.to_dict("records")}
    rows: List[Dict] = []
    for c in cands.to_dict("records"):
        market = c.get("market")
        arow = arows.get(c.get("player_id"))
        actual = _actual_for(arow, market)
        mean = c.get("mean")
        if actual is None or mean is None or float(mean) <= 0:
            continue
        actual = float(actual)
        log_resid = math.log(actual / float(mean)) if actual > 0 else None
        # HOW: volume vs efficiency split, when the market has a count opportunity
        comps = c.get("components") or {}
        proj_vol = comps.get("volume")
        opp_col = _OPP_COL.get(market)
        v_err = e_err = None
        if opp_col and proj_vol and float(proj_vol) > 0 and arow is not None:
            act_vol = arow.get(opp_col)
            if act_vol is not None and float(act_vol) > 0:
                v_err = math.log(float(act_vol) / float(proj_vol))
                e_err = (log_resid - v_err) if log_resid is not None else None
        # Phase 8.4 observation-quality context: a truncated or rest-flagged
        # game is an AVAILABILITY story, not a model-error story -- tagged so
        # the bias learner below never chases it as signal
        ee = int(bool(arow.get("early_exit"))) if arow is not None else 0
        mg = int(bool(arow.get("game_meaningless"))) if arow is not None else 0
        if ee or mg:
            reason = "availability_truncated" if ee else "rest_week_context"
        elif log_resid is not None and abs(log_resid) <= 0.15:
            reason = "on_projection"          # landed close; not a volume/eff story
        elif v_err is None:
            reason = "level"                  # no usage split available for this market
        elif abs(v_err) >= abs(e_err or 0.0):
            reason = "volume"
        else:
            reason = "efficiency"
        rows.append({
            "season": season, "week": week, "player_id": c.get("player_id"),
            "name": c.get("name"), "team": c.get("team"), "opp": c.get("defteam"),
            "home": int(bool(c.get("home"))), "market": market,
            "proj_mean": round(float(mean), 4), "actual": round(actual, 4),
            "log_resid": round(log_resid, 5) if log_resid is not None else None,
            "volume_log_err": round(v_err, 5) if v_err is not None else None,
            "efficiency_log_err": round(e_err, 5) if e_err is not None else None,
            "primary_reason": reason, "early_exit": ee, "game_meaningless": mg,
            "created_at": _now(),
        })
    if rows:
        dbmod.upsert(conn, "player_week_residuals", rows,
                     ["season", "week", "player_id", "market"])
    return len(rows)


# --------------------------------------------------------------------------- #
# 2. The player bias: shrunk, player-SPECIFIC, walk-forward
# --------------------------------------------------------------------------- #
def compute_player_adjustments(conn, before: Optional[Tuple[int, int]] = None,
                               params: Optional[Dict] = None) -> Dict[Tuple[str, str], Dict]:
    """Walk-forward shrunk per-(player, market) mean multiplier.

    ``before=(season, week)``: use only ledger rows strictly before it. Returns
    {(player_id, market): {bias_mult, n_games, shrunk_ratio, market_ratio}}.
    A player-specific ratio is isolated from the market-average ratio, shrunk
    empirical-Bayes toward it, and clipped; players below ``min_games`` are
    omitted (they serve at 1.0)."""
    p = {**DEFAULTS, **(params or {})}
    df = dbmod.query_df(conn, "SELECT * FROM player_week_residuals ORDER BY season, week")
    if df.empty:
        return {}
    # Phase 8.4: context-tagged rows (truncated / rest weeks) are excluded
    # from the player-bias fit -- they measure availability, not model error
    if p.get("clean_context", True):
        for col in ("early_exit", "game_meaningless"):
            if col in df.columns:
                df = df[df[col].fillna(0) == 0]
        if df.empty:
            return {}
    if before is not None:
        s0, w0 = before
        df = df[(df["season"] < s0) | ((df["season"] == s0) & (df["week"] < w0))]
    if df.empty:
        return {}
    # trailing window per player-market handled by tail() after sort; market
    # ratio pooled over the same (pre-`before`) rows for each market.
    market_ratio: Dict[str, float] = {}
    for m, g in df.groupby("market"):
        sp = float(g["proj_mean"].sum())
        if len(g) >= int(p["min_market_n"]) and sp > 0:
            market_ratio[m] = float(g["actual"].sum()) / sp

    K = float(p["shrink_k"])
    lr = float(p["lr"])
    clip = float(p["bias_clip"])
    win = int(p["window_weeks"])
    out: Dict[Tuple[str, str], Dict] = {}
    for (pid, m), g in df.sort_values(["season", "week"]).groupby(["player_id", "market"]):
        g = g.tail(win)
        n = int(len(g))
        if n < int(p["min_games"]):
            continue
        sp = float(g["proj_mean"].sum())
        if sp <= 0:
            continue
        raw_ratio = float(g["actual"].sum()) / sp
        mr = market_ratio.get(m, 1.0)
        # empirical-Bayes shrink the player's ratio toward the market ratio
        shrunk = (n * raw_ratio + K * mr) / (n + K)
        # isolate the player-SPECIFIC part (deviation beyond the market average),
        # so a player who tracks the market gets ~1.0 and never double-counts the
        # market bias that prop_learning already applies.
        player_specific = shrunk / mr if mr > 0 else 1.0
        bias = float(np.clip(1.0 + lr * (player_specific - 1.0), 1.0 - clip, 1.0 + clip))
        out[(pid, m)] = {"bias_mult": round(bias, 4), "n_games": n,
                         "shrunk_ratio": round(shrunk, 4), "market_ratio": round(mr, 4)}
    return out


def persist_player_adjustments(conn, season: int, week: int,
                               adj: Dict[Tuple[str, str], Dict]) -> int:
    """Write the player adjustments that take effect AT (season, week)."""
    rows = [{"as_of_season": season, "as_of_week": week, "player_id": pid,
             "market": m, "bias_mult": a["bias_mult"], "n_games": a["n_games"],
             "shrunk_ratio": a["shrunk_ratio"], "updated_at": _now()}
            for (pid, m), a in adj.items()]
    if rows:
        dbmod.upsert(conn, "player_adjustments", rows,
                     ["as_of_season", "as_of_week", "player_id", "market"])
    return len(rows)


def load_player_adjustments(conn, season: int, week: int) -> Dict[Tuple[str, str], float]:
    """Player mean multipliers EFFECTIVE AT (season, week) — the latest row at or
    before it per (player, market). Same effective-at `<=` semantics as the market
    loop (rows are persisted at week+1 from strictly-prior data)."""
    df = dbmod.query_df(conn, """
        SELECT * FROM player_adjustments
        WHERE (as_of_season < ?) OR (as_of_season = ? AND as_of_week <= ?)
        ORDER BY as_of_season, as_of_week
        """, (season, season, week))
    out: Dict[Tuple[str, str], float] = {}
    for r in df.to_dict("records"):
        out[(r["player_id"], r["market"])] = float(r["bias_mult"])
    return out


def apply_player_bias(cands: pd.DataFrame, player_adj: Dict[Tuple[str, str], float],
                      enabled: bool = True) -> pd.DataFrame:
    """Multiply each candidate's mean by its player bias (recomputing line-relative
    probs), stamping ``player_bias_mult``. No-op when disabled or no adjustments —
    so with zero live data the projection is byte-identical to today."""
    if cands is None or cands.empty:
        return cands
    cands = cands.copy()
    if not enabled or not player_adj:
        cands["player_bias_mult"] = 1.0
        return cands
    from .projection import p_over as p_over_fn

    mult = [float(player_adj.get((pid, m), 1.0))
            for pid, m in zip(cands["player_id"], cands["market"])]
    cands["mean"] = (cands["mean"].astype(float) * pd.Series(mult, index=cands.index)).round(3)
    cands["player_bias_mult"] = [round(x, 4) for x in mult]
    new_po = [
        round(p_over_fn(m, s, l, d), 4) if l is not None and not pd.isna(l) else None
        for m, s, l, d in zip(cands["mean"], cands["sd"], cands["line"], cands["dist"])
    ]
    cands["p_over"] = new_po
    cands["p_under"] = [round(1 - po, 4) if po is not None else None for po in new_po]
    return cands


# --------------------------------------------------------------------------- #
# Read-side: what the ledger has learned (WHERE / HOW), for reporting + tuning
# --------------------------------------------------------------------------- #
def player_residual_report(conn, season: Optional[int] = None) -> Dict:
    """Aggregate the ledger: coverage, and where/how the model has been off."""
    q = "SELECT * FROM player_week_residuals"
    params: tuple = ()
    if season is not None:
        q += " WHERE season=?"
        params = (season,)
    df = dbmod.query_df(conn, q, params)
    if df.empty:
        return {"n": 0, "players": 0}
    home = df[df["home"] == 1]["log_resid"].mean()
    away = df[df["home"] == 0]["log_resid"].mean()
    return {
        "n": int(len(df)), "players": int(df["player_id"].nunique()),
        "how": df["primary_reason"].value_counts().to_dict(),
        "where_mean_log_resid": {"home": round(float(home), 4) if pd.notna(home) else None,
                                 "away": round(float(away), 4) if pd.notna(away) else None},
        "by_market_mean_log_resid": {m: round(float(g["log_resid"].mean()), 4)
                                     for m, g in df.groupby("market") if g["log_resid"].notna().any()},
    }

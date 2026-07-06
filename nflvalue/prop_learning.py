"""The weekly learning loop: grade what published, attribute WHY, adjust.

Every week, after results are in:

  1. GRADE  -- every lean (and every screened candidate) against the actual
              stat: hit/miss at the published line.
  2. ATTRIBUTE -- decompose each miss into the component that caused it:
              log(actual/pred) = log(actual_volume/proj_volume)
                               + log(actual_eff/proj_eff·opp)
              -> primary_reason in {volume_miss, efficiency_miss,
                 availability_surprise (player barely/never played),
                 script_flip (game script inverted vs the pre-game spread),
                 tail_variance (projection fine, outcome in the tail),
                 as_projected (hit)}.
              Stored per lean in ``lean_outcomes`` -- queryable via the RAG
              layer ("why did we miss on Chase in week 14?").
  3. ADJUST -- three bounded, WALK-FORWARD corrections per market, written to
              ``model_adjustments`` keyed by the week they take effect
              (computed ONLY from weeks strictly before it):

      bias_mult    mean(actual)/mean(pred) over the trailing candidate pool,
                   learned at ``lr``, clipped to ±8%. Learned from ALL
                   candidates, never just the picks -- learning bias from a
                   selected sample would bake selection bias into the model
                   (premortem: the ranked tail is partly noise).
      resid_sd     residual SD at the cutoff (the SD the projections publish).
      reliability  trailing hit-rate of the market's LEANS, shrunk hard to
                   0.5 (k=50 pseudo-leans) and clipped to [0.85, 1.15]; the
                   composite may multiply by it when learning is enabled --
                   markets that keep missing rank lower until they earn it.

  All three are deterministic, bounded, and reproducible: rerunning the same
  weeks yields the same adjustments. Nothing here touches context/news --
  personal-context learning lives in ``context_study.py`` behind its own
  evidence gate.

Config (config.json "learning"): {"enabled": true, "lr": 0.35,
"bias_clip": 0.08, "reliability_k": 50, "reliability_clip": 0.15,
"window_candidates": 2000}.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import db as dbmod

DEFAULTS = {"enabled": True, "lr": 0.35, "bias_clip": 0.08,
            "reliability_k": 50.0, "reliability_clip": 0.15,
            "window_weeks": 12}

_USAGE_COL = {"targets": "targets", "carries": "carries", "pass_attempts": "pass_attempts"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# 1+2. Grade + attribute one week's leans
# --------------------------------------------------------------------------- #
def attribute(lean: Dict, player_actual_row: Optional[Dict],
              margin_expected: Optional[float], margin_actual: Optional[float]) -> Dict:
    """Why did this lean hit or miss? Deterministic decomposition."""
    from .projection import MARKETS
    out = {"volume_log_err": None, "efficiency_log_err": None}
    hit = bool(lean["hit"])
    if hit:
        return {**out, "primary_reason": "as_projected",
                "detail": "actual landed on the projected side"}

    if player_actual_row is None:
        return {**out, "primary_reason": "availability_surprise",
                "detail": "no stat line recorded — player effectively absent"}

    spec = MARKETS.get(lean["market"], {})
    opp_key = spec.get("opportunity")
    proj_comps = lean.get("proj_components") or {}
    proj_volume = proj_comps.get("volume")
    actual_volume = player_actual_row.get(_USAGE_COL.get(opp_key, ""), None)

    # a projected contributor who saw ~no usage is an availability story
    if actual_volume is not None and proj_volume:
        if actual_volume <= 0.25 * float(proj_volume) and float(proj_volume) >= 4:
            return {**out, "primary_reason": "availability_surprise",
                    "detail": f"usage collapsed: {actual_volume:g} vs projected {proj_volume:g}"}

    v_err = e_err = None
    if actual_volume and proj_volume and actual_volume > 0 and proj_volume > 0:
        v_err = math.log(actual_volume / float(proj_volume))
        total_err = (math.log(lean["actual"] / lean["mean"])
                     if lean["actual"] > 0 and lean["mean"] > 0 else None)
        e_err = (total_err - v_err) if total_err is not None else None
        out["volume_log_err"] = round(v_err, 4)
        out["efficiency_log_err"] = round(e_err, 4) if e_err is not None else None

    # script flip: pre-game favorite trailed (or dog led) by 10+, and the
    # volume error points the way a flipped script would push it
    if (margin_expected is not None and margin_actual is not None
            and abs(margin_expected) >= 2.5
            and np.sign(margin_actual) != np.sign(margin_expected)
            and abs(margin_actual) >= 10 and v_err is not None and abs(v_err) > 0.2):
        return {**out, "primary_reason": "script_flip",
                "detail": (f"expected margin {margin_expected:+g}, actual {margin_actual:+g}; "
                           f"usage moved {math.exp(v_err)-1:+.0%}")}

    if v_err is not None and e_err is not None:
        if abs(v_err) >= abs(e_err) and abs(v_err) > 0.15:
            return {**out, "primary_reason": "volume_miss",
                    "detail": f"usage off by {math.exp(v_err)-1:+.0%} vs projection"}
        if abs(e_err) > abs(v_err) and abs(e_err) > 0.15:
            return {**out, "primary_reason": "efficiency_miss",
                    "detail": f"per-opportunity efficiency off by {math.exp(e_err)-1:+.0%}"}
    return {**out, "primary_reason": "tail_variance",
            "detail": "projection components were close; outcome landed in the tail"}


def grade_week(conn, season: int, week: int, pw: pd.DataFrame,
               leans: Optional[pd.DataFrame] = None,
               schedules: Optional[pd.DataFrame] = None, clock: str = "wed") -> pd.DataFrame:
    """Grade + attribute a completed week's leans into ``lean_outcomes``.

    ``leans`` defaults to the DB's active leans for (season, week, clock);
    replay callers pass their own frame (with proj_components attached).
    """
    from .candidates import ACTUAL_COL

    if leans is None:
        leans = dbmod.query_df(conn, """
            SELECT * FROM leans WHERE season=? AND week=? AND clock=? AND status='active'
            """, (season, week, clock))
        leans["proj_components"] = None
    if leans.empty:
        return pd.DataFrame()

    wk = pw[(pw["season"] == season) & (pw["week"] == week)]
    actual_rows = {r["player_id"]: r for r in wk.to_dict("records")}

    margins = {}
    if schedules is not None:
        s = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
        for g in s.itertuples(index=False):
            exp = float(g.spread_line) if pd.notna(g.spread_line) else None
            act = float(g.result) if pd.notna(getattr(g, "result", None)) else None
            margins[g.game_id] = (exp, act)

    rows: List[Dict] = []
    for l in leans.to_dict("records"):
        arow = actual_rows.get(l["player_id"])
        if l["market"] == "anytime_td":
            actual = (arow["rush_tds"] + arow["rec_tds"]) if arow else 0.0
            hit = actual >= 1.0
        else:
            col = ACTUAL_COL[l["market"]]
            actual = arow[col] if arow else 0.0
            hit = (actual > l["line"]) if l["side"] == "over" else (actual < l["line"])
        exp_m, act_m = margins.get(l["game_id"], (None, None))
        # margins are home-relative; no per-team flip is needed because the
        # script_flip check compares sign(spread_line) vs sign(result), both
        # home-relative, so the comparison is team-agnostic.
        graded = {**l, "actual": float(actual), "hit": bool(hit)}
        attr = attribute(graded, arow, exp_m, act_m)
        rows.append({
            "season": season, "week": week, "clock": clock, "game_id": l["game_id"],
            "player_id": l["player_id"], "name": l["name"], "market": l["market"],
            "side": l["side"], "line": l["line"], "mean": l["mean"],
            "composite": l["composite"], "actual": float(actual), "hit": int(hit),
            "primary_reason": attr["primary_reason"],
            "volume_log_err": attr.get("volume_log_err"),
            "efficiency_log_err": attr.get("efficiency_log_err"),
            "detail": attr.get("detail", ""), "graded_at": _now(),
        })
    dbmod.upsert(conn, "lean_outcomes", rows,
                 ["season", "week", "clock", "game_id", "player_id", "market"])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Adjustments (pure state-update; DB persistence optional)
# --------------------------------------------------------------------------- #
class LearningState:
    """Walk-forward learning state, identical for replays and live runs.

    Fed WEEKLY AGGREGATES so it is exactly rebuildable from the DB:
      * ``candidate_weeks``: per market, trailing (n, sum_pred, sum_actual)
        of the FULL screened pool (selection-bias guard: never picks-only).
      * ``lean_history``: per market, trailing hit flags of published leans.
    """

    def __init__(self, params: Optional[Dict] = None):
        self.p = {**DEFAULTS, **(params or {})}
        self.candidate_weeks: Dict[str, List] = {}   # [(n, sum_pred, sum_actual), ...]
        self.lean_history: Dict[str, List] = {}
        self.bias_mult: Dict[str, float] = {}

    # -- ingest one completed week ---------------------------------------- #
    def observe(self, market: str, n: int, sum_pred: float, sum_actual: float,
                lean_hits: List[int]) -> None:
        cw = self.candidate_weeks.setdefault(market, [])
        cw.append((int(n), float(sum_pred), float(sum_actual)))
        del cw[:-int(self.p["window_weeks"])]
        lh = self.lean_history.setdefault(market, [])
        lh.extend(int(h) for h in lean_hits)
        del lh[:-500]
        self._update_bias(market)

    def _update_bias(self, market: str) -> None:
        """Direct shrunk estimate from the trailing RAW-prediction window --
        deliberately NOT an iterative update on the previous multiplier, so
        the value is path-independent (rebuildable from the DB) and can never
        compound toward the clip on a stable bias."""
        cw = self.candidate_weeks.get(market) or []
        n_total = sum(n for n, _, _ in cw)
        sum_pred = sum(sp for _, sp, _ in cw)
        if n_total < 100 or sum_pred <= 0:
            return
        ratio = sum(sa for _, _, sa in cw) / sum_pred
        lr, clip = float(self.p["lr"]), float(self.p["bias_clip"])
        self.bias_mult[market] = float(np.clip(1.0 + lr * (float(ratio) - 1.0),
                                               1.0 - clip, 1.0 + clip))

    # -- what next week's run should use ----------------------------------- #
    def adjustments(self) -> Dict[str, Dict]:
        out = {}
        k = float(self.p["reliability_k"])
        clip = float(self.p["reliability_clip"])
        markets = set(self.candidate_weeks) | set(self.lean_history)
        for m in markets:
            lh = self.lean_history.get(m) or []
            n = len(lh)
            h_shrunk = (sum(lh) + k * 0.5) / (n + k) if (n + k) else 0.5
            # slope 1.0: a market must be ~15 points off 50% AFTER shrinkage to
            # hit the clip -- persistent failure gets nudged down, not nuked
            reliability = float(np.clip(1.0 + (h_shrunk - 0.5), 1.0 - clip, 1.0 + clip))
            out[m] = {"bias_mult": self.bias_mult.get(m, 1.0),
                      "reliability": reliability,
                      "n_candidates": sum(x[0] for x in self.candidate_weeks.get(m, [])),
                      "n_leans": n}
        return out

    def persist(self, conn, season: int, week: int) -> None:
        """Write the adjustments that take effect AT (season, week)."""
        rows = [{"as_of_season": season, "as_of_week": week, "market": m,
                 "bias_mult": a["bias_mult"], "resid_sd": None,
                 "reliability": a["reliability"], "n_candidates": a["n_candidates"],
                 "n_leans": a["n_leans"], "updated_at": _now()}
                for m, a in self.adjustments().items()]
        if rows:
            dbmod.upsert(conn, "model_adjustments", rows,
                         ["as_of_season", "as_of_week", "market"])


def load_adjustments(conn, season: int, week: int) -> Dict[str, Dict]:
    """Adjustments EFFECTIVE AT (season, week) per market.

    The `<=` in the SQL is intentional (effective-at semantics): each row is
    persisted at week+1 from data strictly BEFORE it, so the row whose
    (as_of_season, as_of_week) equals the target week already contains only
    pre-week data. Loading `<=` therefore picks the adjustment in effect AT the
    week and never leaks current-week outcomes. Do NOT "fix" this to `<`.
    """
    df = dbmod.query_df(conn, """
        SELECT * FROM model_adjustments
        WHERE (as_of_season < ?) OR (as_of_season = ? AND as_of_week <= ?)
        ORDER BY as_of_season, as_of_week
        """, (season, season, week))
    out: Dict[str, Dict] = {}
    for r in df.to_dict("records"):
        out[r["market"]] = {"bias_mult": r["bias_mult"], "reliability": r["reliability"],
                            "n_candidates": r["n_candidates"], "n_leans": r["n_leans"]}
    return out


def apply_to_candidates(cands: pd.DataFrame, adjustments: Dict[str, Dict],
                        enabled: bool = True) -> pd.DataFrame:
    """Apply bias_mult to means (line-relative probs recomputed) and stamp
    reliability for the composite. No-op when disabled or no adjustments."""
    if not enabled or not adjustments or cands.empty:
        if not cands.empty:
            cands = cands.copy()
            cands["reliability_mult"] = 1.0
        return cands
    from .projection import p_over as p_over_fn

    cands = cands.copy()
    bias = cands["market"].map(lambda m: (adjustments.get(m) or {}).get("bias_mult", 1.0))
    cands["mean"] = (cands["mean"] * bias).round(3)
    new_po = [
        round(p_over_fn(m, s, l, d), 4) if l is not None and not pd.isna(l) else None
        for m, s, l, d in zip(cands["mean"], cands["sd"], cands["line"], cands["dist"])
    ]
    cands["p_over"] = new_po
    cands["p_under"] = [round(1 - p, 4) if p is not None else None for p in new_po]
    cands["bias_mult"] = bias.round(4)
    cands["reliability_mult"] = cands["market"].map(
        lambda m: (adjustments.get(m) or {}).get("reliability", 1.0)).round(4)
    return cands


def record_candidate_aggregates(conn, season: int, week: int,
                                cands: pd.DataFrame, pw: pd.DataFrame) -> int:
    """Per-market (n, Σpred, Σactual) for the full screened pool of a
    completed week — the selection-bias-safe food for bias learning."""
    from .candidates import ACTUAL_COL
    wk = pw[(pw["season"] == season) & (pw["week"] == week)]
    actual_rows = {r["player_id"]: r for r in wk.to_dict("records")}
    rows = []
    for market, grp in cands.groupby("market"):
        n = s_pred = s_act = 0
        for c in grp.itertuples(index=False):
            arow = actual_rows.get(c.player_id)
            if arow is None:
                continue
            actual = ((arow["rush_tds"] + arow["rec_tds"]) if market == "anytime_td"
                      else arow[ACTUAL_COL[market]])
            n += 1
            s_pred += float(c.mean)
            s_act += float(actual)
        if n:
            rows.append({"season": season, "week": week, "market": market,
                         "n": n, "sum_pred": round(s_pred, 3),
                         "sum_actual": round(s_act, 3), "created_at": _now()})
    if rows:
        dbmod.upsert(conn, "candidate_aggregates", rows, ["season", "week", "market"])
    return len(rows)


def rebuild_state(conn, params: Optional[Dict] = None,
                  before: Optional[tuple] = None) -> LearningState:
    """Deterministically rebuild the learning state from the DB (aggregates +
    graded lean outcomes), optionally only weeks strictly before ``before``."""
    state = LearningState(params)
    agg = dbmod.query_df(conn, "SELECT * FROM candidate_aggregates ORDER BY season, week")
    outs = dbmod.query_df(conn, "SELECT * FROM lean_outcomes ORDER BY season, week")
    weeks = sorted({(int(r.season), int(r.week)) for r in agg.itertuples(index=False)})
    for (s, w) in weeks:
        if before is not None and (s, w) >= before:
            break
        wk_agg = agg[(agg["season"] == s) & (agg["week"] == w)]
        wk_out = outs[(outs["season"] == s) & (outs["week"] == w)] if len(outs) else outs
        for r in wk_agg.itertuples(index=False):
            hits = (wk_out[wk_out["market"] == r.market]["hit"].tolist()
                    if len(wk_out) else [])
            state.observe(r.market, r.n, r.sum_pred, r.sum_actual, hits)
    return state


def grade_and_learn(conn, season: int, week: int, inputs, clock: str = "wed",
                    params: Optional[Dict] = None,
                    player_params: Optional[Dict] = None) -> Dict:
    """The Tuesday step: grade last week's leans, attribute the misses,
    fold the week into the learning state, persist next week's adjustments."""
    from .candidates import enumerate_candidates

    # re-enumerate the completed week (deterministic) so misses can be
    # attributed against the same projection components the leans came from
    cands = enumerate_candidates(season, week, inputs=inputs)
    comp_lookup = {(c["player_id"], c["market"]): c.get("components")
                   for c in cands.to_dict("records")}
    leans = dbmod.query_df(conn, """
        SELECT * FROM leans WHERE season=? AND week=? AND clock=? AND status='active'
        """, (season, week, clock))
    if not leans.empty:
        leans["proj_components"] = [comp_lookup.get((p, m))
                                    for p, m in zip(leans["player_id"], leans["market"])]
    outcomes = grade_week(conn, season, week, inputs.pw, leans=leans,
                          schedules=inputs.schedules, clock=clock)
    record_candidate_aggregates(conn, season, week, cands, inputs.pw)

    state = rebuild_state(conn, params=params)
    nxt_week = week + 1
    state.persist(conn, season, nxt_week)

    # player-level sequential layer: record this week's ledger (what each player
    # did, where, how) and persist next week's SHRUNK player adjustments. The
    # ledger accrues even while the APPLY layer is OFF in config -- so the system
    # is "ready to take data" from week one. Guarded so it can never break the
    # market loop above.
    try:
        from . import player_learning as plmod
        n_res = plmod.record_player_residuals(conn, season, week, cands, inputs.pw)
        padj = plmod.compute_player_adjustments(conn, before=(season, nxt_week),
                                                params=(player_params or None))
        plmod.persist_player_adjustments(conn, season, nxt_week, padj)
    except Exception as exc:  # noqa: BLE001
        n_res, padj = 0, {}
        print(f"[learn] player-residual layer skipped ({exc})")
    return {"graded": int(len(outcomes)),
            "hit_rate": (round(float(outcomes["hit"].mean()), 4) if len(outcomes) else None),
            "adjustments_effective": (season, nxt_week),
            "adjustments": state.adjustments(),
            "player_residuals_recorded": n_res,
            "player_adjustments_effective": len(padj),
            "why": why_report(conn, season)}


def why_report(conn, season: int, last_n_weeks: int = 4) -> Dict:
    """Aggregate the attribution ledger: what's been missing and why."""
    df = dbmod.query_df(conn, """
        SELECT * FROM lean_outcomes WHERE season=? ORDER BY week
        """, (season,))
    if df.empty:
        return {"n": 0}
    recent = df[df["week"] >= df["week"].max() - last_n_weeks + 1]
    return {
        "n": int(len(df)), "hit_rate": round(float(df["hit"].mean()), 4),
        "recent_weeks": sorted(recent["week"].unique().tolist()),
        "recent_hit_rate": round(float(recent["hit"].mean()), 4),
        "miss_reasons": df[df["hit"] == 0]["primary_reason"].value_counts().to_dict(),
        "recent_miss_reasons": recent[recent["hit"] == 0]["primary_reason"].value_counts().to_dict(),
        "by_market_hit": {m: round(float(g["hit"].mean()), 4)
                          for m, g in df.groupby("market")},
    }

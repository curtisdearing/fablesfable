#!/usr/bin/env python3
"""Season replay: what would the weekly top-5 leans have done?

For every REG week of a completed season, run the EXACT production path
(enumerate -> composite -> top-5 per game with the per-player cap), then
grade each lean against what the player actually did.

READ THIS BEFORE READING THE NUMBERS (PREMORTEM.md F9 / spec §5):
  * There is no free historical prop-PRICE data, so leans are graded at the
    tool's SYNTHETIC reference line (the player's own trailing mean,
    floor+0.5) -- the same line the report would have printed that week.
  * A hit therefore means "the model's side of its own reference line was
    right", i.e. DIRECTIONAL skill vs a naive trailing-mean baseline.
    It is NOT price-beating, NOT CLV, NOT profit. If these had been real
    -110 lines at these numbers, breakeven would be 52.38%.
  * Selection honesty: the same grading is run over ALL screened candidates,
    so the top-5's hit rate can be compared to "just bet everything" -- the
    composite earns its keep only by beating its own candidate pool.
  * Walk-forward end to end: features, SDs, and lines at week W use only
    weeks < W (season boundaries included via prior seasons).

Run:  python3 lean_backtest.py --season 2025 \
          [--pw /tmp/pw.parquet --opd ... --tw ... --sched ...]
Writes: data/lean_replay_{season}.json + reports/lean_replay_{season}.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from nflvalue import config as cfgmod
from nflvalue.candidates import ACTUAL_COL, WeekInputs, enumerate_candidates
from nflvalue.projection import MARKETS
from nflvalue.shortlist import shortlist_week

BREAKEVEN = 0.5238  # win rate needed at standard -110 juice
UNIT_WIN = 100.0 / 110.0   # profit (in units) per 1u win at standard -110 juice


# --------------------------------------------------------------------------- #
# Walk-forward per-market SD for every week cutoff, in ONE pass per market
# --------------------------------------------------------------------------- #
def precompute_sds(inputs: WeekInputs, markets: List[str],
                   min_history: int = 30) -> Dict[tuple, Dict[str, Optional[float]]]:
    """{(season, week) -> {market -> sd}} where sd is the std of residuals of
    all (season', week') strictly before that cutoff -- identical in
    definition to candidates.market_residual_sd, computed incrementally."""
    import prop_backtest

    out: Dict[tuple, Dict[str, Optional[float]]] = {}
    for market in markets:
        preds = prop_backtest._predictions_for_market(
            inputs.pw, market, inputs.team_idx, inputs.opp_idx)
        if preds.empty:
            continue
        preds = preds.sort_values(["season", "week"], kind="mergesort")
        resid = (preds["actual"] - preds["mean_pred"]).to_numpy(dtype=float)
        keys = list(zip(preds["season"].to_numpy(), preds["week"].to_numpy()))
        n = s = ss = 0.0
        i = 0
        while i < len(resid):
            j = i
            while j < len(resid) and keys[j] == keys[i]:
                j += 1
            # sd at THIS week's cutoff = residuals of all strictly-prior weeks
            sd = None
            if n >= min_history:
                var = (ss - (s * s) / n) / (n - 1)
                sd = float(np.sqrt(var)) if var > 0 else None
            out.setdefault(keys[i], {})[market] = sd
            chunk = resid[i:j]
            chunk = chunk[np.isfinite(chunk)]
            n += len(chunk)
            s += float(chunk.sum())
            ss += float((chunk ** 2).sum())
            i = j
    return out


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def _actuals_for_week(pw: pd.DataFrame, season: int, week: int) -> Dict[tuple, float]:
    wk = pw[(pw["season"] == season) & (pw["week"] == week)]
    idx: Dict[tuple, float] = {}
    for r in wk.itertuples(index=False):
        for market, col in ACTUAL_COL.items():
            idx[(r.player_id, market)] = float(getattr(r, col))
        idx[(r.player_id, "anytime_td")] = float(r.rush_tds + r.rec_tds)
    return idx


def grade(row: Dict, actuals: Dict[tuple, float]) -> Optional[Dict]:
    actual = actuals.get((row["player_id"], row["market"]))
    if actual is None:
        return None
    if row["market"] == "anytime_td":
        hit = actual >= 1.0  # YES side only
    elif row["side"] == "over":
        hit = actual > row["line"]
    else:
        hit = actual < row["line"]
    return {"hit": bool(hit), "actual": actual}


def _rate(frame: pd.DataFrame) -> Dict:
    n = len(frame)
    if not n:
        return {"n": 0, "hit_rate": None, "units": None}
    hits = frame["hit"].astype(bool)
    # standard -110 juice, flat 1u/lean -- NOT a claim these are real tradeable
    # prices (see module docstring: synthetic-line grading); purely so the
    # correlation-aware ablation has a units delta alongside hit-rate.
    units = float(hits.sum()) * UNIT_WIN - float((~hits).sum())
    return {"n": n, "hit_rate": round(float(hits.mean()), 4),
           "units": round(units, 2)}


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def run(season: int, inputs: WeekInputs, weeks: Optional[List[int]] = None,
        write_files: bool = True, learn: bool = False,
        learn_params: Optional[Dict] = None,
        corr_aware: bool = False, corr_discount_strength: float = 1.0,
        out_suffix: str = "") -> Dict:
    """``learn=True`` replays ADAPTIVELY: each week is ranked with the bias/
    reliability adjustments learned from the weeks already graded (walk-
    forward -- week W never sees week W's outcomes), exactly like the live
    Tuesday --grade loop would have applied them.

    ``corr_aware=True`` (Phase 7.6 ablation): selection reads the 7.5
    correlation artifact with a STRICT walk-forward rho (``as_of_season=
    season`` -- only prior seasons inform it, exactly like every other
    walk-forward artifact in this repo). Missing/empty artifact degrades
    loudly to ``corr_aware`` having no effect (identical to False).
    ``out_suffix``: appended to the written filenames so a baseline and a
    corr-aware run don't clobber each other on disk.
    """
    from nflvalue import prop_learning

    cfg = cfgmod.load_config()
    weights = (cfg.get("composite") or {}).get("weights")
    params = (cfg.get("composite") or {}).get("params")
    min_usage = (cfg.get("candidates") or {}).get("min_usage")
    state = prop_learning.LearningState(learn_params or cfg.get("learning")) if learn else None

    corr = None
    if corr_aware:
        from nflvalue.correlation import ART_PATH, CorrelationStructure
        if os.path.exists(ART_PATH):
            cs = CorrelationStructure.load()
            if cs.pair_types:
                corr = cs
            else:
                print("[lean_backtest] correlation artifact is empty -- "
                      "corr_aware has no effect this run.")
        else:
            print(f"[lean_backtest] correlation artifact missing ({ART_PATH}) -- "
                  "corr_aware has no effect this run.")

    all_weeks = sorted(inputs.schedules[
        (inputs.schedules["season"] == season)
        & (inputs.schedules["game_type"] == "REG")]["week"].unique().tolist())
    weeks = weeks or all_weeks

    markets = list(MARKETS)
    sd_map = precompute_sds(inputs, markets)
    from nflvalue.candidates import apply_weather_adjustment, synthetic_lines
    synth_map = {m: synthetic_lines(inputs, m) for m in markets}  # week-independent

    # Phase 6.4: observed per-game weather for the fitted pass-family
    # multiplier (live runs use the kickoff forecast instead -- same
    # train/serve note as the ML weather features)
    wx_map: Dict = {}
    try:
        from nflvalue.weather_study import build_game_weather
        gwx = build_game_weather()
        wx_map = {r.game_id: {"wind_mph": r.wind_mph, "precip_mm": (2.0 if r.precip_flag else 0.0),
                              "effective_outdoor": bool(r.effective_outdoor)}
                  for r in gwx.itertuples(index=False)}
    except Exception as exc:  # noqa: BLE001 -- replay still runs, weather-neutral
        print(f"[lean_backtest] game weather unavailable ({exc}); no weather adjustment")

    # Phase 6.5: opponent-secondary outs (official reports; pre-game info)
    db_outs: Dict = {}
    try:
        from nflvalue.candidates import apply_opp_absence_factor
        from nflvalue.context_features import (DB_POS, OUT_STATUSES,
                                               load_injury_history)
        inj = load_injury_history(sorted(inputs.pw["season"].unique().tolist()))
        d = inj[inj["report_status"].isin(OUT_STATUSES) & inj["position"].isin(DB_POS)]
        db_outs = {(int(s), int(w), str(t)): int(len(g))
                   for (s, w, t), g in d.groupby(["season", "week", "team"])}
    except Exception as exc:  # noqa: BLE001
        print(f"[lean_backtest] injury history unavailable ({exc}); no absence factor")

    lean_rows, cand_rows = [], []
    for wk in weeks:
        cands_raw = enumerate_candidates(season, wk, inputs=inputs, min_usage=min_usage,
                                         sd_by_market=sd_map.get((season, wk), {}),
                                         synth_by_market=synth_map)
        if cands_raw.empty:
            continue
        cands_raw = apply_weather_adjustment(cands_raw, wx_map)
        if db_outs:
            cands_raw = apply_opp_absence_factor(cands_raw, db_outs)
        cands = (prop_learning.apply_to_candidates(cands_raw, state.adjustments(), enabled=True)
                 if state is not None else cands_raw)
        actuals = _actuals_for_week(inputs.pw, season, wk)
        games = shortlist_week(cands, weights=weights, params=params,
                               corr=corr, as_of_season=(season if corr is not None else None),
                               corr_discount_strength=corr_discount_strength)

        lean_keys = set()
        for g in games:
            for rank, l in enumerate(g["leans"], start=1):
                graded = grade(l, actuals)
                if graded is None:
                    continue
                lean_keys.add((l["player_id"], l["market"]))
                lean_rows.append({
                    "season": season, "week": wk, "game_id": g["game_id"],
                    "matchup": g["matchup"], "rank": rank,
                    "player_id": l["player_id"], "name": l["name"], "pos": l["pos"],
                    "team": l.get("team"),
                    "market": l["market"], "side": l["side"], "line": l["line"],
                    "mean": l["mean"], "composite": l["composite"],
                    "corr_discount": l.get("corr_discount", 0.0),
                    "corr_with": (l.get("corr_with") or {}).get("name"),
                    "screened_n": g["screened_n"], **graded,
                })
        # baseline: EVERY screened candidate, model side at the same line
        from nflvalue.composite import score_candidate
        for c in cands.to_dict("records"):
            s = score_candidate(c, weights=weights, params=params)
            graded = grade({**c, "side": s["side"]}, actuals)
            if graded is None:
                continue
            cand_rows.append({"season": season, "week": wk, "market": c["market"],
                              "side": s["side"], "composite": s["composite"],
                              "is_lean": (c["player_id"], c["market"]) in lean_keys,
                              **graded})

        # adaptive mode: fold this week's outcomes into the state AFTER using
        # it (raw prediction sums; lean hits per market) -- strictly walk-forward
        if state is not None:
            wk_leans = [r for r in lean_rows if r["week"] == wk]
            hits_by_market: Dict[str, List[int]] = {}
            for r in wk_leans:
                hits_by_market.setdefault(r["market"], []).append(int(r["hit"]))
            for market, grp in cands_raw.groupby("market"):
                s_pred = n = s_act = 0
                for c in grp.itertuples(index=False):
                    a = actuals.get((c.player_id, market))
                    if a is None:
                        continue
                    n += 1
                    s_pred += float(c.mean)
                    s_act += float(a)
                if n:
                    state.observe(market, n, s_pred, s_act,
                                  hits_by_market.get(market, []))

    leans = pd.DataFrame(lean_rows)
    cands_all = pd.DataFrame(cand_rows)

    report = {
        "season": season, "weeks": weeks, "corr_aware": corr is not None,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "framing": ("Directional grading at SYNTHETIC trailing-mean lines -- the lines the "
                    "tool would have published. NOT price-beating/profit; breakeven at a real "
                    f"-110 line would be {BREAKEVEN:.2%}. Units at flat 1u/lean, standard -110 juice "
                    "-- a units delta between selection policies is informative even though the "
                    "underlying win rate is synthetic-line directional, not price-beating."),
        "leans": {
            "overall": _rate(leans),
            "top1_per_game": _rate(leans[leans["rank"] == 1]),
            "by_market": {m: _rate(g) for m, g in leans.groupby("market")},
            "by_side": {s: _rate(g) for s, g in leans.groupby("side")},
            "by_composite_band": {
                label: _rate(leans[(leans["composite"] >= lo) & (leans["composite"] < hi)])
                for label, lo, hi in (("<35", 0, 35), ("35-45", 35, 45),
                                      ("45-55", 45, 55), ("55+", 55, 1000))},
            "by_week": {int(wk): _rate(g) for wk, g in leans.groupby("week")},
        },
        "baseline_all_candidates": {
            "overall": _rate(cands_all),
            "non_lean_only": _rate(cands_all[~cands_all["is_lean"]]) if len(cands_all) else {},
        },
        "n_games": int(leans["game_id"].nunique()) if len(leans) else 0,
        "avg_screened_per_game": (round(float(leans.groupby("game_id")["screened_n"].first().mean()), 1)
                                  if len(leans) else None),
    }

    if write_files:
        os.makedirs("reports", exist_ok=True)
        cfgmod.save_json(os.path.join(cfgmod.DATA_DIR, f"lean_replay_{season}{out_suffix}.json"),
                         {**report, "lean_rows": lean_rows})
        with open(os.path.join("reports", f"lean_replay_{season}{out_suffix}.md"), "w") as f:
            f.write(render_md(report, leans))
    return {"report": report, "leans": leans, "candidates": cands_all}


def render_md(report: Dict, leans: pd.DataFrame) -> str:
    L = report["leans"]
    lines = [
        f"# Lean replay — {report['season']} season (weeks {report['weeks'][0]}–{report['weeks'][-1]})",
        "",
        "**Leans, not locks.** " + report["framing"],
        "Gambling problem? **1-800-GAMBLER**.",
        "",
        f"- Leans graded: **{L['overall']['n']}** across {report['n_games']} games "
        f"(avg {report['avg_screened_per_game']} candidates screened per game)"
        + (" — **correlation-aware selection**" if report.get("corr_aware") else " — baseline selection"),
        f"- **Overall hit rate: {L['overall']['hit_rate']:.1%}** "
        f"(directional, vs {BREAKEVEN:.1%} breakeven if these were real -110 lines) · "
        f"units: {L['overall']['units']:+.1f}u flat 1u/lean at -110",
        f"- Top-1 pick per game: {L['top1_per_game']['hit_rate']:.1%} (n={L['top1_per_game']['n']})",
        f"- All-candidates baseline: {report['baseline_all_candidates']['overall']['hit_rate']:.1%} "
        f"(n={report['baseline_all_candidates']['overall']['n']}) — the composite must beat this "
        "for the ranking to mean anything.",
        "",
        "| Market | n | hit rate |", "|---|---|---|",
    ]
    for m, r in sorted(L["by_market"].items()):
        lines.append(f"| {m.replace('_', ' ')} | {r['n']} | {r['hit_rate']:.1%} |")
    lines += ["", "| Composite band | n | hit rate |", "|---|---|---|"]
    for band, r in L["by_composite_band"].items():
        if r["n"]:
            lines.append(f"| {band} | {r['n']} | {r['hit_rate']:.1%} |")
    lines += ["", "| Side | n | hit rate |", "|---|---|---|"]
    for s, r in L["by_side"].items():
        lines.append(f"| {s} | {r['n']} | {r['hit_rate']:.1%} |")
    lines += ["", "Weekly: " + ", ".join(
        f"wk{wk} {r['hit_rate']:.0%} (n={r['n']})" for wk, r in sorted(L["by_week"].items())), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--pw"), ap.add_argument("--opd"), ap.add_argument("--tw"), ap.add_argument("--sched")
    ap.add_argument("--weeks", type=int, nargs="*", default=None)
    ap.add_argument("--learn", action="store_true",
                    help="adaptive replay: apply the weekly learning loop walk-forward")
    ap.add_argument("--corr-aware", action="store_true",
                    help="Phase 7.6: correlation-aware selection (walk-forward rho)")
    args = ap.parse_args()
    if args.pw:
        inputs = WeekInputs(pd.read_parquet(args.pw), pd.read_parquet(args.opd),
                            pd.read_parquet(args.tw), pd.read_parquet(args.sched))
    else:
        from nflvalue.candidates import build_week_inputs
        inputs = build_week_inputs()
    res = run(args.season, inputs, weeks=args.weeks, learn=args.learn,
             corr_aware=args.corr_aware, out_suffix=("_corraware" if args.corr_aware else ""))
    r = res["report"]["leans"]["overall"]
    print(f"{args.season}: {r['n']} leans, hit rate {r['hit_rate']:.1%} "
          f"(baseline {res['report']['baseline_all_candidates']['overall']['hit_rate']:.1%})")


if __name__ == "__main__":
    main()

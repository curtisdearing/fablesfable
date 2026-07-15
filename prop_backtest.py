#!/usr/bin/env python3
"""Walk-forward accuracy backtest for the player-prop projection engine.

For every (season, week, player) in 2019-2023 we project each eligible
market using ONLY rolling features built from STRICTLY PRIOR weeks (see
``nflvalue/features.py``), then compare the projected mean against what
actually happened. Reports, per market and per usage-sample-size bucket:

    * MAE / RMSE           -- how far off the point projection tends to be
    * correlation           -- projection vs actual, linear association
    * calibration of P(over) against a SYNTHETIC line (each player's own
      trailing rolling median of that stat -- no real sportsbook price is
      used anywhere in this file)

IMPORTANT, per PROP_SHORTLISTER_SPEC.md §5 / PHASE1_HANDSOFF_DESIGN.md:
this measures PROJECTION ACCURACY, NOT PRICE-BEATING. There is no free
historical player-prop LINE data (props only exist for ~2019+ at paid
providers), so "does the model beat the market" can only be tested forward,
live, once real prop lines are pulled (Phase 3) -- it is NOT tested here and
this script makes no such claim.

The residual SD used for calibration is itself walk-forward: it's an
expanding standard deviation of PAST prediction errors only (``.shift(1)``
before ``.expanding()``), so no future outcome ever sizes a past prediction's
uncertainty band. The LLM synthesis layer (nflvalue/synthesis.py, Phase 1B)
is never invoked here -- backtests run the deterministic model alone.

Run:  python3 prop_backtest.py [--seasons 2019 2020 2021 2022 2023]
Writes: data/prop_backtest.json (+ upserts nflvalue/db.py's prop_backtest table)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from nflvalue import config, db as dbmod
from nflvalue.features import build_opp_pos_def, build_player_week, build_team_week, load_pbp
from nflvalue.projection import DEFAULT_SD_FRACTION, MARKETS, MIN_GAMES_ELIGIBLE, p_over as p_over_fn
from nflvalue.sources import rosters as rostersmod

ACTUAL_COL = {
    "receiving_yards": "rec_yards",
    "receptions": "receptions",
    "rushing_yards": "rush_yards",
    "passing_yards": "pass_yards",
    "pass_attempts": "pass_attempts",
    "rush_attempts": "carries",
}

SAMPLE_BUCKETS = [(0, 0, "0 games (cold start)"), (1, 3, "1-3 games"),
                  (4, 7, "4-7 games"), (8, 10 ** 9, "8+ games")]


def _bucket(n) -> str:
    n = 0 if pd.isna(n) else int(n)
    for lo, hi, label in SAMPLE_BUCKETS:
        if lo <= n <= hi:
            return label
    return "unknown"


def _predictions_for_market(pw: pd.DataFrame, market: str, team_idx: Dict, opp_idx: Dict) -> pd.DataFrame:
    spec = MARKETS[market]
    if market == "anytime_td":
        rows = pw[pw["role"].isin(spec["role"])].copy()
        rows["actual"] = rows["rush_tds"] + rows["rec_tds"]
    else:
        rows = pw[pw["role"].isin(spec["role"])].copy()
        rows["actual"] = rows[ACTUAL_COL[market]]

    if rows.empty:
        return rows

    use_opp_factor = spec["use_opp_factor"]

    means, opp_factors, volumes, effs, eligible = [], [], [], [], []
    for r in rows.itertuples(index=False):
        player_row = r._asdict()
        team_row = team_idx.get((r.season, r.week, r.team))
        # the opponent factor is always looked up under the PLAYER'S OWN role
        # (QB -> pass D, WR/TE -> their own position-specific D, RB -> rush D)
        opp_row = opp_idx.get((r.season, r.week, r.defteam, r.role)) if use_opp_factor else None
        pred = _project_mean_only(player_row, market, team_row, opp_row)
        means.append(pred["mean"])
        opp_factors.append(pred["components"]["opp_factor"])
        volumes.append(pred["components"]["volume"])
        effs.append(pred["components"]["efficiency"])
        eligible.append(pred["eligible_for_shortlist"])

    rows["mean_pred"] = means
    rows["opp_factor"] = opp_factors
    rows["volume_pred"] = volumes
    rows["efficiency_pred"] = effs
    rows["eligible_for_shortlist"] = eligible
    rows["market"] = market
    rows["dist"] = spec["dist"]
    rows["sample_bucket"] = rows["roll_games"].apply(_bucket)
    return rows


def _project_mean_only(player_row: Dict, market: str, team_row: Optional[Dict], opp_row: Optional[Dict]) -> Dict:
    """Thin wrapper around ``projection.project`` -- sd is irrelevant to the
    mean, so we pass a dummy sd here and compute the real (walk-forward) sd
    separately once all predictions for a market are assembled."""
    from nflvalue.projection import project
    return project(player_row, market, team_row=team_row, opp_row=opp_row, line=None, sd=1.0)


def _walk_forward_sd(df_sorted: pd.DataFrame, min_history: int = 10) -> pd.Series:
    """Expanding SD of PAST residuals only (shift(1) before expanding).

    ``df_sorted`` MUST already be sorted by (season, week) with a clean
    0..n-1 index -- the caller owns the single sort so this and the
    caller's row order can never drift apart (pandas' default sort is not
    guaranteed stable, so sorting twice independently is not safe here).
    """
    resid = df_sorted["actual"] - df_sorted["mean_pred"]
    sd_est = resid.shift(1).expanding(min_periods=min_history).std()
    fallback = (df_sorted["mean_pred"] * DEFAULT_SD_FRACTION).clip(lower=0.75)
    return sd_est.fillna(fallback)


def _synthetic_line(pw_market: pd.DataFrame, actual_col_name: str) -> pd.Series:
    """Each player's own trailing rolling MEAN of the actual stat, computed
    leak-free (shift(1) before rolling), used only as a benchmark line for
    the calibration check -- never a real sportsbook price.

    Phase 1B fix: Checkpoint 1 used a rolling MEDIAN here, which structurally
    biases this whole check for right-skewed markets like receiving yards
    (mean > median for a right-skewed stat, so P(actual > median) runs
    meaningfully above 50% almost independent of the model's own P(over) --
    that's a flaw in the calibration BENCHMARK, not necessarily the model).
    A rolling mean is the apples-to-apples comparison: both it and the
    model's projected mean are estimating the same quantity, so P(over) can
    actually be read as "does the model's mean differ from the player's own
    naive trailing mean, and if so, in the right direction."
    """
    g = pw_market.sort_values(["player_id", "season", "week"]).groupby("player_id")
    return g[actual_col_name].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())


def _metrics(df: pd.DataFrame) -> Dict:
    err = df["actual"] - df["mean_pred"]
    n = len(df)
    mae = float(err.abs().mean()) if n else None
    rmse = float(np.sqrt((err ** 2).mean())) if n else None
    corr = float(df["mean_pred"].corr(df["actual"])) if n > 2 else None
    return {"n": n, "mae": round(mae, 3) if mae is not None else None,
            "rmse": round(rmse, 3) if rmse is not None else None,
            "corr": round(corr, 4) if corr is not None else None}


def _calibration(df: pd.DataFrame) -> List[Dict]:
    """10 probability buckets: mean predicted P(over synthetic line) vs the
    actual empirical over-rate in that bucket."""
    d = df.dropna(subset=["synthetic_line", "sd_est"]).copy()
    if d.empty:
        return []
    d["p_over"] = [
        p_over_fn(m, sd, line, dist)
        for m, sd, line, dist in zip(d["mean_pred"], d["sd_est"], d["synthetic_line"], d["dist"])
    ]
    d["actual_over"] = (d["actual"] > d["synthetic_line"]).astype(float)
    d["bucket"] = np.minimum((d["p_over"] * 10).astype(int), 9)
    out = []
    for b, grp in d.groupby("bucket"):
        out.append({
            "bucket": f"{b*10}-{b*10+10}%", "n": int(len(grp)),
            "predicted_p_over": round(float(grp["p_over"].mean()), 4),
            "actual_over_rate": round(float(grp["actual_over"].mean()), 4),
        })
    return sorted(out, key=lambda x: x["bucket"])


def run(seasons: Optional[List[int]] = None, output_path: Optional[str] = None,
        db_path: Optional[str] = None) -> Dict:
    print("=" * 78)
    print("PROP BACKTEST -- measures PROJECTION ACCURACY, not price-beating.")
    print("(No free historical prop-LINE data exists; forward CLV/price tests")
    print(" happen later, live, once real prop lines are pulled -- Phase 3.)")
    print("=" * 78)

    pbp = load_pbp()
    if seasons:
        pbp = pbp[pbp["season"].isin(seasons)]
    print(f"Loaded {len(pbp):,} regular-season plays, seasons {sorted(pbp['season'].unique().tolist())}")

    roster_seasons = sorted(pbp["season"].unique().tolist())
    rosters = rostersmod.fetch_rosters_weekly(roster_seasons)
    pw = build_player_week(pbp, rosters=rosters)
    opd = build_opp_pos_def(pbp, rosters=rosters)
    tw = build_team_week(pbp)
    print(f"player_week: {len(pw):,} rows   opp_pos_def: {len(opd):,} rows   team_week: {len(tw):,} rows")

    team_idx = {(r.season, r.week, r.team): r._asdict() for r in tw.itertuples(index=False)}
    opp_idx = {(r.season, r.week, r.defteam, r.role): r._asdict() for r in opd.itertuples(index=False)}

    run_at = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    report = {"generated": run_at, "n_plays": int(len(pbp)),
              "seasons": sorted(int(s) for s in pbp["season"].unique()),
              "note": "Projection accuracy only -- NOT a price-beating / CLV test. See PROP_SHORTLISTER_SPEC.md §5.",
              "markets": {}}
    db_rows = []

    for market in MARKETS:
        preds = _predictions_for_market(pw, market, team_idx, opp_idx)
        if preds.empty:
            continue
        preds["synthetic_line"] = _synthetic_line(preds, ACTUAL_COL.get(market, "actual"))
        preds = preds.sort_values(["season", "week"], kind="mergesort").reset_index(drop=True)
        preds["sd_est"] = _walk_forward_sd(preds).values

        overall = _metrics(preds)
        by_bucket = {b: _metrics(preds[preds["sample_bucket"] == b]) for _, _, b in SAMPLE_BUCKETS
                     if not preds[preds["sample_bucket"] == b].empty}
        calibration = _calibration(preds)
        # for markets spanning >1 real position (receiving_yards/receptions =
        # WR+TE), also break out accuracy BY position -- this is the number
        # that shows whether the WR/TE-specific opponent factor actually helps.
        by_role = ({r: _metrics(preds[preds["role"] == r]) for r in sorted(preds["role"].unique())}
                   if len(MARKETS[market]["role"]) > 1 else {})
        # cold-start gate (Checkpoint 1B): the subset actually eligible for a
        # future shortlist (>= MIN_GAMES_ELIGIBLE trailing games) -- this is
        # the honest number a ranked shortlist would see, vs. "overall" which
        # includes cold-start rows purely for backtest transparency.
        eligible_only = _metrics(preds[preds["eligible_for_shortlist"]])

        report["markets"][market] = {
            "role": list(MARKETS[market]["role"]), "dist": MARKETS[market]["dist"],
            "low_confidence": MARKETS[market]["low_confidence"],
            "overall": overall, "eligible_only": eligible_only,
            "n_ineligible_cold_start": int((~preds["eligible_for_shortlist"]).sum()),
            "by_sample_size": by_bucket, "by_role": by_role, "calibration": calibration,
        }

        for bucket_label, m in by_bucket.items():
            if m["n"]:
                db_rows.append({
                    "run_at": run_at, "market": market, "sample_bucket": bucket_label,
                    "n": m["n"], "mae": m["mae"], "rmse": m["rmse"], "corr": m["corr"],
                    "calibration_bucket": "overall", "calibration_p_over": None,
                    "calibration_actual_over_rate": None,
                })
        for c in calibration:
            db_rows.append({
                "run_at": run_at, "market": market, "sample_bucket": "all",
                "n": c["n"], "mae": None, "rmse": None, "corr": None,
                "calibration_bucket": c["bucket"], "calibration_p_over": c["predicted_p_over"],
                "calibration_actual_over_rate": c["actual_over_rate"],
            })

    out_path = output_path or os.path.join(config.DATA_DIR, "prop_backtest.json")
    config.save_json(out_path, report)

    conn = dbmod.connect(db_path)
    written = dbmod.upsert(conn, "prop_backtest", db_rows, ["run_at", "market", "sample_bucket", "calibration_bucket"])
    conn.close()

    # ---- console summary ----
    print()
    for market, res in report["markets"].items():
        o = res["overall"]
        e = res["eligible_only"]
        tag = " [low-confidence]" if res["low_confidence"] else ""
        print(f"  {market:18}{tag:18} n={o['n']:6}  MAE={o['mae']:>7}  RMSE={o['rmse']:>7}  corr={o['corr']}")
        print(f"      eligible-for-shortlist  n={e['n']:6}  MAE={e['mae']:>7}  RMSE={e['rmse']:>7}  corr={e['corr']}"
              f"   ({res['n_ineligible_cold_start']} rows gated out, <{MIN_GAMES_ELIGIBLE} trailing games)")
        for label in ("0 games (cold start)", "1-3 games", "4-7 games", "8+ games"):
            if label in res["by_sample_size"]:
                b = res["by_sample_size"][label]
                print(f"      {label:22} n={b['n']:6}  MAE={b['mae']:>7}  RMSE={b['rmse']:>7}  corr={b['corr']}")
        for role, b in res.get("by_role", {}).items():
            print(f"      by-position {role:11} n={b['n']:6}  MAE={b['mae']:>7}  RMSE={b['rmse']:>7}  corr={b['corr']}")
    print(f"\n  Saved: {out_path}  ({written} rows upserted into the prop_backtest table)")
    print("  Reminder: these numbers describe projection accuracy only -- leans, not locks.")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    args = ap.parse_args()
    run(seasons=args.seasons)

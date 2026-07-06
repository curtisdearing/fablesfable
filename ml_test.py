#!/usr/bin/env python3
"""ML improvement test: can RF / gradient boosting beat the tuned composite?

Protocol (same honesty contract as tune_weights.py):
  * WALK-FORWARD by season: models train on seasons strictly before the eval
    season, then rank that season's candidates. A weekly-retrain variant
    additionally folds in the eval season's PRIOR weeks (what a live Tuesday
    retrain would do). ``MLRanker.assert_walk_forward`` hard-fails any
    train/test overlap.
  * Identical candidate pool + identical selection protocol (top-5/game,
    per-player cap, yes-only TD) as production and as the tuned-composite
    baseline it must beat.
  * Metrics: log-loss (the gradient-descent objective the GBDT minimizes) and
    AUC for the classifier itself; hit rate + implied -110 units for what
    the bettor sees. Synthetic-line caveat applies to every number.

Stages:
  python3 ml_test.py --stage frame --seasons 2019 2020 ... [--append]
  python3 ml_test.py --stage eval --season 2021 [--models gbdt rf]
  python3 ml_test.py --stage weekly --season 2025
  python3 ml_test.py --stage report
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from nflvalue import config as cfgmod
from nflvalue import ml_ranker as mlr
from nflvalue.candidates import WeekInputs, enumerate_candidates
from nflvalue.composite import YES_ONLY_MARKETS
from nflvalue.projection import MARKETS

FRAME_PATH = os.path.join(cfgmod.DATA_DIR, "ml_frame.parquet")
EVAL_PATH = os.path.join(cfgmod.DATA_DIR, "ml_eval_results.json")
BREAKEVEN = 0.5238
TUNED = {"conf_share": 0.8, "z_cap": 1.5, "low_conf_mult": 0.8, "markets": "all"}


# --------------------------------------------------------------------------- #
# Stage: frame
# --------------------------------------------------------------------------- #
def build_frame(inputs: WeekInputs, seasons: List[int], append: bool) -> pd.DataFrame:
    import lean_backtest as lb
    from nflvalue.candidates import synthetic_lines
    from nflvalue.context_features import ContextPack
    from nflvalue.sources import rosters as rostersmod

    pkl = os.environ.get("ML_PACKS_PKL")
    if pkl and os.path.exists(pkl + "_a.pkl"):
        import pickle
        with open(pkl + "_a.pkl", "rb") as fh:
            pack, adv = pickle.load(fh)
        with open(pkl + "_chem.pkl", "rb") as fh:
            chem = pickle.load(fh)
        with open(pkl + "_ftn.pkl", "rb") as fh:
            ftn = pickle.load(fh)
        with open(pkl + "_sd.pkl", "rb") as fh:
            sd_map, synth_map = pickle.load(fh)
        min_usage = (cfgmod.load_config().get("candidates") or {}).get("min_usage")
        return _frame_loop(inputs, seasons, append, pack, adv, chem, ftn,
                           sd_map, synth_map, min_usage)

    sd_map = lb.precompute_sds(inputs, list(MARKETS))
    synth_map = {m: synthetic_lines(inputs, m) for m in MARKETS}
    min_usage = (cfgmod.load_config().get("candidates") or {}).get("min_usage")
    all_seasons = sorted(inputs.pw["season"].unique().tolist())
    pack = ContextPack(rostersmod.fetch_rosters_weekly(all_seasons), all_seasons,
                       opd=inputs.opd)
    from nflvalue.advanced_features import AdvancedPack, load_pbp_ext
    _pbp_ext = load_pbp_ext()
    adv = AdvancedPack(pbp=_pbp_ext, schedules=inputs.schedules)
    from nflvalue.chemistry import ChemistryPack
    chem = ChemistryPack(pbp=_pbp_ext, pw=inputs.pw, schedules=inputs.schedules)
    from nflvalue.ftn_features import FTNPack
    ftn = FTNPack(pbp=_pbp_ext)
    if os.environ.get("ML_PACKS_DUMP"):
        import pickle
        with open(os.environ["ML_PACKS_DUMP"], "wb") as fh:
            pickle.dump((pack, adv, chem, ftn, sd_map, synth_map), fh)
    return _frame_loop(inputs, seasons, append, pack, adv, chem, ftn,
                       sd_map, synth_map, min_usage)


def _frame_loop(inputs, seasons, append, pack, adv, chem, ftn,
                sd_map, synth_map, min_usage) -> pd.DataFrame:
    import lean_backtest as lb
    from nflvalue.candidates import (apply_opp_absence_factor,
                                     apply_weather_adjustment)

    # Phase 6.4/6.5 parity: the frame must reflect the same weather/absence
    # steps the replay + live paths apply (means, p_over, stamped factors)
    wx_map, db_outs = {}, {}
    try:
        from nflvalue.weather_study import build_game_weather
        gwx = build_game_weather()
        wx_map = {r.game_id: {"wind_mph": r.wind_mph,
                              "precip_mm": (2.0 if r.precip_flag else 0.0),
                              "effective_outdoor": bool(r.effective_outdoor)}
                  for r in gwx.itertuples(index=False)}
        from nflvalue.context_features import (DB_POS, OUT_STATUSES,
                                               load_injury_history)
        inj = load_injury_history(sorted(inputs.pw["season"].unique().tolist()))
        d = inj[inj["report_status"].isin(OUT_STATUSES) & inj["position"].isin(DB_POS)]
        db_outs = {(int(s), int(w), str(t)): int(len(g))
                   for (s, w, t), g in d.groupby(["season", "week", "team"])}
    except Exception as exc:  # noqa: BLE001
        print(f"[ml_test] weather/absence maps unavailable ({exc})")

    chunks = []
    for season in seasons:
        weeks = sorted(inputs.schedules[
            (inputs.schedules["season"] == season)
            & (inputs.schedules["game_type"] == "REG")]["week"].unique().tolist())
        for wk in weeks:
            try:
                cands = enumerate_candidates(season, wk, inputs=inputs, min_usage=min_usage,
                                             sd_by_market=sd_map.get((season, wk), {}),
                                             synth_by_market=synth_map)
            except ValueError:
                continue
            if cands.empty:
                continue
            cands = apply_weather_adjustment(cands, wx_map)
            if db_outs:
                cands = apply_opp_absence_factor(cands, db_outs)
            actuals = lb._actuals_for_week(inputs.pw, season, wk)
            cands = chem.attach(cands)
            cands = ftn.attach(cands)
            feats = mlr.build_features(cands, inputs.pw, pack=pack, adv=adv)
            feats["y_over"] = mlr.label_over(feats, actuals)
            # baseline-composite fields (tune_weights conventions)
            feats["side"] = np.where(
                feats["market"].isin(YES_ONLY_MARKETS) | (feats["p_over"] >= 0.5),
                "over", "under")
            feats["model_prob"] = np.where(feats["side"] == "over",
                                           feats["p_over"], 1 - feats["p_over"])
            feats["hit"] = np.where(feats["side"] == "over",
                                    feats["y_over"], 1 - feats["y_over"])
            keep = (["season", "week", "game_id", "player_id", "market", "side",
                     "model_prob", "low_confidence", "y_over", "hit"]
                    + mlr.feature_columns())
            keep = list(dict.fromkeys(keep))
            chunks.append(feats[keep].dropna(subset=["y_over"]))
    if not chunks:
        return pd.DataFrame()
    frame = pd.concat(chunks, ignore_index=True)
    if append and os.path.exists(FRAME_PATH):
        old = pd.read_parquet(FRAME_PATH)
        frame = (pd.concat([old, frame], ignore_index=True)
                 .drop_duplicates(subset=["season", "week", "game_id", "player_id", "market"]))
    frame.to_parquet(FRAME_PATH, index=False)
    return frame


# --------------------------------------------------------------------------- #
# Stage: eval (one season, walk-forward by season)
# --------------------------------------------------------------------------- #
def _summarize(leans: pd.DataFrame) -> Dict:
    n, hits = len(leans), int(leans["ml_hit"].sum())
    top1 = leans.groupby(["season", "week", "game_id"]).head(1)
    return {"n": n, "hit_rate": round(hits / n, 4) if n else None,
            "units_at_-110": mlr.implied_units_at_110(hits, n),
            "top1_hit_rate": (round(float(top1["ml_hit"].mean()), 4) if len(top1) else None)}


def _baseline(frame: pd.DataFrame, season: int) -> Dict:
    import tune_weights as tw
    fe = tw._FastEval(frame.rename(columns={"z": "z"}))
    tally = fe.per_season(TUNED)
    h, n = tally.get(season, (0, 0))
    return {"n": n, "hit_rate": round(h / n, 4) if n else None,
            "units_at_-110": mlr.implied_units_at_110(h, n)}


def eval_season(frame: pd.DataFrame, season: int, models: List[str]) -> Dict:
    from sklearn.metrics import log_loss, roc_auc_score

    train = frame[frame["season"] < season]
    test = frame[frame["season"] == season]
    out = {"eval_season": season, "n_train": int(len(train)), "n_test": int(len(test)),
           "baseline_tuned_composite": _baseline(frame, season), "models": {}}
    for name in models:
        model = mlr.MLRanker(model=name).fit(train, train["y_over"])
        p = model.predict_p_over(test)
        y = test["y_over"].to_numpy()
        leans = mlr.rank_and_grade(test, p)
        out["models"][name] = {
            "log_loss": round(float(log_loss(y, p)), 5),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "leans": _summarize(leans),
        }
    return out


# --------------------------------------------------------------------------- #
# Stage: weekly retrain variant (live-cadence simulation)
# --------------------------------------------------------------------------- #
def eval_weekly(frame: pd.DataFrame, season: int, model_name: str = "gbdt") -> Dict:
    """Resumable: per-week graded leans are checkpointed to a scratch parquet,
    so an interrupted run continues where it stopped (18 retrains is long)."""
    from sklearn.metrics import log_loss

    ckpt = os.path.join(cfgmod.DATA_DIR, f"ml_weekly_{season}_{model_name}.parquet")
    done = pd.read_parquet(ckpt) if os.path.exists(ckpt) else pd.DataFrame()
    done_weeks = set(done["week"].unique()) if len(done) else set()

    weeks = sorted(frame[frame["season"] == season]["week"].unique().tolist())
    all_leans = [done] if len(done) else []
    lls = list(done["_ll"].groupby(done["week"]).first()) if len(done) else []
    for wk in weeks:
        if wk in done_weeks:
            continue
        train = frame[(frame["season"] < season)
                      | ((frame["season"] == season) & (frame["week"] < wk))]
        test = frame[(frame["season"] == season) & (frame["week"] == wk)]
        if test.empty:
            continue
        model = mlr.MLRanker(model=model_name).fit(train, train["y_over"])
        p = model.predict_p_over(test)
        ll = float(log_loss(test["y_over"], p, labels=[0, 1]))
        lls.append(ll)
        wk_leans = mlr.rank_and_grade(test, p)
        wk_leans["_ll"] = ll
        all_leans.append(wk_leans)
        pd.concat(all_leans, ignore_index=True).to_parquet(ckpt, index=False)
    leans = pd.concat(all_leans, ignore_index=True)
    return {"eval_season": season, "model": model_name, "mode": "weekly_retrain",
            "avg_log_loss": round(float(np.mean(lls)), 5),
            "leans": _summarize(leans),
            "by_week": {int(w): round(float(g["ml_hit"].mean()), 4)
                        for w, g in leans.groupby("week")}}


def augment_with_real_lines(frame: pd.DataFrame, conn=None) -> pd.DataFrame:
    """SELF-UPDATING LABELS: as real sportsbook lines accrue live, graded
    leans that had an odds_api line get RE-LABELED against that real line
    (y = actual > real line) with the line-dependent features recomputed --
    the model's training target migrates from the synthetic reference to the
    actual betting market, week by week, automatically. Rows without a real
    line keep the synthetic label. No real-line rows yet -> frame unchanged."""
    from nflvalue import db as dbmod
    try:
        conn = conn or dbmod.connect()
        real = dbmod.query_df(conn, """
            SELECT o.season, o.week, o.player_id, o.market, o.actual, l.line AS real_line
            FROM lean_outcomes o
            JOIN leans l ON l.season=o.season AND l.week=o.week AND l.clock=o.clock
                 AND l.game_id=o.game_id AND l.player_id=o.player_id AND l.market=o.market
            WHERE l.line_source='odds_api' AND l.line IS NOT NULL
        """)
    except Exception as exc:  # noqa: BLE001
        print(f"[ml] real-line lookup unavailable ({exc}); synthetic labels only")
        return frame
    if real.empty:
        return frame
    merged = frame.merge(real, on=["season", "week", "player_id", "market"], how="left")
    mask = merged["real_line"].notna()
    n = int(mask.sum())
    if n:
        m = merged.loc[mask]
        merged.loc[mask, "line"] = m["real_line"]
        merged.loc[mask, "mean_minus_line"] = m["mean"] - m["real_line"]
        merged.loc[mask, "sd_over_line"] = m["sd"] / m["real_line"].abs().clip(lower=1.0)
        merged.loc[mask, "z"] = (m["mean"] - m["real_line"]) / m["sd"].clip(lower=1e-6)
        is_td = m["market"] == "anytime_td"
        merged.loc[mask, "y_over"] = np.where(
            is_td, (m["actual"] >= 1.0).astype(float),
            (m["actual"] > m["real_line"]).astype(float))
        print(f"[ml] {n:,} rows re-labeled against REAL lines "
              f"({n / len(merged):.1%} of training data and growing weekly)")
    return merged.drop(columns=["actual", "real_line"])


def _load_results() -> Dict:
    return cfgmod.load_json(EVAL_PATH, {"seasons": {}, "weekly": {}}) or {"seasons": {}, "weekly": {}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["frame", "eval", "weekly", "report", "fit"], required=True)
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=["gbdt", "rf"])
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--calibrate", default="platt_permkt",
                    help="walk-forward calibration for the fit artifact "
                         "(platt_permkt | platt_pooled | none); Phase 7.1")
    ap.add_argument("--pw"), ap.add_argument("--opd"), ap.add_argument("--tw"), ap.add_argument("--sched")
    args = ap.parse_args()

    if args.stage == "frame":
        if args.pw:
            inputs = WeekInputs(pd.read_parquet(args.pw), pd.read_parquet(args.opd),
                                pd.read_parquet(args.tw), pd.read_parquet(args.sched))
        else:
            from nflvalue.candidates import build_week_inputs
            inputs = build_week_inputs()
        frame = build_frame(inputs, args.seasons, args.append)
        print(f"frame: {len(frame):,} rows, seasons {sorted(frame['season'].unique())}")
        return

    frame = pd.read_parquet(FRAME_PATH)
    results = _load_results()
    if args.stage == "fit":
        # production fit: train on EVERYTHING graded so far, save for the pipeline
        frame = augment_with_real_lines(frame)
        model_name = (args.models or ["rf"])[0]
        calibrate = None if str(args.calibrate).lower() == "none" else args.calibrate
        model = mlr.MLRanker(model=model_name, calibrate=calibrate).fit(frame, frame["y_over"])
        path = model.save()
        cal = (f"calibrated={model.calibrate}" if model.calibrator is not None
               else "calibrator=none")
        print(f"fitted {model_name} ({cal}) on {len(frame):,} rows through "
              f"{model.train_max} -> {path}")
        return
    if args.stage == "eval":
        res = eval_season(frame, args.season, args.models)
        key = str(args.season)
        if key in results["seasons"]:                       # merge models, don't clobber
            results["seasons"][key]["models"].update(res["models"])
            results["seasons"][key]["baseline_tuned_composite"] = res["baseline_tuned_composite"]
        else:
            results["seasons"][key] = res
        cfgmod.save_json(EVAL_PATH, results)
        print(json.dumps(res, indent=1))
        return
    if args.stage == "weekly":
        res = eval_weekly(frame, args.season)
        results["weekly"][str(args.season)] = res
        cfgmod.save_json(EVAL_PATH, results)
        print(json.dumps({k: v for k, v in res.items() if k != "by_week"}, indent=1))
        return

    # report
    lines = ["# ML improvement test — RF & gradient boosting vs tuned composite", "",
             "**Leans, not locks.** Walk-forward by season; identical candidate pools and",
             "top-5 selection protocol; graded at synthetic trailing-mean lines (NOT real",
             f"prices; breakeven proxy {BREAKEVEN:.2%} at -110). Log-loss is the gradient-",
             "descent objective the GBDT minimizes. 1-800-GAMBLER.", "",
             "| Season | tuned composite | GBDT hit (units) | RF hit (units) | GBDT log-loss | GBDT AUC |",
             "|---|---|---|---|---|---|"]
    for s in sorted(results["seasons"], key=int):
        r = results["seasons"][s]
        b = r["baseline_tuned_composite"]
        g = r["models"].get("gbdt", {})
        rf = r["models"].get("rf", {})
        gl, rl = g.get("leans", {}), rf.get("leans", {})
        lines.append(
            f"| {s} | {b['hit_rate']:.1%} ({b['units_at_-110']:+.1f}u) "
            f"| **{gl.get('hit_rate', 0):.1%}** ({gl.get('units_at_-110', 0):+.1f}u) "
            f"| {rl.get('hit_rate', 0):.1%} ({rl.get('units_at_-110', 0):+.1f}u) "
            f"| {g.get('log_loss')} | {g.get('auc')} |")
    for s, w in results.get("weekly", {}).items():
        l = w["leans"]
        lines += ["", f"Weekly-retrain GBDT, {s}: **{l['hit_rate']:.1%}** "
                  f"({l['units_at_-110']:+.1f}u, top-1 {l['top1_hit_rate']:.1%}, "
                  f"avg log-loss {w['avg_log_loss']})"]
    md = "\n".join(lines) + "\n"
    os.makedirs("reports", exist_ok=True)
    with open("reports/ml_improvement_test.md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()

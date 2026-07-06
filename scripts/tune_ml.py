#!/usr/bin/env python3
"""Phase 7.2 -- ensemble + walk-forward hyperparameter search + feature
pruning, squeezed against 7.1's FIXED metric (docs/decisions_p7.md): pooled
walk-forward calibrated log-loss is the primary objective; pooled ECE and
per-market Brier are guardrails that must not regress.

Same honesty contract as tune_weights.py and scripts/audit_calibration.py:
  * Search/ablation decisions use CHEAP raw (uncalibrated) walk-forward
    log-loss across expanding-season folds -- refitting the full calibrated
    metric suite for every grid point / feature drop is not affordable on the
    sandbox's cores. Only the FINAL chosen configuration is checked against
    the real calibrated metric suite (log-loss + ECE + per-market Brier,
    pooled 2022-2025 -- the identical rows and method scripts/audit_calibration
    used) before it ships. A change ships only if it beats baseline there.
  * Every fold is walk-forward: a config/feature-set is scored on season S
    using ONLY a model trained on seasons < S. Selection (which config/feature
    set to use) for season S in the reported walk-forward tables draws only on
    prior seasons -- never in-sample argmax.

Stages (each resumable -- checkpoints to data/*.parquet or *.json so repeated
invocations continue rather than restart):
  python3 scripts/tune_ml.py --stage ens_rf_oos      # cache RF walk-forward OOS
  python3 scripts/tune_ml.py --stage ens_analyze      # ensemble bake-off
  python3 scripts/tune_ml.py --stage hp_search  [--batch N]
  python3 scripts/tune_ml.py --stage hp_analyze
  python3 scripts/tune_ml.py --stage prune      [--batch N]
  python3 scripts/tune_ml.py --stage prune_analyze
  python3 scripts/tune_ml.py --stage final_check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from nflvalue import config as cfgmod  # noqa: E402
from nflvalue import ml_ranker as mlr  # noqa: E402
import audit_calibration as cal  # noqa: E402  -- reuse the 7.1 calibration code exactly

FRAME_PATH = os.path.join(cfgmod.DATA_DIR, "ml_frame.parquet")
ENS_OOS_PATH = os.path.join(cfgmod.DATA_DIR, "ens_rf_oos.parquet")
ENS_RESULT_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_ensemble.json")
HP_CKPT_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_hp.json")
HP_RESULT_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_hp_result.json")
PRUNE_CKPT_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_prune.json")
PRUNE_RESULT_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_prune_result.json")
FINAL_RESULT_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_final.json")

# Matches scripts/audit_calibration.py's EVAL_SEASONS/CAL_SEASONS exactly so
# the ensemble bake-off scores the IDENTICAL pooled rows as the 7.1 baseline.
EVAL_SEASONS = [2021, 2022, 2023, 2024, 2025]
CAL_SEASONS = [2022, 2023, 2024, 2025]
# HP search + feature pruning use the fuller walk-forward season range
# (tune_weights.py convention: every season after the first is an eval fold).
HP_EVAL_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
PRUNE_EVAL_SEASONS = [2021, 2022, 2023, 2024, 2025]  # skip the 1-season-train
# 2020 fold for feature-selection stability (disclosed compute-driven choice)

DEFAULT_HP = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                  l2=1.0, max_iter=400)


# --------------------------------------------------------------------------- #
# Ensemble: cache RF walk-forward OOS (GBDT OOS already cached by 7.1's
# scripts/audit_calibration.py --stage oos at data/calib_oos.parquet)
# --------------------------------------------------------------------------- #
def build_rf_oos(frame: pd.DataFrame) -> pd.DataFrame:
    done = pd.read_parquet(ENS_OOS_PATH) if os.path.exists(ENS_OOS_PATH) else pd.DataFrame()
    have = set(done["season"].unique().tolist()) if len(done) else set()
    chunks = [done] if len(done) else []
    for s in EVAL_SEASONS:
        if s in have:
            continue
        t0 = time.time()
        tr = frame[frame["season"] < s]
        te = frame[frame["season"] == s].copy()
        model = mlr.MLRanker("rf").fit(tr, tr["y_over"])
        te["p_rf"] = model.predict_p_over(te, enforce=False)
        keep = ["season", "week", "game_id", "player_id", "market", "y_over", "p_rf"]
        chunks.append(te[keep])
        pd.concat(chunks, ignore_index=True).to_parquet(ENS_OOS_PATH, index=False)
        print(f"  RF OOS season {s}: trained <{s} ({len(tr):,} rows), "
              f"predicted {len(te):,} in {time.time()-t0:.1f}s")
    oos = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return oos.drop_duplicates(subset=["season", "week", "game_id", "player_id", "market"])


def _variant_oos_frame(base_oos: pd.DataFrame, p_col: str) -> pd.DataFrame:
    """Reshape a variant's raw OOS predictions into the shape
    scripts/audit_calibration.py's apply_walk_forward/metric_block expect."""
    f = base_oos[["season", "week", "game_id", "player_id", "market", "y_over"]].copy()
    f["p_raw"] = base_oos[p_col].to_numpy()
    return f


def _calibrate_seasons(oos: pd.DataFrame, method: str, per_market: bool,
                       cal_seasons: List[int]) -> pd.Series:
    """Like scripts/audit_calibration.apply_walk_forward but parameterized on
    the season list (that module hardcodes its own CAL_SEASONS global) -- the
    meta-learner's raw OOS series starts a season later than gbdt/rf/avg (it
    needs its own prior-season seed), so its valid calibrated window is one
    season narrower and must be computed on its own season list, not 7.1's."""
    out = pd.Series(index=oos.index, dtype=float)
    fitter = cal.FITTERS[method]
    markets = sorted(oos["market"].unique().tolist())
    for s in cal_seasons:
        prior = oos[oos["season"] < s]
        cur = oos[oos["season"] == s]
        if prior.empty or cur.empty:
            continue
        if per_market:
            pooled_fn = fitter(prior["p_raw"], prior["y_over"])
            for m in markets:
                idx = cur.index[cur["market"] == m]
                if len(idx) == 0:
                    continue
                pm = prior[prior["market"] == m]
                fn = fitter(pm["p_raw"], pm["y_over"]) if len(pm) >= cal.PERMARKET_MIN else pooled_fn
                out.loc[idx] = fn(cur.loc[idx, "p_raw"].to_numpy())
        else:
            fn = fitter(prior["p_raw"], prior["y_over"])
            out.loc[cur.index] = fn(cur["p_raw"].to_numpy())
    return out


def _walkforward_meta(gbdt_rf: pd.DataFrame) -> pd.Series:
    """Walk-forward logistic meta-learner over [logit(p_gbdt), logit(p_rf)]:
    for eval season s, trained ONLY on pooled (p_gbdt, p_rf, y) of seasons < s
    (within EVAL_SEASONS, so 2021 seeds it) -- never a row it later scores.
    First season with a prior season available is 2022 (mirrors the
    calibrator's 2021-seeds-history rule one level up)."""
    from sklearn.linear_model import LogisticRegression
    out = pd.Series(index=gbdt_rf.index, dtype=float)
    for s in CAL_SEASONS:
        prior = gbdt_rf[gbdt_rf["season"] < s]
        cur = gbdt_rf[gbdt_rf["season"] == s]
        if prior.empty:
            continue
        Xtr = np.column_stack([cal._logit(prior["p_gbdt"]), cal._logit(prior["p_rf"])])
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(Xtr, prior["y_over"].astype(int))
        Xte = np.column_stack([cal._logit(cur["p_gbdt"]), cal._logit(cur["p_rf"])])
        out.loc[cur.index] = lr.predict_proba(Xte)[:, 1]
    return out


def ensemble_analyze() -> Dict:
    gbdt_oos = pd.read_parquet(os.path.join(cfgmod.DATA_DIR, "calib_oos.parquet"))
    rf_oos = pd.read_parquet(ENS_OOS_PATH)
    key = ["season", "week", "game_id", "player_id", "market"]
    merged = gbdt_oos.rename(columns={"p_raw": "p_gbdt"}).merge(
        rf_oos.drop(columns=["y_over"]), on=key, how="inner")
    assert len(merged) == len(gbdt_oos), "GBDT/RF OOS row mismatch -- same folds expected"
    merged["p_avg"] = (merged["p_gbdt"] + merged["p_rf"]) / 2.0
    merged["p_meta"] = _walkforward_meta(merged)   # NaN for 2021 (no prior season pair)

    # gbdt/rf/avg have raw OOS from 2021 on -> calibrated window 2022-2025,
    # IDENTICAL rows to 7.1's baseline. meta needs its own prior-season seed
    # (first meta OOS is 2022) so its calibrated window is one season
    # narrower, 2023-2025 -- disclosed, not smoothed over.
    variants = {"gbdt": ("p_gbdt", EVAL_SEASONS, CAL_SEASONS),
               "rf": ("p_rf", EVAL_SEASONS, CAL_SEASONS),
               "avg": ("p_avg", EVAL_SEASONS, CAL_SEASONS),
               "meta": ("p_meta", CAL_SEASONS, CAL_SEASONS[1:])}
    results: Dict[str, Dict] = {}
    calibrated_p: Dict[str, pd.Series] = {}
    for tag, (col, raw_seasons, cal_seasons) in variants.items():
        vf = _variant_oos_frame(merged[merged["season"].isin(raw_seasons)], col)
        raw_ev = vf[vf["season"].isin(cal_seasons)]
        raw_metrics = cal.metric_block(raw_ev["p_raw"].to_numpy(),
                                       raw_ev["y_over"].to_numpy(),
                                       raw_ev["market"].to_numpy())
        cal_p = _calibrate_seasons(vf, "platt", True, cal_seasons)
        cev = vf[vf["season"].isin(cal_seasons)]
        cal_metrics = cal.metric_block(cal_p.loc[cev.index].to_numpy(),
                                       cev["y_over"].to_numpy(),
                                       cev["market"].to_numpy())
        results[tag] = {"raw": raw_metrics, "calibrated": cal_metrics,
                        "calibrated_window": cal_seasons}
        calibrated_p[tag] = cal_p.loc[cev.index]

    # "beats the best single model" means best of {gbdt, rf} -- NOT assumed
    # to be the currently-shipped gbdt. Significance paired on the COMMON row
    # set each candidate is actually valid over (2022-2025 for gbdt/rf/avg;
    # 2023-2025 for meta, since meta has no 2022 prediction).
    best_single = min(("gbdt", "rf"), key=lambda t: results[t]["calibrated"]["log_loss"])
    sig = {}
    for tag in ("gbdt", "rf", "avg", "meta"):
        if tag == best_single:
            continue
        common_seasons = variants[tag][2]
        base_col = variants[best_single][0]
        base_vf = _variant_oos_frame(merged[merged["season"].isin(EVAL_SEASONS)], base_col)
        base_cal = _calibrate_seasons(base_vf, "platt", True, common_seasons)
        base_idx = base_vf[base_vf["season"].isin(common_seasons)].index
        y = merged.loc[base_idx, "y_over"].to_numpy()
        sig[tag] = cal.paired_t(base_cal.loc[base_idx].to_numpy(),
                                calibrated_p[tag].to_numpy(), y)

    winner = best_single
    best_ll = results[best_single]["calibrated"]["log_loss"]
    for tag in ("avg", "meta"):
        r = results[tag]["calibrated"]
        better_ll = r["log_loss"] < best_ll - 1e-5
        better_ece = r["ece"] <= results[best_single]["calibrated"]["ece"] + 1e-4
        t_ok = sig[tag]["t"] >= 2.0
        if better_ll and better_ece and t_ok:
            winner, best_ll = tag, r["log_loss"]

    payload = {"gbdt_rf_window": EVAL_SEASONS, "gbdt_rf_calibrated_window": CAL_SEASONS,
               "meta_calibrated_window": CAL_SEASONS[1:],
               "results": results, "significance": sig,
               "best_single_model": best_single, "winner": winner,
               "note": ("best_single_model = lower calibrated pooled log-loss "
                       "of {gbdt, rf} -- NOT assumed to be the currently-"
                       "shipped gbdt. winner = that model UNLESS {avg, meta} "
                       "beats it on pooled log-loss AND holds ECE AND clears "
                       "t>=2 on identical rows, in which case the ensemble "
                       "ships instead. meta's window is 2023-2025 (one "
                       "season narrower -- it needs its own prior-season "
                       "seed on top of the base OOS seed).")}
    cfgmod.save_json(ENS_RESULT_PATH, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "results"} |
                     {"log_loss_by_variant": {t: r["calibrated"]["log_loss"]
                                              for t, r in results.items()},
                      "ece_by_variant": {t: r["calibrated"]["ece"]
                                        for t, r in results.items()}},
                     indent=1))
    return payload


# --------------------------------------------------------------------------- #
# Walk-forward hyperparameter search (GBDT) -- mirrors tune_weights.py: each
# eval season's config is chosen from PRIOR seasons' pooled OOS log-loss only.
# --------------------------------------------------------------------------- #
def hp_grid() -> List[Dict]:
    base = dict(DEFAULT_HP)
    grid = [dict(base)]
    for lr in (0.03, 0.1):
        grid.append({**base, "learning_rate": lr})
    for leaves in (15, 63):
        grid.append({**base, "max_leaf_nodes": leaves})
    for mlf in (20, 80):
        grid.append({**base, "min_samples_leaf": mlf})
    for l2 in (0.0, 5.0):
        grid.append({**base, "l2": l2})
    for it in (200, 800):
        grid.append({**base, "max_iter": it})
    # joint combos: conservative/regularized vs aggressive vs a few crosses
    grid += [
        {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 80, "l2": 5.0, "max_iter": 800},
        {"learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2": 0.0, "max_iter": 800},
        {"learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2": 1.0, "max_iter": 800},
        {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 80, "l2": 1.0, "max_iter": 200},
        {"learning_rate": 0.06, "max_leaf_nodes": 63, "min_samples_leaf": 80, "l2": 5.0, "max_iter": 400},
        {"learning_rate": 0.06, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2": 0.0, "max_iter": 400},
        {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2": 0.0, "max_iter": 800},
        {"learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 40, "l2": 1.0, "max_iter": 200},
    ]
    seen, out = set(), []
    for g in grid:
        k = tuple(sorted(g.items()))
        if k not in seen:
            seen.add(k)
            out.append(g)
    return out


def cfg_key(cfg: Dict) -> str:
    return (f"lr{cfg['learning_rate']}_leaves{cfg['max_leaf_nodes']}_"
            f"minleaf{cfg['min_samples_leaf']}_l2{cfg['l2']}_iter{cfg['max_iter']}")


def _fit_fold_logloss(args) -> Tuple[str, int, Dict]:
    cfg, season, frame_path = args
    frame = pd.read_parquet(frame_path)
    tr = frame[frame["season"] < season]
    te = frame[frame["season"] == season]
    m = mlr.MLRanker("gbdt", **cfg).fit(tr, tr["y_over"])
    p = m.predict_p_over(te, enforce=False)
    y = te["y_over"].to_numpy()
    ll = cal.logloss(p, y)
    return cfg_key(cfg), season, {"log_loss": ll, "n": int(len(te))}


def _pending_tasks(grid: List[Dict], seasons: List[int], ckpt: Dict) -> List[Tuple]:
    out = []
    for cfg in grid:
        k = cfg_key(cfg)
        for s in seasons:
            if k in ckpt and str(s) in ckpt[k]:
                continue
            out.append((cfg, s, FRAME_PATH))
    return out


def run_hp_search(batch: int) -> None:
    """Resumable: checkpoints after EVERY completed fit (not just at the end
    of the batch) so a call that times out mid-batch keeps its progress --
    the next invocation picks up exactly where it left off."""
    from concurrent.futures import as_completed
    grid = hp_grid()
    ckpt = cfgmod.load_json(HP_CKPT_PATH, {}) or {}
    pending = _pending_tasks(grid, HP_EVAL_SEASONS, ckpt)
    if not pending:
        print(f"hp_search: all {len(grid)} configs x {len(HP_EVAL_SEASONS)} "
              "seasons already cached")
        return
    todo = pending[:batch]
    done = 0
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_fit_fold_logloss, t): t for t in todo}
        for fut in as_completed(futs):
            key, season, res = fut.result()
            ckpt.setdefault(key, {})[str(season)] = res
            done += 1
            cfgmod.save_json(HP_CKPT_PATH, ckpt)
    cfgmod.save_json(HP_CKPT_PATH, ckpt)
    print(f"hp_search: completed {done}/{len(pending)} pending "
          f"({len(pending) - done} remain)")


def analyze_hp_search() -> Dict:
    grid = hp_grid()
    ckpt = cfgmod.load_json(HP_CKPT_PATH, {}) or {}
    default_key = cfg_key(DEFAULT_HP)

    def pooled_ll(key: str, seasons: List[int]) -> Optional[Tuple[float, int]]:
        rows = ckpt.get(key, {})
        tot_ll, tot_n = 0.0, 0
        for s in seasons:
            r = rows.get(str(s))
            if r is None:
                return None
            tot_ll += r["log_loss"] * r["n"]
            tot_n += r["n"]
        return (tot_ll / tot_n, tot_n) if tot_n else None

    walk_forward = []
    for i, s in enumerate(HP_EVAL_SEASONS[1:], start=1):
        train_seasons = HP_EVAL_SEASONS[:i]
        scored = []
        for cfg in grid:
            k = cfg_key(cfg)
            r = pooled_ll(k, train_seasons)
            if r is not None:
                scored.append((r[0], k, cfg))
        if not scored:
            continue
        scored.sort(key=lambda x: x[0])
        _, best_key, best_cfg = scored[0]
        oos = ckpt.get(best_key, {}).get(str(s))
        default_oos = ckpt.get(default_key, {}).get(str(s))
        if oos is None or default_oos is None:
            continue
        walk_forward.append({"eval_season": s, "chosen_on_train": best_cfg,
                             "train_pooled_log_loss": round(scored[0][0], 5),
                             "oos_log_loss": oos["log_loss"],
                             "default_oos_log_loss": default_oos["log_loss"],
                             "n": oos["n"]})

    pooled_all = []
    for cfg in grid:
        k = cfg_key(cfg)
        r = pooled_ll(k, HP_EVAL_SEASONS)
        if r is not None:
            pooled_all.append({**cfg, "pooled_log_loss": round(r[0], 5), "n": r[1]})
    pooled_all.sort(key=lambda x: x["pooled_log_loss"])

    payload = {"grid_size": len(grid), "eval_seasons": HP_EVAL_SEASONS,
               "walk_forward": walk_forward, "pooled_top5": pooled_all[:5],
               "default": DEFAULT_HP,
               "ship_hp": pooled_all[0] if pooled_all else DEFAULT_HP,
               "note": ("walk_forward rows choose each season's config from "
                       "PRIOR seasons' pooled raw log-loss only (never in-"
                       "sample argmax); pooled_top5 is in-sample across all "
                       "seasons, shown for transparency; ship_hp = pooled "
                       "argmin, same convention as tune_weights.py's "
                       "ship_for_2026. Raw (uncalibrated) log-loss -- the "
                       "cheap search proxy; the final choice is re-checked "
                       "on the real calibrated metric before shipping.")}
    cfgmod.save_json(HP_RESULT_PATH, payload)
    print(json.dumps(payload, indent=1))
    return payload


DEFAULT_RF = dict(n_estimators=400, min_samples_leaf=25, max_features="sqrt")
RF_CKPT_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_rf.json")
RF_RESULT_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_rf_result.json")
# RF wasn't named in 7.2's hyperparameter list (learning_rate/leaves/min-leaf/
# L2/iters are HistGradientBoosting-specific) -- but the ensemble bake-off
# found RF already beats the walk-forward-TUNED GBDT (t=2.44), so it's the
# honest thing to also check whether RF's own knobs move its calibrated
# quality further before shipping it untuned. Kept SMALL (RF fits ~5x GBDT's
# cost) and scoped to CAL_SEASONS -- the window the shipping decision uses.
RF_GRID = [
    dict(DEFAULT_RF),
    {**DEFAULT_RF, "n_estimators": 200},
    {**DEFAULT_RF, "n_estimators": 600},
    {**DEFAULT_RF, "min_samples_leaf": 10},
    {**DEFAULT_RF, "min_samples_leaf": 50},
    {**DEFAULT_RF, "max_features": "log2"},
]


def rf_cfg_key(cfg: Dict) -> str:
    return f"n{cfg['n_estimators']}_minleaf{cfg['min_samples_leaf']}_mf{cfg['max_features']}"


def run_rf_search(time_budget: float = 35.0) -> None:
    """Sequential (RF's n_jobs=-1 already uses all 4 cores per fit -- running
    fits in parallel processes would oversubscribe, not speed up). Checkpoints
    after every fit; call repeatedly until 'all cached' prints. Keeps fitting
    until it has used ``time_budget`` seconds, so cheap (smaller-train) folds
    don't waste a whole 45s call on just one fit."""
    frame = pd.read_parquet(FRAME_PATH)
    ckpt = cfgmod.load_json(RF_CKPT_PATH, {}) or {}
    start = time.time()
    any_done = False
    for cfg in RF_GRID:
        k = rf_cfg_key(cfg)
        for s in CAL_SEASONS:
            if k in ckpt and str(s) in ckpt[k]:
                continue
            if any_done and time.time() - start > time_budget:
                print(f"rf_search: time budget reached, {time.time()-start:.1f}s used")
                return
            t0 = time.time()
            tr = frame[frame["season"] < s]
            te = frame[frame["season"] == s]
            m = mlr.MLRanker("rf", **cfg).fit(tr, tr["y_over"])
            p = m.predict_p_over(te, enforce=False)
            ll = cal.logloss(p, te["y_over"].to_numpy())
            ckpt.setdefault(k, {})[str(s)] = {"log_loss": ll, "n": int(len(te))}
            cfgmod.save_json(RF_CKPT_PATH, ckpt)
            print(f"  rf {k} season {s}: ll={ll:.5f} ({time.time()-t0:.1f}s)")
            any_done = True
    print("rf_search: all configs x seasons already cached")


def analyze_rf_search() -> Dict:
    ckpt = cfgmod.load_json(RF_CKPT_PATH, {}) or {}
    pooled = []
    for cfg in RF_GRID:
        k = rf_cfg_key(cfg)
        rows = ckpt.get(k, {})
        if not all(str(s) in rows for s in CAL_SEASONS):
            continue
        tot_ll = sum(rows[str(s)]["log_loss"] * rows[str(s)]["n"] for s in CAL_SEASONS)
        tot_n = sum(rows[str(s)]["n"] for s in CAL_SEASONS)
        pooled.append({**cfg, "pooled_log_loss": round(tot_ll / tot_n, 5), "n": tot_n})
    pooled.sort(key=lambda x: x["pooled_log_loss"])
    payload = {"eval_seasons": CAL_SEASONS, "grid": pooled, "default": DEFAULT_RF,
               "ship_rf": pooled[0] if pooled else DEFAULT_RF,
               "note": ("raw (uncalibrated) pooled log-loss, 2022-2025 -- the "
                       "same window the ensemble decision and final "
                       "calibrated check use. In-sample argmax across this "
                       "small grid (disclosed; RF's per-fit cost ruled out a "
                       "full walk-forward selection loop here) -- the final "
                       "choice is re-checked on the real calibrated metric.")}
    cfgmod.save_json(RF_RESULT_PATH, payload)
    print(json.dumps(payload, indent=1))
    return payload


# --------------------------------------------------------------------------- #
# Feature pruning: walk-forward leave-one-out ablation.
#
# Ablation vehicle = GBDT at the 7.2-tuned hyperparameters (fast, ~2-4s/fit),
# NOT RF (the model that ends up shipping) -- RF's n_jobs=-1 internal
# parallelism means per-fit cost (~10-20s) times 67 features x 5 folds would
# be 60-90 minutes of wall time the sandbox can't spend on a search proxy.
# Both models are tree ensembles splitting on the identical tabular features,
# so "does this feature move the metric at all" is a reasonably model-
# agnostic question; the FINAL pruned feature set is re-validated on RF (the
# shipped model) in the final_check stage before anything ships.
# --------------------------------------------------------------------------- #
PRUNE_PRED_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_prune_preds.pkl")


def _ablation_hp() -> Dict:
    res = cfgmod.load_json(HP_RESULT_PATH, None)
    if res is None:
        return dict(DEFAULT_HP)
    return {k: v for k, v in res["ship_hp"].items() if k in DEFAULT_HP}


# Combined-GROUP drops: leave-one-out is blind to redundancy -- a cluster of
# near-collinear features (e.g. z/mean/sd/line/mean_minus_line, all
# deterministic transforms of the same mean/sd/line triple that p_over ALSO
# encodes) can each individually show low t purely because the model routes
# around a dropped member through its correlated neighbors, while dropping
# the WHOLE cluster could still hurt a lot. Tested as combined drops, on top
# of the four Phase-6 groups the checkpoint calls out by name.
GROUP_DROPS = {
    "core_belief": ["z", "mean", "sd", "line", "mean_minus_line", "sd_over_line", "opp_factor"],
    "depth_location": ["roll_short_tgt_share", "roll_mid_tgt_share", "roll_short_pass_share"],
    "rz_shares": ["rz_tgt_share", "rz_carry_share", "opp_rz_td_factor"],
    "durability": ["roll_early_exit_rate", "inj_out_count_2y"],
    "opp_absence": ["opp_absence_factor"],
}


def prune_variants() -> List[str]:
    return (["full"] + [f"drop_{f}" for f in mlr.NUMERIC_FEATURES]
            + [f"dropgroup_{g}" for g in GROUP_DROPS])


def _fit_prune_variant(args) -> Tuple[str, int, np.ndarray, np.ndarray, List]:
    variant, season, frame_path, hp = args
    frame = pd.read_parquet(frame_path)
    tr = frame[frame["season"] < season]
    te = frame[frame["season"] == season]
    numeric = None
    if variant.startswith("dropgroup_"):
        grp = GROUP_DROPS[variant[len("dropgroup_"):]]
        numeric = [f for f in mlr.NUMERIC_FEATURES if f not in grp]
    elif variant != "full":
        dropped = variant[len("drop_"):]
        numeric = [f for f in mlr.NUMERIC_FEATURES if f != dropped]
    m = mlr.MLRanker("gbdt", features=numeric, **hp).fit(tr, tr["y_over"])
    p = m.predict_p_over(te, enforce=False)
    y = te["y_over"].to_numpy()
    row_key = list(zip(te["season"], te["week"], te["game_id"],
                       te["player_id"], te["market"]))
    return variant, season, p, y, row_key


def _atomic_pickle_dump(obj, path: str) -> None:
    """Write-then-rename so a kill mid-write (bash timeout) never leaves a
    truncated, unreadable checkpoint -- os.replace is atomic on POSIX."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        import pickle
        pickle.dump(obj, fh)
    os.replace(tmp, path)


def run_prune(batch: int) -> None:
    from concurrent.futures import as_completed
    import pickle
    store: Dict[int, Dict] = {}
    if os.path.exists(PRUNE_PRED_PATH):
        with open(PRUNE_PRED_PATH, "rb") as fh:
            store = pickle.load(fh)
    hp = _ablation_hp()
    pending = []
    for v in prune_variants():
        for s in PRUNE_EVAL_SEASONS:
            if s in store and v in store[s].get("preds", {}):
                continue
            pending.append((v, s, FRAME_PATH, hp))
    if not pending:
        print(f"prune: all {len(prune_variants())} variants x "
              f"{len(PRUNE_EVAL_SEASONS)} seasons already cached")
        return
    todo = pending[:batch]
    done = 0
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_fit_prune_variant, t): t for t in todo}
        for fut in as_completed(futs):
            variant, season, p, y, row_key = fut.result()
            sd = store.setdefault(season, {"y": y, "row_key": row_key, "preds": {}})
            sd["preds"][variant] = p
            done += 1
            _atomic_pickle_dump(store, PRUNE_PRED_PATH)
    print(f"prune: completed {done}/{len(pending)} pending "
          f"({len(pending) - done} remain)")


def analyze_prune() -> Dict:
    import pickle
    with open(PRUNE_PRED_PATH, "rb") as fh:
        store = pickle.load(fh)
    seasons = sorted(store.keys())

    def pooled_ll(variant: str) -> Tuple[float, int]:
        tot_ll, tot_n = 0.0, 0
        for s in seasons:
            sd = store[s]
            p, y = sd["preds"][variant], sd["y"]
            tot_ll += cal.logloss(p, y) * len(y)
            tot_n += len(y)
        return tot_ll / tot_n, tot_n

    full_ll, n_total = pooled_ll("full")
    rows = []
    for f in mlr.NUMERIC_FEATURES:
        v = f"drop_{f}"
        drop_ll, _ = pooled_ll(v)
        # paired per-row log-loss test, pooled across all seasons: positive
        # t => dropping f makes log-loss WORSE (f earns its keep)
        d_all, y_all = [], []
        full_all = []
        for s in seasons:
            sd = store[s]
            full_all.append(sd["preds"]["full"])
            d_all.append(sd["preds"][v])
            y_all.append(sd["y"])
        # paired_t(a, b, y): +t means b beats a. We want +t = dropping HURT
        # (full beats drop) -> paired_t(drop, full, y)
        t = cal.paired_t(np.concatenate(d_all), np.concatenate(full_all), np.concatenate(y_all))
        keep = t["t"] >= 2.0
        rows.append({"feature": f, "drop_log_loss": round(drop_ll, 5),
                    "full_log_loss": round(full_ll, 5),
                    "delta_vs_full": round(drop_ll - full_ll, 5),
                    "t_full_vs_drop": t["t"], "keep": bool(keep),
                    "keep_reason": "individual"})
    rows.sort(key=lambda r: -r["t_full_vs_drop"])

    phase6_groups = {
        "depth_location": ["roll_short_tgt_share", "roll_mid_tgt_share", "roll_short_pass_share"],
        "rz_shares": ["rz_tgt_share", "rz_carry_share", "opp_rz_td_factor"],
        "durability": ["roll_early_exit_rate", "inj_out_count_2y"],
        "opp_absence": ["opp_absence_factor"],
    }
    by_feature = {r["feature"]: r for r in rows}
    group_summary = {g: {"features": feats,
                         "kept": [f for f in feats if by_feature[f]["keep"]],
                         "dropped": [f for f in feats if not by_feature[f]["keep"]]}
                     for g, feats in phase6_groups.items()}

    # combined-group drop check (resolves the leave-one-out/collinearity
    # blind spot for the core-belief cluster and double-checks the four
    # Phase-6 groups as a WHOLE, not just member-by-member)
    group_rows = []
    for g in GROUP_DROPS:
        v = f"dropgroup_{g}"
        if not all(v in store[s]["preds"] for s in seasons):
            continue
        drop_ll, _ = pooled_ll(v)
        d_all = [store[s]["preds"][v] for s in seasons]
        full_all = [store[s]["preds"]["full"] for s in seasons]
        y_all = [store[s]["y"] for s in seasons]
        t = cal.paired_t(np.concatenate(d_all), np.concatenate(full_all), np.concatenate(y_all))
        group_rows.append({"group": g, "features": GROUP_DROPS[g],
                           "drop_log_loss": round(drop_ll, 5), "full_log_loss": round(full_ll, 5),
                           "delta_vs_full": round(drop_ll - full_ll, 5),
                           "t_full_vs_drop": t["t"],
                           "keep_group": t["t"] >= 2.0})
    group_rows.sort(key=lambda r: -r["t_full_vs_drop"])

    # Group override: leave-one-out is blind to redundancy within a
    # collinear cluster (dropping ONE member barely hurts because the model
    # routes around it through its correlated neighbors that are still
    # present). Where the COMBINED group clears t>=2, every member is kept
    # regardless of its individual verdict -- the individual test was
    # confounded, not informative, for that member.
    group_by_name = {r["group"]: r for r in group_rows}
    for g, gr in group_by_name.items():
        if not gr["keep_group"]:
            continue
        for f in gr["features"]:
            if f in by_feature and not by_feature[f]["keep"]:
                by_feature[f]["keep"] = True
                by_feature[f]["keep_reason"] = f"group override ({g}, t={gr['t_full_vs_drop']:+.2f})"

    kept = [r["feature"] for r in rows if r["keep"]]
    dropped = [r["feature"] for r in rows if not r["keep"]]
    payload = {"eval_seasons": seasons, "n_eval": n_total,
               "ablation_model": "gbdt (7.2-tuned hp) -- see module docstring "
                                 "for why RF wasn't used as the ablation vehicle",
               "full_log_loss": round(full_ll, 5), "per_feature": rows,
               "phase6_groups": group_summary, "group_drops": group_rows,
               "kept_features": kept, "dropped_features": dropped,
               "n_kept": len(kept), "n_dropped": len(dropped),
               "note": ("t_full_vs_drop >= 2.0 keeps a feature (dropping it "
                       "would measurably hurt pooled walk-forward log-loss); "
                       "below that bar it's dropped UNLESS a combined-group "
                       "drop test (group_drops) shows the feature's whole "
                       "collinear cluster matters, in which case every member "
                       "is kept (individual leave-one-out can't see "
                       "redundancy within a cluster). Raw (uncalibrated) "
                       f"log-loss, GBDT ablation vehicle, {seasons[0]}-{seasons[-1]}.")}
    cfgmod.save_json(PRUNE_RESULT_PATH, payload)
    print(json.dumps({k: v for k, v in payload.items() if k not in ("per_feature",)},
                     indent=1))
    return payload


# --------------------------------------------------------------------------- #
# Final check: the shipped recipe (RF + pruned features), calibrated, vs the
# 7.1 baseline (GBDT, all features, calibrated) on the REAL metric suite --
# not the cheap raw-log-loss proxies used to search/prune.
# --------------------------------------------------------------------------- #
FINAL_OOS_PATH = os.path.join(cfgmod.DATA_DIR, "ml_tune_final_oos.parquet")


def run_final_oos(features: Optional[List[str]] = None,
                  out_path: str = FINAL_OOS_PATH) -> None:
    """RF walk-forward OOS with the pruned feature set (resumable, one fit
    at a time like rf_search -- RF fits are the expensive step here)."""
    if features is None:
        prune = cfgmod.load_json(PRUNE_RESULT_PATH, None)
        if prune is None:
            raise SystemExit("run --stage prune_analyze first")
        features = prune["kept_features"]
    frame = pd.read_parquet(FRAME_PATH)
    done = pd.read_parquet(out_path) if os.path.exists(out_path) else pd.DataFrame()
    have = set(done["season"].unique().tolist()) if len(done) else set()
    chunks = [done] if len(done) else []
    for s in EVAL_SEASONS:
        if s in have:
            continue
        t0 = time.time()
        tr = frame[frame["season"] < s]
        te = frame[frame["season"] == s].copy()
        m = mlr.MLRanker("rf", features=features).fit(tr, tr["y_over"])
        te["p_raw"] = m.predict_p_over(te, enforce=False)
        keep_cols = ["season", "week", "game_id", "player_id", "market", "y_over", "p_raw"]
        chunks.append(te[keep_cols])
        pd.concat(chunks, ignore_index=True).to_parquet(out_path, index=False)
        print(f"  pruned-RF OOS season {s}: {len(tr):,} train rows, "
              f"{time.time()-t0:.1f}s")
        return  # one season per call -- keeps each invocation under 45s
    print("run_final_oos: all seasons cached")


def final_check(oos_path: str = None, label: str = "rf (default hp, pruned features)") -> Dict:
    """``oos_path=None`` uses the pruned-feature RF OOS (FINAL_OOS_PATH); pass
    ``data/ens_rf_oos.parquet`` (renaming p_rf->p_raw is handled) to check the
    all-features RF instead -- that's what SHIPPED (see docs/decisions_p7.md
    7.2): pruning found real pooled dead weight but broke the passing_yards
    per-market guardrail when actually applied to RF (t=-4.01), a mismatch
    the GBDT-proxy ablation vehicle didn't see. Kept here for reproducing
    either check."""
    baseline_oos = pd.read_parquet(os.path.join(cfgmod.DATA_DIR, "calib_oos.parquet"))
    oos_path = oos_path or FINAL_OOS_PATH
    final_oos = pd.read_parquet(oos_path)
    if "p_rf" in final_oos.columns:
        final_oos = final_oos.rename(columns={"p_rf": "p_raw"})
    if sorted(final_oos["season"].unique()) != EVAL_SEASONS:
        raise SystemExit(f"missing seasons in {oos_path} "
                         f"(have {sorted(final_oos['season'].unique())}, need {EVAL_SEASONS})")

    base_cal = _calibrate_seasons(baseline_oos, "platt", True, CAL_SEASONS)
    final_cal = _calibrate_seasons(final_oos, "platt", True, CAL_SEASONS)
    base_ev = baseline_oos.loc[baseline_oos["season"].isin(CAL_SEASONS)]
    final_ev = final_oos.loc[final_oos["season"].isin(CAL_SEASONS)]

    base_metrics = cal.metric_block(base_cal.loc[base_ev.index].to_numpy(),
                                    base_ev["y_over"].to_numpy(), base_ev["market"].to_numpy())
    final_metrics = cal.metric_block(final_cal.loc[final_ev.index].to_numpy(),
                                     final_ev["y_over"].to_numpy(), final_ev["market"].to_numpy())

    key = ["season", "week", "game_id", "player_id", "market"]
    merged = base_ev[key + ["y_over"]].copy()
    merged["p_base"] = base_cal.loc[base_ev.index].to_numpy()
    merged = merged.merge(
        final_ev[key].assign(p_final=final_cal.loc[final_ev.index].to_numpy()), on=key)
    sig = cal.paired_t(merged["p_base"].to_numpy(), merged["p_final"].to_numpy(),
                       merged["y_over"].to_numpy())

    # hit-rate / top-1 at the frozen selection policy (top-5/game, cap 2/player)
    base_leans = mlr.rank_and_grade(base_ev.assign(low_confidence=False), base_cal.loc[base_ev.index].to_numpy())
    final_leans = mlr.rank_and_grade(final_ev.assign(low_confidence=False), final_cal.loc[final_ev.index].to_numpy())

    def lean_summary(leans: pd.DataFrame) -> Dict:
        top1 = leans.groupby(["season", "week", "game_id"]).head(1)
        n, hits = len(leans), int(leans["ml_hit"].sum())
        return {"n": n, "hit_rate": round(hits / n, 4),
               "units_at_110": mlr.implied_units_at_110(hits, n),
               "top1_hit_rate": round(float(top1["ml_hit"].mean()), 4)}

    per_market_sig = {}
    for mkt in sorted(merged["market"].unique()):
        mm = merged[merged["market"] == mkt]
        per_market_sig[mkt] = cal.paired_t(mm["p_base"].to_numpy(), mm["p_final"].to_numpy(),
                                           mm["y_over"].to_numpy())
    no_market_regression = all(t["t"] >= -2.0 for t in per_market_sig.values())

    payload = {"eval_seasons": CAL_SEASONS, "n_eval": int(len(base_ev)),
               "baseline": {"model": "gbdt (all 67 features, 7.1 shipped)",
                           "calibrated": base_metrics, "leans": lean_summary(base_leans)},
               "final": {"model": label,
                        "calibrated": final_metrics, "leans": lean_summary(final_leans)},
               "significance_vs_baseline": sig,
               "per_market_significance_vs_baseline": per_market_sig,
               "no_market_regression": no_market_regression,
               "beats_baseline": bool(final_metrics["log_loss"] < base_metrics["log_loss"] - 1e-5
                                     and final_metrics["ece"] <= base_metrics["ece"] + 1e-4
                                     and sig["t"] >= 2.0 and no_market_regression)}
    cfgmod.save_json(FINAL_RESULT_PATH, payload)
    print(json.dumps(payload, indent=1))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["ens_rf_oos", "ens_analyze", "hp_search", "hp_analyze",
                            "rf_search", "rf_analyze",
                            "prune", "prune_analyze", "final_oos", "final_check"])
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--rescue", nargs="*", default=None,
                    help="extra features to add back on top of kept_features "
                         "(per-market guardrail rescue check)")
    args = ap.parse_args()

    if args.stage == "ens_rf_oos":
        frame = pd.read_parquet(FRAME_PATH)
        oos = build_rf_oos(frame)
        print(f"RF OOS cache: {len(oos):,} rows, seasons {sorted(oos['season'].unique())}")
        return
    if args.stage == "ens_analyze":
        ensemble_analyze()
        return
    if args.stage == "hp_search":
        run_hp_search(args.batch)
        return
    if args.stage == "hp_analyze":
        analyze_hp_search()
        return
    if args.stage == "rf_search":
        run_rf_search()
        return
    if args.stage == "rf_analyze":
        analyze_rf_search()
        return
    if args.stage == "prune":
        run_prune(args.batch)
        return
    if args.stage == "prune_analyze":
        analyze_prune()
        return
    if args.stage == "final_oos":
        if args.rescue:
            prune = cfgmod.load_json(PRUNE_RESULT_PATH, {})
            feats = prune["kept_features"] + list(args.rescue)
            out = os.path.join(cfgmod.DATA_DIR, "ml_tune_rescue_oos.parquet")
            run_final_oos(features=feats, out_path=out)
        else:
            run_final_oos()
        return
    if args.stage == "final_check":
        final_check(oos_path=os.path.join(cfgmod.DATA_DIR, "ens_rf_oos.parquet"),
                    label="rf (default hp, all 67 features -- SHIPPED)")
        return
    raise SystemExit(f"stage {args.stage!r} not implemented yet")


if __name__ == "__main__":
    main()

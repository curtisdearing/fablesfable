#!/usr/bin/env python3
"""Challenger A gated A/B — hierarchical Bayesian projection vs incumbent
parametric distributions, graded by line-free CRPS on realized stats.

Pre-registered 2026-07-30 (BUILD_PROMPTS_model_challengers_2026-07.md §A,
frozen at commit ded9fc6).  FROZEN GATE §A4, restated verbatim in spirit:

  1. PRIMARY: pooled walk-forward CRPS(challenger) < CRPS(incumbent), eval
     seasons 2021-2024, with P(improvement) >= 0.90 under the paired
     season-week cluster bootstrap (n=4000, seed=20260730), at BOTH seeds
     (7, 1234) of the fitting pipeline.
     Pre-declared expected delta: >= 1.5% relative CRPS improvement.
  2. GUARD (downstream ranker, unchanged features): top-5 hit rate drop
     <= 0.1pp and log-loss increase <= +0.0005 when p_over is fed from the
     challenger distribution.
  3. REPRODUCIBILITY: two identical-seed runs -> byte-identical books.

Mechanics declared before the run (see nflvalue/bayes_projection.py header):
SVI/ADVI inference; 512-draw predictive per row; the SAME sample-based CRPS
estimator scores both arms; anytime_td out of scope (fail-closed).  Guard
arm feeds challenger p_over wherever a walk-forward fit exists (2021+);
2019-2020 ranker-training rows keep incumbent p_over (no challenger exists
for them); all other features untouched.

Run:  python3 analysis/bayes_projection_eval.py [--guard-only] [--out PATH]
Writes book/bayes_projection_eval.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import config as cfgmod                      # noqa: E402
from nflvalue import ml_ranker as mlr                      # noqa: E402
from nflvalue.bayes_projection import (                    # noqa: E402
    MARKETS_IN_SCOPE, BayesProjection, crps_from_samples, incumbent_samples,
    save_book, _stable_hash)

FRAME_PATH = os.path.join(cfgmod.DATA_DIR, "ml_frame.parquet")
BOOK_PATH = os.path.join(ROOT, "book", "bayes_projection_eval.json")

EVAL_SEASONS = [2021, 2022, 2023, 2024]
SEEDS = [7, 1234]
MIN_P_IMPROVE = 0.90
EXPECTED_REL_DELTA = 0.015          # declared 2026-07-30, before any run
TOP5_GUARD_PP = 0.1
LL_GUARD = 0.0005
BOOT_N = 4000
BOOT_SEED = 20260730


def _paired_p(delta, keys, n=BOOT_N, seed=BOOT_SEED):
    """P(mean(delta) < 0) under a season-week cluster bootstrap (house
    pattern, analysis/loc_features_eval.py)."""
    clusters = defaultdict(list)
    for i, k in enumerate(keys):
        clusters[k].append(i)
    idx = [np.array(v) for v in clusters.values()]
    delta = np.asarray(delta)
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        pick = rng.integers(0, len(idx), len(idx))
        j = np.concatenate([idx[p] for p in pick])
        if delta[j].mean() < 0:
            wins += 1
    return wins / n


def build_eval_rows():
    """Per player-week-market rows with a realized actual AND an incumbent
    walk-forward distribution (mean_pred, sd_est, dist) — prop_backtest's own
    machinery, so the incumbent arm is exactly production's distribution."""
    import prop_backtest as pb
    from nflvalue.candidates import build_week_inputs
    inputs = build_week_inputs()
    frames = {}
    for market in MARKETS_IN_SCOPE:
        preds = pb._predictions_for_market(inputs.pw, market, inputs.team_idx,
                                           inputs.opp_idx)
        preds = preds.sort_values(["season", "week"],
                                  kind="mergesort").reset_index(drop=True)
        preds["sd_est"] = pb._walk_forward_sd(preds).values
        keep = preds.dropna(subset=["actual", "mean_pred", "sd_est"]).copy()
        keep = keep[np.isfinite(keep["mean_pred"]) & np.isfinite(keep["sd_est"])]
        frames[market] = keep
    return frames


def run_primary(frames, frame, seed: int):
    """One full walk-forward CRPS pipeline at one seed.  Returns per-market
    numbers + a challenger-p_over table for the downstream guard (computed
    here so predictive sample arrays never accumulate in memory)."""
    per_market = {}
    all_delta = []
    p_over_records = []             # (season, week, player_id, market, p_over)
    for market in MARKETS_IN_SCOPE:
        rows = frames[market]
        m_delta = []
        m_crps_i, m_crps_c, m_n = [], [], 0
        for S in EVAL_SEASONS:
            train = rows[rows["season"] < S]
            test = rows[rows["season"] == S].reset_index(drop=True)
            if test.empty or train.empty:
                continue
            model = BayesProjection(market, seed).fit(train, eval_season=S)
            samples_c = model.predictive_samples(test)
            samples_i = incumbent_samples(
                test["mean_pred"].to_numpy(), test["sd_est"].to_numpy(),
                test["dist"].tolist(),
                seed=_stable_hash(f"inc-{market}-{S}-{seed}") % (2 ** 32))
            actual = test["actual"].to_numpy(dtype=float)
            crps_c = crps_from_samples(samples_c, actual)
            crps_i = crps_from_samples(samples_i, actual)
            delta = crps_c - crps_i                    # negative = challenger better
            m_delta.append(pd.DataFrame({
                "season": test["season"], "week": test["week"], "delta": delta,
                "crps_i": crps_i, "crps_c": crps_c}))
            m_crps_i.append(crps_i.sum())
            m_crps_c.append(crps_c.sum())
            m_n += len(test)
            # challenger p_over at the frame's synthetic lines, computed NOW
            # so the (n_rows x 512) sample array can be freed per season
            sub = frame[(frame["market"] == market) & (frame["season"] == S)]
            if not sub.empty:
                smap = {(int(s), int(w), p): i for i, (s, w, p) in enumerate(
                    zip(test["season"], test["week"], test["player_id"]))}
                for r in sub.itertuples(index=False):
                    j = smap.get((int(r.season), int(r.week), r.player_id))
                    if j is not None:
                        p_over_records.append(
                            (int(r.season), int(r.week), r.player_id, market,
                             float((samples_c[j] > r.line).mean())))
            del samples_c, samples_i
        md = pd.concat(m_delta, ignore_index=True)
        pooled_i = float(np.sum(m_crps_i) / m_n)
        pooled_c = float(np.sum(m_crps_c) / m_n)
        per_market[market] = {
            "n": int(m_n),
            "crps_incumbent": round(pooled_i, 5),
            "crps_challenger": round(pooled_c, 5),
            "rel_delta": round((pooled_c - pooled_i) / pooled_i, 5),
        }
        all_delta.append(md)
    pooled = pd.concat(all_delta, ignore_index=True)
    keys = list(zip(pooled["season"], pooled["week"]))
    p_improve = _paired_p(pooled["delta"].to_numpy(), keys)
    crps_i_pooled = float(pooled["crps_i"].mean())
    crps_c_pooled = float(pooled["crps_c"].mean())
    summary = {
        "per_market": per_market,
        "pooled": {
            "n": int(len(pooled)),
            "crps_incumbent": round(crps_i_pooled, 5),
            "crps_challenger": round(crps_c_pooled, 5),
            "rel_delta": round((crps_c_pooled - crps_i_pooled) / crps_i_pooled, 5),
            "p_improve": round(p_improve, 4),
        },
    }
    p_over_chall = pd.DataFrame(
        p_over_records,
        columns=["season", "week", "player_id", "market", "p_over_chall"])
    return summary, p_over_chall


def run_guard(frame, p_over_chall, seed: int):
    """Downstream ranker guard: identical lean features, p_over fed from the
    challenger wherever a walk-forward fit exists (2021+)."""
    base_features = (cfgmod.load_config().get("ml_ranker") or {}).get("features")
    if not base_features:
        raise SystemExit(
            "config.json ml_ranker.features missing — restore the lean set")

    key_cols = ["season", "week", "player_id", "market"]
    renamed = p_over_chall.rename(columns={"p_over_chall": "_p_over_chall"})
    frame = frame.merge(renamed,
                        on=key_cols, how="left")
    n_swapped = int(frame["_p_over_chall"].notna().sum())

    def _arm(swap: bool):
        mlr._configured_subset = lambda: list(base_features)
        f = frame.copy()
        if swap:
            m = f["_p_over_chall"].notna()
            f.loc[m, "p_over"] = f.loc[m, "_p_over_chall"]
        rows, leans = [], []
        for S in EVAL_SEASONS:
            train = f[f["season"] < S]
            test = f[f["season"] == S]
            model = mlr.MLRanker(model="gbdt", seed=seed).fit(train, train["y_over"])
            p = model.predict_p_over(test)
            y = test["y_over"].to_numpy()
            pc = np.clip(p, 1e-12, 1 - 1e-12)
            rows.append(pd.DataFrame({
                "season": test["season"].to_numpy(), "week": test["week"].to_numpy(),
                "ll": -(y * np.log(pc) + (1 - y) * np.log(1 - pc))}))
            leans.append(mlr.rank_and_grade(test, p))
        ll = pd.concat(rows, ignore_index=True)
        ln = pd.concat(leans, ignore_index=True)
        return {"log_loss": round(float(ll["ll"].mean()), 5),
                "top5_hit": round(float(ln["ml_hit"].mean()), 4),
                "n_leans": int(len(ln))}

    inc = _arm(swap=False)
    cha = _arm(swap=True)
    return {
        "n_frame_rows_swapped": n_swapped,
        "incumbent_fed": inc, "challenger_fed": cha,
        "ll_increase": round(cha["log_loss"] - inc["log_loss"], 5),
        "top5_delta_pp": round((cha["top5_hit"] - inc["top5_hit"]) * 100, 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=BOOK_PATH)
    args = ap.parse_args()

    frames = build_eval_rows()
    frame = pd.read_parquet(FRAME_PATH)
    frame = frame[frame["market"].isin(MARKETS_IN_SCOPE)]
    frame = frame.reset_index(drop=True)

    per_seed = {}
    for seed in SEEDS:
        primary, p_over_chall = run_primary(frames, frame, seed)
        guard = run_guard(frame, p_over_chall, seed)
        per_seed[str(seed)] = {"primary": primary, "guard": guard}
        p = primary["pooled"]
        print(f"seed {seed}: CRPS {p['crps_incumbent']} -> {p['crps_challenger']} "
              f"(rel {p['rel_delta']:+.4f}, P(improve) {p['p_improve']}); "
              f"guard ll +{guard['ll_increase']}, top5 {guard['top5_delta_pp']:+}pp")

    primary_ok = all(
        v["primary"]["pooled"]["rel_delta"] < 0
        and v["primary"]["pooled"]["p_improve"] >= MIN_P_IMPROVE
        for v in per_seed.values())
    expected_delta_met = all(
        v["primary"]["pooled"]["rel_delta"] <= -EXPECTED_REL_DELTA
        for v in per_seed.values())
    guard_ok = all(
        v["guard"]["top5_delta_pp"] >= -TOP5_GUARD_PP
        and v["guard"]["ll_increase"] <= LL_GUARD
        for v in per_seed.values())
    gate = primary_ok and expected_delta_met and guard_ok

    book = {
        "note": ("Challenger A (hierarchical Bayesian projection) vs incumbent "
                 "parametric distributions; line-free CRPS on realized stats; "
                 "anytime_td out of scope (fail-closed). Sample-based CRPS "
                 "(512 draws) scores BOTH arms with the same estimator."),
        "preregistration": ("BUILD_PROMPTS_model_challengers_2026-07.md §A "
                            "(frozen ded9fc6)"),
        "inference_declared": ("SVI/ADVI (numpyro AutoNormal, Adam lr 0.01, "
                               "2000 steps), "
                               "declared before the eval run, fixed across seasons"),
        "eval_seasons": EVAL_SEASONS,
        "markets": list(MARKETS_IN_SCOPE),
        "seeds": per_seed,
        "gate": {
            "rule": (f"pooled CRPS delta < 0 with P >= {MIN_P_IMPROVE} at BOTH seeds "
                     f"(paired season-week bootstrap) AND declared expected delta "
                     f">= {EXPECTED_REL_DELTA:.1%} relative AND downstream guard "
                     f"(top-5 drop <= {TOP5_GUARD_PP}pp, ll increase <= +{LL_GUARD})"),
            "primary_ok": bool(primary_ok),
            "expected_delta_met": bool(expected_delta_met),
            "guard_ok": bool(guard_ok),
            "passed": bool(gate),
        },
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "cluster": "season-week"},
        "holdout_2025": None,
        "policy": ("FAIL -> record everything, flag stays false, no alternative "
                   "priors/likelihoods this checkpoint. 2025 untouched unless PASS."),
    }
    save_book(args.out, book)
    print(f"gate {'PASS' if gate else 'FAIL'}; wrote {args.out}")


if __name__ == "__main__":
    main()

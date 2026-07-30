#!/usr/bin/env python3
"""Pass-location feature lever — pre-registered walk-forward A/B on the ranker.

Challenger = the shipped lean feature set + the three pass_location features
(loc_middle_share, loc_left_share, loc_matchup_epa) built in
nflvalue/advanced_features.py from the free pbp `pass_location` column that
docs/DATA_SOURCES.md flags as the untapped derivation.

Protocol (declared before the first eval run; fablesfable_props track):
  * Walk-forward by season, eval 2021-2024, train strictly prior, identical
    frame/candidates/selection for both arms (only the feature list differs).
  * PRIMARY: pooled per-row log-loss delta (challenger - baseline) < 0 with
    P(improvement) >= 0.90 under a paired season-week cluster bootstrap
    (accuracy_protocol.json acceptance).
  * SECONDARY GUARD: pooled top-5 hit rate must not fall more than 0.1pp.
  * Robustness: both seeds (7, 1234) must agree on the primary direction.
  * 2025 stays untouched unless the gate passes (then ONE holdout look,
    reported, per the locked-benchmark policy).

Writes book/loc_features_eval.json.  Run:
    python3 analysis/loc_features_eval.py [--holdout]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import config as cfgmod            # noqa: E402
from nflvalue import ml_ranker as mlr            # noqa: E402

FRAME_PATH = os.path.join(cfgmod.DATA_DIR, "ml_frame.parquet")
BOOK_PATH = os.path.join(ROOT, "book", "loc_features_eval.json")

LOC_FEATURES = ["loc_middle_share", "loc_left_share", "loc_matchup_epa"]
EVAL_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
SEEDS = [7, 1234]
MIN_P_IMPROVE = 0.90
TOP5_GUARD_PP = 0.1
BOOT_N = 4000
BOOT_SEED = 20260730


def _paired_p(delta, keys, n=BOOT_N, seed=BOOT_SEED):
    """P(mean(delta) < 0) under a season-week cluster bootstrap."""
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


def _row_ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _with_features(features):
    """Patch the config-driven subset for this in-process arm."""
    mlr._configured_subset = lambda: list(features)


def run_arm(frame, features, seed, seasons):
    """One arm: per-season walk-forward fit/predict; per-row ll + lean grades."""
    _with_features(features)
    rows = []
    lean_frames = []
    for s in seasons:
        train = frame[frame["season"] < s]
        test = frame[frame["season"] == s]
        model = mlr.MLRanker(model="gbdt", seed=seed).fit(train, train["y_over"])
        p = model.predict_p_over(test)
        y = test["y_over"].to_numpy()
        rows.append(pd.DataFrame({
            "season": test["season"].to_numpy(), "week": test["week"].to_numpy(),
            "ll": _row_ll(y, p)}))
        leans = mlr.rank_and_grade(test, p)
        lean_frames.append(leans)
    ll = pd.concat(rows, ignore_index=True)
    leans = pd.concat(lean_frames, ignore_index=True)
    top1 = leans.groupby(["season", "week", "game_id"]).head(1)
    return {
        "ll_rows": ll,
        "log_loss": round(float(ll["ll"].mean()), 5),
        "top5_hit": round(float(leans["ml_hit"].mean()), 4),
        "top1_hit": round(float(top1["ml_hit"].mean()), 4),
        "n_leans": int(len(leans)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", action="store_true",
                    help="single 2025 look; only meaningful after a PASS")
    args = ap.parse_args()

    frame = pd.read_parquet(FRAME_PATH)
    base_features = (cfgmod.load_config().get("ml_ranker") or {}).get("features")
    if not base_features:
        raise SystemExit("config.json ml_ranker.features missing — lever needs the lean set")
    chall_features = list(base_features) + LOC_FEATURES
    missing = [f for f in LOC_FEATURES if f not in frame.columns]
    if missing:
        raise SystemExit(f"frame lacks {missing} — rebuild with ml_test --stage frame")

    per_seed = {}
    for seed in SEEDS:
        base = run_arm(frame, base_features, seed, EVAL_SEASONS)
        chall = run_arm(frame, chall_features, seed, EVAL_SEASONS)
        delta = chall["ll_rows"]["ll"].to_numpy() - base["ll_rows"]["ll"].to_numpy()
        keys = list(zip(base["ll_rows"]["season"], base["ll_rows"]["week"]))
        p_improve = _paired_p(delta, keys)
        per_seed[seed] = {
            "baseline": {k: v for k, v in base.items() if k != "ll_rows"},
            "challenger": {k: v for k, v in chall.items() if k != "ll_rows"},
            "ll_delta_pooled": round(float(delta.mean()), 6),
            "p_improve_ll": round(p_improve, 4),
            "top5_delta_pp": round((chall["top5_hit"] - base["top5_hit"]) * 100, 3),
        }
        print(f"seed {seed}: ll {base['log_loss']} -> {chall['log_loss']} "
              f"(delta {per_seed[seed]['ll_delta_pooled']}, P {p_improve}); "
              f"top5 {base['top5_hit']} -> {chall['top5_hit']}; "
              f"top1 {base['top1_hit']} -> {chall['top1_hit']}")

    both_direction = all(v["ll_delta_pooled"] < 0 for v in per_seed.values())
    p_ok = all(v["p_improve_ll"] >= MIN_P_IMPROVE for v in per_seed.values())
    guard_ok = all(v["top5_delta_pp"] >= -TOP5_GUARD_PP for v in per_seed.values())
    gate = both_direction and p_ok and guard_ok

    book = {
        "note": ("pass_location features A/B on the prop ranker; lean set vs "
                 "lean+3; synthetic-line framing applies to hit rates."),
        "features_added": LOC_FEATURES,
        "eval_seasons": EVAL_SEASONS,
        "seeds": {str(k): v for k, v in per_seed.items()},
        "gate": {
            "rule": (f"pooled ll delta < 0 with P >= {MIN_P_IMPROVE} at BOTH seeds "
                     f"(paired season-week bootstrap) AND top-5 drop <= {TOP5_GUARD_PP}pp"),
            "passed": bool(gate),
        },
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "cluster": "season-week"},
        "holdout_2025": None,
    }

    if args.holdout and gate:
        hb = run_arm(pd.read_parquet(FRAME_PATH),
                     (cfgmod.load_config().get("ml_ranker") or {}).get("features"),
                     SEEDS[0], [HOLDOUT_SEASON])
        hc = run_arm(pd.read_parquet(FRAME_PATH), chall_features, SEEDS[0],
                     [HOLDOUT_SEASON])
        book["holdout_2025"] = {
            "baseline": {k: v for k, v in hb.items() if k != "ll_rows"},
            "challenger": {k: v for k, v in hc.items() if k != "ll_rows"},
            "policy": "single touch, reported verbatim, no re-tuning after this look",
        }
        print(f"2025 holdout: ll {hb['log_loss']} -> {hc['log_loss']}; "
              f"top5 {hb['top5_hit']} -> {hc['top5_hit']}; "
              f"top1 {hb['top1_hit']} -> {hc['top1_hit']}")
    elif args.holdout:
        print("gate FAILED — 2025 stays untouched")

    os.makedirs(os.path.dirname(BOOK_PATH), exist_ok=True)
    with open(BOOK_PATH, "w") as fh:
        json.dump(book, fh, indent=1)
    print(f"gate {'PASS' if gate else 'FAIL'}; wrote {BOOK_PATH}")


if __name__ == "__main__":
    main()

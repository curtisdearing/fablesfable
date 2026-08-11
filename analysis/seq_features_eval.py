#!/usr/bin/env python3
"""Challenger B gated A/B — GRU sequence-encoder embeddings as GBDT features
(B1), with the pre-registered conditional direct head (B2) as this
checkpoint's LAST attempt if B1 fails.

Pre-registered 2026-07-30 (BUILD_PROMPTS_model_challengers_2026-07.md §B,
frozen at ded9fc6).  FROZEN GATE §B4 per variant, walk-forward 2021-2024:

  1. PRIMARY: pooled per-row log-loss delta < 0 with P >= 0.90 (paired
     season-week cluster bootstrap, n=4000, seed=20260730) at BOTH seeds
     (7, 1234).  Pre-declared expected delta: -0.003 for B1.
  2. GUARD: pooled top-5 hit rate drop <= 0.1pp.
  3. REPRODUCIBILITY: identical-seed reruns byte-identical.

B1 harness = analysis/loc_features_eval.py verbatim (lean set vs lean+8
``seq_h0..seq_h7``); the encoder for eval season S trains ONLY on seasons
< S (self-supervised, no lines/labels), PCA is fit on train-season hidden
states only.  B2 runs ONLY if B1 fails: small MLP on
[encoder hidden ‖ lean features] trained on y_over, evaluated as a full
ranker replacement under the identical harness.  B2 failing too is the
props track's THIRD consecutive rejection -> registered stop rule trips.

Run:  python3 analysis/seq_features_eval.py [--out PATH]
Writes book/seq_features_eval.json.
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
from nflvalue import seq_encoder as se                     # noqa: E402

# Register the candidate features with the ranker's column whitelist for this
# in-process A/B (the shipped mechanism would be the artifact-carried feature
# list per §B2.4; the eval harness registers them the same way
# loc_features_eval's features are known via advanced_features).
if se.SEQ_FEATURES[0] not in mlr.NUMERIC_FEATURES:
    mlr.NUMERIC_FEATURES = list(mlr.NUMERIC_FEATURES) + list(se.SEQ_FEATURES)

FRAME_PATH = os.path.join(cfgmod.DATA_DIR, "ml_frame.parquet")
BOOK_PATH = os.path.join(ROOT, "book", "seq_features_eval.json")

EVAL_SEASONS = [2021, 2022, 2023, 2024]
SEEDS = [7, 1234]
MIN_P_IMPROVE = 0.90
EXPECTED_LL_DELTA = -0.003          # declared 2026-07-30, before any run
TOP5_GUARD_PP = 0.1
BOOT_N = 4000
BOOT_SEED = 20260730


def _paired_p(delta, keys, n=BOOT_N, seed=BOOT_SEED):
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


def build_seq_features(frame: pd.DataFrame, seed: int):
    """Walk-forward seq_h0..7 for every frame row in train+eval scope:
    for eval season S, encoder_S and PCA_S are fit on seasons < S only, and
    the (train, test) feature columns for that season's A/B both come from
    encoder_S.  Returns {S: DataFrame(season, week, player_id, seq_h*)}."""
    from nflvalue.candidates import build_week_inputs
    inputs = build_week_inputs()
    gl = se.build_game_log(inputs.pw, inputs.schedules)
    per_season = {}
    for S in EVAL_SEASONS:
        enc = se.SeqEncoder(seed).fit(gl[gl["season"] < S], eval_season=S)
        rows = frame[frame["season"] <= S][
            ["season", "week", "player_id"]].drop_duplicates()
        hid = enc.hidden_states(gl, rows)
        hcols = [f"h{i}" for i in range(se.HIDDEN)]
        train_mask = hid["season"] < S
        pca = se.fit_pca(hid.loc[train_mask, hcols].to_numpy(), seed=seed)
        proj = se.project_pca(pca, hid[hcols].to_numpy())
        out = hid[["season", "week", "player_id"]].copy()
        for i in range(se.PCA_DIMS):
            out[f"seq_h{i}"] = proj[:, i]
        per_season[S] = out
        print(f"  seed {seed} S={S}: encoder {enc.meta['n_sequences']} seqs, "
              f"train mse {enc.meta['final_train_mse']}", flush=True)
    return per_season


def run_b1_arm(frame, features, seq_feats, seed):
    """One arm of the B1 A/B.  ``seq_feats`` None = baseline; else the
    per-eval-season walk-forward feature tables."""
    rows, leans = [], []
    for S in EVAL_SEASONS:
        f = frame.copy()
        if seq_feats is not None:
            f = f.merge(seq_feats[S], on=["season", "week", "player_id"], how="left")
            feats = list(features) + se.SEQ_FEATURES
        else:
            feats = list(features)
        mlr._configured_subset = lambda fl=feats: list(fl)
        train = f[f["season"] < S]
        test = f[f["season"] == S]
        model = mlr.MLRanker(model="gbdt", seed=seed).fit(train, train["y_over"])
        p = model.predict_p_over(test)
        y = test["y_over"].to_numpy()
        rows.append(pd.DataFrame({
            "season": test["season"].to_numpy(), "week": test["week"].to_numpy(),
            "ll": _row_ll(y, p)}))
        leans.append(mlr.rank_and_grade(test, p))
    ll = pd.concat(rows, ignore_index=True)
    ln = pd.concat(leans, ignore_index=True)
    return {"ll_rows": ll,
            "log_loss": round(float(ll["ll"].mean()), 5),
            "top5_hit": round(float(ln["ml_hit"].mean()), 4),
            "n_leans": int(len(ln))}


def run_b2_arm(frame, features, seq_feats, seed):
    """B2: direct p_over head — MLP on [seq hidden PCA ‖ lean features],
    trained on y_over, replacing the GBDT entirely.  Same selection
    protocol, same metrics."""
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    rows, leans = [], []
    feats = list(features) + se.SEQ_FEATURES
    for S in EVAL_SEASONS:
        f = frame.merge(seq_feats[S], on=["season", "week", "player_id"], how="left")
        train = f[f["season"] < S]
        test = f[f["season"] == S]
        mask = train["y_over"].notna()
        clf = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), random_state=seed,
                                  max_iter=200, early_stopping=True,
                                  validation_fraction=0.12)),
        ])
        clf.fit(train.loc[mask, feats], train.loc[mask, "y_over"].astype(int))
        p = clf.predict_proba(test[feats])[:, 1]
        y = test["y_over"].to_numpy()
        rows.append(pd.DataFrame({
            "season": test["season"].to_numpy(), "week": test["week"].to_numpy(),
            "ll": _row_ll(y, p)}))
        leans.append(mlr.rank_and_grade(test, p))
    ll = pd.concat(rows, ignore_index=True)
    ln = pd.concat(leans, ignore_index=True)
    return {"ll_rows": ll,
            "log_loss": round(float(ll["ll"].mean()), 5),
            "top5_hit": round(float(ln["ml_hit"].mean()), 4),
            "n_leans": int(len(ln))}


def _gate_variant(frame, base_features, variant_fn, label, seq_feats_by_seed):
    per_seed = {}
    for seed in SEEDS:
        seq_feats = seq_feats_by_seed[seed]
        base = run_b1_arm(frame, base_features, None, seed)
        chall = variant_fn(frame, base_features, seq_feats, seed)
        delta = (chall["ll_rows"]["ll"].to_numpy()
                 - base["ll_rows"]["ll"].to_numpy())
        keys = list(zip(base["ll_rows"]["season"], base["ll_rows"]["week"]))
        p_improve = _paired_p(delta, keys)
        per_seed[str(seed)] = {
            "baseline": {k: v for k, v in base.items() if k != "ll_rows"},
            "challenger": {k: v for k, v in chall.items() if k != "ll_rows"},
            "ll_delta_pooled": round(float(delta.mean()), 6),
            "p_improve_ll": round(p_improve, 4),
            "top5_delta_pp": round((chall["top5_hit"] - base["top5_hit"]) * 100, 3),
        }
        print(f"{label} seed {seed}: ll {base['log_loss']} -> {chall['log_loss']} "
              f"(delta {per_seed[str(seed)]['ll_delta_pooled']}, P {p_improve}); "
              f"top5 {base['top5_hit']} -> {chall['top5_hit']}", flush=True)
    direction = all(v["ll_delta_pooled"] < 0 for v in per_seed.values())
    p_ok = all(v["p_improve_ll"] >= MIN_P_IMPROVE for v in per_seed.values())
    guard_ok = all(v["top5_delta_pp"] >= -TOP5_GUARD_PP for v in per_seed.values())
    return {
        "seeds": per_seed,
        "gate": {
            "rule": (f"pooled ll delta < 0 with P >= {MIN_P_IMPROVE} at BOTH seeds "
                     f"(paired season-week bootstrap) AND top-5 drop "
                     f"<= {TOP5_GUARD_PP}pp"),
            "direction_ok": bool(direction), "p_ok": bool(p_ok),
            "guard_ok": bool(guard_ok),
            "passed": bool(direction and p_ok and guard_ok),
        },
    }


def run_holdout(book_path: str):
    """The registered SINGLE 2025 look, allowed only after a gate PASS and
    only once (reported verbatim, no re-tuning after this look)."""
    import json
    with open(book_path) as fh:
        book = json.load(fh)
    if not (book.get("b1", {}).get("gate", {}).get("passed")
            or (book.get("b2") or {}).get("gate", {}).get("passed")):
        raise SystemExit("gate FAILED — 2025 stays untouched")
    if book.get("holdout_2025") is not None:
        raise SystemExit("holdout already spent — the 2025 look is single-touch")
    frame = pd.read_parquet(FRAME_PATH)
    base_features = (cfgmod.load_config().get("ml_ranker") or {}).get("features")
    seed = SEEDS[0]
    global EVAL_SEASONS
    eval_saved = EVAL_SEASONS
    EVAL_SEASONS = [2025]
    try:
        seq_feats = build_seq_features(frame, seed)
        base = run_b1_arm(frame, base_features, None, seed)
        chall = run_b1_arm(frame, base_features, seq_feats, seed)
    finally:
        EVAL_SEASONS = eval_saved
    book["holdout_2025"] = {
        "seed": seed,
        "baseline": {k: v for k, v in base.items() if k != "ll_rows"},
        "challenger": {k: v for k, v in chall.items() if k != "ll_rows"},
        "ll_delta": round(chall["log_loss"] - base["log_loss"], 5),
        "top5_delta_pp": round((chall["top5_hit"] - base["top5_hit"]) * 100, 3),
        "policy": "single touch, reported verbatim, no re-tuning after this look",
    }
    from nflvalue.bayes_projection import save_book
    save_book(book_path, book)
    print(f"2025 holdout: ll {base['log_loss']} -> {chall['log_loss']} "
          f"(delta {book['holdout_2025']['ll_delta']}); "
          f"top5 {base['top5_hit']} -> {chall['top5_hit']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=BOOK_PATH)
    ap.add_argument("--holdout", action="store_true",
                    help="single 2025 look; only allowed after a PASS")
    args = ap.parse_args()

    if args.holdout:
        run_holdout(args.out)
        return

    frame = pd.read_parquet(FRAME_PATH)
    base_features = (cfgmod.load_config().get("ml_ranker") or {}).get("features")
    if not base_features:
        raise SystemExit(
            "config.json ml_ranker.features missing — lever needs the lean set")

    book = {
        "note": ("Challenger B: GRU trailing-16 game-log encoder + entity "
                 "embeddings; B1 = 8 PCA'd hidden dims as extra GBDT features "
                 "(lean vs lean+8); B2 (conditional, pre-registered LAST "
                 "attempt) = direct p_over head. Synthetic-line framing "
                 "applies to hit rates."),
        "preregistration": ("BUILD_PROMPTS_model_challengers_2026-07.md §B "
                            "(frozen ded9fc6)"),
        "declared_expected_delta_b1": EXPECTED_LL_DELTA,
        "eval_seasons": EVAL_SEASONS,
        "features_added": se.SEQ_FEATURES,
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "cluster": "season-week"},
        "holdout_2025": None,
    }

    seq_feats_by_seed = {}
    for seed in SEEDS:
        print(f"=== building walk-forward seq features, seed {seed} ===", flush=True)
        seq_feats_by_seed[seed] = build_seq_features(frame, seed)

    print("=== Variant B1: embeddings as GBDT features ===", flush=True)
    book["b1"] = _gate_variant(frame, base_features, run_b1_arm, "B1",
                               seq_feats_by_seed)

    if book["b1"]["gate"]["passed"]:
        book["b2"] = {"skipped": "B1 passed; one shipped lever per checkpoint"}
    else:
        print("=== B1 FAILED -> running pre-registered conditional B2 "
              "(the checkpoint's LAST attempt) ===", flush=True)
        book["b2"] = _gate_variant(frame, base_features, run_b2_arm, "B2",
                                   seq_feats_by_seed)

    b1_pass = book["b1"]["gate"]["passed"]
    b2_pass = isinstance(book.get("b2"), dict) and \
        (book["b2"].get("gate") or {}).get("passed", False)
    book["verdict"] = {
        "b1_passed": bool(b1_pass),
        "b2_passed": bool(b2_pass) if not b1_pass else None,
        "stop_rule_tripped": bool(not b1_pass and not b2_pass),
        "stop_rule": ("B1+B2 double rejection = props track's THIRD consecutive "
                      "rejection (after pass_location 2026-07-30) -> "
                      "stop_after_consecutive_rejections=3 trips; the props "
                      "lever hunt pauses. Recorded, not searched past."),
    }

    from nflvalue.bayes_projection import save_book
    save_book(args.out, book)
    verdict = ("PASS(B1)" if b1_pass
               else ("PASS(B2)" if b2_pass else "DOUBLE FAIL -> STOP RULE"))
    print(f"verdict {verdict}; wrote {args.out}")


if __name__ == "__main__":
    main()

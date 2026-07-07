#!/usr/bin/env python3
"""Phase 7.5 — same-game prop correlation, measured walk-forward + shrunk.

Question: when two props live in the same game (a QB's passing over, his WR's
receiving over, an RB's rushing), how much do their outcomes move together? Two
correlated leans are NOT two independent edges, and a same-game parlay's true
price depends on this structure. This script MEASURES it and EXPOSES it as a
walk-forward, shrunk artifact for 7.6 (selection) and 7.7 (staking). It changes
no selection/staking behavior itself.

Method:
  * RESIDUALS, not raw outcomes: r = (actual - projection mean) / projection sd,
    from data/ml_frame.parquet (the projection) joined to player_week (the
    actual). Standardizing makes a QB's 300-yard game and a WR's 90-yard game
    comparable, and strips the projection's own level so we measure co-movement,
    not shared trend.
  * Within each game, form player pairs and label each by TYPE:
    relationship (same_team | opponent | same_player) x each side's pos.family
    (e.g. QB.pass, WR.rec, RB.rush, *.td). Cross-player pairs use one market per
    family (the YARDAGE market + td) so a player can't be double-counted; the
    same-player bucket keeps every market (yards<->attempts).
  * Pearson rho per type, POOLED over standardized residual pairs.
  * WALK-FORWARD: rho for consumption at season S is estimated only from pairs
    in seasons < S (the artifact carries these slices; a leakage test checks it).
  * SHRINKAGE toward zero (so thin/noisy pairs don't invent structure):
    empirical-Bayes in Fisher-z space. z = atanh(rho), SE = 1/sqrt(n-3); the
    between-type signal variance tau^2 is estimated across types, and each z is
    pulled toward 0 by tau^2 / (tau^2 + SE^2). Noisy (small-n) types collapse to
    ~0; well-measured, stable types survive.
  * REAL vs NOISE is an EFFECT-SIZE + STABILITY call, deliberately NOT t>=2:
    with tens of thousands of pairs, t is enormous for economically-zero rho
    (two same-team WRs: rho=0.02 but t=3). A type is REAL iff
    |rho_shrunk| >= RHO_FLOOR AND its per-season sign is stable.

Run: python3 scripts/fit_correlation.py
Outputs: data/correlation_structure.json, reports/correlation_structure.md
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import config as cfgmod                       # noqa: E402
from nflvalue.candidates import ACTUAL_COL, build_week_inputs  # noqa: E402
from nflvalue.correlation import (                           # noqa: E402
    ART_PATH, FAMILY, POSITIONS, classify_pair, eb_fisher_z_shrink)

FRAME_PATH = os.path.join(cfgmod.DATA_DIR, "ml_frame.parquet")
REPORT_MD = "reports/correlation_structure.md"

RESID_CLIP = 8.0        # guard against absurd residuals from a tiny projection sd
CLEAN_LABEL_CONTEXT = True   # Phase 8.4: drop truncated / rest-week rows from
                             # the residual panel (a hamstring-on-drive-one line
                             # or a resting starter's stat pair says nothing
                             # about same-game correlation); same lens/flag
                             # family as features.RECENCY_FIT. Takes effect at
                             # the next artifact regeneration.
MIN_N = 300             # don't even report a type below this many pooled pairs
RHO_FLOOR = 0.05        # |shrunk rho| below this is economically noise
WALK_FORWARD_SEASONS = [2021, 2022, 2023, 2024, 2025]
SIGN_MIN_N = 200        # a per-season slice needs this many pairs to vote on sign


def build_residuals(frame: pd.DataFrame, pw: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for m, col in ACTUAL_COL.items():
        t = pw[["season", "week", "player_id", "team", col]].rename(columns={col: "actual"})
        t["market"] = m
        parts.append(t)
    td = pw[["season", "week", "player_id", "team"]].copy()
    td["actual"] = (pw["rush_tds"] + pw["rec_tds"]).to_numpy()
    td["market"] = "anytime_td"
    parts.append(td)
    act = pd.concat(parts, ignore_index=True)

    d = frame.merge(act, on=["season", "week", "player_id", "market"], how="inner")
    if CLEAN_LABEL_CONTEXT:
        for col in ("early_exit", "game_meaningless"):
            if col in pw.columns:
                tag = pw[["season", "week", "player_id", col]].drop_duplicates(
                    subset=["season", "week", "player_id"])
                d = d.merge(tag, on=["season", "week", "player_id"], how="left")
                d = d[d[col].fillna(0) == 0].drop(columns=[col])
    d["resid"] = ((d["actual"] - d["mean"]) / d["sd"].clip(lower=1e-6)).clip(-RESID_CLIP, RESID_CLIP)
    d["pos"] = "NA"
    for p in POSITIONS:
        d.loc[d[f"pos_{p}"] == 1, "pos"] = p
    d["family"] = d["market"].map(FAMILY)
    return d[["season", "week", "game_id", "player_id", "team", "market",
              "family", "pos", "resid"]].dropna(subset=["resid", "game_id"])


def collect_pairs(d: pd.DataFrame) -> Dict[str, Dict[str, list]]:
    """type -> {'x':[], 'y':[], 'season':[]} of standardized residual pairs.

    Classification (relationship, family, cross-vs-same-player restriction) is
    delegated to ``nflvalue.correlation.classify_pair`` so the read side and the
    measurement side can never disagree."""
    store: Dict[str, Dict[str, list]] = defaultdict(lambda: {"x": [], "y": [], "season": []})
    cols = ["season", "player_id", "team", "market", "pos", "resid"]
    for (_s, _w, _g), grp in d.groupby(["season", "week", "game_id"], sort=False):
        recs = grp[cols].to_dict("records")
        n = len(recs)
        for i in range(n):
            a = recs[i]
            for j in range(i + 1, n):
                b = recs[j]
                ptype = classify_pair(a["pos"], a["market"], a["player_id"], a["team"],
                                      b["pos"], b["market"], b["player_id"], b["team"])
                if ptype is None:
                    continue
                s = store[ptype]
                s["x"].append(a["resid"]); s["y"].append(b["resid"]); s["season"].append(a["season"])
    return store


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def analyze(store: Dict[str, Dict[str, list]]) -> Dict:
    types = {k: v for k, v in store.items() if len(v["x"]) >= MIN_N}
    x = {k: np.asarray(v["x"], float) for k, v in types.items()}
    y = {k: np.asarray(v["y"], float) for k, v in types.items()}
    ssn = {k: np.asarray(v["season"], int) for k, v in types.items()}

    rho_raw = {k: _rho(x[k], y[k]) for k in types}
    n = {k: len(x[k]) for k in types}
    rho_shrunk, tau2 = eb_fisher_z_shrink(rho_raw, n)

    # per-season rho (each season alone) for the sign-stability call
    per_season = {}
    for k in types:
        ps = {}
        for s in sorted(set(ssn[k].tolist())):
            m = ssn[k] == s
            if m.sum() >= SIGN_MIN_N:
                ps[int(s)] = round(_rho(x[k][m], y[k][m]), 3)
        per_season[k] = ps

    # walk-forward slices: rho from seasons < S (what a consumer at S may use).
    # A type is included in slice S ONLY if a PRIOR-ONLY verdict (using strictly
    # seasons < S) judges it real -- |rho_<S| >= RHO_FLOOR AND its per-season
    # sign is stable across the per-season slices restricted to seasons < S
    # (>= 2 qualifying seasons, all sharing the sign of rho_<S). This mirrors
    # the full-history sign-stability logic but sees nothing from seasons >= S,
    # so the INCLUSION decision -- not just the value -- is leak-free.
    walk_forward = {}
    for S in WALK_FORWARD_SEASONS:
        wf = {}
        for k in types:
            m = ssn[k] < S
            if m.sum() < MIN_N:
                continue
            rho_prior = round(_rho(x[k][m], y[k][m]), 4)
            if abs(rho_prior) < RHO_FLOOR:
                continue
            prior_signs = []
            for s in sorted(set(ssn[k][m].tolist())):
                sm = ssn[k] == s
                if sm.sum() >= SIGN_MIN_N:
                    r_s = _rho(x[k][sm], y[k][sm])
                    if abs(r_s) >= 1e-9:
                        prior_signs.append(np.sign(r_s))
            prior_sign = np.sign(rho_prior)
            sign_stable_prior = bool(len(prior_signs) >= 2
                                     and all(sg == prior_sign for sg in prior_signs))
            if sign_stable_prior:
                wf[k] = rho_prior
        walk_forward[str(S)] = wf

    pair_types = {}
    for k in types:
        ps = per_season[k]
        signs = [np.sign(v) for v in ps.values() if abs(v) >= 1e-9]
        pooled_sign = np.sign(rho_shrunk[k])
        sign_stable = bool(len(signs) >= 2 and all(sg == pooled_sign for sg in signs))
        verdict = "real" if (abs(rho_shrunk[k]) >= RHO_FLOOR and sign_stable) else "noise"
        pair_types[k] = {
            "rho_raw": round(rho_raw[k], 4), "rho_shrunk": round(rho_shrunk[k], 4),
            "n_pairs": n[k], "se": round(1.0 / np.sqrt(max(n[k] - 3, 1)), 4),
            "per_season": ps, "sign_stable": sign_stable, "verdict": verdict,
        }
    return {"tau2": round(tau2, 5), "pair_types": pair_types, "walk_forward": walk_forward}


def write_report(payload: Dict) -> None:
    pt = payload["pair_types"]
    real = {k: v for k, v in pt.items() if v["verdict"] == "real"}
    noise = {k: v for k, v in pt.items() if v["verdict"] == "noise"}
    L = ["# Phase 7.5 — same-game prop correlation (walk-forward, shrunk)", "",
         f"Standardized residual correlation `(actual − proj mean)/proj sd`, pooled "
         f"per pair type. Empirical-Bayes Fisher-z shrinkage toward 0 (τ²={payload['tau2']}). "
         f"REAL = |shrunk ρ| ≥ {RHO_FLOOR} AND per-season sign stable — NOT t≥2 "
         "(n is huge; t flags economically-zero ρ as 'significant').", "",
         "Synthetic-line caveat: residuals are vs the projection, not real prices.", "",
         "## REAL correlation structure (consumable by 7.6/7.7)", "",
         "| pair type | n pairs | ρ raw | ρ shrunk | per-season | sign-stable |",
         "|---|---|---|---|---|---|"]
    for k, v in sorted(real.items(), key=lambda kv: -abs(kv[1]["rho_shrunk"])):
        L.append(f"| `{k}` | {v['n_pairs']:,} | {v['rho_raw']:+.3f} | **{v['rho_shrunk']:+.3f}** "
                 f"| {v['per_season']} | {v['sign_stable']} |")
    L += ["", "## NOISE (shrunk ~0 or sign-unstable — treated as 0 downstream)", "",
          "| pair type | n pairs | ρ raw | ρ shrunk | why noise |", "|---|---|---|---|---|"]
    for k, v in sorted(noise.items(), key=lambda kv: -kv[1]["n_pairs"]):
        why = "|ρ|<floor" if abs(v["rho_shrunk"]) < RHO_FLOOR else "sign unstable"
        L.append(f"| `{k}` | {v['n_pairs']:,} | {v['rho_raw']:+.3f} | {v['rho_shrunk']:+.3f} | {why} |")
    L += ["", f"Artifact: `{ART_PATH}` (production shrunk ρ + walk-forward slices).", ""]
    os.makedirs("reports", exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    frame = pd.read_parquet(FRAME_PATH)
    pw = build_week_inputs().pw
    d = build_residuals(frame, pw)
    n_games = d.groupby(["season", "week", "game_id"]).ngroups
    print(f"residuals: {len(d):,} rows, {n_games:,} game-weeks, "
          f"seasons {sorted(d['season'].unique().tolist())}")
    store = collect_pairs(d)
    payload = analyze(store)
    payload = {"built_from": "data/ml_frame.parquet + player_week",
               "seasons": sorted(int(s) for s in d["season"].unique()),
               "n_games": int(n_games), "measure": "standardized_residual_pearson",
               "shrinkage": {"method": "eb_fisher_z_toward_zero", "tau2": payload["tau2"],
                             "rho_floor": RHO_FLOOR, "min_n": MIN_N},
               **payload}
    cfgmod.save_json(ART_PATH, payload)
    write_report(payload)


if __name__ == "__main__":
    main()

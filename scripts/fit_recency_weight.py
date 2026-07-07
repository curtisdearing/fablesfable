"""MEASURED VERDICT: how much should recent games count, and does cleaning the
panel first change the answer?

This is the sweep behind the recency weight -- the one core knob (`EWM_SPAN=4`
in features.py) that shipped on reasoning, never on a fit. It answers three
questions in one walk-forward pass, on the real 2019->now panel:

  1. WEIGHT SHAPE   -- flat window N vs EWM span S vs season-to-date: which
                       trailing weighting predicts next game best (OOS MAE)?
  2. CLEAN-THEN-WEIGHT -- does dropping/down-weighting INJURY-SHORTENED and/or
                       REST/MEANINGLESS prior games (nflvalue.game_context)
                       before averaging beat the raw status quo? (Blowout
                       garbage time is NOT tested here -- already measured and
                       rejected in Phase 6.3.)
  3. CONDITIONING   -- is one global weight right, or should it vary by market,
                       role, player usage tier ("what kind of player"), or team
                       usage volatility ("coaching decision on usage")?

Every prediction for game t uses STRICTLY prior games (shift(1)); cleaning tags
on those prior games use only each game's own data. Baseline to beat = EWM
span 4, raw (production today). Emits data/recency_weight_fit.json + a printed
summary; changes NO production behavior. Nothing here ships until this verdict
says it clears a practical margin -- same discipline as fit_weather.py etc.

    python3 scripts/fit_recency_weight.py                 # full sweep
    python3 scripts/fit_recency_weight.py --quick         # fewer schemes
    python3 scripts/fit_recency_weight.py --markets receiving_yards,receptions
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HIST = os.path.join(ROOT, "historical")
OUT = os.path.join(ROOT, "data", "recency_weight_fit.json")

from nflvalue import features, game_context as gc, ingest  # noqa: E402

# market -> the per-week actual column produced by build_player_week
MARKET_ACTUAL = {
    "receiving_yards": "rec_yards", "receptions": "receptions",
    "rushing_yards": "rush_yards", "rush_attempts": "carries",
    "passing_yards": "pass_yards", "pass_attempts": "pass_attempts",
    "pass_completions": "completions",
}
BASELINE = ("ewm", 4, "raw")   # production today


# --------------------------------------------------------------------------- #
# Trailing prediction for one player's series (strictly prior games)
# --------------------------------------------------------------------------- #
def _trailing_pred(y: np.ndarray, quality: np.ndarray, kind: str, param: int) -> np.ndarray:
    """Prediction for each game from a QUALITY-WEIGHTED trailing average of
    prior games. ``quality`` in [0,1] per prior game (1=keep, 0=drop,
    0<q<1=down-weight the injury/rest games). ``kind`` in {flat, ewm, all}.
    Returns an array aligned to y; index 0 (no history) is NaN."""
    n = len(y)
    out = np.full(n, np.nan)
    for t in range(1, n):
        lo = 0 if (kind == "all") else max(0, t - param) if kind == "flat" else 0
        idx = np.arange(lo, t)
        w = quality[idx].astype(float)
        if kind == "ewm":                      # exponential recency x quality
            age = (t - 1 - idx).astype(float)   # 0 = most recent prior game
            alpha = 2.0 / (param + 1.0)
            w = w * (1.0 - alpha) ** age
        if w.sum() <= 0:
            continue
        out[t] = float(np.sum(w * y[idx]) / np.sum(w))
    return out


def _quality_vector(sub: pd.DataFrame, clean: str) -> np.ndarray:
    """Per-game keep/weight from the cleaning policy applied to that game's own
    tags (used only for PRIOR games inside _trailing_pred)."""
    q = np.ones(len(sub))
    inj = sub.get("injury_shortened", pd.Series(0.0, index=sub.index)).to_numpy()
    mng = sub.get("game_meaningless", pd.Series(0.0, index=sub.index)).to_numpy()
    if clean == "raw":
        return q
    if clean in ("drop_injury", "drop_both"):
        q = np.where(inj > 0, 0.0, q)
    if clean == "dampen_injury":
        q = np.where(inj > 0, 0.25, q)
    if clean in ("drop_rest", "drop_both"):
        q = np.where(mng > 0, 0.0, q)
    return q


def _schemes(quick: bool):
    weights = [("ewm", 4), ("ewm", 3), ("flat", 8)] if quick else [
        ("ewm", 2), ("ewm", 3), ("ewm", 4), ("ewm", 6), ("ewm", 8),
        ("flat", 3), ("flat", 4), ("flat", 6), ("flat", 8), ("all", 0)]
    cleans = ["raw", "drop_injury"] if quick else [
        "raw", "drop_injury", "dampen_injury", "drop_rest", "drop_both"]
    return weights, cleans


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def _score_market(pw: pd.DataFrame, market: str, weights, cleans) -> pd.DataFrame:
    ycol = MARKET_ACTUAL[market]
    keep = ["player_id", "season", "week", "role", "team", ycol,
            "injury_shortened", "game_meaningless"]
    d = pw[[c for c in keep if c in pw.columns]].dropna(subset=[ycol]).copy()
    d = d.sort_values(["player_id", "season", "week"])
    recs = []
    for (kind, param) in weights:
        for clean in cleans:
            preds = np.full(len(d), np.nan)
            pos = 0
            for _, sub in d.groupby("player_id", sort=False):
                y = sub[ycol].to_numpy(dtype=float)
                q = _quality_vector(sub, clean)
                preds[pos:pos + len(sub)] = _trailing_pred(y, q, kind, param)
                pos += len(sub)
            d["_pred"] = preds
            m = d.dropna(subset=["_pred"])
            m = m[m["season"] >= m["season"].min() + 1]   # need >=1 season history
            if m.empty:
                continue
            ae = (m["_pred"] - m[ycol]).abs()
            recs.append({"market": market, "kind": kind, "param": param, "clean": clean,
                         "mae": round(float(ae.mean()), 4), "n": int(len(m)),
                         "by_role": {r: round(float(g.abs().mean()), 4)
                                     for r, g in (m["_pred"] - m[ycol]).groupby(m["role"])}})
    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--markets", default=",".join(MARKET_ACTUAL))
    ap.add_argument("--out", default=OUT,
                    help="output path (per-market shards on slice-limited "
                         "environments merge via scripts/merge_recency_shards.py)")
    ap.add_argument("--panel", default=None,
                    help="tagged-panel parquet cache: loaded if it exists, "
                         "else built once and saved (shard runs share it)")
    args = ap.parse_args()
    markets = [m.strip() for m in args.markets.split(",") if m.strip() in MARKET_ACTUAL]
    weights, cleans = _schemes(args.quick)

    print("[recency] loading panel ...", flush=True)
    if args.panel and os.path.exists(args.panel):
        pw = pd.read_parquet(args.panel)
    else:
        pbp = ingest.load_all_pbp()
        sched = ingest.load_all_schedules()
        snaps_path = os.path.join(HIST, "snap_counts.parquet")
        snaps = pd.read_parquet(snaps_path) if os.path.exists(snaps_path) else None
        pw = features.build_player_week(pbp)
        pw = gc.tag_player_weeks(pw, pbp=pbp, schedules=sched, snap_counts=snaps)
        if args.panel:
            pw.to_parquet(args.panel, index=False)
    print(f"[recency] panel: {len(pw):,} player-weeks, "
          f"{int(pw['injury_shortened'].sum()):,} injury-shortened, "
          f"{int(pw['game_meaningless'].sum()):,} meaningless-tagged", flush=True)

    all_rows = []
    for mk in markets:
        print(f"[recency] sweeping {mk} ...", flush=True)
        all_rows.append(_score_market(pw, mk, weights, cleans))
    res = pd.concat(all_rows, ignore_index=True)

    # ---- verdict assembly --------------------------------------------------- #
    def _mae(market, kind, param, clean):
        r = res[(res.market == market) & (res.kind == kind) & (res.param == param) & (res.clean == clean)]
        return float(r["mae"].iloc[0]) if len(r) else None

    verdict = {"baseline": {"scheme": "ewm4_raw"}, "by_market": {}, "pooled": {}}
    for mk in markets:
        sub = res[res.market == mk].sort_values("mae")
        base = _mae(mk, *BASELINE)
        best = sub.iloc[0].to_dict() if len(sub) else {}
        best_clean_same_weight = sub[(sub.kind == "ewm") & (sub.param == 4)].sort_values("mae")
        verdict["by_market"][mk] = {
            "baseline_ewm4_raw_mae": None if base is None else round(base, 4),
            "best_scheme": {k: best.get(k) for k in ("kind", "param", "clean", "mae", "n")},
            "best_improvement_vs_baseline": None if (base is None or not best)
            else round(base - best["mae"], 4),
            "best_at_ewm4_cleaning_only": (
                None if best_clean_same_weight.empty else
                {k: best_clean_same_weight.iloc[0][k] for k in ("clean", "mae")}),
        }
    # pooled n-weighted MAE per (weight,clean) to read the global winner
    pooled = (res.assign(wsum=res.mae * res.n).groupby(["kind", "param", "clean"])
              .agg(mae_n=("wsum", "sum"), n=("n", "sum")))
    pooled["mae"] = (pooled["mae_n"] / pooled["n"]).round(4)
    pooled = pooled.drop(columns="mae_n").sort_values("mae").reset_index()
    verdict["pooled"]["ranking"] = pooled.head(12).to_dict("records")
    base_pooled = pooled[(pooled.kind == "ewm") & (pooled.param == 4) & (pooled.clean == "raw")]
    verdict["pooled"]["baseline_ewm4_raw_mae"] = (
        None if base_pooled.empty else float(base_pooled["mae"].iloc[0]))

    # full grid rides along so shard runs can be merged with exact pooled math
    verdict["grid"] = res.drop(columns=["by_role"]).to_dict("records")
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(verdict, f, indent=2)

    # ---- printed summary ---------------------------------------------------- #
    print("\n=== POOLED (n-weighted MAE, lower = better) — top 8 ===")
    for r in pooled.head(8).to_dict("records"):
        tag = "  <-- baseline" if (r["kind"], r["param"], r["clean"]) == BASELINE else ""
        print(f"  {r['kind']:>4} {r['param']:>2}  {r['clean']:<14} MAE={r['mae']:.4f}{tag}")
    print("\n=== PER MARKET (baseline EWM4-raw -> best) ===")
    for mk, v in verdict["by_market"].items():
        b = v["best_scheme"]
        print(f"  {mk:<16} {v['baseline_ewm4_raw_mae']} -> "
              f"{b.get('kind')}{b.get('param')}/{b.get('clean')} {b.get('mae')} "
              f"(Δ{v['best_improvement_vs_baseline']})")
    print(f"\n[recency] wrote {out_path}")
    print("VERDICT RULE: only promote a change whose pooled MAE gain is both "
          "consistent across seasons and larger than the noise you'd tune away; "
          "wire the winner into features.py behind a flag + a leakage test, the "
          "way garbage_filter already is.")


if __name__ == "__main__":
    main()

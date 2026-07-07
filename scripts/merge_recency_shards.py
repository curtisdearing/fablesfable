"""Merge per-market shards of fit_recency_weight.py into one verdict file,
recomputing the pooled ranking exactly as the single-run script does, and add
the PROMOTION-RULE evidence the sweep itself doesn't produce: per-season MAE
deltas of the candidate winner(s) vs the EWM4-raw baseline (a winner must be
consistent across seasons, not just pooled).

    python3 scripts/merge_recency_shards.py --shards "/tmp/rw_*.json" \
        --panel /tmp/rw_panel.parquet          # writes data/recency_weight_fit.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.fit_recency_weight import (BASELINE, MARKET_ACTUAL,  # noqa: E402
                                        _quality_vector, _trailing_pred)

OUT = os.path.join(ROOT, "data", "recency_weight_fit.json")


def _per_season_delta(pw: pd.DataFrame, market: str, scheme: tuple) -> dict:
    """MAE(baseline) - MAE(scheme) per season, positive = scheme better."""
    ycol = MARKET_ACTUAL[market]
    d = pw.dropna(subset=[ycol]).sort_values(["player_id", "season", "week"]).copy()
    out = {}
    for label, (kind, param, clean) in (("base", BASELINE), ("cand", scheme)):
        preds = np.full(len(d), np.nan)
        pos = 0
        for _, sub in d.groupby("player_id", sort=False):
            y = sub[ycol].to_numpy(dtype=float)
            q = _quality_vector(sub, clean)
            preds[pos:pos + len(sub)] = _trailing_pred(y, q, kind, param)
            pos += len(sub)
        d[f"_p_{label}"] = preds
    m = d.dropna(subset=["_p_base", "_p_cand"])
    m = m[m["season"] >= m["season"].min() + 1]
    for season, grp in m.groupby("season"):
        base = float((grp["_p_base"] - grp[ycol]).abs().mean())
        cand = float((grp["_p_cand"] - grp[ycol]).abs().mean())
        out[int(season)] = round(base - cand, 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="/tmp/rw_*.json")
    ap.add_argument("--panel", default="/tmp/rw_panel.parquet")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    grid_rows, by_market = [], {}
    for path in sorted(glob.glob(args.shards)):
        with open(path) as f:
            shard = json.load(f)
        grid_rows.extend(shard.get("grid", []))
        by_market.update(shard.get("by_market", {}))
    res = pd.DataFrame(grid_rows)
    if res.empty:
        raise SystemExit("no shard grids found")

    pooled = (res.assign(wsum=res.mae * res.n).groupby(["kind", "param", "clean"])
              .agg(mae_n=("wsum", "sum"), n=("n", "sum")))
    pooled["mae"] = (pooled["mae_n"] / pooled["n"]).round(4)
    pooled = pooled.drop(columns="mae_n").sort_values("mae").reset_index()
    base_pooled = pooled[(pooled.kind == BASELINE[0]) & (pooled.param == BASELINE[1])
                         & (pooled.clean == BASELINE[2])]

    verdict = {
        "baseline": {"scheme": "ewm4_raw"},
        "by_market": by_market,
        "pooled": {
            "ranking": pooled.head(12).to_dict("records"),
            "baseline_ewm4_raw_mae": (None if base_pooled.empty
                                      else float(base_pooled["mae"].iloc[0])),
        },
        "grid": grid_rows,
    }

    # ---- promotion-rule evidence: per-season deltas for each market winner --- #
    pw = pd.read_parquet(args.panel)
    consistency = {}
    for mk, v in by_market.items():
        b = v.get("best_scheme") or {}
        if not b.get("kind"):
            continue
        scheme = (b["kind"], int(b["param"]), b["clean"])
        deltas = _per_season_delta(pw, mk, scheme)
        consistency[mk] = {
            "winner": f"{b['kind']}{b['param']}/{b['clean']}",
            "per_season_delta_vs_ewm4raw": deltas,
            "seasons_improved": sum(1 for d in deltas.values() if d > 0),
            "seasons_total": len(deltas),
        }
        print(f"{mk:<17} {consistency[mk]['winner']:<16} "
              f"improved {consistency[mk]['seasons_improved']}/{len(deltas)} seasons  {deltas}")
    verdict["season_consistency"] = consistency

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nmerged {res['market'].nunique()} markets -> {args.out}")
    print("pooled top 5:", pooled.head(5).to_dict("records"))


if __name__ == "__main__":
    main()

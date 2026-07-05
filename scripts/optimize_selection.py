"""Phase 6.7: walk-forward SELECTION-policy optimizer -- "most bets won".

Framing (same correction as the original ML ask): the GBDT probability layer
already IS gradient descent -- boosting minimizes log-loss, retrained weekly.
What was never optimized is the BET-SELECTION policy sitting on top of it:
which candidates become bets. That policy is 5 discrete dials, so exhaustive
walk-forward search dominates gradient methods here (nothing to
differentiate):

    p_min       minimum model probability on the chosen side
    top_k       bets per game
    per_player  max bets per player per game
    markets     all / no anytime_td / core-4 yardage-count markets
    low_conf    include low-confidence candidates or not

Protocol (pre-committed, no peeking):
  * inputs = per-season OUT-OF-SAMPLE GBDT probabilities (model for season S
    trained strictly on seasons < S);
  * for each eval season S in 2022-2025, the policy is chosen ONLY on pooled
    prior seasons' OOS results (2021..S-1), objective = units at -110 with a
    minimum-volume floor (>=150 bets/season pro-rated) so a lucky 10-bet
    policy can't win;
  * the chosen policy is then applied to season S untouched.

Synthetic-line caveat applies to every number, as everywhere in this repo.

Run:  python3 scripts/optimize_selection.py [--preds /tmp/p6/oos_preds.parquet]
      (regenerate preds: fit gbdt per season on data/ml_frame.parquet with
       train = seasons < S, save p_ml per row -- see docs/decisions_p6.md)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CORE4 = {"receiving_yards", "receptions", "rushing_yards", "passing_yards"}
BASELINE = {"p_min": 0.50, "top_k": 5, "per_player": 2, "markets": "all", "low_conf": True}
GRID = {
    "p_min": [0.50, 0.54, 0.58, 0.62, 0.66],
    "top_k": [1, 2, 3, 5],
    "per_player": [1, 2],
    "markets": ["all", "no_td", "core4"],
    "low_conf": [True, False],
}
MIN_BETS_PER_SEASON = 150      # volume floor during POLICY SELECTION
UNIT_WIN = 100.0 / 110.0       # -110


def select_bets(df: pd.DataFrame, pol: dict) -> pd.DataFrame:
    d = df.copy()
    d["p_side"] = np.where(d["market"].eq("anytime_td") | (d["p_ml"] >= 0.5),
                           d["p_ml"], 1.0 - d["p_ml"])
    d["bet_side"] = np.where(d["market"].eq("anytime_td") | (d["p_ml"] >= 0.5),
                             "over", "under")
    d["won"] = np.where(d["bet_side"] == "over", d["y_over"], 1 - d["y_over"])
    if pol["markets"] == "no_td":
        d = d[d["market"] != "anytime_td"]
    elif pol["markets"] == "core4":
        d = d[d["market"].isin(CORE4)]
    if not pol["low_conf"]:
        d = d[~d["low_confidence"].astype(bool)]
    d = d[d["p_side"] >= pol["p_min"]]
    d = d.sort_values("p_side", ascending=False)
    d = d.groupby(["season", "week", "game_id", "player_id"]).head(pol["per_player"])
    d = d.groupby(["season", "week", "game_id"]).head(pol["top_k"])
    return d


def score(bets: pd.DataFrame) -> dict:
    n = len(bets)
    if n == 0:
        return {"n": 0, "hit": None, "units": 0.0}
    hits = float(bets["won"].sum())
    return {"n": n, "hit": round(hits / n, 4),
            "units": round(hits * UNIT_WIN - (n - hits), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="/tmp/p6/oos_preds.parquet")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "selection_opt.json"))
    args = ap.parse_args()
    preds = pd.read_parquet(args.preds).dropna(subset=["y_over", "p_ml"])
    seasons = sorted(preds["season"].unique().tolist())
    policies = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(preds):,} OOS predictions, seasons {seasons}, {len(policies)} policies")

    results = {"per_season": {}, "frontier": [], "baseline": {}, "grid": GRID,
               "objective": f"pooled prior-season units@-110, volume floor {MIN_BETS_PER_SEASON}/season"}
    for S in seasons[1:]:
        prior = preds[preds["season"] < S]
        n_prior_seasons = prior["season"].nunique()
        best, best_units = None, -1e9
        for pol in policies:
            sc = score(select_bets(prior, pol))
            if sc["n"] < MIN_BETS_PER_SEASON * n_prior_seasons:
                continue
            if sc["units"] > best_units:
                best, best_units = pol, sc["units"]
        oos = score(select_bets(preds[preds["season"] == S], best))
        base = score(select_bets(preds[preds["season"] == S], BASELINE))
        results["per_season"][int(S)] = {"policy": best, "oos": oos, "baseline_policy": base}
        print(f"{S}: policy {best}")
        print(f"     OOS {oos}   vs baseline(top5) {base}")

    # descriptive hit-vs-volume frontier on pooled OOS (each row's MODEL is
    # OOS; the frontier itself is descriptive, not a walk-forward promise)
    pooled = preds
    print("\npooled 2021-2025 OOS frontier (p_min sweep at top5/pp2/all):")
    for pm in (0.50, 0.54, 0.58, 0.62, 0.66, 0.70):
        sc = score(select_bets(pooled, {**BASELINE, "p_min": pm}))
        sc["p_min"] = pm
        results["frontier"].append(sc)
        bph = sc["n"] / pooled["week"].nunique() / pooled["season"].nunique()
        print(f"  p_min {pm:.2f}: n={sc['n']:>6,}  hit {sc['hit']}  units {sc['units']:>8}  (~{bph:.1f} bets/wk)")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()

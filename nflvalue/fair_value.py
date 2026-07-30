"""Fair-value market blend for game lines (the "nfelo move", it5 shipped).

fair = alpha * model + (1 - alpha) * market, with alpha fit walk-forward on
strictly-prior seasons.  This is the measured best margin/total *forecaster*
(analysis/line_engine.py it5: MAE 9.81 vs model-alone 10.21, corr .428) — a
PRICE-CONTEXT engine, never a side-picker.  The line-dissection doc's lesson
stands: blending the market back in collapses ATS edge (it6), so the fair
value is displayed next to the market line and the model's own edge, and it
never feeds ats_pick / total_pick / EV math.

Pre-registered ship gate, per analysis/accuracy_protocol.json `acceptance`
(paired_resampling_unit season-week, minimum_probability_of_improvement 0.9):
a market ("spread", "total") ships its alpha ONLY if, pooled walk-forward,

    blend MAE <= market MAE            (must beat the close as a forecaster)
    and P(blend beats market)  >= 0.90 (paired season-week bootstrap)
    and blend MAE <= model MAE         (must not be worse than the model alone)

Fit inputs come from real dumped sim outputs (backtest.py --dump-predictions
-> data/backtest_predictions.json), not re-derived ratings.  Results persist
to book/fair_value.json; consumers load ONLY gate-passed alphas via
load_shipped() and show nothing otherwise (fail-closed, like every other
unmeasured number in this repo).

Run:  python3 -m nflvalue.fair_value          # (re)fit + gate + write book
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_PATH = os.path.join(ROOT, "data", "backtest_predictions.json")
BOOK_PATH = os.path.join(ROOT, "book", "fair_value.json")

ALPHA_GRID = [round(a, 2) for a in np.arange(0.0, 1.01, 0.05)]
MIN_P_IMPROVE = 0.90        # protocol acceptance.minimum_probability_of_improvement
BOOTSTRAP_N = 4000
BOOTSTRAP_SEED = 20260730   # deterministic re-runs

MARKETS = {
    # market key -> (model prediction field, market line field, actual field)
    "spread": ("margin_mean", "spread_line", "margin"),
    "total": ("total_mean", "total_line", "total_pts"),
}


def _rows(preds, pred_key, market_key, actual_key):
    """Valid (season, week, model, market, actual) rows; skip missing fields."""
    out = []
    for p in preds:
        m, k, y = p.get(pred_key), p.get(market_key), p.get(actual_key)
        if m is None or k is None or y is None:
            continue
        out.append((int(p["season"]), int(p["week"]), float(m), float(k), float(y)))
    return out


def fit_alpha(rows, grid=None):
    """Alpha minimizing blend MAE on the given rows (train side of the walk)."""
    grid = grid if grid is not None else ALPHA_GRID
    m = np.array([r[2] for r in rows])
    k = np.array([r[3] for r in rows])
    y = np.array([r[4] for r in rows])
    best_a, best_mae = 0.0, float("inf")
    for a in grid:
        mae = float(np.mean(np.abs(a * m + (1 - a) * k - y)))
        if mae < best_mae - 1e-12:
            best_a, best_mae = float(a), mae
    return best_a, best_mae


def _paired_p_improve(err_market, err_blend, cluster_keys,
                      n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """P(blend MAE < market MAE) under a season-week cluster bootstrap."""
    clusters = defaultdict(list)
    for i, key in enumerate(cluster_keys):
        clusters[key].append(i)
    keys = sorted(clusters)
    if not keys:
        return 0.0
    idx_by_cluster = [np.array(clusters[k]) for k in keys]
    err_market = np.asarray(err_market)
    err_blend = np.asarray(err_blend)
    rng = np.random.default_rng(seed)
    wins = 0
    n_clusters = len(idx_by_cluster)
    for _ in range(n):
        pick = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([idx_by_cluster[j] for j in pick])
        if err_blend[idx].mean() < err_market[idx].mean():
            wins += 1
    return wins / n


def walk_forward(preds, market):
    """Walk-forward alphas + MAEs for one market; strictly-prior training only."""
    pred_key, market_key, actual_key = MARKETS[market]
    rows = _rows(preds, pred_key, market_key, actual_key)
    seasons = sorted({r[0] for r in rows})
    per_season, e_model, e_market, e_blend, clusters = [], [], [], [], []
    for s in seasons[1:]:
        train = [r for r in rows if r[0] < s]
        test = [r for r in rows if r[0] == s]
        if not train or not test:
            continue
        alpha, _ = fit_alpha(train)
        m = np.array([r[2] for r in test])
        k = np.array([r[3] for r in test])
        y = np.array([r[4] for r in test])
        blend = alpha * m + (1 - alpha) * k
        per_season.append({
            "season": s, "n": len(test), "alpha": alpha,
            "mae_model": round(float(np.mean(np.abs(m - y))), 4),
            "mae_market": round(float(np.mean(np.abs(k - y))), 4),
            "mae_blend": round(float(np.mean(np.abs(blend - y))), 4),
        })
        e_model.extend(np.abs(m - y)); e_market.extend(np.abs(k - y))
        e_blend.extend(np.abs(blend - y))
        clusters.extend((r[0], r[1]) for r in test)
    if not per_season:
        return None
    pooled = {
        "n": len(e_blend),
        "mae_model": round(float(np.mean(e_model)), 4),
        "mae_market": round(float(np.mean(e_market)), 4),
        "mae_blend": round(float(np.mean(e_blend)), 4),
        "p_blend_beats_market": round(
            _paired_p_improve(e_market, e_blend, clusters), 4),
    }
    gate_pass = (pooled["mae_blend"] <= pooled["mae_market"]
                 and pooled["mae_blend"] <= pooled["mae_model"]
                 and pooled["p_blend_beats_market"] >= MIN_P_IMPROVE)
    ship_alpha, _ = fit_alpha(rows)          # all seasons -> live alpha
    return {
        "per_season": per_season,
        "pooled": pooled,
        "gate": {
            "rule": ("mae_blend <= mae_market AND mae_blend <= mae_model AND "
                     f"p_blend_beats_market >= {MIN_P_IMPROVE} "
                     "(paired season-week bootstrap)"),
            "passed": bool(gate_pass),
        },
        "shipped_alpha": ship_alpha if gate_pass else None,
    }


def fit_and_write(predictions_path=PREDICTIONS_PATH, book_path=BOOK_PATH):
    with open(predictions_path) as fh:
        preds = json.load(fh)
    report = {
        "note": ("fair = alpha*model + (1-alpha)*market; price context ONLY, "
                 "never a side-picker (see docs/analysis_line_dissection.md it5/it6). "
                 "Only gate-passed markets ship an alpha; consumers fail closed."),
        "source": os.path.basename(predictions_path),
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                      "cluster": "season-week"},
        "markets": {},
    }
    for market in MARKETS:
        res = walk_forward(preds, market)
        if res is not None:
            report["markets"][market] = res
    os.makedirs(os.path.dirname(book_path), exist_ok=True)
    with open(book_path, "w") as fh:
        json.dump(report, fh, indent=1)
    return report


_SHIPPED_CACHE = {"path": None, "value": None}


def load_shipped(book_path=BOOK_PATH):
    """{market: alpha} for gate-PASSED markets only; {} when unmeasured (fail closed)."""
    if _SHIPPED_CACHE["path"] == book_path and _SHIPPED_CACHE["value"] is not None:
        return _SHIPPED_CACHE["value"]
    shipped = {}
    try:
        with open(book_path) as fh:
            book = json.load(fh)
        for market, res in (book.get("markets") or {}).items():
            alpha = res.get("shipped_alpha")
            if alpha is not None and res.get("gate", {}).get("passed"):
                shipped[market] = float(alpha)
    except (OSError, ValueError, KeyError, TypeError):
        shipped = {}
    _SHIPPED_CACHE["path"] = book_path
    _SHIPPED_CACHE["value"] = shipped
    return shipped


def fair_line(model_value, market_value, alpha):
    """Blend one line; None unless both sides and alpha exist (fail closed)."""
    if model_value is None or market_value is None or alpha is None:
        return None
    return float(alpha) * float(model_value) + (1.0 - float(alpha)) * float(market_value)


def main():
    report = fit_and_write()
    for market, res in report["markets"].items():
        p = res["pooled"]; g = res["gate"]
        print(f"{market:7s} pooled n={p['n']}  MAE model {p['mae_model']}  "
              f"market {p['mae_market']}  blend {p['mae_blend']}  "
              f"P(beat market) {p['p_blend_beats_market']}  "
              f"gate {'PASS' if g['passed'] else 'FAIL'}  "
              f"shipped_alpha {res['shipped_alpha']}")
    print(f"wrote {BOOK_PATH}")


if __name__ == "__main__":
    main()

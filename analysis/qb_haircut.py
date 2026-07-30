#!/usr/bin/env python3
"""QB backup-start haircut — dissection integration item 3, measured.

The line-dissection doc (docs/analysis_line_dissection.md) records that the
game sim makes NO QB adjustment while a backup start moves real openers 3-10
points.  This lab measures whether a walk-forward-fitted haircut on the sim's
own margin forecast survives its pre-registered gate.

Detection (pre-declared, pregame-knowable by kickoff):
    incumbent(team, game) = modal starting qb_id over the team's previous 8
    games (strictly prior, cross-season), requiring the incumbent started >= 5
    of those 8.  backup_start = this game's starter != incumbent.
    Fewer than 8 priors, or no >=5/8 incumbent -> NOT flagged (fail closed).

Haircut fit (walk-forward): for eval season S, h is fit on seasons < S by
minimizing margin MAE of  adj = margin_mean - h*backup_home + h*backup_away
over a 0..10 x 0.25 grid.  Sim forecasts come from real dumped predictions
(data/backtest_predictions.json), never re-derived.

Pre-registered ship gate (analysis/accuracy_protocol.json acceptance):
    pooled walk-forward MAE(adj) <= MAE(base)
    AND P(adj beats base) >= 0.90 under a paired season-week bootstrap.
Diagnostics on the flagged subset are reported but are NOT the gate.

Writes book/qb_haircut.json.  Consumers (weekly.py live board) apply a
haircut ONLY if this book says the gate passed (fail closed).

Run:  python3 analysis/qb_haircut.py
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHED_PATH = os.path.join(ROOT, "historical", "schedules_qb.parquet")
PRED_PATH = os.path.join(ROOT, "data", "backtest_predictions.json")
GAMES_PATH = os.path.join(ROOT, "data", "backtest_games.json")
BOOK_PATH = os.path.join(ROOT, "book", "qb_haircut.json")

WINDOW = 8            # trailing games defining the incumbent
MIN_STARTS = 5        # incumbent must have started >= this many of WINDOW
GRID = np.arange(0.0, 10.01, 0.25)
MIN_P_IMPROVE = 0.90
BOOT_N = 4000
BOOT_SEED = 20260730


def backup_flags(sched: pd.DataFrame, mode: str = "modal8") -> dict:
    """{(season, week, home, away): (backup_home, backup_away)} walk-forward.

    modal8  (v1): starter != modal of prior 8 team games (incumbent >= 5/8).
    abrupt  (v2, pre-registered AFTER v1's gate failure): starter != modal of
            prior 6 (incumbent >= 4/6) AND the incumbent started the team's
            IMMEDIATELY previous game — i.e. the first game(s) of an abrupt
            absence, where the ratings are most stale and the market moves
            most.  Settled regime changes are deliberately NOT flagged.
    """
    sched = sched.sort_values(["gameday", "season", "week"]).reset_index(drop=True)
    history: dict[str, list] = defaultdict(list)   # team -> [qb_id ...] in time order
    flags = {}
    for r in sched.itertuples(index=False):
        out = []
        for team, qb in ((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id)):
            flagged = False
            if qb is not None and not pd.isna(qb):
                if mode == "modal8":
                    prior = history[team][-WINDOW:]
                    if len(prior) == WINDOW:
                        (inc, n_inc), = Counter(prior).most_common(1)
                        flagged = n_inc >= MIN_STARTS and qb != inc
                elif mode == "abrupt":
                    prior = history[team][-6:]
                    if len(prior) == 6:
                        (inc, n_inc), = Counter(prior).most_common(1)
                        flagged = (n_inc >= 4 and qb != inc and prior[-1] == inc)
                else:
                    raise ValueError(mode)
            out.append(flagged)
        flags[(int(r.season), int(r.week), r.home_team, r.away_team)] = tuple(out)
        for team, qb in ((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id)):
            if qb is not None and not pd.isna(qb):
                history[team].append(qb)
    return flags


def _join(preds, games, flags):
    """Rows: (season, week, margin_mean, margin, bh, ba). Preds carry no team
    ids, so join through backtest_games.json order (same source, same filter)."""
    ready = [g for g in games if g.get("ready")]
    if len(ready) != len(preds):
        raise SystemExit(f"cannot align: {len(ready)} ready games vs {len(preds)} predictions")
    rows = []
    for g, p in zip(ready, preds):
        if (g["season"], g["week"]) != (p["season"], p["week"]):
            raise SystemExit("alignment drift between backtest_games and predictions")
        bh, ba = flags.get((g["season"], g["week"], g["home"], g["away"]), (False, False))
        rows.append((g["season"], g["week"], float(p["margin_mean"]),
                     float(p["margin"]), bool(bh), bool(ba)))
    return rows


def fit_h(rows):
    m = np.array([r[2] for r in rows]); y = np.array([r[3] for r in rows])
    bh = np.array([r[4] for r in rows]); ba = np.array([r[5] for r in rows])
    best_h, best_mae = 0.0, float("inf")
    for h in GRID:
        mae = float(np.mean(np.abs(m - h * bh + h * ba - y)))
        if mae < best_mae - 1e-12:
            best_h, best_mae = float(h), mae
    return best_h, best_mae


def _paired_p(err_base, err_adj, keys, n=BOOT_N, seed=BOOT_SEED):
    clusters = defaultdict(list)
    for i, k in enumerate(keys):
        clusters[k].append(i)
    idx = [np.array(v) for v in clusters.values()]
    err_base, err_adj = np.asarray(err_base), np.asarray(err_adj)
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        pick = rng.integers(0, len(idx), len(idx))
        j = np.concatenate([idx[p] for p in pick])
        if err_adj[j].mean() < err_base[j].mean():
            wins += 1
    return wins / n


def walk_forward(rows):
    seasons = sorted({r[0] for r in rows})
    per_season, eb, ea, keys = [], [], [], []
    flag_resid = []          # (residual on flagged-backup games, side-signed)
    for s in seasons[1:]:
        train = [r for r in rows if r[0] < s]
        test = [r for r in rows if r[0] == s]
        if not any(r[4] or r[5] for r in train):
            continue
        h, _ = fit_h(train)
        for (season, week, m, y, bh, ba) in test:
            adj = m - h * bh + h * ba
            eb.append(abs(m - y)); ea.append(abs(adj - y)); keys.append((season, week))
            if bh != ba:      # exactly one side has a backup
                signed = (y - m) * (-1 if bh else 1)   # >0 = backup team underperformed sim
                flag_resid.append(signed)
        n_flag = sum(1 for r in test if r[4] or r[5])
        per_season.append({"season": s, "h": h, "n": len(test), "n_flagged": n_flag})
    pooled = {
        "n": len(eb),
        "mae_base": round(float(np.mean(eb)), 4),
        "mae_adj": round(float(np.mean(ea)), 4),
        "p_adj_beats_base": round(_paired_p(eb, ea, keys), 4),
    }
    flagged = {
        "n_one_sided_backup": len(flag_resid),
        "mean_signed_residual_vs_backup": round(float(np.mean(flag_resid)), 3) if flag_resid else None,
        "note": "positive = the backup-QB team did worse than the sim expected (the haircut's premise)",
    }
    gate = (pooled["mae_adj"] <= pooled["mae_base"]
            and pooled["p_adj_beats_base"] >= MIN_P_IMPROVE)
    ship_h, _ = fit_h(rows)
    return per_season, pooled, flagged, bool(gate), (ship_h if gate else None)


def _run_variant(sched, preds, games, mode):
    rows = _join(preds, games, backup_flags(sched, mode))
    n_flagged = sum(1 for r in rows if r[4] or r[5])
    per_season, pooled, flagged, gate, ship_h = walk_forward(rows)
    print(f"[{mode}] games {len(rows)}  flagged(any side) {n_flagged}")
    for ps in per_season:
        print(f"  {ps['season']}: h={ps['h']:.2f}  n_flagged={ps['n_flagged']}")
    print(f"  pooled MAE base {pooled['mae_base']} adj {pooled['mae_adj']}  "
          f"P(improve) {pooled['p_adj_beats_base']}")
    print(f"  flagged one-sided n={flagged['n_one_sided_backup']}  "
          f"signed residual {flagged['mean_signed_residual_vs_backup']}")
    print(f"  gate {'PASS' if gate else 'FAIL'}  shipped h = {ship_h}")
    return {
        "detection_mode": mode,
        "n_games": len(rows), "n_flagged_any_side": n_flagged,
        "per_season": per_season,
        "pooled": pooled,
        "flagged_diagnostics": flagged,
        "gate": {"rule": f"mae_adj <= mae_base AND p >= {MIN_P_IMPROVE} "
                         "(paired season-week bootstrap)",
                 "passed": gate},
        "shipped_haircut_points": ship_h,
    }


def main():
    sched = pd.read_parquet(SCHED_PATH)
    sched = sched[sched["game_type"] == "REG"]
    with open(PRED_PATH) as fh:
        preds = json.load(fh)
    with open(GAMES_PATH) as fh:
        games = json.load(fh)
    book = {
        "note": ("QB backup-start haircut on the sim margin; gate per "
                 "accuracy_protocol acceptance; consumers fail closed. "
                 "Sequencing: v1 modal8 ran first and FAILED its pooled gate; "
                 "v2 abrupt was then pre-registered as a sharper detection "
                 "(first games of an abrupt absence only) and run ONCE."),
        "variants": {
            "v1_modal8": _run_variant(sched, preds, games, "modal8"),
            "v2_abrupt": _run_variant(sched, preds, games, "abrupt"),
        },
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "cluster": "season-week"},
    }
    os.makedirs(os.path.dirname(BOOK_PATH), exist_ok=True)
    with open(BOOK_PATH, "w") as fh:
        json.dump(book, fh, indent=1)
    print(f"wrote {BOOK_PATH}")


if __name__ == "__main__":
    main()

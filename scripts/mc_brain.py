"""Phase 6.8: Monte Carlo the brain -- what its record actually implies.

Inputs: the WEEKLY-RETRAIN graded leans (data/ml_weekly_{season}_gbdt.parquet,
the live Tuesday cadence -- the model's weights re-fit every week), 2024-2025.

What it computes:
  * cluster bootstrap BY WEEK (correlated leans within a week stay together)
    -> distribution of season hit rate and units at -110, P(profitable
    season), P(beating the 52.38% breakeven);
  * losing-streak / drawdown distributions and bankroll paths (flat 1u and
    quarter-Kelly) from within-week-shuffled bet sequences;
  * the realism section: these hit rates are vs SYNTHETIC trailing-mean
    lines. Real sportsbook lines price most of what this model knows. The
    report therefore also answers: what would various TRUE hit rates (52-58%,
    the plausible real-line band) mean in units, variance, and the number of
    bets needed to distinguish skill from luck. CLV remains the only
    accepted proof of real edge (kill-check unchanged).

Run:  python3 scripts/mc_brain.py            # writes docs/mc_brain_eval.md
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEASONS = (2024, 2025)
B = 10_000
UNIT_WIN = 100.0 / 110.0
BREAKEVEN = 0.5238
RNG = np.random.default_rng(20260705)


def load_leans() -> pd.DataFrame:
    frames = []
    for s in SEASONS:
        p = os.path.join(ROOT, "data", f"ml_weekly_{s}_gbdt.parquet")
        d = pd.read_parquet(p)
        d["season"] = s
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def bootstrap_weeks(leans: pd.DataFrame) -> dict:
    out = {}
    for s, grp in leans.groupby("season"):
        weeks = [g["ml_hit"].to_numpy() for _, g in grp.groupby("week")]
        nw = len(weeks)
        hits, units = [], []
        for _ in range(B):
            sample = [weeks[i] for i in RNG.integers(0, nw, nw)]
            arr = np.concatenate(sample)
            h = arr.mean()
            hits.append(h)
            units.append(arr.sum() * UNIT_WIN - (len(arr) - arr.sum()))
        hits, units = np.array(hits), np.array(units)
        out[int(s)] = {
            "n_bets": int(len(grp)), "hit": round(float(grp["ml_hit"].mean()), 4),
            "hit_ci90": [round(float(np.quantile(hits, q)), 4) for q in (0.05, 0.95)],
            "units": round(float(grp["ml_hit"].sum() * UNIT_WIN
                                 - (len(grp) - grp["ml_hit"].sum())), 1),
            "units_ci90": [round(float(np.quantile(units, q)), 1) for q in (0.05, 0.95)],
            "p_profitable": round(float((units > 0).mean()), 4),
            "p_beat_breakeven": round(float((hits > BREAKEVEN).mean()), 4),
        }
    return out


def streaks_and_bankroll(leans: pd.DataFrame) -> dict:
    """Within-week order is unknown pre-game -> shuffle within week, keep
    week order; flat 1u and quarter-Kelly bankroll paths from 100u."""
    seq_by_season = {}
    for s, grp in leans.groupby("season"):
        seq_by_season[int(s)] = [g["ml_hit"].to_numpy() for _, g in
                                 grp.groupby("week", sort=True)]
    out = {}
    for s, weeks in seq_by_season.items():
        max_lose, max_dd, end_flat = [], [], []
        for _ in range(B // 4):                                    # 2,500 paths
            path = np.concatenate([RNG.permutation(w) for w in weeks])
            # flat 1u
            pnl = np.where(path == 1, UNIT_WIN, -1.0).cumsum()
            eq = np.concatenate([[0.0], pnl])
            max_dd.append(float((np.maximum.accumulate(eq) - eq).max()))
            end_flat.append(float(pnl[-1]))
            # losing streak
            runs, cur = 0, 0
            for x in path:
                cur = cur + 1 if x == 0 else 0
                runs = max(runs, cur)
            max_lose.append(runs)
        out[s] = {
            "max_losing_streak_p50_p95": [int(np.quantile(max_lose, .5)),
                                          int(np.quantile(max_lose, .95))],
            "max_drawdown_units_p50_p95": [round(float(np.quantile(max_dd, .5)), 1),
                                           round(float(np.quantile(max_dd, .95)), 1)],
            "flat_end_units_p5_p50_p95": [round(float(np.quantile(end_flat, q)), 1)
                                          for q in (.05, .5, .95)],
        }
    return out


def realism_table() -> list:
    """What TRUE hit rates mean at -110, flat 1u, 300 bets/season -- the
    honest bridge from synthetic-line records to real-line expectations.
    Quarter-Kelly lives HERE, at plausible real-line rates: running Kelly at
    the synthetic 66-68% compounds into absurdity and would be pure fiction."""
    rows = []
    n = 300
    for p in (0.52, 0.5238, 0.54, 0.55, 0.56, 0.58):
        ev = n * (p * UNIT_WIN - (1 - p))
        sd = float(np.sqrt(n * p * (1 - p)) * (UNIT_WIN + 1))
        # bets to distinguish p from breakeven at 80% power, one-sided 5%
        n_dist = (int(np.ceil(((1.645 + 0.842) ** 2 * p * (1 - p))
                              / (p - BREAKEVEN) ** 2)) if p > BREAKEVEN else None)
        # quarter-Kelly over a 300-bet season from 100u
        kf = max((UNIT_WIN * p - (1 - p)) / UNIT_WIN, 0.0) / 4.0
        end_med, p_halve = 100.0, 0.0
        if kf > 0:
            banks = []
            halves = 0
            for _ in range(2000):
                wins = RNG.random(n) < p
                bank = 100.0
                low = bank
                for w in wins:
                    stake = bank * kf
                    bank += stake * UNIT_WIN if w else -stake
                    low = min(low, bank)
                banks.append(bank)
                halves += int(low <= 50)
            end_med, p_halve = float(np.median(banks)), halves / 2000
        rows.append({"true_hit": p, "ev_units_300bets": round(ev, 1),
                     "sd_units_300bets": round(sd, 1),
                     "p_losing_season": round(float(1 - 0.5 * (1 + math.erf(
                         ev / (sd * math.sqrt(2))))), 3) if sd else None,
                     "bets_to_prove_skill": n_dist,
                     "qkelly_f_pct": round(kf * 100, 2),
                     "qkelly_end_median": round(end_med, 1),
                     "qkelly_p_halve": round(p_halve, 3)})
    return rows


def main() -> None:
    leans = load_leans()
    boot = bootstrap_weeks(leans)
    paths = streaks_and_bankroll(leans)
    real = realism_table()
    res = {"inputs": "weekly-retrain GBDT leans (live cadence), synthetic-line graded",
           "bootstrap_by_week": boot, "paths": paths, "realism_at_real_lines": real,
           "breakeven": BREAKEVEN, "B": B}
    with open(os.path.join(ROOT, "data", "mc_brain.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    L = ["# Monte Carlo evaluation of the brain (Phase 6.8)", "",
         "**Inputs:** weekly-retrain GBDT leans (the live Tuesday cadence -- the",
         "model's weights re-fit every week), 2024-2025, graded at synthetic",
         "trailing-mean reference lines. **Every number below inherits that",
         "caveat**; real sportsbook lines price most of what this model knows.",
         "Breakeven at -110 = 52.38%. 1-800-GAMBLER.", ""]
    for s, r in boot.items():
        L += [f"## {s} (n={r['n_bets']} bets)", "",
              f"- hit rate {r['hit']:.1%} (week-bootstrap 90% CI "
              f"{r['hit_ci90'][0]:.1%}-{r['hit_ci90'][1]:.1%})",
              f"- units at -110 flat: {r['units']:+.1f} (90% CI {r['units_ci90'][0]:+.1f}"
              f" to {r['units_ci90'][1]:+.1f})",
              f"- P(profitable season) {r['p_profitable']:.1%}, "
              f"P(hit > breakeven) {r['p_beat_breakeven']:.1%}", ""]
        p = paths[s]
        L += [f"- max losing streak (median / p95): {p['max_losing_streak_p50_p95'][0]} / "
              f"{p['max_losing_streak_p50_p95'][1]} bets",
              f"- max drawdown flat-1u (median / p95): {p['max_drawdown_units_p50_p95'][0]} / "
              f"{p['max_drawdown_units_p50_p95'][1]} units", ""]
    L += ["## What this means at REAL lines", "",
          "The synthetic-line hit rates above measure directional skill vs a",
          "naive trailing-mean line, NOT price-beating. Against real, sharp",
          "-110 lines the plausible band for a good model is 52-58%. At flat",
          "1u, 300 bets/season:", "",
          "| true hit | EV (units) | SD (units) | P(losing season) | bets to prove skill | ¼-Kelly f | ¼-Kelly median end (100u) | P(halve) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in real:
        L.append(f"| {r['true_hit']:.2%} | {r['ev_units_300bets']:+.1f} | "
                 f"{r['sd_units_300bets']:.1f} | {r['p_losing_season']} | "
                 f"{r['bets_to_prove_skill'] or '—'} | {r['qkelly_f_pct']}% | "
                 f"{r['qkelly_end_median']} | {r['qkelly_p_halve']} |")
    n55 = next(r["bets_to_prove_skill"] for r in real if r["true_hit"] == 0.55)
    n54 = next(r["bets_to_prove_skill"] for r in real if r["true_hit"] == 0.54)
    L += ["",
          f"Read 'bets to prove skill': even a genuinely skilled 55% bettor",
          f"needs ~{n55:,} bets to statistically separate from breakeven (80%",
          f"power) -- multiple full seasons at this volume. A 54% bettor needs",
          f"~{n54:,}. This is why the kill-check uses CLV (n>=150 resolved,",
          "avg CLV>0, 52%+ positive-CLV rate) rather than won-bet counts:",
          "closing-line value converges orders of magnitude faster than",
          "profit does.", "",
          "**The honest chain:** synthetic-line skill (measured, strong) ->",
          "real-line hit rate (unknown until live prices accrue) -> profit",
          "(variance-dominated at any realistic volume). The brain's numbers",
          "justify running the live CLV experiment; they do not yet justify",
          "conviction about profit."]
    md = "\n".join(L) + "\n"
    with open(os.path.join(ROOT, "docs", "mc_brain_eval.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()

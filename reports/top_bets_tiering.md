# Accuracy-tiered per-game bet ranking (Top Bets)

Product rule: per game, "Best Bets" = up to top 5, ranked list capped at top 10.
Rank is driven by measured accuracy, fail-closed:
- BEST tier: rank ≤5 AND the pick's confidence band has graded accuracy ≥67%.
- VALUE tier: rank ≤10 AND band accuracy >50% AND edge >0.
- Bands with n<20 are "unproven" and excluded from both tiers.
- Games show FEWER bets when bands do not qualify; thresholds never relax.

## Measured bands (settled graded picks, data/weekly.json replay)

Admission is gated on the **Wilson 95% lower bound** (`accuracy_lb`), not point
accuracy, so thin/lucky bands cannot clear a tier. BEST needs `lb ≥ 67%`; VALUE
needs `lb > 50%` and edge > 0; bands with n < 20 are excluded.

**Shipped calibration — full walk-forward pool 2019-2025, n=1960 games
(recalibrated 2026-07-18; `build_ratings.py` now ingests 2024-25):**

| Band | Accuracy | 95% LB | n | Tier |
|---|---:|---:|---:|---|
| p>=0.7 (moneyline) | 74.4% | 69.9% | 395 | **BEST** |
| p 0.62-0.7 (moneyline) | 70.5% | 66.2% | 468 | VALUE |
| p 0.55-0.62 | 59.6% | 55.6% | 592 | VALUE |
| p 0.5-0.55 | 54.7% | 50.3% | 499 | VALUE (marginal) |
| edge>=4.0 (ATS/total) | 51.7% | 48.3% | 828 | excluded |
| edge 0.0-0.5 | 51.3% | 46.7% | 446 | excluded |
| edge 1.5-2.5 | 49.5% | 46.1% | 818 | excluded |
| edge 0.5-1.5 | 49.1% | 45.9% | 937 | excluded |
| edge 2.5-4.0 | 47.0% | 43.7% | 829 | excluded |

Emits **1959 tiered bets over 152 weeks (398 BEST / 1561 VALUE)**.

### Held-out forward test (the registered check) — PASSES

Bands were trained on **2021-2024**, frozen, then applied to **2025**; realized
2025 accuracy of every emitted pick (`analysis/lever3_holdout_recal.py`):

| | train 2021-24 (lb) | realized 2025 |
|---|---|---|
| **BEST tier** (all `p≥0.7`) | lb 67.2%, n=235 | **50/66 = 75.8%** |
| VALUE tier (pooled) | — | 95/147 = 64.6% |
| band `p≥0.7` | 73.2% (lb 67.2%) | 75.8% |
| band `p 0.62-0.7` | 68.6% (lb 63.1%) | 71.8% |
| band `p 0.55-0.62` | 61.5% (lb 56.3%) | 57.9% |

Every moneyline band held its label out-of-sample; the BEST tier exceeded its
≥67% floor at 75.8%. All ATS/total edge bands trained below the 50% LB gate and
so emitted nothing in 2025 — the fail-closed exclusion is correct, not a miss.

Key honest findings:
- **High-probability moneylines are the only real signal, and it is now
  out-of-sample validated.** `p≥0.7` clears BEST on 2021-24 training (lb 67.2%)
  and delivers 75.8% on held-out 2025.
- **The ATS/total edge bands are coin-flips.** Every edge band lands at 47-52%
  point / ≤48.3% LB across 1960 games and is excluded. Edge magnitude remains
  NON-monotone in accuracy. The 2023-only `edge 0.5-1.5` band (61.2%, n=139) that
  once cleared VALUE was small-sample noise — over seven seasons it is 49.1%.
- `p 0.5-0.55` only marginally clears VALUE (lb 50.3%) and was NOT emitted in the
  2025 holdout; treat it as provisional.

Prior single-season baseline (2023, n=285) is archived at
`data/weekly.single-season-2023.bak.json`; its bands were what the earlier report
labeled "2025 season" — that label was incorrect; the artifact was the 2023 replay.

## Provenance and limits

- `build_ratings.py` now builds walk-forward ratings across seven seasons
  (2019-2025) from `historical_lines.parquet` + `historical/lines_extra.parquet`
  (schedules) and `historical/historical_pbp.parquet` + `pbp_{2024,2025}.parquet`
  (EPA/drives). Drive-outcome rates are preserved from `data/league_priors.json`
  because the reduced pbp parquets no longer carry `fixed_drive_result`; every
  other prior is remeasured. The 2019-2023 ratings drift <0.06 pts vs the prior
  `backtest_games.json`, so the extension is additive, not a regime change.
- Shipped bands pool all seven seasons; the 2021-24 train / 2025 verify split
  above is the honesty check on the method, not the shipped table.
- The prop ML ranker artifact (`data/ml_ranker.joblib`) is NOT involved in this
  game-line band recal and was not required.
- Synthetic/replay hit rates are research evidence, not betting edge; the
  fail-closed exact-market gate for props is untouched. Prop leans are NOT in
  the ranked tiers until their markets carry approved graded accuracy.
- Deeper ranker evidence (pooled WF 2021-24: top-1 70.10%, top-5 66.92%;
  2025 holdout: 76.47/69.93) lives in data/accuracy_registry.json.

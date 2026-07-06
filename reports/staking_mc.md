# Phase 7.7 — advisory staking, bankroll Monte Carlo

**ADVISORY ONLY — no bet is ever placed.** Sizes come from `nflvalue.staking`; this MC shows what they imply for the bankroll at **plausible real-line edges (52-58%), NOT** the synthetic 66-68%. Start 100u, ~306 bets/season, within-game legs correlated at ρ=0.30 (7.5). Ruin = lost ≥80%. Synthetic-line caveat on every input. 1-800-GAMBLER.

`shrunk` = shipped rule (edge-shrink × ¼-Kelly × correlation × caps); `qkelly` = plain ¼-Kelly on the raw edge; `flat` = 1u non-compounding.

| true hit | strategy | median end | p5 | p95 | p95 max DD | P(ruin) | P(halve) |
|---|---|---|---|---|---|---|---|
| 52.00% | flat | 97.5 | 70.8 | 126.2 | 36.7% | 0.0 | 0.004 |
| 52.00% | qkelly | 100.0 | 100.0 | 100.0 | 0.0% | 0.0 | 0.0 |
| 52.00% | shrunk | 100.0 | 100.0 | 100.0 | 0.0% | 0.0 | 0.0 |
| | | | | | | | |
| 52.38% | flat | 99.5 | 70.8 | 130.0 | 36.1% | 0.0 | 0.002 |
| 52.38% | qkelly | 100.0 | 100.0 | 100.0 | 0.0% | 0.0 | 0.0 |
| 52.38% | shrunk | 100.0 | 100.0 | 100.0 | 0.0% | 0.0 | 0.0 |
| | | | | | | | |
| 54.00% | flat | 109.0 | 80.4 | 139.5 | 29.7% | 0.0 | 0.0 |
| 54.00% | qkelly | 106.9 | 83.8 | 138.6 | 24.1% | 0.0 | 0.0 |
| 54.00% | shrunk | 103.3 | 93.1 | 114.6 | 11.0% | 0.0 | 0.0 |
| | | | | | | | |
| 55.00% | flat | 114.7 | 88.0 | 145.3 | 25.4% | 0.0 | 0.0 |
| 55.00% | qkelly | 119.3 | 82.6 | 181.6 | 32.9% | 0.0 | 0.0 |
| 55.00% | shrunk | 107.7 | 92.9 | 128.4 | 14.5% | 0.0 | 0.0 |
| | | | | | | | |
| 56.00% | flat | 120.5 | 91.8 | 151.0 | 23.3% | 0.0 | 0.0 |
| 56.00% | qkelly | 140.3 | 81.4 | 250.8 | 39.4% | 0.0 | 0.0015 |
| 56.00% | shrunk | 111.8 | 95.1 | 131.3 | 13.3% | 0.0 | 0.0 |
| | | | | | | | |
| 58.00% | flat | 131.9 | 103.3 | 160.5 | 18.8% | 0.0 | 0.0 |
| 58.00% | qkelly | 227.4 | 97.6 | 530.0 | 49.6% | 0.0 | 0.0015 |
| 58.00% | shrunk | 119.3 | 101.7 | 139.8 | 11.3% | 0.0 | 0.0 |
| | | | | | | | |

Reading it: at plausible edges the shipped `shrunk` rule grows the bankroll far slower than raw quarter-Kelly but with a much smaller p95 drawdown and near-zero ruin — the point of shrinking an estimated edge. At 52.38% (breakeven) every strategy drifts flat-to-down; there is no sizing that manufactures an edge that isn't there.


# Monte Carlo evaluation of the brain (Phase 6.8)

**Inputs:** weekly-retrain GBDT leans (the live Tuesday cadence -- the
model's weights re-fit every week), 2024-2025, graded at synthetic
trailing-mean reference lines. **Every number below inherits that
caveat**; real sportsbook lines price most of what this model knows.
Breakeven at -110 = 52.38%. 1-800-GAMBLER.

## 2024 (n=1360 bets)

- hit rate 66.3% (week-bootstrap 90% CI 64.3%-68.3%)
- units at -110 flat: +362.0 (90% CI +304.9 to +421.2)
- P(profitable season) 100.0%, P(hit > breakeven) 100.0%

- max losing streak (median / p95): 6 / 8 bets
- max drawdown flat-1u (median / p95): 8.3 / 11.4 units

## 2025 (n=1360 bets)

- hit rate 68.2% (week-bootstrap 90% CI 65.8%-70.7%)
- units at -110 flat: +409.7 (90% CI +342.5 to +480.9)
- P(profitable season) 100.0%, P(hit > breakeven) 100.0%

- max losing streak (median / p95): 6 / 8 bets
- max drawdown flat-1u (median / p95): 8.2 / 11.3 units

## What this means at REAL lines

The synthetic-line hit rates above measure directional skill vs a
naive trailing-mean line, NOT price-beating. Against real, sharp
-110 lines the plausible band for a good model is 52-58%. At flat
1u, 300 bets/season:

| true hit | EV (units) | SD (units) | P(losing season) | bets to prove skill | ¼-Kelly f | ¼-Kelly median end (100u) | P(halve) |
|---|---|---|---|---|---|---|---|
| 52.00% | -2.2 | 16.5 | 0.553 | — | 0.0% | 100.0 | 0.0 |
| 52.38% | -0.0 | 16.5 | 0.5 | — | 0.0% | 100.0 | 0.0 |
| 54.00% | +9.3 | 16.5 | 0.287 | 5855 | 0.85% | 107.1 | 0.0 |
| 55.00% | +15.0 | 16.5 | 0.181 | 2231 | 1.38% | 119.8 | 0.001 |
| 56.00% | +20.7 | 16.4 | 0.103 | 1163 | 1.9% | 141.2 | 0.002 |
| 58.00% | +32.2 | 16.3 | 0.024 | 478 | 2.95% | 229.8 | 0.005 |

Read 'bets to prove skill': even a genuinely skilled 55% bettor
needs ~2,231 bets to statistically separate from breakeven (80%
power) -- multiple full seasons at this volume. A 54% bettor needs
~5,855. This is why the kill-check uses CLV (n>=150 resolved,
avg CLV>0, 52%+ positive-CLV rate) rather than won-bet counts:
closing-line value converges orders of magnitude faster than
profit does.

**The honest chain:** synthetic-line skill (measured, strong) ->
real-line hit rate (unknown until live prices accrue) -> profit
(variance-dominated at any realistic volume). The brain's numbers
justify running the live CLV experiment; they do not yet justify
conviction about profit.

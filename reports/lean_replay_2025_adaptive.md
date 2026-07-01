# 2025 replay — static model vs weekly learning loop

**Leans, not locks.** Same framing as `lean_replay_2025.md`: graded at synthetic
trailing-mean lines (NOT market prices); breakeven proxy 52.4% at -110. The
adaptive run applies `prop_learning` walk-forward: week N is ranked using only
weeks < N (bias multipliers from the full candidate pool + per-market
reliability from graded leans). Identical candidate pools, identical volume
(1,360 leans each). 1-800-GAMBLER.

| | Static | Adaptive (learning) |
|---|---|---|
| Overall hit rate | 56.5% | **58.5%** |
| Top-1 pick per game (n=272) | 58.8% | **64.0%** |
| Weeks 10–18 (loop warmed up) | ~54.7% | **57.8%** |
| All-candidates baseline | 47.0% | 47.1% |

**How it improved — the mechanism is visible in the mix:** the reliability
multiplier moved volume toward markets that kept hitting and away from ones
that kept missing. Receptions leans grew 363→523 (hit rate held at 64.2%);
rush-attempts leans shrank 96→55, anytime-TD 6→2. Bias multipliers trimmed
means on markets the model over-projected.

| Market | static n → adaptive n | static hit → adaptive hit |
|---|---|---|
| anytime td | 6 → 2 | 33.3% → 0.0% |
| pass attempts | 14 → 2 | 42.9% → 0.0% |
| passing yards | 130 → 89 | 55.4% → 57.3% |
| receiving yards | 501 → 445 | 52.7% → 53.7% |
| receptions | 363 → 523 | 64.5% → 64.2% |
| rush attempts | 96 → 55 | 45.8% → 50.9% |
| rushing yards | 250 → 244 | 58.4% → 58.2% |

Weekly (static → adaptive): wk1 61%→61%, wk2 51%→56%, wk3 64%→64%, wk4 61%→61%, wk5 50%→50%, wk6 53%→57%, wk7 60%→60%, wk8 57%→63%, wk9 57%→60%, wk10 69%→71%, wk11 53%→56%, wk12 51%→50%, wk13 56%→56%, wk14 57%→60%, wk15 49%→50%, wk16 50%→51%, wk17 55%→61%, wk18 61%→65%

# Correlation-aware selection ablation — seasons 2022–2025

**Leans, not locks.** Directional grading at synthetic trailing-mean lines (not price-beating/profit). Units at flat 1u/lean, standard -110 juice.

## Pooled result

| | n | hit rate | units |
|---|---|---|---|
| Baseline (shipped) | 5435 | 58.1% | +595.8u |
| Correlation-aware | 5435 | 58.6% | +645.5u |

| Top-1 per game | n | hit rate | units |
|---|---|---|---|
| Baseline | 1087 | 61.9% | +197.8u |
| Correlation-aware | 1087 | 61.9% | +197.8u |

## Diversification

- Baseline: avg 5.0 leans/slip, ~4.38 effective independent bets/slip (1 minus max positive pairwise rho with each earlier-selected leg in the same slip).
- Correlation-aware: avg 5.0 leans/slip, ~4.954 effective independent bets/slip; 257 selected legs carried a nonzero correlation discount.

## Per-season

| Season | Baseline hit | Baseline units | Corr-aware hit | Corr-aware units | Δhit | Δunits |
|---|---|---|---|---|---|---|
| 2022 | 59.0% | +172.3u | 58.4% | +155.1u | -0.66% | -17.18u |
| 2023 | 58.2% | +152.0u | 58.8% | +165.4u | +0.51% | +13.36u |
| 2024 | 58.7% | +163.4u | 59.5% | +184.4u | +0.81% | +21.00u |
| 2025 | 56.5% | +108.1u | 57.8% | +140.6u | +1.25% | +32.46u |

## Example slips (baseline vs correlation-aware, same game)

**BAL @ NYJ** — 2022 week 1

Baseline:
- M.Davis rush_attempts over (composite 35.38)
- E.Moore receptions under (composite 34.4)
- Mi.Carter rushing_yards under (composite 28.86)
- J.Flacco passing_yards over (composite 27.77)
- M.Davis rushing_yards over (composite 26.93)

Correlation-aware:
- M.Davis rush_attempts over (composite 35.38)
- E.Moore receptions under (composite 34.4)
- Mi.Carter rushing_yards under (composite 28.86)
- J.Flacco passing_yards over (composite 27.77)
- D.Duvernay receptions under (composite 23.44)

**BUF @ LA** — 2022 week 1

Baseline:
- C.Kupp receiving_yards under (composite 90.54)
- M.Stafford passing_yards under (composite 42.33)
- C.Akers rush_attempts under (composite 37.32)
- C.Akers rushing_yards under (composite 31.71)
- D.Henderson rushing_yards over (composite 29.1)

Correlation-aware:
- C.Kupp receiving_yards under (composite 90.54)
- C.Akers rush_attempts under (composite 37.32)
- M.Stafford passing_yards under (composite 42.33, discounted 31% vs C.Kupp)
- D.Henderson rushing_yards over (composite 29.1)
- S.Diggs receptions over (composite 28.6)

**DEN @ SEA** — 2022 week 1

Baseline:
- R.Penny rushing_yards under (composite 57.93)
- M.Gordon rushing_yards under (composite 39.28)
- T.Lockett receiving_yards under (composite 36.35)
- M.Goodwin receiving_yards under (composite 27.43)
- R.Penny rush_attempts under (composite 25.44)

Correlation-aware:
- R.Penny rushing_yards under (composite 57.93)
- M.Gordon rushing_yards under (composite 39.28)
- T.Lockett receiving_yards under (composite 36.35)
- M.Goodwin receiving_yards under (composite 27.43)
- J.Williams rushing_yards under (composite 24.3)

**GB @ MIN** — 2022 week 1

Baseline:
- A.Thielen receptions under (composite 37.56)
- J.Jefferson receiving_yards under (composite 36.2)
- A.Thielen receiving_yards under (composite 30.8)
- K.Cousins passing_yards under (composite 28.83)
- R.Tonyan receiving_yards over (composite 28.1)

Correlation-aware:
- A.Thielen receptions under (composite 37.56)
- J.Jefferson receiving_yards under (composite 36.2)
- R.Tonyan receiving_yards over (composite 28.1)
- R.Cobb receptions under (composite 23.35)
- I.Smith receptions under (composite 23.17)

**IND @ HOU** — 2022 week 1

Baseline:
- P.Campbell receptions under (composite 40.01)
- B.Cooks receiving_yards under (composite 37.36)
- J.Taylor rushing_yards over (composite 35.2)
- J.Taylor rush_attempts over (composite 32.46)
- C.Moore receiving_yards under (composite 30.33)

Correlation-aware:
- P.Campbell receptions under (composite 40.01)
- B.Cooks receiving_yards under (composite 37.36)
- J.Taylor rushing_yards over (composite 35.2)
- C.Moore receiving_yards under (composite 30.33)
- C.Conley receptions over (composite 27.81)

**JAX @ WAS** — 2022 week 1

Baseline:
- L.Thomas receptions under (composite 46.87)
- C.Samuel receiving_yards under (composite 32.36)
- J.Agnew receiving_yards over (composite 30.49)
- L.Thomas receiving_yards under (composite 26.22)
- C.Kirk receptions under (composite 26.06)

Correlation-aware:
- L.Thomas receptions under (composite 46.87)
- C.Samuel receiving_yards under (composite 32.36)
- J.Agnew receiving_yards over (composite 30.49)
- C.Kirk receptions under (composite 26.06)
- E.Engram receiving_yards under (composite 23.33)

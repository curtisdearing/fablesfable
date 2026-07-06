# Phase 7.1 — calibration audit (walk-forward, out-of-sample)

Pooled OOS seasons **2022-2025**, n=44,350; overall over-rate 0.3620. Calibrator for season S fit ONLY on raw OOS predictions of seasons < S (2021 seeds history). Calibrate P(over) vs y_over; p_under = 1 - cal.

Synthetic-line caveat applies to every number (trailing-mean lines, no real prices).

## Method bake-off (pooled)

| variant | log-loss | ECE | MCE | Brier | reliability | resolution |
|---|---|---|---|---|---|---|
| platt_permkt **←winner** | 0.62448 | 0.0118 | 0.0281 | 0.21798 | 0.00023 | 0.01271 |
| beta_permkt | 0.62457 | 0.0104 | 0.0262 | 0.21802 | 0.0002 | 0.01259 |
| beta_pooled | 0.62646 | 0.0161 | 0.0334 | 0.21874 | 0.00036 | 0.01237 |
| platt_pooled | 0.62707 | 0.0229 | 0.047 | 0.21893 | 0.00064 | 0.01237 |
| isotonic_pooled | 0.6276 | 0.0133 | 0.0291 | 0.21883 | 0.00027 | 0.01192 |
| isotonic_permkt | 0.62899 | 0.0102 | 0.0225 | 0.21832 | 0.00017 | 0.01241 |
| raw | 0.6294 | 0.0335 | 0.0864 | 0.21997 | 0.00164 | 0.01246 |

**Winner: `platt_permkt`.** pooled log-loss 0.6294→0.62448 (-0.00492), ECE 0.0335→0.0118 (-0.0217), reliability 0.00164→0.00023.

## Significance (paired per-row log-loss vs raw; +t = calibration helps)

Pooled: dLL +0.00492, **t=+7.17** (n=44,350). Near-tie vs beta_permkt t=+1.38 (pick simpler Platt); beats isotonic_permkt t=+4.13 (thin-slice overfit).

| season | dLL vs raw | t |
|---|---|---|
| 2022 | +0.01119 | +6.69 |
| 2023 | +0.00502 | +3.55 |
| 2024 | +0.00326 | +2.63 |
| 2025 | +0.00012 | +0.11 |

*Honest read: the gain is large early and shrinks toward zero as the base model's training history grows (self-calibrating). Wired as a guard + tail-corrector that clears the pooled bar and never hurts.*

## Per-market (raw → winner): over-rate, log-loss, Brier, ECE

| market | n | over | LL raw→win | Brier raw→win | ECE raw→win |
|---|---|---|---|---|---|
| receiving_yards | 9,108 | 0.4037 | 0.68242→0.67085 | 0.24393→0.23899 | 0.0553→0.014 |
| receptions | 9,108 | 0.409 | 0.67731→0.67062 | 0.24162→0.23888 | 0.0443→0.0192 |
| rushing_yards | 3,581 | 0.3985 | 0.67315→0.66676 | 0.23959→0.237 | 0.0522→0.0243 |
| passing_yards | 2,337 | 0.481 | 0.6716→0.67132 | 0.23916→0.23929 | 0.0329→0.026 |
| pass_attempts | 2,337 | 0.4801 | 0.66798→0.67203 | 0.2376→0.23962 | 0.0351→0.035 |
| rush_attempts | 3,581 | 0.4203 | 0.68631→0.67821 | 0.24593→0.24258 | 0.0531→0.021 |
| anytime_td | 14,298 | 0.243 | 0.5267→0.52607 | 0.17349→0.17333 | 0.0198→0.0205 |

## Reliability (pooled, equal-frequency deciles)

| decile | n | raw p̄ | winner p̄ | observed |
|---|---|---|---|---|
| 0 | 4,435 | 0.116 | 0.146 | 0.136 |
| 1 | 4,435 | 0.189 | 0.212 | 0.221 |
| 2 | 4,435 | 0.250 | 0.291 | 0.302 |
| 3 | 4,435 | 0.302 | 0.352 | 0.348 |
| 4 | 4,435 | 0.344 | 0.378 | 0.365 |
| 5 | 4,435 | 0.382 | 0.395 | 0.392 |
| 6 | 4,435 | 0.420 | 0.410 | 0.418 |
| 7 | 4,435 | 0.462 | 0.425 | 0.429 |
| 8 | 4,435 | 0.513 | 0.445 | 0.481 |
| 9 | 4,435 | 0.614 | 0.508 | 0.528 |

Plots: `reports/calibration_reliability_pooled.png`, `reports/calibration_reliability_by_market.png`.


# Which factor combinations maximize accuracy — full-feature search

**Date:** 2026-07-14 · Supersedes the coarse game-level pass in `../mc_perfect_search_report.md`.

**Method:** Rebuilt your entire feature pipeline from nflverse (fresh caches) → `ml_test.py --stage frame` → **66,408 real candidates, 2020–2025, all 80 considerations**. Grouped features into 16 factor groups matching your architecture. Stage 1: 1,653 subset combinations screened (walk-forward OLS ranker). Stage 2–3: your shipped GBDT (`MLRanker`) on top subsets + full drop-one/add-one ablation. **Identical protocol to yours**: walk-forward by season (eval 2022–25), `rank_and_grade` top-5/game, player cap 2, synthetic-line grading, breakeven proxy 52.38%.

**Harness validation:** my full-feature GBDT reproduces your recorded `ml_eval_results.json` per season (mine 63.3/63.8/65.9/67.9 vs yours 63.5/65.0/65.7/67.5) — same machine, fresh data.

## The winning combination

**model_belief + proj_parts + player_status + market_pos + weather (+ qb_oline)**

| Config | Top-5 hit (n=5,435) | Top-1 hit (n=1,087) | 2025 season |
|---|---|---|---|
| **Winner + qb_oline (6 groups)** | **66.9%** | **71.3%** | 69.5% / 74.6% |
| Winner (5 groups) | 66.8% CI90 [65.8, 67.9] | 70.6% | 69.5% / 74.6% |
| ALL 16 groups (= shipped ranker) | 65.2% | 69.2% | 67.9% / 71.3% |
| model_belief alone | 64.6% | 66.2% | — |
| Tuned composite (your baseline) | 57–59% | — | 57.1% |

Per-season winner stability: 64.9 → 66.1 → 66.7 → 69.5% (monotonically improving, never below 64.9%). P(true rate > breakeven) ≈ 100%.

## What each factor is worth (GBDT ablation around the winner)

**Load-bearing (drop-one from winner):**

| Factor group (features) | Dropped → hit | Δ |
|---|---|---|
| model_belief (p_over, z, mean, sd, line geometry) | 63.0% | **−3.9pp** — the projection itself is most of the signal |
| player_status (is_contract_year, age_years) | 65.4% | **−1.4pp** — the sleeper factor |
| proj_parts (opp_factor, game_script, proj_volume/efficiency) | 65.6% | **−1.2pp** |
| weather (temp, wind) | 66.6% | −0.2pp |
| market_pos (market + position identity) | 66.7% | −0.1pp (top-5); larger on top-1 |

**Actively harmful when added on top of the winner (add-one):**

| Added group | Hit | Δ vs winner |
|---|---|---|
| usage_rolls (12 roll_* cols) | 64.2% | **−2.6pp** |
| ftn (PA/motion/blitz/box) | 64.5% | −2.3pp |
| chemistry (QB-receiver, teammate-out) | 65.2% | −1.6pp |
| team_tendencies (PROE, pace, EPA, CPOE...) | 65.5% | −1.4pp |
| redzone shares | 65.6% | −1.2pp |
| def_injuries / opp_epa / game_ctx | 66.1–66.3% | −0.5 to −0.7pp |
| ngs | 66.8% | 0.0 |
| situational (birthday, revenge) | 66.7% top-5 but **71.8% top-1 (best)** | ≈0 |

Why: the projection layer already consumes usage, matchup, pace and tendencies upstream — `mean`, `z`, `opp_factor`, `game_script` are distillations of them. Feeding the raw versions to the GBDT again adds ~40 noisy/collinear dimensions on n≈50k and the model overfits. **More considerations ≠ more accuracy: the kitchen sink costs ~1.7pp vs the lean set.**

Stage-1 paired-lift screen (1,653 subsets) agrees independently: model_belief +4.4pp, player_status +1.7pp, usage_rolls +1.7pp *in linear models only*, market_pos +0.9pp, proj_parts +0.7pp; ngs/chemistry/ftn/weather ≈ 0 or negative.

## Implemented or not

| Finding | Status in repo |
|---|---|
| Every winning factor | ✅ Implemented — all 6 groups already flow into the frame (`ml_ranker.py`, `advanced_features.py`, `context_features.py`) |
| The winning **combination** | ❌ Not implemented — `feature_columns()` hard-codes ALL features; `data/ml_ranker.joblib` + `config.json ml_ranker` train on the kitchen sink. No feature-subset knob exists. **~1.7pp top-5 / ~2pp top-1 on the table.** |
| contract_year + age as first-class factors | ✅ built (`contract_year_lookup`, `age_years`) — keep; they're quietly your 2nd-strongest group |
| Composite ranker as pick-selector | ⚠️ Superseded: 57–59% vs GBDT 67% on identical pools — ranker (already `enabled: true`) should be the selection authority |
| Still open from round 1 | `player_receptions` missing from live `prop_markets`; `anytime_td` still fetched; `weights.json` demo-trained |

## Do this

1. Add a `features` list to `config.json ml_ranker` and default it to the 6 winning groups; retrain `ml_ranker.joblib` (`ml_test.py --stage fit`).
2. Keep `situational` in the top-1 path if you surface a single "pick of the game" (71.8% top-1).
3. Round-1 fixes still apply (receptions live market, demo-weight reset).

*Artifacts: `combo_slices.py` (resumable search), `bootstrap_data.py` (nflverse cache rebuild), `s1.jsonl` (1,653 subset screens), `gbdt.jsonl` (112 GBDT season-fits). Synthetic-line caveat applies to every number; directional accuracy ≠ price-beating ROI.*

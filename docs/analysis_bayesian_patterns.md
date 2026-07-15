# Bayesian factor weighting + pattern mining — what to weight together, what stands out

**Date:** 2026-07-14 · Data: 66,408 candidates (2020–25) for weighting; 32,131 real player-weeks (2020–25) joined to schedules/injuries/rosters/birthdays for patterns. All outcomes graded vs the player's own trailing mean (your synthetic-line convention, base over-rate ≈ 43–50% depending on stat). Empirical-Bayes shrinkage everywhere (skeptical prior, k=60) — raw rates are shrunk toward baseline before anything is called real. 37 patterns tested; at P>0.95 expect ~1–2 false positives among the nulls, so the headline effects are the ones with P≈1.0 and n>200.

## 1. Bayesian factor weights (Laplace posterior, prior β~N(0, 0.2²))

**Credible pooled factors (90% CI excludes 0)** — top of the list: `roll_pass_attempts` +0.47, `roll_carry_share` +0.34, `proj_volume` −0.34, `p_over` +0.27, `roll_target_share` +0.27, `sd` +0.26, `z` +0.19. Then meaningfully: `opp_factor` −0.08, `team_margin` +0.08, **`is_contract_year` −0.076 (contract-year players go UNDER — fade the hype)**, `total_line` +0.05, `wind` −0.04, `key_teammate_absent` +0.04, `age_years` −0.03, `home` +0.02.

**Not credible** (posterior straddles 0): `revenge_game` (team-level), `is_birthday_week`, `qb_continuity` (pooled), `oline_outs`, `temp`, all NGS, FTN pa/motion, `rz_tgt_share`, `opp_epa_factor`. Matches the GBDT ablation from the combo search.

**Weight these together (correlation blocks → collapse to indices):**

| Block | Members | Suggested index |
|---|---|---|
| Role/volume | roll_targets, target_share, carries, carry_share, adot, air_yards, ypt, catch_rate, rz shares | one "role volume" score |
| Projection internals | mean, sd, line, proj_volume, proj_efficiency, roll_pass_attempts | already distilled — don't re-feed raws |
| Belief | p_over, z | keep as-is (core) |
| Defense outs | def_out_total, def_out_db | one "defense depleted" score |
| Aggression | team_neutral_proe, team_edp | one score |
| Efficiency | team_epa_play, team_cpoe | one score |

**Weight these DIFFERENTLY by market (hierarchical partial pooling, τ=0.1):**

| Market | Deviating factor | Direction vs pooled |
|---|---|---|
| pass_attempts | qb_continuity −0.18 (z=−4.0) | continuity → attempts UNDER; new QB → attempts UP |
| rush_attempts | roll_ypc +0.31 (z=+4.3) | efficient runners get fed; also wind flips POSITIVE here |
| receptions | roll_catch_rate −0.18 (z=−4.0), roll_ypt +0.10 | high catch-rate → reception UNDERS (line inflation) |
| passing_yards | opp_factor sign flips +0.08; cpoe/edp negative | matchup works opposite for QB yards |
| anytime_td | roll_air_yards +0.18, roll_carries +0.13 | TD market has its own volume logic |

Your `weights.json` is one global vector — the data says props need per-market vectors with shrinkage to global. (And it's still demo-trained; see round 1.)

## 2. Your three hypotheses — verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| "WR birthday → +40% TDs" | **Rejected.** If anything negative | 565 birthday-week skill player games: TD rate 21.1% vs 23.3% base, lift −2.0pp, P(>0)=0.12. Bayesian coef also ~0. Kill the birthday feature. |
| "RB1 out → QB attempts over 80% of the time" | **Rejected for QB… but the cascade is REAL and lands on the RB room** | QB attempts over when RB1 out: 50.0% vs 49.8% (nothing). **Backup RB carries over: 63.9% vs 44.7% base, +15.4pp, P≈1.000, n=241** — the strongest pattern in the entire battery. Teams replace the runs, they don't abandon them. |
| "Proven QB-WR duo performs better when WR2 out" | **Not supported at prop level** | Duo + WR2 out → WR1 rec yds over: 46.9% vs 43.0% (n=49, P=0.64). WR2-out alone: +1.3pp WR1 yds, +0.3pp receptions. TE targets actually DROP −3.6pp. Redistribution is flat — lines already price it. |

## 3. What stands out (the discoveries, ranked by credibility × size)

| Pattern | n | Rate vs base | Lift | P(real) |
|---|---|---|---|---|
| **RB1 out → backup RB carries OVER** | 241 | 63.9% vs 44.7% | **+15.4pp** | ~1.000 |
| **Wind ≥15mph → QB pass yards UNDER** | 207 | 33.3% over vs 49.9% | **−12.8pp** | ~1.000 |
| **Backup QB starting → QB rush yards OVER** | 350 | 52.0% vs 42.3% | **+8.3pp** | ~1.000 |
| Wind ≥15 → pass attempts UNDER | 207 | 40.1% vs 50.6% | −8.2pp | 0.996 |
| Dome game → WR rec yards OVER | 3,013 | 45.7% vs 41.8% | +3.9pp | ~1.000 |
| Cold ≤32°F → WR rec yards UNDER | 431 | 38.1% vs 43.3% | −4.6pp | 0.980 |
| Post-bye → WR/TE rec yards UNDER (rust > rest) | 1,022 | 40.4% vs 43.9% | −3.3pp | 0.986 |
| Big underdog → QB attempts OVER / WR targets OVER | 418/1,619 | +3.7/+2.1pp | | 0.95/0.96 |
| Turf → WR rec yards OVER | 4,141 | 44.3% vs 42.1% | +2.2pp | 0.998 |
| Short week → WR/TE rec yards OVER | 962 | 46.4% vs 43.4% | +2.8pp | 0.961 |
| Primetime → RB rush yards UNDER | 1,345 | 41.0% vs 43.8% | −2.7pp | 0.978 |
| RB revenge game → rush yards OVER (suggestive) | 86 | 52.3% vs 43.0% | +5.5pp | 0.909 |
| Referee crews (totals): Hochuli 44.1% over (shrunk) vs Blake/Rogers 51%+ | 48–98/crew | ±3–4pp spread | | exploratory |

Nothing-burgers worth knowing: division-rematch familiarity (−1.8pp, P=0.23), Denver altitude for visiting WRs (+1.4pp, P=0.65), backup QB → TE bump (folk wisdom, −0.8pp), primetime TD boost (none).

## 4. Implementation plan (maps to your architecture)

Your repo already has the designed slot for exactly this: **`context_features.py` tags + `config.json context_learning.enabled_tags` + `prop_learning.py` reliability learning**. It's currently empty (`enabled_tags: []`). Ship patterns as context tags with EB-posterior priors, let the reliability loop keep them honest.

**Phase 1 — wire the three big ones (data already cached, ~1 day):**

1. `rb1_out_rb_room` tag: RB1 (top trailing carries, roster-of-record) listed Out → boost `rush_attempts`/`rushing_yards` OVER for remaining RBs. Prior logit shift ≈ ln(2.2·odds) → start conservative at +0.35, reliability-gated. Note: your `absence_matrix.json` already MEASURES this (`RB1_out→RB1_active`) but `pipeline_weekly.py` only applies absence to QB passing markets (`apply_absence_qb_adjustment`) — point it at the RB room, where the effect actually is.
2. `wind15_pass_under` tag: wind ≥15 (already fetched by `sources/weather.py`) → QB passing yards/attempts UNDER prior −0.30 logit. Your `p_weather_pass` weight is −0.013 (demo-trained noise); this is the real prior for it.
3. `backup_qb_rush` tag: schedule `qb_id` ≠ trailing modal starter → QB rush yards OVER prior +0.25. New signal, nothing in repo touches it. Data: schedules (cached) — 5 lines in `context_features.py`.

**Phase 2 — venue/schedule features into the ML frame (~1 day):** add `roof_dome`, `surface_turf`, `rest_days`, `primetime`, `post_bye`, `big_dog` to `NUMERIC_FEATURES` (join from schedules in `build_features`; the frame build already touches schedules). Retrain ranker. These are broad, mild, and the GBDT will find the interactions (e.g., dome × WR × adot).

**Phase 3 — per-market hierarchical weights (~2 days):** replace global `weights.json` p_* entries with per-market vectors shrunk toward global (τ≈0.1) in `learn.py` — the per-market deviation table above is the empirical justification and the initialization.

**Phase 4 — kill/monitor:** delete `is_birthday_week`; leave revenge as RB-only context tag (P=0.91, monitor); add referee to game-notes display (not the model) until a season of CLV data exists. Route every new tag through your `killcheck.py`/CLV loop — a tag that doesn't beat close gets its reliability zeroed by `prop_learning` automatically.

**New data worth ingesting later:** nflverse `participation` (offense_personnel/formation, 2016–2023) for the trips/personnel hypotheses — testable historically but discontinued for live 2024+; snap counts (`load_snap_counts`) as a cleaner "out" detector than injury reports; `load_officials` for ref crews (schedules `referee` covers it).

*Artifacts: `bayes_weights.py`, `pattern_battery.py`, `bayes_weights.json`, `patterns.json`, `refs.json` alongside this file. Caveats: synthetic-line grading (directional, not price-beating); books shade obvious spots (wind, RB1 out) — the CLV tracker, not the hit rate, decides if these survive at real prices.*

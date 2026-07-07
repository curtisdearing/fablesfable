# Decisions — Phase 8 (post-7 hardening: completions, surgical spend, observation quality)

Session goal: add QB completions, cut the Odds-API volume problem without paying,
and put the recency-weight / injury / game-worth objections on a measured
footing. Same discipline as p6/p7 — nothing enters the deterministic mean or
calibration until a fit clears it.

## 8.1 `pass_completions` market (shipped)

Eighth prop market, end to end. Mean = projected `pass_attempts` × a shrunk
trailing completion rate (`roll_comp_rate` = completions/attempts, shift-1-then-
roll, shrunk toward the QB archetype prior with `SHRINK_K` — byte-identical
machinery to `roll_ypa`/`roll_catch_rate`). Fallback SD fraction **0.22**
(vs the generic 0.45): completions is a bounded count off a stable attempt base,
so realized totals are far less dispersed. A measured walk-forward residual SD
still supersedes it. Odds mapping `player_pass_completions → pass_completions`;
config + selector rows added.

**Caveat (retrain-gated):** `MARKETS7` in `ml_ranker.py` and `FAMILY` in
`correlation.py` are frozen artifacts tied to the shipped RF's 67-feature space.
Completions is deliberately NOT added to either, so it projects and screens on
the deterministic + selector path but gets **no ML-calibrated probability and no
same-game correlation discount** until the next RF retrain. `opp_factor` is off
for completions (the only QB opp factor is yards-per-play; applying it to a count
would double-count the pass-D signal already in projected attempts).

## 8.2 Surgical spend — the free answer to the volume limit (shipped, opt-in)

The free tier caps at 450 usable credits/mo and props bill per market × region,
so full weekly coverage was impossible and the puller rotated. **Surgical spend**
(`oddsapi_props.surgical_markets` + `_pull_surgical`, config `surgical_spend`,
OFF by default) requests, per game, only the markets holding a candidate whose
pre-pull conviction vs its synthetic line (`|p_over − 0.5|`) clears
`min_conviction` (0.06). Trimming e.g. 8 markets to the 2–3 promising ones
multiplies game coverage on the same budget, and never spends a credit on a
market with no plausible edge.

Budget safety is preserved by construction: every entry reserves a **full-cost
close** (`entry_spent + Σ full_cost ≤ remaining_at_start`) so CLV still resolves,
and the hard ceiling check is unchanged — proven in
`test_surgical_pull_reserves_closes_and_never_exceeds_budget`. The rotating
full-market path is byte-identical when disabled (all prior budget tests green).
Follow-up (not done): make `resnap_lines` close only the markets actually entered
per game, to reclaim the conservative over-reservation.

Verification vs coverage, on the record: proving the strategy is **not** odds-
constrained (projections grade against free nflverse actuals; the CLV kill-check
needs ~150 resolved leans/season, which rotation already reaches). Surgical spend
is about *acting on more games*, not *proving edge*. A paid Odds-API tier stays
deferred behind the live-CLV GO gate (p7 §7.9); no scraping (H11).

## 8.3 Observation-quality layer + recency-weight sweep (measurement only — verdict-gated)

Two objections to the projection's memory, made measurable rather than assumed:

1. **The recency weight was never fit.** `EWM_SPAN=4` shipped in 1B to fix a
   flat-8 average lagging real usage shifts; the p3-5 log lists it as *queued*.
   It is one global constant over every player/market/situation.
2. **Contaminated games silently nerf the mean.** Injury-shortened games enter
   `roll_*` at full weight (only `roll_early_exit_rate` exists, as a durability
   *feature*, never a cleaner). Rest/meaningless games (seed locked/eliminated)
   are unmodeled — only `short_week` rest exists; team records feed narrative
   only.

**`nflvalue/game_context.py`** (new, tested on synthetic frames —
`tests/test_game_context.py`): per-observation tags — `injury_shortened`
(early-exit signature OR a leak-safe snap-share collapse vs the player's prior
trailing median) and `game_meaningless` (a COARSE, explicitly-labeled proxy from
records-to-date + a conference 7-seed-cut approximation; no tiebreakers). Nothing
consumes these in production yet.

**`scripts/fit_recency_weight.py`** (new, standalone verdict — run locally; the
sandbox has no parquet reader): one walk-forward sweep answering, per market,
weight shape (flat-N vs EWM span vs season-to-date) × cleaning (raw /
drop-injury / dampen-injury / drop-rest / drop-both) × conditioning (market /
role / player usage tier / team volatility), scored on OOS next-game MAE vs the
`EWM4-raw` baseline. Emits `data/recency_weight_fit.json`.

**Blowout garbage time is NOT re-tested** — measured and REJECTED in 6.3 (no MAE
gain; receiving 24.77 vs 24.58, receptions 1.670 vs 1.658). Injury-shortened and
rest cleaning are the genuinely untested ideas. **Promotion rule:** only wire a
winner into `features.py` (behind a flag + a leakage test, like `garbage_filter`)
if its pooled MAE gain is consistent across seasons and beyond tuning noise.

## 8.4 Other areas that need the same observation-quality lens

The contamination isn't confined to `projection.py`; one shared tagging layer
should feed all of these (do NOT patch each in isolation):

- **`opp_pos_def` (opponent-vs-role defense factors)** — a defense that faced
  three rested/backup offenses looks better than it is; same clean-then-average
  question as the player means.
- **Team pace / PROE (`_team_week`)** — a team resting starters tanks its own
  volume basis; game-worth tagging belongs here too.
- **The ML training frame** — the RF learns on the same dirty rows; the tags
  should also exist as *features* (a game the model knows was injury-shortened /
  a likely rest week from clinch status), not only as a cleaning mask.
- **`player_learning.py` residual ledger** — an availability miss (player left
  early / rested) is currently attributable as *model error*; the ledger should
  tag context so the player-bias learner doesn't chase noise.
- **Calibration + correlation** — both fit on the same panel; contaminated
  player-weeks bias `p_over` calibration and same-game ρ. Cleaning (if 8.3
  clears) should extend to the calibration/correlation frames, not just the mean.

Nothing in 8.4 is built — it's the scope map for after the 8.3 verdict.

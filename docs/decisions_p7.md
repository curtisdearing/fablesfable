# Phase 7 — decision log

Scope: `PHASE7_BUILD_PROMPTS.md`. Branch cut from `phase6`. Phase 7 turns the
walk-forward synthetic-line engine into one that can prove or disprove its own
edge, sizes bets honestly, and states plainly what it cannot claim. Every number
here is at synthetic trailing-mean lines unless explicitly labeled otherwise.

---

## 7.1 — Probability calibration + quality-metric spec

The GBDT ranker was tuned for ordering (AUC ~.64) and log-loss, but nobody had
checked whether `P(over)=0.62` actually lands 62%. Edge (`composite.py`),
correlation weighting (7.5), and Kelly sizing (7.7) all assume it does. 7.1
audits that assumption, wires a walk-forward calibration layer if the numbers
justify it, and — most importantly for the jobs that follow — **freezes the
probability-quality metric suite that 7.2 onward optimizes against**, so the
target stops moving.

### The probability-quality metric suite (THE fixed target for 7.2+)

Calibrate and score **P(over)** against `y_over` (actual landed over the line;
`anytime_td` = scored). `p_under = 1 - P(over)` downstream, so the composite
edge interface is unchanged: `edge = calibrated P(side) − de-vigged market prob`.

- **Primary objective — pooled walk-forward calibrated log-loss.** Lower is
  better. A strictly proper scoring rule, the GBDT's own training loss, and
  sensitive to both calibration and resolution at once. 7.2 ships a change only
  if it lowers this out-of-sample.
- **Guardrails (a change may not regress these):**
  - **Pooled ECE** — Expected Calibration Error, 10 **equal-frequency** bins,
    `Σ (n_k/N)·|p̄_k − ō_k|`. Must not rise beyond paired noise.
  - **Per-market Brier** — one Brier per market; no market may worsen materially
    (paired per-row test, the t≥2 culture). Per-market log-loss + ECE reported
    beside it.
- **Diagnostics (reported, not optimized):** reliability curves (equal-frequency
  deciles), pooled + per-market + by predicted-prob decile; Brier decomposition
  (`Brier = reliability − resolution + uncertainty`, Murphy).

### Walk-forward evaluation protocol (the leakage discipline)

Two levels, and the whole point is that **no fold ever calibrates itself**:

1. **Base.** The GBDT for season S trains on seasons `< S` and predicts S — the
   existing `assert_walk_forward` guard (`train_max < predict rows`) still holds.
2. **Calibrator.** The map that corrects season S is fit **only on the
   out-of-sample base predictions of seasons `< S`**. It never sees a row it
   later corrects. So the calibrated series runs **2022–2025** (2021 has no prior
   OOS season; it seeds the calibrator's history). Raw metrics are reported on
   the identical pooled rows so before/after is same-data.

For the **shipped production artifact** (base trained on all history H), the
calibrator's training pairs are generated the same way — expanding-season folds
*inside* `MLRanker.fit`, each fold training strictly before the season it
predicts (`MLRanker.cal_fold_spans` is the witness; a leakage test asserts every
fold's `train_max < predict_season`). Applied to future weeks, the whole object
still satisfies `train_max < predict rows`.

Reproduce everything with `python3 scripts/audit_calibration.py` (stages `oos` →
`analyze`); outputs `reports/calibration_audit.md`, two reliability PNGs, and
`data/calibration_audit.json`. OOS only, never in-sample argmax (same honesty
contract as `tune_weights.py`).

### Audit — is miscalibration real, and where?

Pooled OOS 2022–2025, n=44,350, overall over-rate 0.362. The raw GBDT is
**well-calibrated in aggregate but overconfident at the tails**: mean `P(over)`
0.362 ≈ base rate, yet the top predicted-probability decile predicts p̄=0.614
and observes only 0.528 (MCE 0.0864) — and the bottom decile predicts 0.116,
observes 0.136. It stretches probabilities away from the base rate too far,
**exactly in the high-confidence region that edge and Kelly read.** So even a
modest pooled gain concentrates where it costs money.

Method bake-off (calibrator fit strictly on prior-season OOS; pooled log-loss is
the primary objective, ECE the guardrail):

| variant | log-loss | ECE | MCE | Brier | reliability |
|---|---|---|---|---|---|
| **platt_permkt (shipped)** | **0.62448** | 0.0118 | 0.0281 | 0.21798 | 0.00023 |
| beta_permkt | 0.62457 | 0.0104 | 0.0262 | 0.21802 | 0.00020 |
| beta_pooled | 0.62646 | 0.0161 | 0.0334 | 0.21874 | 0.00036 |
| platt_pooled | 0.62707 | 0.0229 | 0.0470 | 0.21893 | 0.00064 |
| isotonic_pooled | 0.62760 | 0.0133 | 0.0291 | 0.21883 | 0.00027 |
| isotonic_permkt | 0.62899 | 0.0102 | 0.0225 | 0.21832 | 0.00017 |
| raw | 0.62940 | 0.0335 | 0.0864 | 0.21997 | 0.00164 |

**Method choice — per-market Platt**, chosen from the numbers, not by default:

- It wins the **primary metric** (pooled log-loss 0.62448) and cuts ECE 65%
  (0.0335→0.0118) and MCE 67% (0.0864→0.0281). Paired per-row log-loss vs raw:
  **t=+7.2** — clears the t≥2 bar decisively.
- **beta** is a statistical tie (Δlog-loss 9e-5, t=1.38) — so the simpler,
  lower-variance 2-parameter map is chosen over beta's 3.
- **isotonic** is *significantly worse* than Platt (t=+4.1): per-market isotonic
  has the best ECE but the worst log-loss of the per-market methods — the classic
  thin-slice overfit (step artifacts on the ~2.3k passing / one-sided TD slices).
  This is precisely the isotonic-needs-sample-size prior, confirmed on this data.
- **Pooled** variants lose to their per-market counterparts because the
  distortion is market-specific (different base rates, different stretch);
  pooling averages the correction away.

Per-market (raw → platt_permkt): the continuous-yardage markets carry the
miscalibration and the fix lands there; the already-calibrated `anytime_td` is
left alone (Platt ≈ identity), and the thin passing slices barely move.

| market | n | over | log-loss raw→cal | Brier raw→cal | ECE raw→cal |
|---|---|---|---|---|---|
| receiving_yards | 9,108 | 0.404 | 0.68242→0.67085 | 0.24393→0.23899 | 0.0553→0.0140 |
| receptions | 9,108 | 0.409 | 0.67731→0.67062 | 0.24162→0.23888 | 0.0443→0.0192 |
| rushing_yards | 3,581 | 0.399 | 0.67315→0.66676 | 0.23959→0.23700 | 0.0522→0.0243 |
| rush_attempts | 3,581 | 0.420 | 0.68631→0.67821 | 0.24593→0.24258 | 0.0531→0.0210 |
| passing_yards | 2,337 | 0.481 | 0.67160→0.67132 | 0.23916→0.23929 | 0.0329→0.0260 |
| pass_attempts | 2,337 | 0.480 | 0.66798→0.67203 | 0.23760→0.23962 | 0.0351→0.0350 |
| anytime_td | 14,298 | 0.243 | 0.52670→0.52607 | 0.17349→0.17333 | 0.0198→0.0205 |

### Honest verdict — did calibration move anything?

**Yes, materially, but with a shrinking trend that must be reported.** The gain
is large early and decays toward zero as the base model's training history grows
and it self-calibrates:

| season | Δlog-loss (raw→cal) | t |
|---|---|---|
| 2022 | +0.01119 | +6.69 |
| 2023 | +0.00502 | +3.55 |
| 2024 | +0.00326 | +2.63 |
| 2025 | +0.00012 | +0.11 |

By 2025 the raw GBDT (trained on six seasons) is essentially self-calibrated —
calibration adds nothing (t=0.1) but, critically, **never hurts** (Δ ≥ 0 every
season; Platt on an already-calibrated slice is near-identity). It is shipped for
three reasons: (a) it clears the pooled bar (t=7.2) and is free when unneeded;
(b) the tail-overconfidence it removes is in the exact region edge/Kelly consume;
(c) 7.2 will ensemble, re-tune, and prune — any of which can re-introduce
miscalibration — so the layer is a standing guard, not a one-time patch. An
honest reader should note the marginal value on the most recent season is ~0.

### What shipped

- `scripts/audit_calibration.py` — reproducible audit (reliability + ECE +
  Brier-decomposition, pooled/per-market/per-decile, method bake-off, paired
  significance), cached OOS preds, plots, JSON.
- `nflvalue/ml_ranker.py` — `Calibrator` (per-market Platt) + `MLRanker(...,
  calibrate="platt_permkt")`. `predict_p_over` returns calibrated probabilities;
  `raw=True` exposes the base for auditing; the calibrator is saved/loaded with
  the artifact. **Production call site (`pipeline_weekly._maybe_stamp_ml`) and
  `composite.py` are unchanged** — the seam is a wrapped `predict_p_over`.
- `ml_test.py --stage fit --calibrate platt_permkt` (default) builds the
  calibrated production artifact; `--calibrate none` disables.
- `tests/test_leakage.py` — two new guards: no fold trains on the season it
  calibrates (`cal_fold_spans`), and calibrated predictions for season S are
  byte-identical when later seasons are removed. `tests/test_ml.py` — calibrator
  contract (bounded, monotone, base-preserving, save/load).

Default: calibration **on** in the fit artifact. ml_ranker itself remains
flag-gated in config, unchanged by this job.

---

## 7.2 — Ensemble + walk-forward tuning + feature pruning

Squeezed the probability model against 7.1's fixed metric (pooled walk-forward
calibrated log-loss, primary; ECE + per-market Brier, guardrails; pooled OOS
2022-2025, n=44,350 -- identical rows to 7.1's audit throughout). Search and
ablation decisions used cheap raw (uncalibrated) walk-forward log-loss as the
proxy (`tune_weights.py`'s convention); only the final chosen recipe was
checked against the real calibrated metric suite before shipping.
Reproduce: `scripts/tune_ml.py` (`ens_rf_oos`/`ens_analyze`, `hp_search`/
`hp_analyze`, `rf_search`/`rf_analyze`, `prune`/`prune_analyze`, `final_oos`/
`final_check`) and `scripts/ship_rf.py` for the checkpointed production fit.

### Ensemble bake-off -- the headline finding

Tested a simple average and a walk-forward logistic meta-learner (fit only on
prior-season pooled OOS pairs of GBDT+RF, `[logit(p_gbdt), logit(p_rf)]`) over
GBDT+RF's out-of-sample probabilities, each then calibrated (7.1's per-market
Platt) the identical way. The "beats the best single model" bar was measured
against best-of-{gbdt, rf}, not assumed to be the currently-shipped GBDT:

| variant | calibrated log-loss | ECE | t vs best single |
|---|---|---|---|
| gbdt (7.1 shipped) | 0.62448 | 0.0118 | -6.75 |
| **rf** | **0.62118** | **0.0108** | -- (best single) |
| avg (gbdt+rf) | 0.62171 | 0.0111 | -1.71 |
| meta (logistic, 2023-2025\*) | 0.62261 | 0.0132 | -1.63 |

\*meta's own walk-forward fit needs a prior-season seed on top of the base
OOS seed, so its valid window is one season narrower (2023-2025, n=33,152).

**RF alone, uncombined, is the best model found** -- it beats the shipped
GBDT decisively (t=+6.75) by more than either combination method closes.
Neither the average nor the meta-learner beats RF alone (both show negative
t vs RF, i.e. worse); the ensemble does **not** ship. This wasn't obvious
going in but isn't shocking in hindsight: RF's hit-rate had already led GBDT's
in every season of `reports/ml_improvement_test.md` (e.g. 2025: 68.2% vs
67.1%) -- the calibrated-log-loss audit confirms the same ordering on the
metric that actually governs edge and Kelly.

### Walk-forward hyperparameter search

GBDT (the parameters 7.2 was scoped around -- learning_rate, max_leaf_nodes,
min_samples_leaf, l2, max_iter): 19-config curated grid (not full 5-dim
cartesian -- one-at-a-time variations off the default plus a few joint
combos), walk-forward selected exactly per `tune_weights.py`'s convention
(each eval season's config chosen from prior seasons' pooled raw log-loss
only). Tuning **helped GBDT materially** (pooled raw log-loss 0.6394 ->
0.63073 across 2020-2025; calibrated pooled log-loss on 2022-2025 improved
0.62448 -> 0.62212) but **tuned GBDT still loses to untuned RF**
(t=+2.44, clears the bar) -- HP search closed most of the gap, not all of it.
Shipped-for-2026 GBDT config (pooled argmin, in-sample like `ship_for_2026`,
shown for the record though not shipped): `lr=0.03, leaves=63, min_leaf=40,
l2=1.0, iter=200`.

RF wasn't named in 7.2's hyperparameter list, but since it won the ensemble
bake-off, shipping it untuned without checking felt like leaving evidence on
the table. A small grid (6 configs: n_estimators, min_samples_leaf,
max_features, kept deliberately small -- RF's per-fit cost is ~5x GBDT's)
found **no improvement**: default (400 trees, min_leaf 25, `sqrt` features)
is already the pooled-best config (0.62067, tied with `max_features="log2"`
at floating-point identical odds since sqrt(67)~8.2 vs log2(67)~6.1 are close
splits). An honest "no effect" -- RF ships with its library defaults.

### Feature pruning -- ablation found real dead weight, but it didn't survive shipping

Walk-forward leave-one-out ablation across all 67 `NUMERIC_FEATURES`, 5 folds
(2021-2025), using **GBDT at the 7.2-tuned hyperparameters as the ablation
vehicle** (not RF -- RF's cost made a 67-feature x 5-fold refit grid
impractical on the sandbox; both are tree ensembles on the identical tabular
inputs, stated as a disclosed methodological choice, not a silent one).
Paired per-row log-loss t-test, t>=2 keeps a feature.

Leave-one-out is blind to redundancy: several near-collinear features (the
"core belief" cluster `z/mean/sd/line/mean_minus_line/sd_over_line/opp_factor`
-- all deterministic transforms of the same triple `p_over` also encodes)
each individually looked droppable, because the model routed around any ONE
of them through its still-present correlated neighbors. A combined-group
drop test (removing the whole cluster at once) caught this: t=+3.22, the
group matters, so every member is kept despite weak individual signal. Same
rescue logic applied to the four Phase-6 additions named in the checkpoint:

| group (Phase 6 addition) | features | individually kept | combined-group t | verdict |
|---|---|---|---|---|
| depth/location (6.1) | roll_short/mid_tgt_share, roll_short_pass_share | 2/3 | +2.61 | **group rescued -- all 3 kept** |
| rz shares (6.2) | rz_tgt_share, rz_carry_share, opp_rz_td_factor | 2/3 | +1.97 (borderline, doesn't clear) | rz_tgt_share dropped, other 2 kept individually |
| durability (6.5) | roll_early_exit_rate, inj_out_count_2y | 1/2 | -0.65 (no rescue) | inj_out_count_2y kept (t=3.03 alone), roll_early_exit_rate dropped |
| opp_absence (6.5) | opp_absence_factor | 0/1 | t=1.86, just under the bar | **dropped** as an ML feature |

Net: 46 kept, 21 dropped (full per-feature delta table in
`data/ml_tune_prune_result.json`). `opp_absence_factor`'s ML-feature drop is
specific to the classifier's feature list -- it's untouched as a deterministic
composite multiplier, which was measured and shipped separately in Phase 6.5
and isn't in scope here.

**Applied to the actual shipped model (RF), pruning broke a per-market
guardrail and was NOT shipped.** Refitting RF with the pruned 46-feature set
and re-running the real calibrated check (not the cheap proxy) found pooled
metrics still looked good (log-loss 0.62285, better than baseline) but
`passing_yards` regressed significantly against the GBDT baseline (t=-4.01,
n=2,337 -- the thinnest market). Checking whether the full-feature RF also
carries this regression ruled out a plain model-choice effect (full RF vs
GBDT on `passing_yards`: t=-0.54, noise, no violation) -- the regression is
specific to pruning-applied-to-RF. Tried rescuing the two most plausible
passing-specific dropped features (`team_cpoe`, `oline_outs`) and it didn't
fix it (t=-3.57, barely moved), meaning the true cause is some other feature
or combination the pooled, GBDT-proxy ablation didn't surface for this
~5%-of-rows market. Rather than spend more compute chasing it, the honest
call is the same one 7.1 made for calibration's near-zero 2025 gain: **an
honest negative result is a valid result.** Feature pruning is reported in
full above and **not applied** to the shipped artifact; all 67 features ship.
Future work: repeat the ablation per-market, or directly on RF with more
compute, before revisiting this.

### Final validation -- what shipped

**RF, default hyperparameters, all 67 features, platt_permkt calibration
(7.1, unchanged).** Checked on the real calibrated metric suite, pooled OOS
2022-2025 (n=44,350, identical rows to 7.1's baseline):

| metric | baseline (GBDT, 7.1 shipped) | shipped (RF) | delta | significance |
|---|---|---|---|---|
| calibrated log-loss (pooled) | 0.62448 | **0.62118** | -0.00330 | t=+6.75 |
| ECE (pooled) | 0.0118 | **0.0108** | -0.0010 | guardrail holds |
| Brier (pooled) | 0.21798 | 0.21724 | -0.00074 | guardrail holds |
| hit rate (frozen top-5/game policy) | 65.85% | **67.84%** | +1.99pt | -- |
| units at -110 (n=5,435) | +1,397.6u | **+1,603.8u** | +206.2u | -- |
| top-1 hit rate | 68.72% | **71.57%** | +2.85pt | -- |

Per-market significance (paired t, calibrated, RF vs GBDT baseline) -- **no
market regresses**: anytime_td t=+4.68, receptions t=+4.77, receiving_yards
t=+2.70 (all improve materially); pass_attempts t=+1.74, rush_attempts
t=+0.78, rushing_yards t=+0.48, passing_yards t=-0.54 (all flat, within
noise, no guardrail violation). Full per-market table in
`data/ml_tune_final.json`.

### What shipped

- `nflvalue/ml_ranker.py` -- `feature_columns(numeric=None)` accepts a pruned
  subset (persisted per-model via `MLRanker(..., features=[...])`, unused by
  the shipped artifact since pruning didn't ship); `MLRanker(model="ensemble",
  members=[...], combiner="avg"|"meta")` (unused by the shipped artifact since
  the ensemble didn't win, but built + tested since 7.2 required checking it);
  calibrator fold generation refactored behind `_oos_fold_predict` so it works
  identically for a single model or a nested ensemble.
- `scripts/tune_ml.py` -- the full search/ablation harness (ensemble bake-off,
  HP grid, RF check, feature ablation with group-rescue, final calibrated
  check), every stage resumable/checkpointed for the sandbox's per-call time
  limit.
- `scripts/ship_rf.py` -- checkpointed production-artifact build (RF's
  full-history fit + 6 calibration folds exceed one call; this reproduces
  exactly what `MLRanker.fit()` does, in pieces, then assembles + saves the
  identical object). `ml_test.py --stage fit --models rf` remains the
  canonical one-shot regeneration path on an environment without the
  sandbox's call-length limit.
- `config.json` -- `ml_ranker.model` -> `"rf"`, provenance updated.
- `nflvalue/ml_ranker.py` build_features -- defensive fix (unrelated to the
  search/tuning work, surfaced by finally having a real artifact on disk to
  test against): a `pw` frame missing a Phase-6 roll column now NaN-fills
  instead of raising `KeyError`.
- `tests/test_ml.py` (+5), `tests/test_leakage.py` (+2) -- pruned-feature
  contract, ensemble fit/predict/save-load (avg + meta), meta-learner fold-span
  leakage guard (mirrors the calibrator's), calibration-wraps-ensemble
  leakage guard. Full suite: **236 tests green** (two ~40s halves, within the
  sandbox's per-call limit).

Artifact regenerable via `python3 ml_test.py --stage fit --models rf
--calibrate platt_permkt` (gitignored, per `data/*.joblib`).

---

## 7.3 — Real-line capture + re-label + CLV harness — design

The binding constraint on every number in this repo is that grading is
**synthetic** (trailing-mean lines), not real. This job designs the pipeline
that lets the model be graded and *re-trained* against real sportsbook prices,
and specifies the CLV accrual + kill-check that is the project's actual
referendum. **7.4 builds it; this is design + spec only** — no production logic
changes here. A worked example (`scripts/clv_worked_example.py`) proves the math
end to end on the synthetic fixture, since no live key exists and July is the
offseason.

Most of the machinery already exists (Phase-3 "Block A", `docs/decisions_p3-5.md`).
This section **pins the ambiguous definitions, names the three gaps 7.4 must
close, and writes out the leakage/look-ahead proof** — it does not re-invent
what works.

**Reuse (Block A, unchanged):** `sources/oddsapi_props.py` (`CreditBudget`
450/mo ledger, `pull_week_props` rotating entry pull, `resnap_lines` targeted
close pull, `to_prop_lines_frame`/`consensus_two_way` sharp-weighted de-vig),
`clv.py` (`snapshot_prob`, `log_close_for_week` → `clv` table, `rolling_clv`),
`killcheck.py` (the pre-committed verdict), `ml_test.augment_with_real_lines`
(the re-label path), and the `lines` / `leans` / `lean_outcomes` / `clv` /
`api_credits` tables in `db.py`.

### 1. Real prop-line capture

- **Markets:** the 7 shipped markets via `ODDS_TO_MARKET` (receiving_yards,
  receptions, rushing_yards, rush_attempts, passing_yards, pass_attempts,
  anytime_td). **Books:** the user's own books (config `books`, e.g.
  draftkings/fanduel/betmgm) — a specific book list is both a comparable price
  basis and cheaper than a whole-region pull; fall back to `regions="us"`.
- **Cadence (open → close), two snapshots per resolvable lean:**
  - **ENTRY** (`wed` clock) — the earliest affordable pull, taken as early as
    props are posted (props post later than sides, typically ~2–3 days out).
    `pull_week_props`, rotating least-recently-pulled first, `max_prop_games_
    per_run`. This is the number we would transact at.
  - **CLOSE** (`t90` clock) — `resnap_lines` targeted at games that already have
    an entry line and kick soon; the last snapshot before kickoff.
- **Storage schema:** the `lines` table, unchanged — `(ts, game_id, book,
  market, player_name, side, point, price, player_id)`, PK `(ts, game_id, book,
  market, player_name, side)` (snapshots idempotent). Book `player_name` is
  matched to gsis ids by normalized name (+ first-initial variant); unmatched
  rows persist with `player_id=NULL` — visible, never guessed, and can never
  mint an edge or a CLV row.
- **The exact "closing" snapshot (GAP #1):** the latest `lines` snapshot whose
  `ts` falls in the window **`[kickoff − CLOSE_WINDOW_H, kickoff]`**
  (`CLOSE_WINDOW_H` default **6h**, config). If no snapshot lands in that
  window, the lean is **UNRESOLVED** — *not* resolved against a stale entry-era
  snapshot (which would fake CLV ≈ 0). Today `snapshot_prob(at_or_before=kickoff)`
  accepts any old ≤-kickoff snapshot; 7.4 adds the `at_or_after_ts` floor.
- **Budget (GAP #2 — coupled reservation):** 450 credits/mo, hard-stopped by
  the existing ledger; cost = `markets × regions` per event (= 5 × 1 = **5
  credits/event** with a book list). Every *resolvable* lean needs **two** pulls
  (entry + close), so entries must not consume the whole budget:

  | quantity | value |
  |---|---|
  | monthly ceiling | 450 credits |
  | cost / event-pull | 5 (5 markets × 1 region) |
  | event-pulls / month | 90 |
  | in-season weeks / month | ~4.3 → ~21 pulls/week |
  | fully-resolved games/week (entry+close) | ~10 |
  | resolved leans/week (≈3–4 survive two-sided at both snaps) | ~30–40 |
  | **weeks to reach n ≥ 150** | **~4** |

  **Reservation rule:** cap entry events so ≥50% of the remaining weekly budget
  is held for closes — `max_entry_events_per_week = floor(weekly_budget /
  (2 · cost_per_event))`; never pull an entry for a game you can't also afford
  to close. Without this the rotation can spend everything on entries and
  resolve *nothing*.

### 2. Label migration (synthetic → real) + leakage proof

- **Which line is the label:** the **decision-time (entry) real line** =
  `leans.line` where `line_source='odds_api'` — the point we can actually
  transact at. **Not** the closing line (that is CLV's job, a different
  question). `augment_with_real_lines` already reads `leans.line`; this is
  correct and pinned. `anytime_td`: `y = scored` (nominal point 0.5), unchanged.
- **When a row flips:** exactly when **both** hold — (a) a real decision-time
  line exists for that lean, and (b) the game is **graded** (an `actual` exists
  in `lean_outcomes`). The `lean_outcomes ⋈ leans` join enforces both. Rows with
  no real line keep the synthetic label. Migration is automatic and weekly.
- **What changes on flip:** `line ← real`; `y_over ← (actual > real line)`;
  recompute `z`, `mean_minus_line`, `sd_over_line` against the real line.
  `mean`, `sd`, and every usage/efficiency/context feature are **UNCHANGED** —
  they never depended on the line.
- **Leakage / look-ahead — the proof (this is the whole game):**
  1. Re-labeling changes only a row's own label and its own line-derived
     features; it never changes the row's `(season, week)`. A re-labeled week-t
     row is still a week-t row, so the walk-forward split (`train < T`) is
     untouched.
  2. The real line is captured **before kickoff** (decision-time), so it carries
     no in-game information about its own outcome.
  3. A row can flip only **after its game is graded**, which is strictly in the
     past of any training cutoff `T` that would use it (you retrain for week `T`
     only once weeks `< T` are complete). A live retrain for `T` therefore sees
     real labels only for weeks `< T` — automatically walk-forward.
  4. **Forbidden and structurally prevented:** re-labeling the current/future
     week from a closing line before its game is played. No `lean_outcomes` row
     exists until the game is graded, so the join yields nothing and the row
     stays synthetic. *7.4 must not add any path that writes a real label from a
     line without a graded `actual`.*
  5. The 7.1 calibrator inherits this cleanly: it fits on OOS base predictions
     of seasons `< S`; those rows may now carry real labels but are still `< S`,
     so the `train_max < predict_season` fold-span guard is unaffected.
- **Worked example (`clv_worked_example.py`, step 2):** synthetic line 47.0,
  actual 50.0 → synthetic `y=1`; real decision line 52.5 → **`y` flips to 0**;
  `z` 0.444 → 0.139, `mean_minus_line` +8.00 → +2.50, `mean`/`sd` unchanged.
  Asserted.

### 3. CLV harness — complete + monitorable

- **Per-lean CLV, de-vigged consensus prob space:** `entry_prob` = consensus
  fair P(side) from the latest snapshot ≤ `lean.as_of`; `close_prob` = consensus
  fair P(side) from the closing snapshot (§1); **`clv_prob = close_prob −
  entry_prob`**. Comparing de-vigged probabilities means a book fattening its
  margin can't masquerade as line movement. `anytime_td` is one-sided → raw
  implied at both ends (`prob_kind='raw_implied'`); the *difference* is still
  meaningful, and the flag prevents mixing the two kinds.
- **Two-snapshot resolution rule:** a lean resolves **iff** it has (a) an entry
  snapshot ≤ `as_of`, (b) a distinct closing snapshot in the pre-kickoff window
  with `close.ts > entry.ts`, and (c) `status='active'` (voided leans — player
  ruled out at t90 — never resolve). Otherwise UNRESOLVED: visibly absent, never
  faked.
- **Gaps 7.4 closes:**
  - **#1 close-window floor** (§1) — add `at_or_after_ts` to `snapshot_prob`.
  - **#3 clock dedup** — the `clv` PK omits `clock`, so a `wed` + `t90` pair for
    the same `(season, week, game, player, market, side)` collide. **Rule:**
    resolve against the **earliest active `as_of`** (the `wed` entry captures the
    most line movement — premortem F5), one `clv` row per key. 7.4 selects
    `MIN(as_of)` among active leans before computing `entry_prob`.
- **Monitorability — the 7.4 dashboard tab reads exactly these** (from `clv` +
  `leans`): `resolved_n = COUNT(clv)`; `logged_n = COUNT(leans WHERE
  status='active')`; **coverage = resolved_n / logged_n** (a low value warns the
  close-snapshot budget is too thin to resolve the log); `lifetime_mean` &
  `rolling(50)` avg `clv_prob`; `positive_rate = AVG(clv_prob > 0)`;
  `avg_point_move`; and the GO/NO_GO/INSUFFICIENT banner. `rolling_clv` and
  `killcheck.report` already return all but `coverage`.
- **Kill-check — pre-committed thresholds, NOT moved** (Phase-3 / PREMORTEM
  §248): **n ≥ 150** resolved leans; **GO** iff lifetime avg CLV > 0 **AND**
  positive-CLV rate ≥ **52%**; otherwise **NO_GO** — "revert to
  projection/entertainment tool, stop staking." `n < 150` → INSUFFICIENT_SAMPLE.

### 4. The prop-CLV honesty caveat

PREMORTEM §285: props are lower-liquidity than main markets, where CLV is a
**weaker** edge proxy. So positive prop-CLV is **necessary, not sufficient** — a
GO is "consistent with edge," not proof of profit. Two forces still bite even at
positive CLV: soft books limit winners in ~20 bets (F4), and a Wednesday entry
already sits near-closing (F5, the worst moment for the stated edge). The
referendum remains CLV, framed this honestly; a real-money decision is a 7.9
call, not a CLV-crosses-zero reflex.

### Worked example + acceptance tests for 7.4

`scripts/clv_worked_example.py` drives the **real** code paths on the synthetic
fixture (`tests/fixtures/oddsapi_event_props_synthetic.json`) + a throwaway DB:
capture two snapshots → re-label a row (y flips 1→0) → CLV (`+0.0236` prob-points,
market moved toward our over) → monitor (`INSUFFICIENT_SAMPLE`, n=1). All
assertions pass. This is the fixture proof; it changes no production logic.

7.4 is done when, on recorded/synthetic fixtures: (a) an entry+close capture runs
within a coupled budget that reserves close credits; (b) `snapshot_prob` honors
the `CLOSE_WINDOW_H` floor and marks out-of-window leans UNRESOLVED; (c) CLV
dedups to one row per lean key against the earliest active `as_of`; (d) the
re-label path flips only graded rows with a real line, proven by an extended
`tests/test_leakage.py` case; (e) the dashboard shows resolved-n vs 150,
coverage, avg CLV, positive-rate, and the GO/NO-GO banner; (f) the kill-check
thresholds are the unchanged pre-committed ones.

---

## 7.4 — Real-line capture + re-label + CLV harness — implementation

Built 7.3's design against the real code paths (`clv.py`, `killcheck.py`,
`oddsapi_props.py`, `ml_test.augment_with_real_lines`, `pipeline_weekly.py`,
`dashboard.py`, `scripts/auto_weekly.py`). No live Odds API key exists in this
environment and it's the offseason, so every number in this section is proven
on fixtures/synthetic data or on the real (empty) local warehouse -- never
presented as a live result. Checkpoint B (dry-run) evidence: `python3
scripts/clv_worked_example.py`.

### GAP #1 -- close-window floor (`clv.py`)

`snapshot_prob` gained an `at_or_after_ts` floor and `log_close_for_week` now
computes `window_start = kickoff - close_window_hours` (config
`clv.close_window_hours`, default 6.0) and requires the close snapshot to fall
in `[window_start, kickoff]`. A snapshot that predates the window no longer
silently passes as a "close" -- it resolves to nothing (visibly absent, not a
faked ~0 CLV). Proven in `tests/test_clv_killcheck.py`
(`test_close_window_floor_rejects_stale_snapshot`,
`test_close_window_floor_accepts_in_window_snapshot`) and worked-example step 5.

### GAP #2 -- entry-event budget reservation (`oddsapi_props.py`)

`entry_event_cap(budget, cost_per_event)` implements
`floor(weekly_budget / (2 * cost_per_event))` with `weekly_budget =
budget.remaining / 4.3` (weeks/month), so an in-season week never pulls more
ENTRY events than leaves room for a paired CLOSE pull per game later that same
week. `pull_week_props`'s per-run cap is now `min(config
max_prop_games_per_run, entry_event_cap(...))` and returns the reserved cap
for visibility (`entry_cap_reserved`). Also fixed a real, previously-silent
bug while doing this: `config.json` had no `prop_markets_internal` key, so the
capture path always fell back to a hardcoded 5-market list missing
`rush_attempts` and `pass_attempts` -- both are now in the config's 7-market
list. Proven in `tests/test_oddsapi_props.py`
(`test_entry_event_cap_formula`,
`test_entry_pulls_reserve_half_the_budget_for_closes`, and the rewritten
`test_budget_never_exceeded_over_a_simulated_month`) and worked-example step 7.

### GAP #3 -- clock dedup (`clv.py`)

The `clv` table's primary key omits `clock` (a `wed` and `t90` lean for the
same game/player/market/side would collide on upsert), so
`log_close_for_week` now dedups the input leans to one row per
`(game_id, player_id, market, side)`, keeping the **earliest** active
`as_of` (the Wednesday entry captures the most line movement, matching
PREMORTEM F5's framing of the entry point). Proven in
`tests/test_clv_killcheck.py` (`test_clock_dedup_resolves_one_row_against_earliest_as_of`)
and worked-example step 6.

### Re-label leakage proof

Extended `tests/test_leakage.py` with
`test_augment_with_real_lines_only_flips_graded_rows_with_real_line`: three
rows -- (a) real line + graded -> flips, season/week unchanged, non-line
features (`mean`, `sd`) untouched; (b) real line but **ungraded** (no
`lean_outcomes` row -- a future/in-progress week) -> stays fully synthetic;
(c) graded but `line_source != 'odds_api'` -> stays fully synthetic. This is
the structural proof that a real label can never be written without both a
real line AND a graded outcome (7.3 §2, point 4's "forbidden path").

### Monitorability: coverage + dashboard tab

`killcheck.report` gained `coverage = resolved_n / logged_n` (warns when the
close-snapshot budget is too thin to resolve the log even though leans are
being generated). `nflvalue/dashboard.py` gained a dedicated **CLV /
Kill-Check** tab: a colored GO/NO_GO/INSUFFICIENT_SAMPLE banner, a progress
bar toward the pre-committed n=150 gate, cards for resolved-n, lifetime and
rolling(50) avg CLV, positive-CLV rate (vs the 52% bar), and coverage, plus
the §4 honesty caveat (prop CLV is a weaker edge proxy than main-market CLV).
Verified by rendering the template with jsdom for all three verdict states
(GO / NO_GO / INSUFFICIENT_SAMPLE) and by regenerating the committed
`dashboard.html` from the real (currently empty -- offseason, no key) local
`data/latest.json`, which honestly shows `INSUFFICIENT_SAMPLE`, n=0 resolved,
70 leans logged.

### Scheduler

`scripts/auto_weekly.py`'s three jobs (`wed`, `t90`, `tuesday`) already
self-detect the offseason and no-op cleanly (verified: no upcoming REG week
within 8 days / no kickoffs in the T-90 window / no completed week all exit 0
with a one-line log). One real gap found while wiring in-season cadence:
`resnap_lines` has no internal dedup, so a `t90` job firing hourly across a
game's ~2.75h pre-kickoff window would pay for the close snapshot repeatedly.
Added `clv.has_close_snapshot(conn, game_id, kickoff, close_window_hours)` and
wired it into `job_t90` as a filter so at most one close pull happens per game
per week, matching GAP #2's assumption. Proven in `tests/test_clv_killcheck.py`
(`test_has_close_snapshot_false_when_nothing_in_window`,
`test_has_close_snapshot_true_once_in_window`). Installed three Cowork
scheduled tasks that shell out to `scripts/auto_weekly.py`: `wed` (Wednesdays
10am ET), `t90` (hourly on Thu/Sun/Mon -- NFL gamedays), `tuesday` (Tuesdays
8am ET) -- all run year-round and rely on the script's own no-op guards rather
than the schedule encoding season boundaries.

### Checkpoint B status

Done, on fixtures: `scripts/clv_worked_example.py` runs capture → re-label →
CLV → monitor → all three gaps end to end (7 steps, all assertions pass); the
dashboard renders the referendum state (verified GO/NO_GO/INSUFFICIENT_SAMPLE
banners + regenerated from real, honestly-empty local data); the scheduler is
installed and self-detects the offseason. Full test suite: 250/250 passed
(`tests/`, run in batches due to sandbox time limits). No number in this
section, the dashboard, or the worked example came from a live sportsbook --
this environment has no Odds API key and it's the offseason.

---

## 7.5 — Same-game prop correlation modeling

Two leans in the same game are not two independent edges, and a same-game
parlay's true price depends on how the legs move together. This job **measures**
that structure walk-forward, shrinks it so thin pairs can't invent correlation,
and **exposes** it as an artifact for 7.6 (selection) and 7.7 (staking). It
changes no selection/staking behavior — measure and expose only. Plain-language
companion: `docs/EXPLAINER_correlation.md`.

### Method

- **Standardized residuals, not raw outcomes.** For every prop we take
  `r = (actual − projection mean) / projection sd` from `data/ml_frame.parquet`
  (projection) joined to `player_week` (actual). Standardizing makes a QB's
  300-yard game and a WR's 90-yard game comparable and strips the projection's
  own level, so we measure *co-movement*, not shared trend.
- **Pair taxonomy.** Within each game, pairs are typed as
  `relationship | posA.familyA ~ posB.familyB`, relationship ∈ {sameplayer,
  sameteam, opponent}, families {pass, rec, rush, td}. Cross-player pairs use one
  market per family (yardage + td) so a player can't be double-counted;
  same-player keeps every market. Classification lives in `nflvalue.correlation.
  classify_pair` (shared by the measurement script and the read side).
- **Pooled Pearson ρ per type**, over 1,823 game-weeks (2019–2025).
- **Walk-forward.** ρ for consumption at season S is estimated only from pairs
  in seasons `< S`; the artifact carries these slices and a leakage test proves
  each slice is byte-identical when seasons ≥ S are removed.
- **Shrinkage toward zero** (empirical-Bayes, Fisher-z): `z=atanh(ρ)`,
  `SE²=1/(n−3)`; between-type signal variance `τ²=0.071` estimated across types;
  each `z` pulled toward 0 by `τ²/(τ²+SE²)`. Thin/noisy types collapse to ~0.
- **REAL vs NOISE is an effect-size + stability call, deliberately NOT t≥2.**
  With tens of thousands of pairs, t is enormous for economically-zero ρ (two
  same-team WRs: ρ=0.03 but t≈3). A type is **REAL** iff `|ρ_shrunk| ≥ 0.05`
  **and** its per-season sign is stable; else NOISE (consumed as 0).

### Measured structure — what's real

*Reproduce: `python3 scripts/fit_correlation.py` → `reports/correlation_structure.md`,
`data/correlation_structure.json`. Synthetic-line caveat: residuals are vs the
projection, not real prices.*

**Same-player (mechanical — near-duplicate legs):** two markets on one player are
almost the same bet.

| type | n | ρ raw → shrunk |
|---|---|---|
| QB passing_yards ↔ pass_attempts | 3,813 | +0.785 → **+0.783** |
| TE receiving_yards ↔ receptions | 4,202 | +0.775 → **+0.774** |
| WR receiving_yards ↔ receptions | 11,045 | +0.765 → **+0.764** |
| RB rushing_yards ↔ carries | 5,974 | +0.764 → **+0.763** |
| RB rushing_yards ↔ anytime_td | 11,948 | +0.353 → **+0.352** |
| WR receiving_yards ↔ anytime_td | 22,090 | +0.332 → **+0.332** |
| TE receiving_yards ↔ anytime_td | 8,404 | +0.269 → **+0.268** |

**Same-team, cross-player (the SGP-relevant structure):**

| type | n | ρ raw → shrunk | reading |
|---|---|---|---|
| QB pass ↔ WR rec | 11,660 | +0.297 → **+0.297** | QB throws well → his WR gains |
| QB pass ↔ TE rec | 4,443 | +0.245 → **+0.244** | same, tight end |
| QB pass ↔ WR td | 12,001 | +0.113 → **+0.113** | more passing → more WR TDs |
| QB pass ↔ TE td | 4,495 | +0.090 → **+0.090** | weak but stable |
| QB pass ↔ RB rush | 6,291 | −0.080 → **−0.080** | pass game vs run game (script trade-off) |

**Opponent, cross-team (game flow):**

| type | n | ρ raw → shrunk | reading |
|---|---|---|---|
| QB pass ↔ opp QB pass | 2,014 | +0.110 → **+0.110** | shootouts: both QBs throw |
| RB rush ↔ opp RB rush | 4,919 | −0.100 → **−0.100** | one team runs (leading) → other passes |
| QB pass ↔ opp WR rec | 11,568 | +0.052 → **+0.052** | faint shootout echo |
| QB pass ↔ opp WR td | 11,912 | +0.051 → **+0.051** | faint shootout echo |

### What's NOISE (consumed as 0)

- **Two same-team WRs' receiving: ρ=0.03** (n=17,978, t≈3). The DFS "stack"
  intuition does **not** survive here — within a game, target competition roughly
  cancels the shared game-script lift. This is the headline noise finding, and
  the clearest case of why the t≥2 culture had to be dropped for effect size.
- Two opponent WRs (0.03), essentially every TD↔TD cross pair (|ρ|<0.05), and
  rare types (two same-team passing QBs, n=393, sign-unstable) → all 0.

### Real vs noise, stated plainly

The structure a bettor actually needs is small and stable: **same-player
multi-market pairs are near-duplicates (~0.76)**; a **same-team QB + his
pass-catcher move together at ~0.30**; **run vs pass legs (same team, or
opposing RBs) hedge at ~−0.08 to −0.10**; **shootout (opposing QBs) is a mild
+0.11**. Everything else — including the popular two-WR stack — is noise. All
real types are sign-stable across all seven seasons and across the walk-forward
slices (e.g. QB↔WR: 0.31, 0.31, 0.30, 0.29, 0.30 for as-of 2021→2025).

### Recommended use (7.6 / 7.7) — honest scope

- **Correlation-aware selection (7.6): clearly worth building.** Don't count two
  correlated leans as two edges. The load-bearing cases: (a) a player's own two
  markets are ~one bet (ρ≈0.76) — never let a slip carry both as independent;
  (b) a same-team QB-over + pass-catcher-over is ~1.3 bets, not 2 (ρ≈0.30) —
  discount or cap. Negative pairs (QB-pass vs RB-rush; opposing RBs) are
  *diversifying*, not redundant — leave them.
- **SGP joint pricing (7.7): worth it only for the handful of real types, not a
  general engine.** A Gaussian copula on the standardized residuals, using these
  ρ, can price the same-team QB↔WR/TE receiving stack and the same-player pairs
  honestly. It should **not** price arbitrary same-game parlays — most leg-pairs
  are noise, and a copula fed noise invents precision. Recommendation: expose the
  joint estimate as an **optional, clearly-labeled readout for the real cross-
  player types only**, never as a synthetic-line "edge."

### Artifact interface (what 7.6 / 7.7 read)

- `data/correlation_structure.json` — per pair type: `rho_raw`, `rho_shrunk`,
  `n_pairs`, `se`, `per_season`, `sign_stable`, `verdict`; plus `walk_forward`
  slices (ρ from `< S` for each S) and the shrinkage metadata (`tau2`,
  `rho_floor`, `min_n`). Gitignored/regenerable (`data/*` derived).
- `nflvalue/correlation.py` — `classify_pair(...) → type key`;
  `CorrelationStructure.load()`; `.rho(ptype, as_of_season=None)` and
  `.rho_for(pos_i, market_i, player_i, team_i, pos_j, market_j, player_j, team_j,
  as_of_season=None)` → **shrunk ρ, and 0.0 for any NOISE or unknown type** (a
  consumer is never handed structure the audit called noise); `.real_types()`.
  `as_of_season` returns the strict walk-forward slice for a backtest; omit it
  for the production (all-history) value live.
- Leakage-tested in `tests/test_correlation.py`: order-independent
  classification, cross-player volume-market exclusion, shrinkage collapses thin
  types, the accessor zeroes noise/unknown, and the **walk-forward slice for S is
  unchanged when seasons ≥ S are deleted**.

### Done

Correlations measured walk-forward with shrinkage and a leakage test; the
real-vs-noise call is explicit and effect-size based; the artifact is ready for
7.6/7.7 to consume without further design. No selection or staking behavior
changed in this job.

---

## 7.6 — Correlation-aware selection + reporting

Wires 7.5's shrunk-rho artifact into the shortlist itself, so a top-5 isn't
secretly five bets on one game outcome, and surfaces the effect everywhere a
human reads a slip. No change to composite scoring, the ML ranker, or the
data going INTO a candidate's score -- this job only changes which candidates
get selected, and what's displayed alongside them.

### Selection rule

`shortlist.rank_game`/`shortlist_week` gain optional `corr`, `as_of_season`,
`corr_discount_strength` params. `corr=None` (the default) reproduces the
exact pre-7.6 selection byte-for-byte -- every existing caller is unaffected
until it's explicitly opted in.

When `corr` is given, selection becomes an MMR-style greedy walk: at each of
the `top_n` slots, pick the REMAINING candidate with the highest *discounted*
score, where the discount is `redundancy_discount(rho) = clip(max(0, rho), 0,
0.95) * strength` -- the largest POSITIVE shrunk rho vs any lean ALREADY
selected in that game (same-player pairs ~0.76, same-team QB+pass-catcher
~0.30). A near-duplicate leg that would have filled a slot on raw composite
alone can lose it to a lower-composite but genuinely independent candidate
further down the list. Negative/diversifying rho (QB-pass vs RB-rush,
opposing RBs) is never penalized -- 7.5 found those legs *help*, exactly as
recommended. A discount is never total (capped at 0.95): a fully-duplicate
leg can still fill an otherwise-empty slot rather than being hard-banned.
Each selected lean carries `corr_discount` (0 if none) and `corr_with`
(`{player_id, name, market, rho}` of the leg it was discounted against, or
`None`) for the report to explain itself.

`as_of_season`: strict walk-forward rho (only seasons `< as_of_season`
informed it) for the backtest; omitted (production/all-history value) for
every live call site (`pipeline_weekly.run_week`/`run_t90` via
`load_correlation`, which degrades loudly -- one printed line, `corr=None` --
if the artifact is disabled, missing, or empty, never inventing structure).

### Reporting

`game_notes.correlation_notes_for_game` flags pairs of SELECTED leans whose
type cleared 7.5's REAL bar, appended into the SAME display-only `g["notes"]`
list every renderer already reads (`report.py` markdown, `document.py` HTML
drop, and -- newly wired this phase, since it turned out `dashboard.py` never
rendered `notes` at all before now -- the dashboard's Weekly Leans tab). A
positive pair reads "move together"; a negative pair reads "hedge against
each other". The dashboard additionally shows a small "corr −N%" badge next
to a discounted lean's market, with the partner leg on hover.

### SGP joint-probability readout (7.5's narrow green-light)

`correlation.sgp_joint_prob(p_i, side_i, p_j, side_j, rho)`: a Gaussian
copula over each leg's OWN model probability (never a synthetic-line "edge")
and the measured rho, via the correct bivariate-normal quadrant
(inclusion-exclusion on `scipy.stats.multivariate_normal`'s CDF, side-aware).
`shortlist.sgp_readouts` computes this for every pair of SELECTED leans whose
type cleared the REAL bar, returning `{leg_a, leg_b, rho, independent_joint_prob,
copula_joint_prob, label}` -- `label` states plainly this is informational
only and prices nothing until a real SGP market exists. Wired into
`report.generate` (`g["sgp"]`, rendered in the markdown and the dashboard) and
`pipeline_weekly.run_t90`; gated by config `correlation.sgp_readout` (default
true). Verified: independence (`rho=0`) falls back to the plain product;
positive rho raises an over/over joint above independence and lowers an
over/under joint below it (the hedge direction); symmetric under leg swap;
degenerate probabilities (0/1) return `None` rather than a meaningless number.

### Ablation (Checkpoint C) — honest result: KEEP, pooled positive

`scripts/ablate_correlation.py` replays `lean_backtest.run` twice per season
(baseline vs `corr_aware=True`, same frozen policy: `top_n=5,
max_per_player=2`) over 2022-2025 on real historical data (not fixtures --
this is the same walk-forward directional-hit-rate replay every other phase's
backtest uses, with the same synthetic-line caveat: NOT price-beating/profit).
`reports/correlation_ablation.md` / `data/correlation_ablation.json`:

| | n | hit rate | units (flat 1u, -110) |
|---|---|---|---|
| Baseline (shipped) | 5,435 | 58.1% | +595.8u |
| Correlation-aware | 5,435 | **58.6%** | **+645.5u** |

| Season | Baseline hit / units | Corr-aware hit / units | Δhit | Δunits |
|---|---|---|---|---|
| 2022 | 59.0% / +172.3u | 58.4% / +155.1u | **−0.66%** | **−17.2u** |
| 2023 | 58.2% / +152.0u | 58.8% / +165.4u | +0.51% | +13.4u |
| 2024 | 58.7% / +163.4u | 59.5% / +184.4u | +0.81% | +21.0u |
| 2025 | 56.5% / +108.1u | 57.8% / +140.6u | +1.25% | +32.5u |

Top-1-per-game is (correctly) unchanged -- a single pick per game has nothing
to discount against. Diversification: baseline slips average ~4.38 "effective
independent bets" per 5-leg slip (1 minus max positive pairwise rho vs each
earlier-selected leg); correlation-aware slips average ~4.95 -- measurably
closer to 5 genuinely independent bets, with 257 selected legs (of ~5,435,
~4.7%) carrying a nonzero discount over the 4 seasons.

**Honest characterization:** pooled result is positive on BOTH hit-rate and
units, and diversification improved as designed -- 3 of 4 seasons beat
baseline on both metrics; 2022 alone is mildly worse (−0.66pt hit,
−17u), plausibly season-level noise given the same order of magnitude of
year-to-year variance already documented in 6.7/6.8's policy-search work. This
clears the pre-committed bar ("keep it only if it helps or is
neutral-with-better-diversification") on the stronger of the two conditions
-- it helps, pooled and in most seasons, not just diversifies. Shipped
enabled by default (`config.json` `"correlation": {"enabled": true,
"discount_strength": 1.0, "sgp_readout": true}`).

### Tests

`tests/test_correlation.py`: `redundancy_discount` (positive-only, strength
scaling, 0.95 cap), `sgp_joint_prob` (independence fallback, degenerate-input
None, leg-swap symmetry, positive rho raises the aligned joint / lowers the
hedge joint) -- 7 new tests, 11 total in the file.
`tests/test_shortlist.py`: `corr=None` byte-identical to baseline; a
near-duplicate leg loses its slot to an independent one; discount/partner
tagging; negative rho never penalizes; determinism under input reordering;
`shortlist_week` threads `corr` per game; `sgp_readouts` returns the real
pair and empty when `corr` is `None` or no real type is present -- 8 new
tests, 20 total. `tests/test_game_notes_auto.py`: correlation notes appended
only when `corr` is given, negative rho reads "hedge", existing story notes
untouched by default -- 5 new tests, 10 total. Full suite: 270/270 passed.

### Done

Selection consumes the 7.5 artifact (byte-identical when not opted in); the
report/dashboard/document all show correlation flags and, where a real
cross-player pair exists among the selected legs, a clearly-labeled SGP
readout; tests cover the discount math, the selection behavior change, and
the reporting; the walk-forward ablation is honestly reported and the feature
ships because it helped, not merely because it was neutral.

---

## 7.7 — Staking / bankroll module

Bet sizing is where a good model still goes broke. This job turns calibrated
edges into **advisory** stake sizes that respect estimation error, correlation
(7.5), and a hard drawdown tolerance, sized to survive the 6.8 variance
envelope. **The module never places a bet, moves money, or initiates a transfer**
— output is a recommendation a human may act on, ignore, or override. Every input
probability is model-estimated and, until real CLV accrues, graded at synthetic
lines. Plain-language companion: `docs/EXPLAINER_staking.md`.

### The sizing rule (`nflvalue/staking.py`, deterministic + pure)

Per lean, given calibrated `p` (model prob of the side), the real price `d`
(`b = d−1`), and de-vigged `market_prob`:

1. `edge = p − market_prob`; **`≤ 0` → stake 0** (never size a non-edge).
2. **Estimation-error shrink** (`s_edge = 0.5`): regress the edge toward the
   market prior — the market is efficient and `p` is a noisy estimate, so size as
   if the edge is half what it looks. `p_s = market_prob + s_edge·edge`.
3. **Full Kelly on the shrunk prob** at the real price: `f* = (b·p_s−(1−p_s))/b`.
4. **Fractional Kelly** (`κ = 0.25`, quarter-Kelly — matches 6.8; absorbs
   parameter + correlation uncertainty). Effective ≈ ⅛ of raw Kelly.
5. **Correlation adjust** (7.5): `f / (1 + Σ_{j≠i, same game} max(ρ_ij, 0))`.
   Two same-team QB+WR overs (ρ≈0.30) each shrink ×1/1.30; a player's own two
   markets (ρ≈0.76) shrink so together they ≈ one bet. Negative (hedging) ρ gets
   **no bonus** — conservative.
6. **Per-bet cap** `min(f, 0.02)` (below the 58% quarter-Kelly of 2.95%).
7. **Portfolio**: scale a slate down if stakes sum past `max_slate = 0.10`; then a
   global `dd_scale ≤ 1` lets the 6.8 MC pin the fraction to a drawdown cap.

`1u ≡ 1% of bankroll` (6.8 convention). A p=0.55 side at −110 (fair 0.5238) sizes
to **0.69% of bankroll** (0.69u) — half of plain quarter-Kelly's 1.37%, the
estimation-error discount made concrete.

### Bankroll Monte Carlo (`scripts/staking_mc.py`)

The 6.8 machinery, run at **plausible real-line edges (52–58%), NOT the synthetic
66–68%** (which compounds to fiction — 6.8 makes the point, we honor it). A ~306-
bet season with within-game legs correlated at ρ=0.30 (Gaussian copula from the
7.5 artifact, so the correlation adjustment is exercised). Start 100u; ruin = lost
≥80%.

| true hit | strategy | median end | p95 max DD | P(ruin) |
|---|---|---|---|---|
| 54% | plain ¼-Kelly | 106.9 | 24.6% | 0 |
| 54% | **shrunk (shipped)** | **103.6** | **11.1%** | **0** |
| 55% | plain ¼-Kelly | 119.3 | 33.4% | 0 |
| 55% | **shrunk** | **108.2** | **14.6%** | **0** |
| 58% | plain ¼-Kelly | 227.4 | 48.1% | 0 |
| 58% | **shrunk** | **119.7** | **10.7%** | **0** |

The shipped rule grows the bankroll far slower than raw quarter-Kelly but holds
p95 drawdown to **~10–15%** (inside the 20% tolerance, so the default `dd_scale=1`
needs no further scaling) with **zero ruin** — the whole point of shrinking an
edge you only estimated. At 52.38% (breakeven) the rule bets nothing (edge ≤ 0):
**no sizing manufactures an edge that isn't there.** Reproduce:
`python3 scripts/staking_mc.py` → `reports/staking_mc.md`, `data/staking_mc.json`.

### Advisory-only, and the honest chain

The module's docstring and the report both carry the disclaimer, and
`recommend_stakes` returns it in every payload. Staking is a **new advisory
layer**: it consumes composite/model fields + the 7.5 artifact and changes no
selection or composite behavior. The 6.8 honest chain still governs — synthetic-
line skill (measured) → real-line hit rate (unknown until CLV accrues) → profit
(variance-dominated). These sizes say *how much to risk IF the edge is real*; the
CLV kill-check (7.3/7.4), not this module, decides whether it is.

### Done

Staking is deterministic, shrunk for estimation error, correlation- and
drawdown-aware, capped, advisory-only, and tested (`tests/test_staking.py`, 9
cases: determinism, no-edge→0, monotonic in edge, shrink reduces size, per-bet +
slate caps, correlation reduces correlated stakes, negative-ρ gets no bonus,
disclaimer/units). Every projection is at real-line-plausible edges with the
synthetic caveat stated.

---

## 7.8 — End-to-end integration, tests, docs

**One end-to-end test** (`tests/test_e2e_phase7.py`) walks a fixture slate through
the whole Phase-7 chain using each component's real interface — calibrated
`MLRanker` → correlation-aware `shortlist.rank_game(corr=…)` → advisory
`staking.recommend_stakes` → fixture `lines`/`leans`/`lean_outcomes` →
`clv.log_close_for_week` → `killcheck.report` — and asserts every hand-off
connects (calibrated probs bounded/walk-forward; the aware shortlist differs from
the pre-7.6 baseline; stakes respect per-bet + slate caps and carry the advisory
disclaimer; CLV = close − entry > 0; the kill-check reads `INSUFFICIENT_SAMPLE`
at n=1). A regression anywhere in the chain trips it.

**Leakage suite covers every Phase-7 surface** (`tests/test_leakage.py`):
calibration fit (7.1), ensemble meta-learner (7.2), real-line re-label (7.3/7.4),
correlation estimate (7.5) — each proven to use only data strictly before the
season/week it informs — plus an explicit note+test that advisory staking (7.7)
has **no** temporal/rolling surface (pure point-in-time; its only history-derived
input is the walk-forward-guarded correlation ρ).

**Docs brought to reality:** `README.md`, `docs/HOW_A_PICK_IS_MADE.md`, and
`docs/DATA_SOURCES.md` now describe the calibrated RF ranker, correlation-aware
selection, advisory staking, and the completed CLV referendum (n≥150 gate +
coverage + dashboard), with the synthetic-vs-real distinction stated plainly.
Full suite: **283 tests**, green (run in batches for the sandbox's per-call time
limit; whole suite ~90s of compute). *(Grew to 332 after the §7.10 post-ship
hardening pass below.)*

---

## 7.9 — Go/no-go framework + honest final assessment

This section ships the project. It does two things and nothing else: it
**pre-commits** the exact evidence that flips this tool from "entertainment" to
"staked," and it states, as plainly as an honest skeptic would, what Phases 1–7
do and do not prove. Not one number in the go/no-go decision is a synthetic-line
figure — **the referendum is closing-line value.**

### A. The go/no-go framework (pre-committed; CLV only)

Why CLV and not won bets: 6.8 showed even a genuinely skilled 55% bettor needs
~2,231 bets to separate from breakeven, a 54% bettor ~5,855 — many seasons at NFL
volumes. CLV converges orders of magnitude faster (~50–150 resolved leans). So
the decision rests entirely on CLV.

**GO requires ALL of the following** (the first three are the Phase-3 / kill-check
thresholds, unchanged; the last two are additional *validity* guards that make GO
strictly harder, never looser):

1. **n ≥ 150 resolved leans** — entry and in-window close both captured, lean
   active (not voided). *(Phase-3, unchanged.)*
2. **Lifetime average CLV > 0** in de-vigged probability space. *(unchanged.)*
3. **Positive-CLV rate ≥ 52%.** *(unchanged.)*
4. **Coverage ≥ 0.5** (`resolved_n / logged_n`): at least half of logged leans
   actually resolved, so the 150 isn't a cherry-picked, biased subset of a thin
   close-snapshot budget. Below this, any verdict is provisional until capture is
   fixed.
5. **Span ≥ 8 distinct in-season weeks** of live capture, so the sample crosses
   real line-movement regimes rather than one fluke slate.

**NO-GO** is declared the moment n ≥ 150 with (avg CLV ≤ 0 OR positive-CLV rate <
52%). Pre-committed consequence, in writing, in the report and on the dashboard:
**revert to a projection/entertainment tool and stop staking.** The check is
*continuous*, not one-shot — even after a GO, a sustained negative rolling-window
CLV re-triggers NO-GO. `n < 150` (today's state) is **INSUFFICIENT_SAMPLE**: keep
logging, stake nothing, conclude nothing.

**What a GO authorizes — and what it does not.** GO permits *small, disciplined,
monitored* staking at the 7.7 advisory sizes (shrunk quarter-Kelly, capped), with
the kill-check still armed. It is **not** a profit guarantee: prop CLV is a weaker
edge proxy than main-market CLV (PREMORTEM §285), soft books limit winners in
~20 bets (F4), and a Wednesday entry already sits near the close (F5). A GO means
"consistent with real edge, worth risking a little to find out more" — no more.
And in every state the tool still **never places a bet or moves money.**

### B. The paid-data decision (FTN API)

The FTN Data API (participation + charting: man/zone, alignment, personnel) is
the single largest remaining free-data gap and the **one pre-authorized paid
purchase** — but **only after a GO (§A)**. The logic is strict: buying data to
improve a tool that has not yet demonstrated real edge is spending money on an
unproven bet. The trigger is a GO **plus** a stated, testable hypothesis that FTN
features would lift the specific markets where measured edge is thinnest. Until
both hold: no paid source, and no scraping/ToS-circumvention (design H11). This is
the only door through which paid data may ever enter, and it stays locked until
CLV opens it.

### C. Honest assessment — what Phases 1–7 built

**What it demonstrably does** (measured, walk-forward, reproducible from scripts):

- **Walk-forward directional skill at synthetic lines** — calibrated RF 63–69% /
  season vs the tuned composite's 57–59%, calibrated pooled log-loss 0.621,
  +1,600u at −110 over 2022–25 top-5 selection. **All at synthetic trailing-mean
  lines.**
- **Calibrated probabilities** — per-market Platt, ECE ≈ 0.011; `P(over)=0.62`
  lands ~62%, so edge and stake read a trustworthy number (7.1/7.2).
- **A measured, shrunk same-game correlation structure** — QB↔his-WR ≈ 0.30,
  same-player markets ≈ 0.76, run/pass hedges ≈ −0.08 to −0.10, and the two-WR
  "stack" honestly called noise (7.5).
- **A quantified variance envelope** (6.8) and **advisory sizing that survives
  it** — shrunk, correlation- and drawdown-aware, ~10–15% worst-case drawdown,
  near-zero ruin at plausible edges (7.7).
- **Discipline as a feature** — every constant measured with printed provenance;
  leakage guards on every surface; and honest *negative* results kept rather than
  buried (garbage-time filter, opponent-pace term, birthdays/revenge, feature
  pruning, the ~0 calibration gain on 2025).

**What it does NOT prove:**

- **Real-line profit.** Synthetic trailing-mean lines structurally favor unders
  and price far less than a real sharp market; part of the ML edge is likely
  learned exploitation of that construction (documented in `decisions_p3-5.md`).
  The honest chain is **synthetic-line skill (measured, strong) → real-line hit
  rate (unknown until CLV accrues) → profit (variance-dominated at any realistic
  NFL volume)**. No synthetic number in this repo is a profit claim, and none is
  admissible in the go/no-go decision.

**Bottom line.** The project is complete in the only sense that is honest here —
not "guaranteed to win," but a disciplined instrument that makes calibrated,
correlation-aware, honestly-sized recommendations *and is built to prove or
disprove its own edge*, with a pre-committed answer either way. Its current state
is **INSUFFICIENT_SAMPLE / paper-trade** (no live Odds API key, offseason, 0
resolved leans). Until the CLV GO in §A, it is an entertainment and research tool
— and §A says exactly, in advance, what would change that. Top-level snapshot:
`PROJECT_STATUS.md`.

---

## 7.10 — Post-ship hardening pass

After 7.9 shipped, a multi-agent audit re-read **every** module — the Phase-7
surfaces AND the older Phase 1–6 learning/feature/IO code — for correctness (not
style). No correctness-critical or leakage kill-bug was found anywhere; the issues
below are the real-but-bounded ones that were fixed. The honest-negative culture
applies here too (most were low-severity and are recorded as such). Full suite
after this pass: **332 tests**, green (331 passed, 1 skipped — `test_ingest` skips
cleanly when the optional `nflreadpy` isn't installed, via `pytest.importorskip`).

**Correctness fixes (real):**

- **Correlation walk-forward verdict-gate leak** (`correlation.py`,
  `fit_correlation.py`). `rho(as_of_season=S)` returned a leak-free prior-only ρ
  *value*, but gated inclusion on the type's `verdict`, which was computed from
  *full-history* (2019–2025) sign-stability. So a backtest at season S could
  include/exclude a pair based on data ≥ S. Fixed: the `walk_forward[S]` slices
  now carry only pair types that clear a **prior-only** verdict (|ρ_<S| ≥ floor
  AND sign-stable across seasons < S); `rho(as_of_season=S)` gates solely on
  presence in that slice. Production (all-history) consumption is unchanged. New
  test asserts the *inclusion set* (not just each value) is byte-identical when
  seasons ≥ S are removed. The shipped `sameteam|QB.pass~WR.rec` survives every
  slice.
- **CLV prob-kind mixing** (`clv.py`, `db.py`). `log_close_for_week` never checked
  `snapshot_prob`'s `prob_kind`, so a one-sided `anytime_td` close (raw-implied)
  could be differenced against a two-sided entry (de-vigged) — an apples-to-
  oranges CLV feeding the kill-check. Fixed: a lean now resolves only when entry
  and close share `prob_kind`; the shared kind is persisted on a new `clv.prob_kind`
  column; the module docstring was corrected to match. New test covers the
  mixed-kind skip. (Continuous-yardage markets are de-vigged at both ends and are
  unaffected.)
- **Staking Monte-Carlo used independent draws per strategy** (`staking_mc.py`).
  The `flat`/`qkelly`/`shrunk` comparison now uses **common random numbers** —
  the same per-path/week correlated outcomes drive all three — so the drawdown/
  growth comparison is properly paired. Numbers shifted trivially; the qualitative
  story is unchanged (shrunk grows slower but ~10–15% p95 drawdown, zero ruin;
  breakeven bets nothing).

**Defensive guards (latent — not triggered by real data, fixed anyway):**

- `staking.recommend_stakes` now sanitizes non-finite/negative bankroll to a
  zero-dollar (still advisory) readout, treats NaN/inf or out-of-`[0,1]` `p` /
  `market_prob` / `price` as unstakeable (never propagating NaN into a stake), and
  clamps `dd_scale` to `[0,1]` so a stray value can't breach the caps.
- `report._lean_row_md` tolerates a present-but-`None` composite; `projection`'s
  survival functions return a finite `p_over` in `[0,1]` on a NaN mean/sd (never a
  spurious 1.0); `oddsmath.american_to_decimal(0)` no longer divides by zero; and
  `consensus_two_way`'s reported `n_books` counts only price-contributing books.
- Doc test counts corrected across `README.md`, `PROJECT_STATUS.md`, and this file.

**Older-code audit (Phase 1–6) fixes:**

- **Synthesis confidence cap** (`synthesis.py`). The post-client wrapper re-enforced
  the RISK→medium and stale→low confidence caps "no matter what the client
  decided," but not the `EXCLUDED`→low cap — a non-mock LLM client returning
  `EXCLUDED`+`high` would have survived. Now capped client-independently. (The
  core H6 contract — synthesis can never mutate a published number — was
  re-verified sound; this was a confidence-label gap only.)
- **Network-IO robustness** (`sources/_http.py`, `sources/oddsapi.py`). `get_json`
  gained a bounded retry-with-backoff and a parse/URL-error guard (raising a typed
  `HttpJsonError` the callers already degrade on); the unguarded `fetch_game_odds`
  / `fetch_scores` now degrade to empty results like `fetch_event_props`, and
  `_normalize_game_odds` uses `.get("price")` so a book omitting a price is skipped
  rather than crashing the live slate. Feeds now degrade loudly, never crash.
- **Notify footguns** (`notify.py`). `post_weekly`'s default flipped to
  `dry_run=True` (safe by default; the pipeline always passes an explicit value),
  and the live Discord POST is wrapped so a 5xx/timeout returns an error status
  instead of blocking the pipeline — with the webhook URL still never logged.
- **`context_study` mixed-dtype column**. `context_mult` defaulted untagged rows to
  `None`, yielding an object-dtype float/None column; now defaults to a `1.0`
  no-op multiplier (behavior-preserving for the score, clean dtype).
- **Defense-in-depth leakage tests.** Added end-to-end truncation-invariance guards
  (on the as-of-read values) for `AdvancedPack` (red-zone shares) and
  `ChemistryPack` (formation tilts); `ContextPack` was correctly skipped (its
  features are immutable DOB / strictly-before roster lookups already covered).
- Two doc/comment corrections in `prop_learning.py`: `load_adjustments`'s docstring
  now documents the intentional effective-at `<=` SQL (so no one "fixes" it to `<`
  and breaks it), and a misleading dead "flip for away teams" comment was removed.

**RAG / CLI audit fixes:**

- **SQL-whitelist validator bypass** (`rag/nl2sql.py`) — *the one critical (though
  latent) finding*. The table-whitelist regex only captured the identifier
  immediately after each `FROM`/`JOIN`, so a comma cross-join
  (`SELECT * FROM leans, sqlite_master`) slipped a non-whitelisted table past the
  check and could read `sqlite_master` / `api_credits`. Not exploitable with the
  shipped rule-based generator (it only emits safe single-table SQL), but
  `validate_sql` is by design the sole boundary for any future LLM SQL backend, so
  it was closed now: every table in every FROM/JOIN clause (including
  comma-separated) is validated, `sqlite_*` is explicitly denied, and the
  read-only guards are unchanged. Tests prove the bypass is rejected while
  legitimate single-table/JOIN queries still pass.
- **`weekly.py --season <no-games>`** raised `ValueError` on `max([])`; now returns
  cleanly with a "No games for that season" message. `tune_weights`'s walk-forward
  OOS integrity, both Monte Carlos, `build_ratings`, `backtest`, and CLI secret
  hygiene (no key/webhook ever printed) were audited and are clean.

No correctness-critical defect or leakage kill-bug was found in the actual scoring
path — the pipeline, report, dashboard, oddsmath core, projection distributions,
candidate adjustments, the learning loop (walk-forward + selection-bias guards
verified), the feature builders (all strictly prior-week), the RAG layer, the
Monte Carlos, and the weight-tuning were all audited across four rounds and are
clean. Every fix above is behavior-preserving on valid inputs and covered by a
test.

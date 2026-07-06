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

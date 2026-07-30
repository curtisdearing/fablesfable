<!--
HOW TO USE: These are two PRE-REGISTERED model challengers for the fablesfable_props
track, written 2026-07-30 (offseason). Run them in a data-equipped session (network +
`analysis/bootstrap_data.py` + `ml_test.py --stage frame` with the config feature subset
set to null). Open the repo, paste ONE challenger's section below the divider, and run it
to its verdict before touching the other. The GATES IN §A4/§B4 ARE FROZEN BY THIS COMMIT:
running a challenger and then adjusting its gate is protocol violation, not iteration.

Track bookkeeping (analysis/accuracy_protocol.json `acceptance`):
- Both challengers are fablesfable_props-track levers. ONE lever per checkpoint:
  Challenger A runs at the next checkpoint, Challenger B at the one after.
- The props track carries ONE standing consecutive rejection (pass_location,
  2026-07-30, book/loc_features_eval.json). If A and B both reject, that is three
  consecutive -> stop_after_consecutive_rejections trips and the props lever hunt
  pauses exactly as the game-line hunt did on 2026-07-30.
- The game-line track is STOPPED and neither challenger may touch it.
-->

---

# Build Prompts — Model challengers under gates (Bayesian projection · sequence model)

**North star (unchanged):** honest weekly value leans. These challengers exist to
improve two different links in that chain — **A** attacks the *distribution* every
P(over) is read from; **B** attacks the *features* the ranker sees. Neither may ship
a number that hasn't beaten the incumbent under its frozen gate.

## 0. Read first (both challengers)

- `docs/ACCURACY_PROTOCOL.md` + `analysis/accuracy_protocol.json` — acceptance rules
  (P(improvement) ≥ 0.90, paired season-week resampling, expected delta declared
  before the run, 2025 = locked single-touch benchmark, 2026 = prospective judge).
- `docs/HOW_A_PICK_IS_MADE.md` — where projections, distributions, and the ranker sit.
- `docs/decisions_p3-5.md` 2026-07-30 entries — the seeded-backtest determinism fix and
  the review-pass verification lessons (headless render; no unseeded MC verdicts).
- `analysis/loc_features_eval.py` — the harness pattern Challenger B must reuse
  (walk-forward A/B, both seeds, paired season-week bootstrap, book output).
- `nflvalue/gate_registry.py` — your verdict book gets a collector so the dashboard's
  Honest Record tab shows it, PASS or FAIL.

Post a short plan and wait for approval before coding.

## 0.1 Non-negotiables inherited by both (re-read `PHASE1_BUILD_PROMPT.md §1`)

- **No leakage.** Every input strictly pregame; sequences and pooled posteriors end at
  week w−1; `AsOfLookup` semantics everywhere; add mutation tests proving your new
  path fails when the guard is weakened (house precedent: tests/test_guard_mutations.py).
- **Determinism.** Fixed seeds end-to-end (data order, init, training). Two full runs
  must produce byte-identical eval artifacts, and a test must prove seed-stability at
  small scale (house precedent: tests/test_backtest_determinism.py).
- **Fail closed.** The incumbent path must be bit-identical when the challenger is
  absent, gate-failed, or errors. A missing artifact silently reverting to the
  incumbent is CORRECT; a missing artifact crashing the weekly run is a bug; a
  gate-failed artifact being consumed anywhere is a protocol violation with a test.
- **Free data only.** Nothing beyond the existing nflverse/Odds-API stack. CPU-only
  training; the weekly job must stay under ~2 min added wall time.
- **Negative results are first-class.** A FAIL writes the same book, decision-log
  entry, accuracy-ledger row, and gate-registry entry a PASS would.
- **Synthetic-line framing** applies to every hit rate you print. CRPS and log-loss
  are line-free and preferred wherever possible.

---

# Challenger A — Hierarchical Bayesian projection layer (distributions, not points)

## A1. Why this lever, declared before the run

Every P(over) is currently read from a parametric family (gamma / negative binomial /
Poisson) wrapped around a point projection with a heuristically scaled sd. The known
symptom is the O/U calibration asymmetry (means→median was measured and REJECTED —
the family's shape, not its center, is the suspect). Partial pooling
(player ⊂ role ⊂ team ⊂ league) should shrink small-sample players honestly and give
week-1 / post-trade players a principled prior instead of a cold-start gate.

**Pre-declared expected delta:** pooled walk-forward CRPS improves ≥ 1.5% relative
vs the incumbent distributions. (Protocol requires this line to exist before any run.)

## A2. Scope — build in order

**A2.1 `nflvalue/bayes_projection.py`** — numpyro (preferred; jax CPU) or PyMC model,
per market family: hierarchical location + scale with player-level random effects
partially pooled through position-role and team offense levels; season-block
covariates only from strictly-prior weeks. Fit walk-forward per eval season on
seasons < S (expanding). ADVI is acceptable if NUTS exceeds the compute budget —
declare which BEFORE the eval run and keep it fixed across seasons.

**A2.2 Predictive interface** — `predictive(player_id, market, season, week) ->
{quantiles, cdf(x), mean}` matching the incumbent distribution API so `p_over`
consumers swap without code changes. Persist per-season fitted artifacts
(`data/bayes_proj_{S}.json` or joblib ≤ 5 MB total) carrying their own metadata:
seasons trained, seed, inference method, package versions.

**A2.3 `analysis/bayes_projection_eval.py`** — the gated A/B:
per player-week-market with a realized actual, score incumbent vs challenger
predictive distributions by **CRPS on the actual stat value** (line-free), pooled
walk-forward over eval seasons 2021–2024. Paired season-week cluster bootstrap
(n=4000, seed=20260730 — reuse the existing `_paired_p` pattern). Also recompute
`p_over` from the challenger distribution and rerun the UNCHANGED ranker to measure
the downstream guard metrics. Writes `book/bayes_projection_eval.json`.

**→ CHECKPOINT: show the book with per-market and pooled CRPS + the gate line, and wait.**

**A2.4 Integration (ONLY on gate PASS)** — config flag `projection.bayes` default
false; when true and the artifact validates, `projection.py` reads the hierarchical
predictive; otherwise byte-identical incumbent behavior. Add the gate_registry
collector, decision-log entry, ledger row either way.

## A3. Out of scope

Game-line anything (track stopped). TD/exact markets stay fail-closed regardless of
CRPS (exact-market approval is a separate, harder gate). No calibration layers on
top (already rejected 2026-07-16); the posterior IS the calibration claim.

## A4. FROZEN GATE (pre-registered 2026-07-30)

Ship the challenger distribution only if ALL hold on the walk-forward 2021–2024 pool:

1. **Primary:** pooled CRPS(challenger) < CRPS(incumbent) with
   P(improvement) ≥ 0.90 under the paired season-week bootstrap, at BOTH seeds
   (7, 1234) of the fitting pipeline.
2. **Guard (downstream ranker, unchanged features):** top-5 hit rate drop ≤ 0.1pp
   and log-loss increase ≤ +0.0005 when p_over is fed from the challenger.
3. **Reproducibility:** two identical-seed runs byte-identical books.

On PASS: one single 2025 holdout look (CRPS + guards), reported verbatim, no
re-tuning after the look. On FAIL: record everything, revert nothing (flag stays
false), stop — do NOT try alternative priors/likelihoods in the same checkpoint;
that is how preregistration dies.

## A5. Tests & definition of done

Leakage: posterior for (S, w) invariant to poisoning weeks ≥ w (mutation-style test).
Fail-closed: missing/corrupt artifact -> incumbent path bit-identical (golden test).
Shrinkage sanity: a 2-observation player's predictive sd strictly wider than a
100-observation player's at equal role. Book internally consistent
(gate.passed ⇔ shipped flag), collector renders it, suite green from fresh clone.

---

# Challenger B — Sequence model over game logs + entity embeddings (learned features)

## B1. Why this lever, declared before the run

The ranker's ~50 features are hand-rolled EWMs/rolls of the same game logs a sequence
model could read raw. The hypothesis is NOT "deep learning beats GBDT on tabular"
(it usually doesn't at n≈74k — that naive version is expected to fail); it is that a
small recurrent encoder over the trailing-16-game log, with learned player/team/
opponent-defense embeddings, captures temporal shape (usage trajectory, role change
momentum) that fixed-span EWMs cannot, and that its representation helps THE EXISTING
GBDT as features. pass_location's rejection (2026-07-30) sharpened this: single
hand-derived channels are priced; learned temporal representations are the untested
class.

**Pre-declared expected delta:** pooled walk-forward log-loss −0.003 (the
player_depth_rank scale) for B1.

## B2. Scope — build in order

**B2.1 `nflvalue/seq_encoder.py`** — torch (CPU), GRU (1–2 layers, hidden ≤ 64) over
each candidate's trailing 16 player-games (per-game stat vector + home flag + rest
days + opponent-defense embedding ≤ 16 dims + team embedding ≤ 8 dims; player
embedding ≤ 16 dims). Sequence strictly ends at week w−1 (reuse AsOfLookup index
construction). Head: predict next-game stat vector (self-supervised — it never sees
lines or labels the ranker is graded on). Trained walk-forward per eval season on
seasons < S. Fixed seeds; `torch.use_deterministic_algorithms(True)`; single thread.
Artifact ≤ 5 MB per season, carrying its own metadata.

**B2.2 Variant B1 (run first): embeddings as GBDT features** — extract the encoder's
final hidden state (≤ 8 PCA-reduced dims, PCA fit on train seasons only) as
`seq_h0..seq_h7` candidate features; A/B exactly like `analysis/loc_features_eval.py`
(lean set vs lean+8), same seeds, same bootstrap, writing
`book/seq_features_eval.json`.

**→ CHECKPOINT: show B1's book and gate line, and wait.**

**B2.3 Variant B2 (run ONLY if B1 fails, and it is the LAST attempt of this
checkpoint): direct p_over head** — small MLP on [encoder state ‖ line features]
trained on y_over, evaluated as a full ranker replacement under the same harness.
Pre-registering B2 now, conditional on B1's failure, is what makes running it
legitimate; inventing further variants after two failures is not. If B1 passes, B2
is not run (one shipped lever per checkpoint).

**B2.4 Integration (ONLY on gate PASS)** — B1: features join the config lean list
with provenance, artifact-carried feature list (the existing mechanism), retrain via
the normal Tuesday path; encoder artifact validated at load, missing/corrupt ->
features NaN -> GBDT handles them (fail-safe, tested). B2 (if ever): behind
`ml_ranker.model = "seq"` config with the GBDT as automatic fallback.

## B3. Out of scope

Transformers over play-by-play tokens (compute), any market/line input to the
encoder (leakage surface for the self-supervised task), game-line usage, paid data.

## B4. FROZEN GATE (pre-registered 2026-07-30)

Per variant, walk-forward 2021–2024 pool, ranker metrics:

1. **Primary:** pooled log-loss delta < 0 with P ≥ 0.90 (paired season-week
   bootstrap) at BOTH seeds (7, 1234).
2. **Guard:** top-5 hit rate drop ≤ 0.1pp.
3. **Reproducibility:** identical-seed reruns byte-identical.

PASS -> one single 2025 holdout look, verbatim, then integrate per B2.4.
B1 FAIL -> run B2 once under the same gate. B2 also FAIL -> that is the props
track's third consecutive rejection: record both books, stop the props lever hunt,
and update `hot.md`/ledger to say so.

## B5. Tests & definition of done

Sequence leakage: encoder output for (S, w) invariant to poisoning games ≥ w.
Determinism: same-seed encoder training twice -> identical extracted features (small
fixture). Fail-safe: absent artifact -> NaN features -> pipeline + dashboard run
green (golden test). PCA fit proven train-only (poison eval season, projection
unchanged). Book/collector/ledger/decision-log written on either verdict. Suite
green from a fresh clone with `FABLESFABLE_STRICT_FIXTURES=1`.

---

## Session checklist for whoever runs these

1. `analysis/bootstrap_data.py` (network session), then rebuild the full-width frame:
   set `ml_ranker.features` to null, `ml_test.py --stage frame --seasons 2019 ... 2025`,
   restore config (see 2026-07-30 worklog for the exact dance).
2. Run ONE challenger to verdict. Books + registry + ledger + worklog, PASS or FAIL.
3. Ship via the proven path: branch -> bundle in `drops/` -> osascript push/PR/merge
   from the owner's Mac (see vault memory note; the sandbox cannot push directly).
4. Do not start the other challenger in the same checkpoint.

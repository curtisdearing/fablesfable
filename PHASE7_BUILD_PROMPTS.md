<!--
HOW TO USE THIS FILE
- This is the Phase 7 plan: the sequence of jobs that finishes the project,
  each routed to the model best suited to it. Paste ONE job block at a time
  into a FRESH session of the model named on that block.
- Every job follows the house protocol: read the named docs -> POST A PLAN
  and wait for approval -> build in small commits -> hit the CHECKPOINT and
  wait. Do not skip the plan/approve/checkpoint gates.
- Branch: cut `phase7` from `phase6` once `phase6` is pushed/merged. Small
  commits per sub-item, same as Phase 6.
- The "why this model" line on each job is the routing rationale, not a rule
  you must justify again to the user — just proceed on the tagged model.
- Source of truth for everything below: docs/decisions_p6.md (what shipped and
  why), docs/AUDIT_stats_context_2026-07-03.md (the gaps), docs/DATA_SOURCES.md
  (the paywall boundary), PREMORTEM.md (the kill conditions). Read them.
-->

# Phase 7 — finish the project: probability model → real lines/CLV → correlation → staking

Phases 1–6 built a disciplined, walk-forward NFL prop engine that grades at
**synthetic** trailing-mean lines. Checkpoint 2 (docs/decisions_p6.md) settled
two things that define Phase 7:

1. **Selection is already at its frontier** — accuracy now has to come from the
   *probability model*, not from re-slicing which candidates become bets.
2. **The binding constraint on every number in this repo is that grading is
   synthetic, not real.** Real sportsbook lines price most of what the model
   knows; the only accepted proof of edge is forward CLV. Phase 7 turns the
   engine into a machine that can *prove or disprove* its own edge, sizes bets
   honestly if it does, and states plainly what it cannot claim.

Phase 7 completes the project across four threads — probability quality, the
real-line/CLV referendum, same-game correlation, and staking — then closes with
the honest go/no-go framework.

---

## Model routing principle

**Opus** owns jobs whose core is *deciding what is correct*: statistical
methodology, calibration inside a walk-forward loop, estimating and using a
correlation structure from small samples, Kelly sizing under estimation error,
leakage judgment on a novel pipeline, and the final honesty assessment. Errors
in these are **silent and expensive** — no test catches a subtly-leaked
calibration fit or an over-aggressive bet size until it costs money.

**Sonnet** owns jobs whose core is *implementing a pinned spec*: search grids,
ablations, API plumbing, dashboards, schedulers, tests, docs. The design is
already fixed by the preceding Opus job, and **tests catch the errors.**

Every job below inherits ALL Phase 1–6 non-negotiables (restated once under
"Shared non-negotiables"). The split doubles as cost discipline: Opus only
where judgment is load-bearing.

---

## Sequence & dependencies

| # | Job | Model | Depends on | Gate after |
|---|---|---|---|---|
| 7.1 | Probability calibration + quality-metric spec | **Opus** | phase6 | — |
| 7.2 | Ensemble + walk-forward tuning + feature pruning | **Sonnet** | 7.1 | **Checkpoint A** |
| 7.3 | Real-line capture + re-label + CLV harness — design | **Opus** | 7.2 | — |
| 7.4 | Live line capture + CLV dashboard + scheduler — build | **Sonnet** | 7.3 | **Checkpoint B** |
| 7.5 | Same-game correlation modeling | **Opus** | 7.2 | — |
| 7.6 | Correlation-aware selection + reporting | **Sonnet** | 7.5 | **Checkpoint C** |
| 7.7 | Staking / bankroll module | **Opus** | 7.5, 6.8 | — |
| 7.8 | End-to-end integration, tests, docs | **Sonnet** | 7.1–7.7 | **Checkpoint D** |
| 7.9 | Go/no-go framework + honest project assessment | **Opus** | all | ship |

7.3–7.4 (live/CLV) and 7.5–7.6 (correlation) are independent branches after
7.2 — run them in either order, or in parallel sessions. 7.7 needs 7.5's
correlation output. 7.8 and 7.9 close everything out.

---

## Shared non-negotiables (apply to EVERY job)

- **Walk-forward, always.** Every new feature, calibration fit, correlation
  estimate, and hyperparameter choice at time T uses only data strictly before
  T. Extend tests/test_leakage.py to cover it. The calibration and correlation
  fits are new leakage surfaces — guard them explicitly.
- **Measured, not guessed.** Any constant, weight, or threshold is fit from this
  project's own history with printed provenance (a `scripts/fit_*.py` that
  prints the number), and ships only if it clears the significance bar it's
  measured against (the t≥2 / BH-q<0.05 culture from Phase 6). If it doesn't
  clear, say so and don't ship it — an honest "no effect" is a valid result.
- **Synthetic-line honesty.** Until real prices accrue, every hit rate / unit
  figure is vs the synthetic trailing-mean line and must be labeled as such.
  Do not let a synthetic-line number masquerade as a real-line or profit claim.
- **Narrative gate holds.** No context tag influences the composite or the ML
  features until it clears n≥100 + BH-q<0.05 AND a human promotes it in config
  (context_study). Phase 7 does not relax this.
- **No new paid data sources** without an explicit, approved justification.
  Real prop lines come from the Odds API free tier (500 credits/mo, hard-capped
  at 450), already integrated. The paid FTN API remains the first justified
  purchase *if and only if* live CLV proves edge worth funding — that's a 7.9
  decision, not a silent import.
- **The tool never places a bet, moves money, or initiates a transfer.** It
  sizes, ranks, and recommends. Staking output is advisory. This is absolute.
- **Protocol:** read the named docs → post a plan → wait for approval →
  build in small commits → show the checkpoint deltas → wait.

---

## 7.1 — Probability calibration + quality-metric spec  ·  **OPUS**

> Why Opus: calibrating a classifier *inside* a walk-forward loop is a leakage
> trap (the calibrator must never see the fold it corrects), and choosing +
> validating the method (isotonic vs Platt vs beta, pooled vs per-market) is a
> judgment call that every downstream job depends on. Get this wrong silently
> and the edge math, correlation weighting, and Kelly sizing all inherit the
> error.

**Objective.** Make the GBDT's output probabilities *trustworthy as
probabilities*, and pin the single probability-quality metric suite that jobs
7.2 onward optimize against. Today the ranker is tuned for ordering (AUC ~.64)
and log-loss, but nobody has checked whether P(over)=0.62 actually hits 62% —
and edge, correlation weighting, and Kelly all assume it does.

**Read first:** docs/decisions_p6.md (ML sections), nflvalue/ml_ranker.py,
ml_test.py, nflvalue/composite.py (how edge consumes model_prob).

**Scope (what/why; the how is yours):**
- Audit calibration walk-forward: reliability curves + Expected Calibration
  Error + Brier decomposition, pooled AND per-market AND by probability decile,
  on the out-of-sample season predictions. Establish whether miscalibration is
  real and where it concentrates.
- Design and wire a calibration layer fit **strictly walk-forward** (the
  calibrator for season/week T trains only on < T; prove no fold sees its own
  correction). Choose the method from the audit, not by default — isotonic
  needs sample size, Platt/beta are steadier on thin per-market slices; justify
  the choice with numbers.
- Specify the **probability-quality metric suite** (the objective 7.2 targets):
  log-loss + ECE + per-market Brier, with the walk-forward evaluation protocol.
  Write it into docs/decisions_p7.md so 7.2 optimizes a fixed target, not a
  moving one.
- Confirm calibrated probabilities flow into composite edge unchanged in
  interface (edge = calibrated P(side) − de-vigged market prob).

**Checkpoint:** reliability curves before/after, ECE + log-loss deltas
(pooled + per-market), and the written metric spec. Wait.

**Done when:** calibration is wired walk-forward with a leakage test; the audit
is reproducible from a script; the metric suite is documented; and you've
stated honestly whether calibration moved anything or the raw GBDT was already
well-calibrated (a real possibility — report it either way).

---

## 7.2 — Ensemble + walk-forward tuning + feature pruning  ·  **SONNET**

> Why Sonnet: once 7.1 fixes the metric and the walk-forward protocol, this is
> disciplined search and ablation execution — mechanical, high-volume, and
> fully test-guarded. No open methodology questions remain.

**Objective.** Squeeze the probability model against 7.1's fixed metric:
ensemble the models, search hyperparameters walk-forward, and prune dead
features — keeping only changes that measurably improve calibrated probability
quality out-of-sample.

**Read first:** docs/decisions_p7.md (7.1's metric spec), ml_test.py,
nflvalue/ml_ranker.py, tune_weights.py (the walk-forward search convention).

**Scope:**
- Ensemble GBDT + RF (both already implemented) — test a simple average and a
  walk-forward-fit logistic meta-learner over their out-of-sample probabilities.
  Ship the ensemble only if it beats the best single model on 7.1's metric OOS.
- Walk-forward hyperparameter search (learning rate, leaves, min-leaf, L2,
  iters) with each season's config chosen only from prior seasons — mirror
  tune_weights.py's honesty contract exactly; report OOS, never in-sample argmax.
- Feature pruning by walk-forward ablation: drop features that don't move the
  metric; report the kept/dropped list with per-feature deltas. Watch the
  Phase-6 additions specifically (depth/location, rz shares, durability,
  opp_absence) — keep them only where they earn it, drop honestly where they
  don't.
- Refit + save the production artifact; keep it gitignored/regenerable.

**Checkpoint (CHECKPOINT A — accuracy):** before/after calibrated log-loss +
ECE + Brier (pooled + per-market), hit-rate + top-1 at the frozen selection
policy, kept/dropped feature table. Wait.

**Done when:** every shipped change beats baseline on 7.1's OOS metric; the
search + ablation are reproducible from scripts; the artifact refits on the
2-core sandbox; results are honestly reported including anything that didn't help.

---

## 7.3 — Real-line capture + re-label + CLV harness — design  ·  **OPUS**

> Why Opus: this is the referendum's integrity. Migrating training labels from
> the synthetic line to the real closing line without look-ahead, deciding what
> "closing" means per book, and specifying CLV timing are subtle validity
> questions where a quiet mistake fakes an edge that isn't there. The existing
> augment_with_real_lines path is unproven — its correctness is the whole game.

**Objective.** Design the pipeline that lets the model be graded and
*re-trained* against real sportsbook prices instead of synthetic lines, and
specify the CLV accrual + kill-check monitoring that is the project's actual
referendum. No live keys needed to design it; fixtures stand in.

**Read first:** docs/decisions_p3-5.md (Block A — odds, CLV, kill-check),
docs/DATA_SOURCES.md (paywall boundary), ml_test.py (augment_with_real_lines),
nflvalue/oddsapi_props.py, nflvalue/clv.py, nflvalue/killcheck.py, PREMORTEM.md.

**Scope (design + spec; 7.4 builds it):**
- Specify real prop-line capture: which markets/books, snapshot cadence
  (open → close), storage schema, and the exact definition of the "closing"
  snapshot per the existing budget cap (450 credits/mo). Reuse Block A where it
  exists; name what's missing.
- Design **label migration**: as real lines accrue, graded leans with a real
  line get re-labeled (y = actual vs real line) with line-dependent features
  recomputed — walk-forward, no look-ahead, synthetic label only where no real
  line exists. Pin down exactly when a row flips and prove it can't leak.
- Specify the CLV harness: de-vigged-probability CLV per lean, the
  two-snapshot resolution rule, and the kill-check (n≥150 resolved, avg CLV>0,
  ≥52% positive-CLV rate → GO, else the premortem's "revert to entertainment"
  language). This already partly exists — make it complete and monitorable.
- Write the whole design into docs/decisions_p7.md as the spec 7.4 implements.

**Checkpoint:** the written design + a worked example on recorded/synthetic
fixtures showing a row correctly re-labeled and a CLV correctly computed. Wait.

**Done when:** the design is unambiguous enough for 7.4 to build without further
judgment calls, every leakage/look-ahead question is answered in writing, and
the kill-check thresholds are the pre-committed Phase-3 ones (not moved).

---

## 7.4 — Live line capture + CLV dashboard + scheduler — build  ·  **SONNET**

> Why Sonnet: plumbing to 7.3's pinned spec — API hardening, storage, a
> dashboard tab, scheduled jobs. Fixture-testable, no design latitude.

**Objective.** Implement 7.3's spec: capture real lines within budget, run the
re-labeling + CLV pipeline, surface it, and schedule the in-season cadence.

**Read first:** docs/decisions_p7.md (7.3's spec), the Block A modules named in
7.3, scripts/auto_weekly.py, dashboard.html / nflvalue/dashboard.py.

**Scope:**
- Harden Odds API prop-line capture to the spec (budget-aware rotation already
  exists — extend, don't rewrite); store open + close snapshots.
- Implement the re-labeling pipeline and CLV accrual exactly per 7.3; wire the
  kill-check to run on the accrued sample.
- Add a CLV / kill-check dashboard tab: rolling CLV, positive-CLV rate,
  resolved-n vs the 150 gate, GO/NO-GO banner.
- Install the in-season scheduled jobs (line pull, CLV resolve, weekly grade)
  guarded to no-op in the offseason, as the existing scheduler does.
- Everything degrades loudly to fixtures when no key/slate exists (the repo has
  no live key and it's the offseason — prove the math on fixtures).

**Checkpoint (CHECKPOINT B — live infra):** dry-run against recorded/synthetic
fixtures showing capture → re-label → CLV → kill-check end to end, dashboard
rendering, scheduler self-detecting. Wait.

**Done when:** the pipeline runs on fixtures with tests, the dashboard shows the
referendum state, the scheduler is installed and offseason-safe, and no number
is presented as real that came from a synthetic fixture.

---

## 7.5 — Same-game correlation modeling  ·  **OPUS**

> Why Opus: estimating a correlation structure among a game's props from limited
> history, shrinking it so small samples don't produce fake structure, and
> deciding how it feeds selection vs parlay pricing are genuine statistical
> judgment. A naïve correlation estimate is worse than none.

**Objective.** Measure how a game's props move together (a QB's passing over,
his WR's receiving over, the team total, the game script) and produce a
shrunk, walk-forward correlation structure that selection (7.6) and staking
(7.7) can consume.

**Read first:** docs/decisions_p6.md (game-script + red-zone wiring — the
mechanisms that create correlation), nflvalue/candidates.py,
nflvalue/composite.py, nflvalue/projection.py.

**Scope (what/why):**
- Estimate pairwise correlation of prop *outcomes* (and/or residuals vs
  projection) within a game, by market-pair type (QB-pass ↔ WR-rec,
  RB-rush ↔ team script, two WRs same team, etc.), walk-forward and
  **shrunk toward zero** so thin pairs don't invent structure. Report which
  pairs carry real, stable correlation and which are noise.
- Decide and document how the structure is *used*: at minimum, a
  correlation-aware view for selection (don't treat two correlated leans as
  independent edges); optionally, a same-game-parlay joint-probability
  estimate. Recommend scope honestly — SGP pricing is only worth building if
  the correlations are stable enough to price.
- Expose the structure as a clean, walk-forward artifact (leakage-tested) for
  7.6/7.7 to read. No selection/staking behavior changes in THIS job — this
  job measures and exposes; 7.6 wires.

**Checkpoint:** the measured correlation table (by pair type, with n and
shrinkage), a plain statement of what's real vs noise, and the exposed
artifact's interface. Wait.

**Done when:** correlations are measured walk-forward with shrinkage and a
leakage test, the "real vs noise" call is explicit, and the artifact is ready
for consumption without further design.

---

## 7.6 — Correlation-aware selection + reporting  ·  **SONNET**

> Why Sonnet: wire 7.5's artifact into the selection path and surface it —
> execution to a pinned spec, test-guarded.

**Objective.** Make the shortlist correlation-aware per 7.5, and show the user
why (so a top-5 isn't secretly five bets on one game outcome).

**Read first:** docs/decisions_p7.md (7.5's output + intended use),
nflvalue/shortlist.py, nflvalue/composite.py, nflvalue/game_notes.py.

**Scope:**
- Implement the correlation-aware selection rule 7.5 specified (e.g. discount or
  cap correlated leans within a slip / per game), walk-forward and deterministic.
- Surface correlation in the report/panel (flag when leans are correlated).
- If 7.5 green-lit SGP pricing, expose the joint estimate as a labeled,
  optional readout — never as a synthetic-line "edge."
- Ablate: does correlation-aware selection change the OOS hit/units at the
  frozen policy? Report honestly; keep it only if it helps or is neutral-with-
  better-diversification (state which).

**Checkpoint (CHECKPOINT C — correlation):** before/after selection deltas,
example slips showing the de-correlation, any SGP readout. Wait.

**Done when:** selection consumes the correlation artifact, the report shows it,
tests cover it, and the effect is honestly characterized.

---

## 7.7 — Staking / bankroll module  ·  **OPUS**

> Why Opus: bet sizing is where a good model still goes broke. Kelly under
> estimation error, correlation (from 7.5), and a drawdown constraint informed
> by the 6.8 variance envelope is subtle math with ruinous downside if
> mis-set. This needs judgment about shrinkage, caps, and honesty — not a
> textbook Kelly formula dropped in.

**Objective.** Turn edges + probabilities into *advisory* stake sizes that
respect estimation error, correlation, and a hard drawdown tolerance — sized so
the 6.8 variance envelope is survivable, and clearly labeled as recommendations
the tool never acts on.

**Read first:** docs/mc_brain_eval.md (the 6.8 variance/streak/drawdown
envelope), scripts/mc_brain.py, 7.5's correlation artifact, nflvalue/composite.py.

**Scope (what/why):**
- Implement fractional-Kelly sizing off calibrated edge, **shrunk for
  estimation error** (the edge is an estimate; size as if it's smaller than it
  looks), with correlation-adjusted sizing when multiple leans hit the same
  game (7.5), and a max-per-bet + max-drawdown cap tied to the 6.8 envelope.
- Show the bankroll implications: expected growth, drawdown p95, P(ruin) at the
  chosen fraction — reuse the 6.8 Monte Carlo machinery, run at *plausible
  real-line* edges, NOT the synthetic 66–68% (which compounds to fiction; 6.8
  already makes this point — honor it).
- Output is advisory only: a recommended unit size per lean, with the risk
  readout. **No bet is ever placed.** State this in the module and the report.

**Checkpoint:** the sizing rule with its shrinkage + caps, and the bankroll
Monte Carlo at plausible edges (growth / drawdown / ruin). Wait.

**Done when:** staking is deterministic, shrunk, correlation- and drawdown-
aware, advisory-only, tested, and every projection is at real-line-plausible
edges with the synthetic caveat stated.

---

## 7.8 — End-to-end integration, tests, docs  ·  **SONNET**

> Why Sonnet: well-specified consolidation — full-pipeline smoke test across all
> new pieces, extend the leakage suite, update the docs. The design is done.

**Objective.** Prove the whole Phase-7 pipeline runs end to end, lock it with
tests, and bring the docs current so the project is genuinely finishable.

**Read first:** docs/decisions_p7.md (all prior 7.x specs), tests/ (existing
suite), README.md, docs/DATA_SOURCES.md, docs/HOW_A_PICK_IS_MADE.md.

**Scope:**
- End-to-end smoke test: enumerate → calibrated ensemble → correlation-aware
  selection → staking readout → (fixture) real-line capture → CLV → kill-check,
  on a recorded slate. One test that would catch a regression anywhere in the
  chain.
- Extend tests/test_leakage.py to every Phase-7 surface (calibration fit,
  correlation estimate, real-line re-label, any new rolling feature).
- Update README + HOW_A_PICK_IS_MADE + DATA_SOURCES to describe the finished
  system truthfully, including the synthetic-vs-real distinction and the CLV
  referendum.
- Full suite green; note runtime so it stays within the sandbox slices.

**Checkpoint (CHECKPOINT D — integration):** green suite, the end-to-end test
output, doc diffs. Wait.

**Done when:** the pipeline runs start to finish on fixtures, leakage tests
cover every new surface, docs match reality, and the suite is green.

---

## 7.9 — Go/no-go framework + honest project assessment  ·  **OPUS**

> Why Opus: the highest-judgment, most-honesty-critical job. This defines the
> evidence that flips the tool from "entertainment" to "staked," and writes the
> plain-language account of what the finished project can and cannot claim.
> This is where the project either stays intellectually honest or quietly
> oversells itself.

**Objective.** Write the pre-committed decision framework and the honest final
assessment that complete the project.

**Read first:** everything in docs/ (especially decisions_p6, decisions_p7,
mc_brain_eval, PREMORTEM), nflvalue/killcheck.py.

**Scope (what/why):**
- The go/no-go framework: the exact, pre-committed evidence that moves the tool
  from paper to staked — CLV thresholds (n, avg, positive rate), a minimum live
  duration, and what NO-GO triggers (revert to entertainment, stop staking).
  Nothing here is a synthetic-line number; the referendum is CLV.
- The paid-data decision: whether/when the FTN API purchase is justified, tied
  explicitly to CLV evidence — the one place a paid source may enter, and only
  then.
- The honest assessment: what Phase 1–7 actually built, what it demonstrably
  does (walk-forward synthetic-line skill, calibrated probabilities, quantified
  variance), and what it does NOT yet prove (real-line profit) — the chain
  from synthetic skill → real-line hit rate → profit, with the variance reality
  from 6.8. No overclaiming.
- Ship docs/decisions_p7.md's closing section + a top-level project status the
  user can act on.

**Checkpoint:** the framework + assessment for review. Wait — this one ships the
project, so the user reads it in full before it's final.

**Done when:** the go/no-go thresholds are pre-committed and unambiguous, the
paid-data trigger is defined, and the assessment is one an honest skeptic would
sign — stating the limits as plainly as the strengths.

---

## Closing note

Run them in order (7.1 → 7.9), honoring the four checkpoints. The two middle
branches (7.3–7.4 live/CLV, 7.5–7.6 correlation) can go in parallel Opus/Sonnet
sessions after 7.2 if you want to move faster. Keep the house rules: measured
constants with provenance, walk-forward everything, synthetic-line honesty, the
narrative gate, no paid data without the 7.9 trigger, and the tool never places
a bet. When 7.9 ships, the project is complete in the only sense that matters
here — not "guaranteed to win," but *honest about exactly what it knows and
built to find out the rest.*

# 2026 accuracy protocol

`analysis/accuracy_protocol.json` is the machine-readable source of truth. This
document explains how a model change earns promotion. A plausible story is not
evidence and a better retrospective chart is not a release decision.

## Truth windows and scorecards

- Develop with expanding, season-forward folds through 2024. The 2025 season is
  a locked regression benchmark inspected only at a declared checkpoint. It has
  already been audited and must not be described as untouched. Unmodified 2026
  prospective predictions are the final judge.
- Fablesfable reports log loss, Brier score, ECE, overconfidence ECE, top-k hit
  rate, and exact selection n. Real entry-to-close CLV is the market scorecard:
  mean probability CLV, share beating the same-side consensus close, resolved n,
  and unresolved n.
- Tailstail reports MAE/RMSE by position, Spearman rank correlation, interval
  coverage and width, and distribution scores when samples are retained.
- Public consensus is an external challenger until source, retrieval timestamp,
  scoring rules, player coverage, and redistribution rights are recorded. It is
  never silently substituted for missing truth.

## One-lever experiment contract

Before a run, add an immutable ledger entry containing hypothesis, affected
track, expected delta, feature clock, sample definition, compute budget, and
accept/reject gate. Fantasy and props may run one lever each in parallel.
Anything that changes the shared projection core serializes both tracks and
requires both suites. Three consecutive rejected single levers trigger a stop
and one preregistered ensemble/blend challenger; they do not justify searching
more combinations on the benchmark.

Suggested initial deltas are hypotheses, not promised gains:

| Lever | Expected useful delta | Primary gate |
|---|---:|---|
| Ranker calibration | ECE -0.005 | no log-loss/Brier regression |
| Lean feature subset | log loss -0.002 | paired season-week improvement ≥90% |
| Absence cascade | top-5 hit +0.5 pp | matched control and 2025 checkpoint |
| Fantasy availability split | MAE -0.05 | paired season-week improvement ≥90% |
| Fantasy market shrinkage | MAE -0.05 | public baseline provenance complete |
| TD hurdle component | MAE -0.03 | no interval undercoverage |

After every candidate run, execute `analysis/sanity_diff.py BASE CANDIDATE`. The
top ten must retain at least 50% identity overlap unless the ledger predicts and
explains the churn. Side flips, rank changes, additions, removals, and score
deltas are review alarms—not proof of accuracy.

## Calibration and forward CLV

ECE uses ten fixed probability bins. Overconfidence ECE only accumulates bins
where mean confidence exceeds observed frequency, making unjustified certainty
visible. Calibration is evaluated on held-out probabilities, never training
fits. A calibrator must improve its preregistered calibration target without
materially worsening discrimination or log loss.

Synthetic lines are `floor(strictly-prior trailing player mean) + 0.5` (with
anytime TD fixed at 0.5). They support trend and regression tests only; no profit,
ROI, market-edge, or CLV claim may be derived from them. Forward CLV compares a
recorded entry with the same side at cross-book consensus close. The kill check
requires 150 resolved entries, mean CLV above zero, and at least 52% beating the
close; unresolved entries are counted but make no claim.

## Factor and matched-control gate

An exposed cohort needs at least 100 observations and a matched unexposed cohort
needs at least 100. Match position/market, depth role, season phase, and favored,
neutral, or underdog game script. Cluster uncertainty by team-season, apply
Benjamini-Hochberg q<0.05 across the registered family, and require a later
season-forward replication. If any required field is absent, the result is
research-only.

Birthday, revenge, homecoming, primetime, stadium, referee and similar narrative
factors remain shadow features under the same gate. Even after passing, their
live effect is capped at 3% until prospective replication. Absence and roster
cohorts must include both short-notice outs and structural long-term vacancies;
they are separate estimands.

## Freeze, checkpoints, and failure modes

Use `docs/FROZEN_PROTOCOL_TEMPLATE.md` for every accepted release. Freeze the
Week 1 protocol by 2026-09-05. After Week 8, perform a bye-week data/schema,
calibration, drift, and compute audit; retraining does not waive promotion
gates. Explicit risks are NFL rule changes, offseason roster churn, vendor
schema changes, subjective lever selection, simulation cost growth, and stale
market/player identifiers. Fail closed when identity, clock, or provenance is
ambiguous.

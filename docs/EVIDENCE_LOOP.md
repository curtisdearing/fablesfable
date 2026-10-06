# Evidence loop: from settled results to research decisions

`analysis/evidence_loop.py` is the cumulative result-to-research loop. It reads
immutable issued/offered-line records and native final outcomes, scores them,
and screens at most two registered mechanism challengers under criteria frozen
in `analysis/research_registry.json`. It changes no production weight,
calibration, default, line or delivered card, and it makes no odds/API call.

## What it measures, kept apart

| Quantity | Where | Rule |
|---|---|---|
| Forecast calibration | `score_arms` | Brier, log loss, ECE, reliability bins, per-market n; bootstrap resampling whole games (and season-weeks), unavailable below 5 clusters |
| Point/performance error | settlement grader (`nflvalue/issued_grading.py`) | not recomputed here |
| Hypothetical ROI | `flat_roi` | flat 1u at the actual offered decimal price; `None` when the source has no price receipt |
| Actual account P/L | `account_pl` | always `None` until a ticket ledger exists |

Coverage is complete: every received row lands in exactly one of `scored`,
`push`, `void`, `pending`, `missing_probability`, `not_in_scope` (e.g. PASS rows),
and in exactly one chronology window.

## Chronology

| Window | Rule |
|---|---|
| `prospective_confirmation` | 2026 week >= 5, explicit-offset decision clock strictly before kickoff, immutable capture sha256 |
| `retrospective_exploratory` | every earlier season and 2026 weeks 1-4 (already read; never an untouched holdout) |
| `excluded_late` | decision at/after kickoff |
| `unknown_clock` | no machine decision clock (hand-graded chat cards); never prospective |

Naive timestamps are `unknown_clock`, never assumed UTC.

## Registry and gates

`research_registry.json` keeps every prior verdict (C2 rejected, agentB window
rejected, incumbent/game-line no-edge, C1m active under its own prospective v3
protocol) next to new candidates. `criteria_sha256` covers the protocol block and
every candidate's `criteria`; editing a gate after the freeze makes
`validate_registry` fail and `run` refuse (exit 2). Status changes are an outcome
log and do not alter the hash. At most two candidates may be
`active_prospective`. A screening pass only makes a candidate eligible for
prospective confirmation; nothing is promoted by this tool.

## First execution (2026-10-06, criteria frozen 04:07:37Z, run 04:07:46Z)

Results in `analysis/evidence_loop_results.json`. All retrospective/exploratory.

* **Offered prop lines, 2026 W1-2 (307 events, 20 games):** reproduces the
  2026-10-05 receipt exactly: Brier incumbent .2680, C1m .2656, C2 .2680,
  market consensus .2475; flat 1u at offered prices incumbent 137-134 −12.08u,
  C1m 138-131 −8.86u, C2 142-135 −8.36u. No arm approaches the market.
* **Delivered MNF card (ATL@NO, 2026-10-05):** 2 recommendations 1-1,
  −0.115u hypothetical at the issued prices, 1 PASS counted and not scored, all
  three `unknown_clock`. Account P/L unknown.
* **QBROLE-TOTAL-v1** (QB-role change state vs real closing total, nflverse
  schedules): fit 2024 beta +0.047 (wrong sign); 2025 dBrier +0.00096
  [−0.00293, +0.00494] on 272 games (106 exposed). **Retired.**
* **OPPFORM-TOTAL-v1** (as-of opponent-adjusted scoring form vs closing total):
  2025 dBrier +0.00003 [−0.00029, +0.00034]. **Retired.**

No statistically supported edge was found. The closing total already carries
both mechanisms at game level.

## Settlement → analysis integration contract

The settlement worker calls, after `nflvalue.issued_grading.grade()`:

```python
from analysis.evidence_loop import append_evidence, from_issued_grading
append_evidence(EVIDENCE_LEDGER, from_issued_grading(grade_output))
```

or writes `grade_output` to JSON and runs
`python -m analysis.evidence_loop run --settlement-grade grade.json ...`.
Required fields per row of `sections.recommendations_given.rows`: `record_id`,
`season`, `week`, `game_id`, `market`, `side`, `line`, `pick_class`, `tier`,
`decision_ts` and `kickoff` (explicit offsets), `settlement`
(win/loss/push/void/unresolved), `actual`, `quote_price` (decimal), `quote_book`,
`model_p_side`. The ledger is sha256-chained JSONL: identical rows are skipped,
a changed row (stat correction) is appended as a revision, and a tampered line
fails `read_ledger`.

## Commands

```bash
python -m analysis.evidence_loop freeze           # print criteria hash + registry errors
python -m analysis.evidence_loop run --historical <dir with lines_extra/pbp/injuries parquet> \
    [--real-lines real_lines_rows.csv.gz] [--issued-grade grade.json --card-game-id 2026_04_ATL_NO] \
    [--settlement-grade issued_grade_output.json] --ledger evidence.jsonl --output report.json
pytest -q tests/test_evidence_loop.py
```

Prospective confirmation needs pre-kickoff captured, settled rows for 2026
week >= 5; none exist in the inputs above, and the report says so.

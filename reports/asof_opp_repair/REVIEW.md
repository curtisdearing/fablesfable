# Repair-only review change: live opponent-vs-role factor (2026-10-05, receipt v2)

Branch `repair/asof-opp-factor-20261005`, base = deployed `44de45e`. Commits above base, in order:
`8cfbde5` (the repair + tests + first receipts) → `ca7d470` (review doc v1) → `2398d51` (receipt script records
commit, dirty state, code and input hashes) → this commit (regenerated receipts + this document). The branch carries
ONLY the serving repair, its tests, the receipt runner and receipts. C1, C2, pick records, protocols and the
prospective capture live on `shadow/oct4-postmortem-20261005`. Not merged, not deployed. **Owner authorization and
exact-head required CI are prerequisites for any merge.**

## Defect

`features.build_opp_pos_def` emits rows only for played weeks, so for a live week `candidates.enumerate_candidates`
found no `(season, week, opp, role)` row and `projection.project` ran every yards market with `opp_factor = 1.0`,
while every backtest and the dispersion/football-only validations applied the prior-weeks factor (0.72–1.16 on 2026
W4). The first repair (`ca7c495`, on the shadow branch) built as-of rows only when the whole week had no opponent
row; independent review showed that once one game completed, all 293 remaining fixture rows dropped to `missing`.

## Diff vs `44de45e` (7 files, +631 / −2 at `2398d51`; this commit adds only receipt JSON and this file)

| file | change |
|---|---|
| `nflvalue/features.py` (+47) | `build_opp_pos_def` = raw aggregation + `_add_rolling_opp_features` (moved verbatim); new `asof_opp_pos_def(opd, season, week, defteams)` — NaN-raw placeholder rows for the target week, identical rolling/league-prior/factor code, strictly-prior history |
| `nflvalue/candidates.py` (+35 / −2) | `missing_keys = {(opp, role) ∉ opp_idx}`; as-of rows built for exactly those defteams and used only for those keys; played rows keep precedence; yards rows stamped `opp_source ∈ {played, asof, missing}` and `opp_roll_games` (0 = league-prior neutral, distinct from missing); counting markets `None` |
| `tests/test_asof_opp_factor.py` (+205, 7 tests) | equality as-of vs as-played (2020 W8, all defteams × roles); prior-weeks poison; live path carries the factor; completed week unchanged; **mixed week**; **sparse role**; completed week equals baseline factors |
| `analysis/asof_opp_parity_receipt.py` (+140) | three-scenario receipt runner; records executed commit, dirty tracked files, sha256 of `features.py` / `candidates.py` / `projection.py` / itself, sha256 of every input parquet (symlinks resolved) |
| `reports/asof_opp_repair/parity_2020w08_fixture.json`, `parity_2026w04_real.json` | receipts (below) |
| `reports/asof_opp_repair/REVIEW.md` | this document |

Production code touched: `features.py`, `candidates.py` only. No selection, stake, model, state-asset or schema
migration change; `opp_source` / `opp_roll_games` are additive columns on in-memory candidate rows.

## RED / GREEN (behavioural)

Against `ca7c495` (first repair): the current test file → **2 failed, 5 passed**;
`test_mixed_week_keeps_asof_rows_for_the_unplayed_games` fails by assertion `{'missing': 293}` (the reviewer's
construction); `test_sparse_role…` fails on the new `opp_roll_games` column. Against `44de45e` the file cannot import
`asof_opp_pos_def`. On `8cfbde5` and later: **7 passed**. Independent re-review on `ca7d470`: 7 passed.

## Parity receipts (regenerated at `2398d51`; `required_invariant_passes: true` in both)

Both receipts name commit `2398d51`, code hashes `features.py b28472da…`, `candidates.py b1c394fa…`,
`projection.py b4c87443…`, and the input hashes (fixture `pbp_2019_2020 c58e1a2b…`, `schedules_2019_2020
7b9a25fd…`; real `historical_pbp b5e4ed93…`, `pbp_2024 b9fee1bd…`, `pbp_2025 5abe85ae…`, `pbp_2026 3911a7cc…`,
`lines_extra ad7a3eb7…`). The real-data receipt's dirty list contains only the fixture receipt JSON written by the
preceding run in the same session; no code or input file was dirty. The reviewer's own receipts at `ca7d470`
(`~/.hermes/profiles/football-genius/artifacts/oct4-shadow-repair-rereview-20261005/`) report identical counts and
the same code hashes for `features.py` / `candidates.py`.

| data | scenario | yards rows | played | as-of informed | as-of neutral | missing | factor comparisons | max abs diff |
|---|---|---|---|---|---|---|---|---|
| 2020 W8 fixture | wholly unplayed | 314 | 0 | 314 | 0 | 0 | 314 | 4.9e-5 |
| 2020 W8 fixture | after ATL–CAR | 314 (293 remaining) | 21 | 293 | 0 | 0 | 314 | 4.9e-5 |
| 2020 W8 fixture | fully played | 192 | 192 | — | — | 0 | 192 | 4.9e-5 |
| 2026 W4 real | wholly unplayed | 600 | 0 | 600 | 0 | 0 | 523 | 5.0e-5 |
| 2026 W4 real | after PIT–CLE (Thu) | 600 | 36 | 564 | 0 | 0 | 523 | 5.0e-5 |
| 2026 W4 real | fully played | 193 | 193 | — | — | 0 | 193 | 5.0e-5 |

The 5e-5 is the card's 4-decimal rounding of `components.opp_factor`. **Zero missing forecast factors is verified;
complete as-played comparison coverage is not:** 2026 W4 compares 523 of 600 unplayed rows because the cached
play-by-play lacks two week-4 games (no as-played row to compare against). No neutral factors occurred; when they
do, `opp_roll_games == 0` names the reason and the receipt lists the keys.

## Tests: exact commands and exclusions

Repair branch at `2398d51` (tests and production code identical to this head):

```
pytest -q tests -p no:cacheprovider -W ignore -rs --deselect tests/test_backtest_smoke.py --deselect tests/test_club_report_wiring.py
→ 1247 passed, 5 skipped, 5 deselected in 130.67s
```

Clean baseline `44de45e` (same interpreter, same data links):

```
pytest -q tests -p no:cacheprovider -W ignore -rs --deselect tests/test_backtest_smoke.py
→ 1 failed, 1242 passed, 5 skipped, 2 deselected in 113.79s
```

Exclusions and reasons, identical on both heads:
- deselected `tests/test_backtest_smoke.py` (2 tests): runs the full historical backtest, minutes-long, no behaviour under change;
- deselected on the repair head only `tests/test_club_report_wiring.py` (3 tests): its persistence case
  `test_club_statuses_and_estimates_persist_under_the_run_id` **fails on the clean baseline** (`reed["verified"] is True`
  assertion), so it was run on the baseline (1 failed, 2 passed) and excluded from the repair run to keep the counts
  comparable; it is unrelated to this change;
- skipped (5, same on both): `test_bayes_projection.py:24` (numpyro not installed), `test_seq_encoder.py:16` (torch not
  installed), `test_explain_cards.py:207`, `:225` and `test_golden_output.py:125` (`ml_frame.parquet` build artifact absent).

Net: +7 passed (the new file), −2 (club-report tests excluded), otherwise the same population. Required CI
(`.github/workflows/ci.yml`, full `pytest -q tests/` on Python 3.11 with the committed fixtures) has not run on this
head; it runs when the branch is pushed.

## Effect on live numbers

Every live yards forecast changes toward the validated path (2026 W4 live replay, pbp cut before W4: Hampton rushing
yards 58.6 → 49.5, Worthy receiving yards 34.2 → 39.7, Cousins passing yards 214 → 180, Purdy 231 → 189). Correctness
repair; no accuracy or profit claim; selection and stake posture unchanged. Counting markets (attempts, receptions,
anytime TD) are untouched by construction (`use_opp_factor=False`).

## Rollback plan (narrowed to what was tested)

- Code: `git revert <repair commit>` restores `44de45e` behaviour. What was tested: for a fully played week every
  yards row is `played` and its `opp_factor` equals the full-history table (`test_completed_week_matches_the_baseline_path`);
  this is opponent-factor equality on completed weeks, **not** a byte-for-byte comparison of every candidate field
  against an independently executed baseline run. The two new columns are additive; nothing reads them yet.
- Published artefacts: a repaired Wednesday/T-90 run that has already been published stays published and recorded
  (`published` events are append-only, keyed to that run's `code_sha`); reverting code does not and must not rewrite
  them. After a rollback the next scheduled run produces pre-repair forecasts under its own `code_sha`; the hub shows
  both runs with their shas. If a repaired board must be withdrawn before kickoff, the existing path is a new run
  (new `run_id`, revision + 1 superseding the earlier records), never deletion.
- Pre-merge gate suggested for the owner: replay the most recent published week with
  `analysis/asof_opp_parity_receipt.py --season 2026 --week <W>` and keep the receipt beside the publication; a
  non-zero `missing` count or `max_abs_diff > 1e-4` blocks the run.

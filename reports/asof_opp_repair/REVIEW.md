# Repair-only review change: live opponent-vs-role factor (2026-10-05)

Branch `repair/asof-opp-factor-20261005` = deployed `44de45e` + one commit `8cfbde5`. Contains ONLY the serving repair,
its tests and its receipts. C1, C2, the pick-record module, protocols and reports stay on
`shadow/oct4-postmortem-20261005` (which carries the same repair commit's files at `7730611`). Not pushed, no PR,
nothing deployed. **Stop for review.**

## Defect

`features.build_opp_pos_def` emits rows only for played weeks, so for a live week `candidates.enumerate_candidates`
found no `(season, week, opp, role)` row and `projection.project` ran every yards market with `opp_factor = 1.0`,
while every backtest and the dispersion/football-only validations applied the prior-weeks factor (0.72–1.16 on 2026
W4). The first repair (`ca7c495`) built as-of rows only when the whole week had no opponent row; independent review
showed that once one game completed, all 293 remaining fixture rows dropped to `missing` / 1.0.

## Change (6 files, +532 / −2)

- `nflvalue/features.py` (+47): `build_opp_pos_def` split into raw aggregation + `_add_rolling_opp_features` (moved
  verbatim); new `asof_opp_pos_def(opd, season, week, defteams)` — placeholder rows with NaN raw stats for the target
  week, identical rolling/league-prior/factor code, strictly-prior history, returns target-week rows.
- `nflvalue/candidates.py` (+35 / −2): `missing_keys = {(opp, role) not in opp_idx}`; as-of rows built for exactly
  those defteams and used only for those keys; played rows keep precedence; every yards row stamped `opp_source`
  (`played | asof | missing`) and `opp_roll_games` (0 = league-prior neutral, a justified neutral distinguishable
  from a missing row). Counting markets carry `None`.
- `tests/test_asof_opp_factor.py` (7 tests): as-of == as-played equality (2020 W8, all defteams × roles); prior-weeks
  poison; live path carries the factor; completed week unchanged; **mixed week** (prior weeks + first completed game,
  player/team inputs constant: completed game `played`, every other game `asof` with factors equal to the as-played
  table and unchanged from the wholly-unplayed enumeration); **sparse role** (a defteam with no prior TE rows → as-of
  TE row present, `roll_games 0`, factor 1.0, never `missing`; other roles informed); completed week equals baseline.
- `analysis/asof_opp_parity_receipt.py` + `reports/asof_opp_repair/parity_*.json`: receipts below.

## RED / GREEN (behavioural, not import failures)

Against `ca7c495` (first repair), the new test file: **2 failed, 5 passed** —
`test_mixed_week_keeps_asof_rows_for_the_unplayed_games` fails by assertion `{'missing': 293}` (the reviewer's
construction exactly); `test_sparse_role…` fails on the new `opp_roll_games` column.
Against `44de45e`: the file cannot import `asof_opp_pos_def` (the defect predates any as-of row).
On `8cfbde5`: **7 passed**.

## Partial-week parity receipts (`required_invariant_passes: true` in both)

| data | scenario | yards rows | played | asof informed | asof neutral | missing | max abs diff vs as-played table |
|---|---|---|---|---|---|---|---|
| 2020 W8 fixture | wholly unplayed | 314 | 0 | 314 | 0 | 0 | 4.9e-5 |
| 2020 W8 fixture | after ATL–CAR completed | 314 (293 remaining) | 21 | 293 | 0 | 0 | 4.9e-5 |
| 2020 W8 fixture | fully played | 192 | 192 | — | — | 0 | 4.9e-5 |
| 2026 W4 real | wholly unplayed | 600 | 0 | 600 | 0 | 0 | 5.0e-5 |
| 2026 W4 real | after PIT–CLE (Thu) completed | 600 | 36 | 564 | 0 | 0 | 5.0e-5 |
| 2026 W4 real | fully played | 193 | 193 | — | — | 0 | 5.0e-5 |

The 5e-5 is the card's 4-decimal rounding of `components.opp_factor`. 2026 W4 compares 523 of 600 unplayed rows: the
other 77 face the two week-4 games not yet in the play-by-play cache (no as-played row to compare). No neutral
factors occurred on either dataset; when they do, `opp_roll_games == 0` names the reason (no prior games for that
defteam/role, league prior by design) and the receipt lists the keys.

## Baseline-aware test results

Deployed baseline `44de45e`: `tests/test_club_report_wiring.py::test_club_statuses_and_estimates_persist_under_the_run_id`
fails on a clean checkout (pre-existing, unrelated). Repair branch full suite (same deselections):
**1243 passed, 9 skipped, 5 deselected** (the four extra skips are data-dependent tests that resolved to the committed fixture because the cache links were added mid-run; they pass against the cache on the shadow branch). Targeted suites touched by the change (serving-skew, football-only, leakage, projection,
golden output, explain cards, accuracy-review regressions): green on the first repair (115 passed, 3 skipped) and
re-run here within the full suite.

## Effect on live numbers

Every live yards forecast changes toward the validated path (2026 W4: Hampton rushing yards 58.6 → 49.5, Worthy
receiving yards 34.2 → 39.7, Cousins passing yards 214 → 180, Purdy 231 → 189). Correctness repair; no accuracy or
profit claim; selection and stake posture unchanged.

## Rollback plan

Single commit, additive schema (`opp_source`, `opp_roll_games` on candidate rows; no DB migration, no state-asset
change): `git revert 8cfbde5` restores `44de45e` behaviour exactly; played-week outputs are byte-identical before and
after (completed-week test), so no published artefact needs regeneration on rollback. If merged before a Wednesday
run, replay the previous published week with `analysis/asof_opp_parity_receipt.py --season 2026 --week <W>` and keep
the receipt beside the publication; a non-zero `missing` count or `max_abs_diff > 1e-4` blocks the run.

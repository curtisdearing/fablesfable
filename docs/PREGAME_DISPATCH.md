# Pregame (T-90) dispatch runbook

GitHub starts this repository's scheduled runs anywhere from on time to 5.6 h
late, so the cron entries in `live-weekly.yml` cannot be relied on to land a
T-90 refresh inside `[kickoff - 90 min, kickoff)`. The dependable trigger is a
`workflow_dispatch` with `job=t90` sent inside that window.
`scripts/pregame_dispatch.py` sends exactly one such dispatch, for exactly one
named game, and reads the result back.

Nothing here installs a scheduler. Someone, a person or an agent that is awake
on a machine with an authenticated `gh`, has to run the command inside the
window.

## Modes

| mode | network | effect |
|---|---|---|
| (default) dry run | none | validates identity and timing (`--at` to simulate), prints the plan and the activation command |
| `--check` | read-only (GitHub API, state release download to a temp dir, ESPN scoreboard) | readiness report; allowed before the window, refused at or after `kickoff - min-lead` |
| `--execute` | read + one `gh workflow run` | all checks, local one-shot lock, dispatch, wait (bounded) and read back |
| `--readback RUN_ID` | read-only | what that run actually did |
| `--fallback-after RUN_ID` | read + at most one `gh workflow run` | re-dispatch only if RUN_ID failed before its `Run weekly job` step |

## Refusals

These are decided before any network call:
- game id not `SEASON_WW_AWAY_HOME` for the given season/week;
- kickoff without a timezone;
- now before `kickoff - 90 min - early-minutes` (default early-minutes 0);
- now after `kickoff - min-lead-minutes` (default 35);
- now at or after kickoff. A run started then would be retrospective, and its output must not be called pregame.

## Readiness checks

`--execute` dispatches only if all of these pass:
- `remote_main_sha`: remote `main` equals `--expect-sha`, the SHA the release owner deployed.
- `ci_green_on_sha`: the Tests workflow has a successful run on that SHA and no unsuccessful one.
- `workflow_active`, `workflow_dispatch_t90`: `live-weekly.yml` is enabled and offers `job=t90` at that SHA.
- `no_active_production_run`: no queued or running run of `live-weekly.yml` or `publication-ingest.yml`, the `nfl-live-production` group.
- `no_prior_dispatch_this_window`: no `workflow_dispatch` run of `live-weekly.yml` since one hour before the window.
- `processed_state_guard`:
  - The current production state is checksum-verified, restored into a temp dir and read read-only.
  - There must be zero `leans` rows with `clock='t90'` for the game.
  - This is the same processed-state guard `job_t90` uses.
- `official_kickoff`: the ESPN week scoreboard lists away@home at exactly `--kickoff`, with status `STATUS_SCHEDULED`.

## Duplicate protection

There are four layers, and none replaces the workflow's own guard:
- the prior-dispatch and active-run checks above;
- a local lock `<receipt-dir>/dispatch-<game>.lock` created with `O_EXCL` (fallback uses `<receipt-dir>/fallback/`);
- `job_t90` resnaps odds and processes only games without `t90` leans;
- the serialized `nfl-live-production` concurrency group.

## Fallback cannot duplicate odds pulls

`job_t90` requests the closing odds before it runs the game, and it publishes state only if the whole run succeeds. So a run that failed *inside* `Run weekly job` may already have paid for odds that no published state records. A second run would pull them again.

`--fallback-after` therefore refuses if:
- the earlier run reached that step, whatever the step's conclusion;
- the earlier run is still running;
- the state already shows the game processed.

It re-dispatches only when the failure happened earlier (checkout, install, history, state restore). Anything else is a human decision.

## Read-back

The dispatch call returning 0 is not treated as success. The run counts as `processed` only if all of these hold:
- the run concluded `success` on the expected SHA;
- the gate chose `job=t90`;
- the log contains `[auto] t90 <game>:` and no `FAILED` line for the game;
- the release pointer names this run's `state-<id>-<attempt>` archive;
- that archive has `t90` lean rows for the game.

## What stays time-dependent

These can only be verified at run time:
- the final official source;
- inactives (the ESPN inactives endpoint has returned 404; that availability hold is separate);
- live prices;
- the published site.

A processed T-90 run does not by itself make every card executable.

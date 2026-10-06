# Pregame (T-90) dispatch runbook

GitHub starts this repository's scheduled runs anywhere from on time to 5.6 h
late, so the cron entries in `live-weekly.yml` cannot be relied on to land a
T-90 refresh inside `[kickoff - 90 min, kickoff)`. The dependable trigger is a
`workflow_dispatch` with `job=t90` sent inside that window.
`scripts/pregame_dispatch.py` sends exactly one such dispatch, for exactly one
named game, and reads the result back.

`pregame_dispatch.py` installs no scheduler; `scripts/pregame_scheduler.py` is the
one that runs it (see "Scheduler" below). Whatever runs it must be awake on a machine
with an authenticated `gh` inside the window.

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

## Scheduler

`scripts/pregame_scheduler.py` decides only *when* to call `--execute`; every check above
still applies. Each tick (every 5 minutes):

- reads the ESPN scoreboard for the current and next regular-season week (scheduled games only);
- groups kickoffs at most 40 min apart into one slot, since one `job_t90` run processes every
  game due within 90 minutes and the wrapper refuses a second dispatch within an hour of a
  window. Sunday 4:05 + 4:25 ET is one dispatch, sent when the 4:25 window opens;
- launches the wrapper, detached, for the slot's earliest game from `dispatch_at`
  (latest kickoff - 90 min) until `last_launch` (earliest kickoff - 40 min), unless the
  wrapper's lock already exists or an attempt is still running. Refused and not-ready attempts
  retry on the next tick;
- passes the whole slot to the wrapper (`--slot-games`): readiness and read-back cover every
  member game, and the slot counts as processed only when each game has its own
  `[auto] t90 <game>: N voided` line, no `FAILED` line and `t90` leans in the state this run
  published. The processed-state guard refuses only when *every* slot game already has `t90`
  leans (reported as `already processed`, not as a miss); `job_t90` skips processed games and
  resnaps only unprocessed ones, so a partially processed slot is never pulled twice;
- validates the ESPN payload: season/type/week from the root, else `leagues[0].season`
  (`type` is an object there), else each event's own `season`/`week`. A payload that cannot
  be resolved is an error in the heartbeat, never an empty week;
- persists every slot's plan, so a slot is settled even after its games leave the
  scoreboard (kicked off while the Mac slept, ESPN rolled the week, scoreboard unreadable):
  `dispatched and processed`, `already processed`, `superseded` (kickoff moved into another
  slot, or postponed/canceled), or `MISSED` with the evidence (the last decision, or "no tick
  between dispatch_at and last_launch; previous tick ...");
- notifies once per slot when a dispatch did not process, or the slot closed undispatched
  (including "never attempted" when the machine was asleep or off);
- treats a stored PID as the running wrapper only if that PID's command is
  `pregame_dispatch.py` (after a restart the PID may belong to something else);
- writes `state/heartbeat.json` (tick time, previous tick, released SHA, board errors, next
  slots).

Health, read-only (no lock, no sync, no network, no writes); exit 1 on a stale heartbeat
(> 15 min), a future heartbeat (clock skew), a board error on the last tick, an unknown SHA, or
a slot in the last 8 days that ended in anything but processed / already processed /
superseded:

    python3 ~/fablesfable-ops/runner/scripts/pregame_scheduler.py --ops-dir ~/fablesfable-ops --health

### Host limits

The scheduler is a LaunchAgent: it ticks only while the user is logged in and the Mac is
awake. A sleeping, closed-lid or powered-off Mac runs nothing; launchd coalesces the missed
intervals into one tick on wake, and that tick reports every slot whose window passed as
`MISSED`. Nothing here wakes the Mac. If a window must be covered, keep the Mac awake and on
power through it (for example `caffeinate -s` in a terminal, or a one-off
`sudo pmset schedule wake "MM/DD/YY HH:MM:SS"` before `dispatch_at`; both are operator
choices, not installed by this script). A healthy installed agent is not evidence that a
future slot will be processed: only the slot's read-back is.

Known limit: kickoffs 41-100 minutes apart (e.g. a 7:15 + 8:15 PM ET Monday doubleheader)
form separate slots, and the wrapper's `no_prior_dispatch_this_window` check (any
`workflow_dispatch` since one hour before the window) refuses the second one after the first
dispatch; the second slot then ends `MISSED` and must be dispatched by hand. Weeks 5-6 of 2026
have no such pair.

Install on macOS as a LaunchAgent: `scripts/install_pregame_scheduler.sh` (default ops
directory `~/fablesfable-ops`: `runner/` checkout fast-forwarded to `origin/main` each tick,
`receipts/`, `logs/`, `state/`). Remove with `--uninstall`. Plan without launching:
`python3 scripts/pregame_scheduler.py --no-sync --dry-run`.

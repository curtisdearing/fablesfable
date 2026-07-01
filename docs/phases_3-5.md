# Phases 3–5 — live lines + CLV, weekly automation + Discord, RAG: how to run it

Everything below assumes Phase 1/1B/2 setup (see `docs/phase1.md`). No new
Python dependencies were added for Phases 2–5 — `requirements.txt` is
unchanged except for comments.

**Leans, not locks.** Every surface this pipeline produces is research, not
advice, and says so. Gambling problem? **1-800-GAMBLER**.

---

## The weekly loop (in season)

```bash
# WEDNESDAY ~10:00 — provisional slate run (live feeds + budgeted odds pull)
python3 pipeline_weekly.py --season 2026 --week 3 --mode live --live-odds

# GAME DAY, ~90 min before each kickoff — final availability gate per game
python3 pipeline_weekly.py --season 2026 --week 3 --clock t90 --game 2026_03_CLE_BAL

# AFTER the slate — resolve closing-line value + kill-check
python3 pipeline_weekly.py --season 2026 --week 3 --resolve-clv
```

What each run does:

| Run | Effect |
|---|---|
| `--mode live` (wed) | features → availability (ESPN) → projections → budgeted odds pull (rotating ≤4 games) → composite (edge where a real line exists, `no_market` otherwise) → top-5/game with "5 of N screened" → synthesis context panel → `reports/props_week_{S}_{W}.md` + `data/weekly_props.json` + `leans` table → dashboard **Weekly Leans** tab → optional Discord |
| `--clock t90 --game X` | re-pulls injuries + per-event `active` flags for that game, **auto-voids** Wednesday leans on OUT/inactive players (`leans.status='voided'` with reason), re-ranks the game without them, writes `reports/props_week_{S}_{W}_t90_{game}.md`, refreshes dashboard |
| `--resolve-clv` | approximates each lean's close (last `lines` snapshot ≤ kickoff), writes the `clv` table, prints rolling CLV + the kill-check verdict |
| `--mode historical` | replays a completed week from the parquet (candidate set = players who recorded usage; features still strictly prior-week). Live feeds are marked "not applicable", lines are synthetic (†), edge is `no_market`. |

Freshness gate: in live mode, stale/missing injuries (>36h) or an empty feed
sets `publish=false` — the report renders with a **NOT PUBLISHED** banner and
Discord gets at most a gate notice. That is working as intended; fix the feed,
rerun (runs are idempotent — same week+clock overwrites itself).

Kill-check any time: `python3 -m nflvalue.killcheck` (verdict is
INSUFFICIENT_SAMPLE until ~150 leans have resolved CLV; then GO / NO_GO with
the pre-committed stop-staking language).

## The Odds API (Block A)

- Put your free key in the environment (`ODDS_API_KEY=...`) or
  `config.local.json` — **never** in `config.json` (tracked).
- Budget: `config.json → odds_budget {monthly_credits: 500, reserve: 50}` —
  the client hard-stops at 450/month (ledger in the `api_credits` table;
  API-reported usage headers override local estimates). Per event call the
  cost is `markets × regions` (5 × 1 by default).
- `max_prop_games_per_run` (default 4) caps each pull; rotation is
  least-recently-pulled-first, so coverage cycles across weeks. Un-pulled
  games are tagged `no_market` and ranked on confidence + matchup only.
- Calibration gate: `config.json → composite.params.calibration_passed`.
  Set `false` to force everything to `no_market` (e.g., if a future
  re-calibration fails); edge only ranks when this is `true`.

## Discord (Block B — off by default)

1. Create a **private** webhook in your own server (Server Settings →
   Integrations → Webhooks).
2. Provide it via env `DISCORD_WEBHOOK_URL=...` **or** create
   `config.local.json` (gitignored):
   `{"discord_webhook": "https://discord.com/api/webhooks/..."}`
3. Flip `"discord_enabled": true` in `config.json`.
4. Add `--discord` to the pipeline run. Default is a **dry run** (builds the
   embeds, posts nothing); add `--discord-live` to actually post.

Every message carries the disclaimer + 1-800-GAMBLER footer. `publish=false`
weeks post a gate notice, never picks. Personal and unmonetized by design —
no affiliate links, nothing monetized, and the notifier contains no wagering
code of any kind.

## Scheduling

Cron (adjust timezone; NFL Wednesdays + a game-day loop):

```cron
# Wednesday 10:00 provisional run (in season: Sep–Jan)
0 10 * 9-12,1 3  cd /path/to/repo && python3 pipeline_weekly.py --season $(date +\%Y) --week $WEEK --mode live --live-odds --discord >> logs/wed.log 2>&1

# Game days: check every 15 min; a small wrapper decides which games are ~T-90
*/15 10-22 * 9-12,1 0,1,4,6  cd /path/to/repo && python3 scripts/t90_wrapper.py >> logs/t90.log 2>&1

# Tuesday cleanup: resolve CLV for last week
0 9 * 9-12,1 2  cd /path/to/repo && python3 pipeline_weekly.py --season $(date +\%Y) --week $LASTWEEK --resolve-clv >> logs/clv.log 2>&1
```

`$WEEK` derivation and the T-90 wrapper (compare `gameday`+`gametime` from the
schedules parquet to now, fire `--clock t90 --game ...` once per game) are
deliberately left as a 20-line site-specific script — kickoff timezones and
hosting differ per machine. Alternatively, in Cowork you can ask Claude to
"run my Wednesday props pipeline every Wednesday at 10am" and it will create a
scheduled task that shells out to the same commands.

## RAG query layer (Block C)

```bash
python3 -m nflvalue.rag.nl2sql "show me the leans for week 10 2023"
python3 -m nflvalue.rag.nl2sql "what's our average CLV"
python3 -m nflvalue.rag.nl2sql "which leans were voided in 2023"
```

Output always shows the generated SQL, the tables it read, the row count, and
an answer composed only from returned rows. Safety: SELECT-only, single
statement, no comments, table whitelist (`leans`, `lines`, `clv`,
`player_week`, `opp_pos_def`, `projections`, `prop_backtest`,
`manual_notes`), and a structural row cap (`config.json → rag.row_cap`, 200).

Semantic recall over past reports is flag-gated OFF
(`rag.vectorstore_enabled`) and uses a dependency-free TF-IDF index over
`reports/*.md`; enable it and call `nflvalue.rag.vectorstore.search("...")`.

## Manual context notes

Insert rows into `manual_notes` (season, week, scope `player|team|game`,
ref = gsis id / abbr / game_id, tag, note) and they appear in the context
panel — **display-only, never scored** (`weight` exists in the schema for
compatibility and is ignored by ranking on purpose).

## Tests

```bash
python3 -m pytest tests/ -q          # full suite (~139 tests)
# or in chunks (tight shell timeouts):
python3 -m pytest tests/test_leakage.py tests/test_reproducibility.py -q
python3 -m pytest tests/test_projection.py tests/test_positions.py tests/test_backtest_smoke.py -q
python3 -m pytest tests/test_freshness.py tests/test_sleeper.py tests/test_availability.py tests/test_synthesis.py -q
python3 -m pytest tests/test_shortlist.py tests/test_report_phase2.py -q
python3 -m pytest tests/test_oddsapi_props.py tests/test_clv_killcheck.py tests/test_rag.py -q
python3 -m pytest tests/test_pipeline_weekly.py tests/test_notify_secrets_app.py -q
```

Guardrail coverage highlights: monthly budget can't be exceeded (simulated
month), de-vig/CLV math against hand-computed numbers, T-90 voids an inactive
player end-to-end, freshness halt blocks publishing, hostile LLM client that
edits a projection raises `SynthesisContractViolation`, SQL injection/DDL/
non-whitelisted tables rejected, no secrets in tracked files, and the
existing game-line app (`build_ratings.py`, `backtest.py`, `run.py`,
`weekly.py`) still compiles and the dashboard still renders legacy payloads.

# Phases 3–5 (+ Phase 2) — autonomous-run decision log

Defaults chosen without approval gates, per the autonomous build prompt. Each is
reversible; config keys are noted where one exists.

## State correction (biggest call of the run)

- **Phase 2 was NOT merged when the Phase 3–5 prompt arrived** — the repo had
  just passed Checkpoint 1B-b. The prompt's State line assumed
  `composite.py`/`shortlist.py`/`report.py` existed. Decision: build Phase 2
  first (its plan was already user-approved), skip its show-and-wait
  checkpoint under autonomous mode, then Blocks A→C. Branches:
  `phase1b-part2` → `phase2` → `phases3-5`, each merged to `main`.

## Phase 2 modeling defaults

- **Candidate set, historical weeks:** players with a `player_week` row at
  (season, week) — the reviewed `prop_backtest.py` convention. Features remain
  strictly prior-week; only the SET uses week-W participation.
  **Live weeks:** `roster_mode="carry_forward"` (players from each team's most
  recent prior week; zero week-W information). Honest cost: debuts/trades are
  invisible until availability/live rosters trim or add.
- **Game script:** from the nflverse schedule's pre-game `spread_line`
  (verified: positive = home margin, e.g. 2023_01_DET_KC = 4.0 with KC home)
  through the existing `game_script_multipliers`. The Monte-Carlo hookup can
  replace it live; the spread is deterministic, pre-game information.
- **Per-market SD:** std of walk-forward residuals over all weeks strictly
  before the target (what `prop_backtest`'s expanding SD converges to at that
  cutoff); `DEFAULT_SD_FRACTION` fallback below 30 residuals
  (`sd_source` tags which one was used).
- **Synthetic line:** player's trailing 8-game mean (shift-1, min 3), snapped
  `floor(x)+0.5` so it can't push; tagged `synthetic_trailing_mean` and
  rendered with a † everywhere. Synthetic lines never mint an edge
  (`no_market`).

## Composite (config.json `composite`)

- Weights 0.5 edge / 0.3 confidence / 0.2 matchup; with no market the
  remaining two renormalize (0.6/0.4 effective) and the row is tagged
  `no_market`. Edge cap 0.10 (a 10-point prob edge = full component), z cap 2.
- **Calibration gate** `params.calibration_passed` (default **true**, based on
  the Phase 1B Checkpoint 1B-a calibration fix being reviewed and accepted).
  Setting false forces every candidate to no_market behavior — the switch the
  Phase-3 hard rule demands.
- **anytime_td is YES-only** and confidence credit is zeroed whenever the
  model's own side probability is < 50%; `low_confidence` markets take a
  ×0.8 composite penalty. (Found in the first 2023-wk10 render: the top-5s
  were degenerate "no-TD" unders — untradeable leans. Fixed, with a
  regression test.)
- Max **2 leans per player** per game (`shortlist.max_per_player`) so
  correlated markets (rec yds + receptions) can't fill a top-5.

## Block A — odds, CLV, kill-check

- **Budget:** ceiling = 500 − 50 reserve = **450 credits/month**
  (`odds_budget`), persisted in the `api_credits` table; cost model =
  markets × regions per event call; the events listing is treated as free
  (documented Odds API behavior) but the reserve absorbs drift; API
  `x-requests-used` headers override local estimates when present.
- **Rotation:** least-recently-pulled game first (from `lines` history),
  capped by `max_prop_games_per_run` (4). Skipped games run `no_market`.
- `lines` rows key on the book's **player_name** (a book string may not match
  a gsis id; unmatched rows are stored with `player_id=NULL`, visible, and
  can never mint an edge). Name→gsis matching is exact-normalized (+
  first-initial variant), else None — never fuzzy-guessed.
- **CLV:** compared in de-vigged probability space (consensus across books at
  each snapshot); `anytime_td` (one-sided) uses raw implied probability,
  marked `prob_kind="raw_implied"`. Close = last snapshot ≤ kickoff; a lean
  needs two distinct snapshots to resolve. Points may move between entry and
  close; `point_moved` is recorded and the prob-space comparison is the
  headline metric.
- **Kill-check:** n ≥ 150 resolved leans; GO iff lifetime avg CLV > 0 AND
  positive-CLV rate ≥ 52%; otherwise NO_GO with the spec's pre-committed
  "revert to entertainment tool, stop staking" language. No API key in this
  environment (and July = offseason), so **current status:
  INSUFFICIENT_SAMPLE (0 resolved)** — fixtures prove the math instead.
- A real odds-api payload could not be recorded (no key, no in-season events);
  `tests/fixtures/oddsapi_event_props_synthetic.json` is SYNTHETIC, labeled,
  shaped per the documented v4 response.

## Block B — pipeline, dashboard, Discord

- Dashboard gets a **new "Weekly Leans" tab**; the existing "Player Props"
  tab (old game-line app EV props) is untouched, and the legacy payload still
  renders (tested).
- **Two clocks:** `wed` = provisional full-slate run; `t90 --game` re-pulls
  availability + per-event actives, **auto-voids** leans whose player is
  OUT/inactive (`leans.status='voided'`, reason + provenance), re-ranks that
  game without them, writes a t90 addendum report, refreshes the dashboard.
  RISK (Questionable) players are kept but carried in the context panel.
- **Freshness gate (live mode):** stale/missing injuries (36h threshold) ⇒
  `publish=false` ⇒ NOT PUBLISHED banner in the report/dashboard and Discord
  gets at most an explicit gate notice — never picks. Historical mode marks
  live feeds "not applicable" instead of pretending they were checked.
- **Synthesis** runs post-ranking, on the ranked leans only, feeding the
  context panel (score impact structurally zero — the ranking is already
  final). `news[]` is empty by default: there is no free real-time news
  source (design H4); ESPN news wiring is a future addition.
- **Discord:** `discord_enabled=false` by default; webhook ONLY from
  `DISCORD_WEBHOOK_URL` env or gitignored `config.local.json`; pipeline
  default is dry-run (`--discord-live` to actually post); embeds carry the
  disclaimer + 1-800-GAMBLER footer; ≤10 embeds/message, ≤25 fields/embed.
- **Scheduling** is documented (cron + Cowork scheduled-task option) rather
  than installed — installing a live schedule against a 2026 offseason slate
  would fire on nothing; see docs/phases_3-5.md.

## Block C — RAG

- NL→SQL translator is **rule-based and deterministic by default** (canned
  patterns for leans/CLV/screen-count/voids/stat-leaders/backtest); a real
  LLM can plug in behind `NL2SQLClient`, but the **validator is the security
  boundary** either way: SELECT-only, single statement, no comments, table
  whitelist (`api_credits` and `sqlite_master` excluded), structural outer
  `LIMIT` (config `rag.row_cap`, 200).
- Answers are composed only from returned rows; empty result ⇒ "no rows"
  answer, never an inference.
- Vectorstore = dependency-free TF-IDF over `reports/*.md`, flag-gated OFF
  (`rag.vectorstore_enabled`). Chroma/FAISS deliberately not added to
  requirements; the interface allows swapping later.

## Learning loop (added post-Phase-5, user-requested)

- User asked for iterative week-over-week learning INCLUDING personal context
  (birthdays etc.). That collides with the locked spec decision (context is
  never scored). Resolution: quantitative learning ships live
  (bias/reliability/reallocation/news-to-context), while personal context is
  **measured in a hypothesis ledger** and may only gain a bounded multiplier
  after n≥100, BH-q<0.05, AND explicit human promotion in config — the spec's
  zero-weight default remains the shipped behavior.
- Bias is learned from the FULL screened candidate pool, never the picks
  (selection-bias guard), as a direct shrunk estimate (path-independent, no
  compounding), clipped ±8%; reliability slope 1.0, shrunk k=50, clipped ±15%.
- Reallocation boosts: share-ratio based, capped ×1.35, halved when the basis
  is a proportional guess. QB replacement remains unsupported (different
  problem, not a share shift).
- Miss attribution thresholds: usage <25% of projection ⇒ availability
  surprise; |log-err| >0.15 picks volume vs efficiency; script_flip needs an
  actual-margin sign flip ≥10 pts against a ≥2.5-pt expectation.
- 2025 adaptive-vs-static validation: 56.5%→58.5% overall, top-1 58.8%→64.0%
  at identical volume; mechanism = market-mix shift toward receptions +
  trimmed over-projections. Same synthetic-line caveats as the static replay.

## Environment notes

- The mounted project folder forbade file deletion until Cowork's
  delete-permission was granted (needed for `git init`); git history exists
  from this run onward (baseline commit = pre-existing Phase 1A/1B-1 state).
- `historical/sleeper_players.parquet` (sleeper_id↔gsis map, 3,893 rows) was
  recorded live during the 1B build and is committed as a cache.

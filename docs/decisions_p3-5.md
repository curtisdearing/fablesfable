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

## Hands-off ingest + weight tuning (2026-07-01, user-requested)

- `nflvalue/ingest.py`: live runs auto-refresh current-season pbp/schedules/
  rosters (per-season parquet caches; frozen 2019-2023 base untouched);
  failures degrade loudly to cache, never silently. `--no-refresh` to skip.
- Weight tuning is WALK-FORWARD (tune_weights.py): each season's config chosen
  only from prior seasons, scored out-of-sample (57.5–59.5%/season). Shipped
  2026 config = walk-forward majority: conf_share 0.8 → weights
  {edge .5, confidence .4, matchup .1}, z_cap 1.5, low_confidence_mult 0.8,
  all markets ("core4" beat "all" pooled by 0.1pt — inside noise; keeping
  TD/attempts preserves live-price optionality). In-sample pooled argmax is
  reported for transparency but NOT what shipped.
- Combined validation, 2025 replay at identical volume (synthetic-line
  caveats apply): static default 56.5% → learning 58.5% → tuned+learning
  59.9% overall; top-1 per game 58.8% → 64.0% → 64.7%.

## ML ranking layer (2026-07-01, user-requested)

- Request: "random forest + ML improvement test, best gradient descent score
  on bet payouts." Framing correction applied: RF has no gradient descent;
  gradient boosting minimizes log-loss (the reported "gradient descent
  score"). Both were tested.
- Architecture: stacked CLASSIFIER P(actual > line) over the deterministic
  model's own beliefs + walk-forward usage/context features. Projection
  NUMBERS stay deterministic-model-owned; ML supplies ranking probability +
  side. Structural anti-leakage: predict refuses any week ≤ train cutoff
  (WalkForwardViolation); pipeline falls back to composite for past-week
  replays only.
- Walk-forward OOS (identical pools/protocol, synthetic-line grading):
  tuned composite 57.1–59.5%/season; GBDT 63.2–67.1% (log-loss .626–.639,
  AUC .62–.64); RF 63.8–69.0%. Weekly-retrain GBDT 2025: 66.5%, top-1 69.5%.
- MATERIAL CAVEAT recorded: part of the ML gap is learned exploitation of the
  synthetic-line construction (mean-anchored lines ⇒ under-skew); transfer to
  real bookmaker lines is unproven until live CLV accrues. Kill-check remains
  the referendum; once real lines exist, y should be re-labeled against them.
- Shipped: config `ml_ranker.enabled=true`, GBDT artifact (fits on this
  2-core sandbox; RF upgrade = `python3 ml_test.py --stage fit --models rf`
  offline). Artifact is gitignored (regenerable, data-derived). When ML is
  on, the learning loop's bias-mean correction is skipped (classifier trained
  on raw beliefs; reliability/context multipliers remain display-consistent).
- RF n_jobs=-1 is reproducible to one float ULP (parallel vote averaging);
  GBDT byte-reproducible. Seed 20260701.

## Deterministic context features (2026-07-01, user-requested)

- "Base projections on birthdays, revenge, defensive injuries" implemented as
  FACTS, not news: DOBs (nflverse load_players), roster-history former teams
  (≥3 weeks, walk-forward, current team excluded), official injury-report
  Out/Doubtful counts for the OPPOSING defense (total + DB-only). All enter
  as ML-ranker features (the classifier weights them from outcomes) and as
  context-panel lines feeding the context_ledger; the deterministic composite
  still never scores them.
- Empirical answers, 2019–2025 candidates (n=73,925): birthday weeks
  (n=2,275) over-rate 36.3% vs 36.6% baseline — nothing; revenge (n=1,111)
  33.7% vs 36.7% — if anything slightly negative; 2+ secondary players Out
  (n=3,540 pass-family rows) 43.9% vs 41.4% — real, directionally sensible.
- Ablation (GBDT OOS): 2025 flat (67.1%→67.1%, log-loss .62942→.62846);
  2024 +0.4pt (63.2%→63.6%). Features kept: free, def-outs earns it, the
  rest is now measured instead of mythologized.
- Correction to the design doc: nflverse injuries data is empirically
  available through 2025 via nflreadpy (the FEED concern stands for live
  T-90; ESPN remains the live backstop). Ingest now refreshes injuries +
  players_meta each run.

## Advanced process features (2026-07-01, user-requested spec)

- Implemented per the user's PBP spec: neutral-situation PROE (1st/2nd down,
  Q1-Q3, score ±7, wp 20-80% — nflverse `pass_oe`), early-down passing,
  neutral pace (median snap gap ≤45s within drive), rolling EPA/play,
  shotgun/no-huddle rates, team CPOE; NGS receiving (avg separation,
  intended-air-yards share, YAC-above-expected; 2016+, 32% row coverage =
  qualifying receivers); red-zone target/carry shares; O-line Out/Doubtful
  counts; QB continuity (projected starter's share of trailing attempts —
  schedule QB ids are recorded starters, effectively pre-game-knowable);
  contract final-year flag (nflverse/OTC; specific bonus clauses aren't
  structured free data → news keywords 'incentive/bonus/escalator' feed the
  context_ledger instead); age from DOB; temp/wind from schedules (domes
  neutralized 70F/0mph; HISTORICAL VALUES ARE OBSERVED, live uses forecast —
  small train/serve mismatch, accepted + noted). PROE-vs-total edge left to
  the GBDT as a learned interaction (it sees both columns).
- **CAUGHT LEAK (the important lesson):** first build joined RZ/NGS rolls at
  exact (season, week) keys, but those rows only EXIST for weeks a player had
  RZ usage / NGS qualification — the NaN pattern itself revealed current-week
  involvement. GBDT feasted: 2025 OOS jumped to an impossible 88.2%
  (AUC .764). Fixed with `AsOfLookup` (strictly-before as-of joins; regression
  test pins it) — a NaN now only ever means 'no prior history'.
- Honest post-fix ablation (GBDT OOS): 2024 63.6%→65.7% (AUC .6312→.6332);
  2025 67.1%→67.3% (AUC .6298→.6341, log-loss .62846→.62786), top-1 69.9%.
  Artifact refit on the full feature set; ingest refreshes NGS + contracts +
  extended pbp columns each run.

## Cross-book lines + matchup strengthening (2026-07-02, user-requested)

- to_prop_lines_frame now does REAL line comparison: consensus point (most
  two-sided books, deterministic tie-break), sharp-weighted de-vigged
  consensus fair prob (oddsmath.consensus_two_way, pinnacle 2x), best price
  per side line-shopped with books named. Edge = model − CONSENSUS;
  ev_best_price + n_books surfaced in components. A soft book's vig can no
  longer pose as fair value; its off-market price is instead CAPTURED.
- Matchup clause strengthened: opp_epa_allowed factor (computed since the
  context build, previously unscored) is now a fourth directional matchup
  sub-score. Known remaining matchup depth: defense VOLUME-allowed factor
  and FTN man/zone splits (2022+) — queued as early-season ablations.
- DOC/CODE MISMATCH found in features.py: docstring claimed EWM for player
  rolling means; code ships a flat 8-game window (all validation/tuning was
  performed against the flat version, so behavior is consistent — the
  docstring was the bug, now fixed). EWM recency weighting queued as a
  controlled ablation rather than a silent core change.

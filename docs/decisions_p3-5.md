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

## Your books + chemistry/formation tier (2026-07-02, user-requested)

- Odds pulls now target config `books`: ["draftkings", "betmgm",
  "hardrockbet"] via the API's bookmakers param (falls back to regions when
  empty). Consensus + line-shopping runs across exactly those three.
- Formation/chemistry ask triaged by data trust: exact formations/personnel/
  on-field-22 (NGS participation) is free only through 2023 and discontinued
  — REJECTED for live features (train/serve mismatch). Shipped the trustable
  generalizations, all-seasons and live-capable: shotgun-vs-under-center
  usage tilts (per player, 15-play-per-bucket trust gate), QB-specific
  target-share chemistry vs the schedule's projected starter, top-teammate
  absence + historical with/without boost (Chase-sans-Higgins as a standing
  feature), defense pressure rate (sacks+hits/dropback). All strictly-before
  via AsOfLookup; NaN = insufficient history, never current-week info.
- 2025 OOS with chemistry tier: 67.3% -> 67.9% (log-loss .628, AUC .632 —
  no leak signature). Panel/game notes surface tilts >=5%, chemistry >=4%,
  and teammate-absent bumps. FTN charting (play-action/motion/blitz, 2022+)
  queued as the next tier.

## Second-order injuries + formations workaround + FTN (2026-07-02, user-requested)

- MEASURED second-order translation (not asserted): teammate-absent
  beneficiaries (n=297 player-weeks 2019-25) gain volume but lose ~31%
  efficiency per opportunity -> reallocation now applies eff dampening
  (slope .29/unit boost, floor .85) on top of the capped volume boost.
  Backup-QB weeks (n=162): volume ~FLAT (pass x1.02, rush x0.98 — the "more
  handoffs" intuition is empirically false) but passing efficiency x0.916 ->
  pass-family means x0.92 when projected starter threw <50% of trailing
  attempts. User's directional instincts confirmed on efficiency, corrected
  on volume.
- Formations workaround shipped: free FTN charting (2022->now, weekly) gives
  own play-action + motion rates, opponent blitz rate (5+ rushers) and
  DEFENDERS-IN-BOX faced — the live-safe formation-adjacent tier. Cached per
  season, ingest-refreshed, AsOf walk-forward, NaN pre-2022. 2025 OOS: 67.5%
  overall (within noise of 67.9), top-1 69.9% -> 71.3%, log-loss .6276.
- Paid options priced (2026-07): FTN Data CSV $599 (3 seasons + pbp feeds),
  API tier custom-priced (participation + charting since 2019), site sub
  $69.99/yr (no API); PFF+ $79.99/yr or $9.99/mo (browsable premium stats
  incl alignment/slot; no API). Decision: stay free until live CLV proves
  edge worth funding; the paid FTN API is the natural first purchase.
- Ops: pack construction pickled/cached for chunked frame rebuilds; FTN
  cross-join OOM caught+fixed (game-only merge guard removed); AsOfLookup
  made picklable.

## 2026-07-17 — Dev/hardening session (opening lines, Wilson-LB bands, O/U negative result)

- **P0 durable opening/closing line record.** `clv.py` previously logged
  entry→close only for *published* leans, so no opens/closes existed for the
  markets we never bet — and the accuracy notes call a real-line record the
  blocker for every honest edge claim. Added `clv.opening_prob()` (consensus
  fair prob from the EARLIEST snapshot, mirror of `snapshot_prob`'s latest) and
  `clv.log_open_close_for_week()`, which persists open+close point/prob/n_books
  for EVERY snapshotted `(game,market,player,side)` into the new
  `line_open_close` table. Single-snapshot rows store open==close so coverage
  stays visible; nothing is ever fabricated. Wired into
  `pipeline_weekly.resolve_clv` (the t90/close job). This is the prerequisite
  data capture for a retroactive real-line reliability/CLV backtest.

- **O/U 64/44 "means→median" fix — REJECTED (negative result).** The vault note
  proposed switching the synthetic reference line from the trailing mean to the
  trailing median to cure the under bias. Measured on 2019-2023 player-weeks:
  switching mean→median barely moves the over-rate (receiving_yards 0.360→0.415;
  receptions 0.374→0.355; rushing_yards 0.221→0.232; passing_yards 0.062→0.063).
  It does NOT fix the bias and, per the long-standing comment in
  `prop_backtest._synthetic_line`, a median benchmark reintroduces the opposite
  bias (P(actual>median) runs >50% for right-skewed stats). Conclusion, in line
  with the load-bearing caveat "do not tune accuracy against synthetic lines":
  the synthetic-line over/under split is an artifact, not a calibration target.
  The real fix is real lines (P0 above) + calibration diagnostics, not the
  synthetic line's central-tendency choice. No code change to synthetic lines.

- **Lever #3 (Top-Bets band recal), methodology half.** Tier admission gated on
  the raw point estimate let a thin/lucky band (14/20 = 70%, but 95% Wilson
  LB ≈ 48%) clear the 67% Best bar. Added `wilson_lower_bound()`; every band now
  carries `accuracy_lb`; `rank_game` admits on the LB (Best: LB≥67%, Value:
  LB>50%). Because wider n tightens the LB toward the point estimate, accruing
  seasons can only ADMIT more bets, never relax the bar — the lever's "proper
  CIs / never relax" guarantee, enforced per band. Populating multi-season
  graded replays into `data/weekly.json` remains data-gated (needs the ML
  artifact + a historical run).

- **Env note.** The vault-embedded checkout (`20-projects/nfl-sim/fablesfable`)
  was found stale at `2e9cb14`, ~10 merged commits behind GitHub `main`
  (`9d7c99e`, PRs #4/#5/#6). Two fixes drafted against the snapshot
  (`load_fixture` CSV fallback; a `reliability.py` ECE module) were discovered
  to ALREADY exist upstream (`rosters.load_fixture` fallback; `nflvalue/calibration.py`)
  and were dropped. This session's commits are rebased on the true `9d7c99e`.

### 2026-07-17 — Red-team / pressure-test of the same session

- **Leak fixed in `log_open_close_for_week`.** Adversarial probe: with a game's
  kickoff missing from the dict, the function recorded a POST-game snapshot as
  the "close" (a 6.00/1.05 post-kick price → close_prob 0.149). `log_close_for_week`
  guards this (`if not kickoff: continue`); the new function did not. Fixed:
  CLOSE is only recorded when bounded by a real kickoff; with no kickoff (or no
  pre-kickoff snapshot) we store the legitimate open and leave close NULL rather
  than fabricate. Rows where EVERY snapshot is post-kickoff are dropped. +2
  regression tests (`test_open_close_never_records_postkick_close_without_kickoff`,
  `test_open_close_drops_row_when_all_snapshots_post_kickoff`).
- **Wilson LB verified** exact against an independent closed-form re-derivation
  (14/20→0.48102, 90/90→0.95906, 52/75→0.58169) and shown monotone (LB ≤ point
  estimate on 10k random draws, 0 violations) — the gate can only tighten.
- **Consequence to flag:** the currently-shipped Best band (p≥0.62 ML, 52/75 =
  69% point) has Wilson LB 0.582 < 0.67, so under LB gating it NO LONGER clears
  the Best tier. This is the lever behaving as intended (fail-closed on thin n),
  but it means the in-season Best tab can show fewer/no bets until multi-season
  replays widen n. Thresholds tighten, never relax — by design.
- **Re-verified 262 tests green** on a clean tree (data restored to upstream;
  the one earlier `test_all_data_audit` failure was a stale-JSON overlay from the
  vault snapshot, not a code regression).

## 2026-07-18 — Phase 7 hardening (reliability, speed, traceability)

Scope was explicitly *not* predictive: no new features, no new markets, no
model-family changes. The gate for the whole phase was a **no-delta
attestation** — `ml_test --stage frame` rebuilt across 2019-2025 before and
after the full change set, compared column-by-column NaN-aware: **73,925 rows
x 39 columns byte-identical**. The `2023 wk10` pipeline report is likewise
byte-identical modulo the `as_of` provenance stamp. Suite 262 -> 367.

### 7.1 Performance — profiled, then deliberately NOT optimized

| Target | Wall | Peak RSS |
|---|---:|---:|
| `pipeline_weekly` 2023 wk10, ML off | 5.15 s | — |
| `pipeline_weekly` 2023 wk10, ML on | 8.18 / 8.22 / 8.18 s | 1.12 GB |
| `ml_test --stage frame` (7 seasons, 73,925 rows) | 20.0 s | 1.11 GB |
| `ml_test --stage fit` (GBDT) | 2.7 s | 0.37 GB |
| test suite | 19.7 s (262) / 24 s (367) | — |

cProfile puts the cost in pandas' Python-UDF fallback: `_transform_general`
4.31 s over 61 calls, `_rolling_shifted` 24,096 calls, ~97k `Series.__init__`.
That vectorizes — and the decision was **not to**. The prize is ~3 s on a job
that runs once a week, and the code in question is the `shift(1)`-then-roll
primitive that every anti-leakage guarantee in this repo rests on. Spending
the highest-risk surface in the codebase to buy an unmeasurable speedup is a
bad trade. The table above is published as the *evidence for* leaving it
alone. The real headroom risk is peak RSS (1.12 GB, and an OOM was already
caught in this path on 2026-07-02), not latency.

### 7.2 Failure-mode hardening — three real defects

All five external deps (Odds API, nflreadpy, Open-Meteo, Discord, GH release
asset) are now forced to fail in `tests/test_failure_modes.py` across timeout /
malformed / partial / auth / budget-exhaustion. Found and fixed:

- **Phantom line from a malformed payload.** An outcome with no `price` and no
  `point` produced a persisted `lines` row with `price=None` and `point`
  defaulted to **0.5** — i.e. a "receiving yards over 0.5" quote that never
  existed, sitting in the DB indistinguishable from a real one. 0.5 is the
  anytime-TD convention *only*. Such outcomes are now dropped.
- **One flaky HTTP call killed the weekly run.** `pull_week_props` let fetch
  exceptions propagate and the pipeline did not wrap the call, so a single
  timed-out event aborted the run *after* the candidate pool was built. Now
  degrades per game into `skipped_error` and the game falls through to the
  documented `no_market`. `BudgetExceeded` is deliberately still a hard stop —
  overspending a metered free tier is not a degradation.
- **Discord failure discarded a completed week.** `post_weekly` let `urlopen`
  raise, losing the caller's return payload after the report, dashboard and DB
  writes had already succeeded. Now returns `status="error"`, with the webhook
  URL redacted out of the error string (urllib puts the full URL in the
  exception text, and error strings get logged and pasted into reports).

### 7.3 Schema + artifact integrity

`PRAGMA user_version` with forward-only, additive-only migrations. This matters
because the live DB is a durable release asset: `CREATE TABLE IF NOT EXISTS` is
a no-op against an existing table, so a newly added column would never have
appeared on the deployed database. A pre-versioning DB (user_version=0) is
*adopted and stamped*, not rebuilt — tested against a populated legacy DB
asserting every prior row survives byte-for-byte, and that a new column reads
NULL rather than a fabricated default. A DB from a *newer* release is refused
outright rather than silently downgraded.

Model artifacts get a SHA-256 sidecar at `save()` and are verified at
`load()`, refusing to score on mismatch (tampered and truncated cases both
tested). A *missing* sidecar is tolerated — pre-7.3 artifacts have none, and
absence is not evidence of corruption; only disagreement is fatal.

### 7.4 Numerical edge cases — the serious one

**`p_over` turned missing data into certainty.** `max(0.0, min(1.0, nan))`
evaluates to **1.0** in CPython, because every comparison against NaN is False.
So a NaN mean/sd/line — this codebase's own documented encoding for "no prior
history" (see `AsOfLookup`) — came out of `p_over` as `p_over=1.0000` with
`eligible_for_shortlist=True`. That is maximum edge and maximum confidence
simultaneously: the row does not error, **it ranks first on the board**.
Reachable through `expected_volume` whenever no usage basis exists.

Corpus audit: **0 of 73,925 historical candidate rows were affected** — the bug
was latent, which is precisely why the frame stayed byte-identical after the
fix, and precisely why it needed a corpus-wide assertion rather than a unit
test alone. Now: `p_over=None`, `mean=None`, `eligible_for_shortlist=False`.

Also fixed: `_SF.get(dist, _norm_sf)` silently substituted a normal for any
unrecognised distribution name (a typo'd market spec would have scored against
the wrong family); `devig_multiplicative` returned a flat `1/n` prior on
all-unusable prices (a fabricated probability with no market behind it);
`consensus_two_way` accumulated in dict-insertion order, yielding **2 distinct
`p_a` values across 24 orderings of 4 books** (1 ULP apart) in violation of the
determinism rule; and `n_books` counted every key handed in rather than the
books that actually contributed a usable two-sided price, over-stating
published market support.

### 7.5 Test-suite quality

Coverage 57% -> 58% overall, but the movement that matters is on the modules in
scope: `oddsmath` 56->90, `weather` 38->69, `notify` 82->90, `ml_ranker`
76->80, `db` 80->83, `oddsapi_props` 80->81. (`projection.py` reads 90->86
because the fail-closed guards added 26 statements to the denominator.)

The leakage guards were **mutation-tested**: the 2026-07-02 leak was
re-injected (`AsOfLookup` `bisect_left`->`bisect_right`, strictly-before ->
inclusive), `shift(1)` was removed from `_rolling_shifted`, and
`assert_walk_forward` was weakened `<=`->`<`. Each mutation provably changes
observable behaviour, so all three alarms are wired to something real.

Suite hygiene: `test_ingest.py::test_refresh_degrades_loudly_not_silently`
*failed* rather than skipped when the optional `nflreadpy` was absent, which
made the headline "262 green" silently conditional on an optional install.
Now `pytest.importorskip`.

### Not done, deliberately

No performance optimization (see 7.1). No change to any projection value, any
market, any model family. The synthetic-line over/under split was not touched
and was not tuned against — the recorded means->median negative result stands.

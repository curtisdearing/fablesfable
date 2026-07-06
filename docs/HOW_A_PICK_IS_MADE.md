# How a pick is made — the complete, no-black-box walkthrough

Every number on a published lean is traceable through these nine steps. File
references point at the exact code.

## 1. Data comes in (`nflvalue/ingest.py`, runs before every live pipeline)

Play-by-play, schedules (with pre-game spread/total and projected starting
QBs), weekly rosters, official injury reports, NGS tracking, FTN charting,
contracts, player DOBs — all free nflverse/ESPN/Open-Meteo feeds, cached as
parquet under `historical/`. Coverage and trust grades: `DATA_SOURCES.md`.

## 2. Walk-forward features (`features.py`)

For each (player, week): rolling usage (targets, target share, carries,
attempts), rolling efficiency (yards/target, catch rate, yards/carry, YPA)
shrunk toward position means on small samples; for each (defense, position):
rolling yards- and EPA-allowed factors (WR and TE tracked separately); for
each team: rolling pass/rush volume. **Everything is `shift(1)`-then-roll: a
row at week W aggregates only weeks < W.** `tests/test_leakage.py` poisons
future weeks and asserts nothing moves.

## 3. The deterministic projection (`projection.py`)

```
expected volume    = team rolling volume × player usage share × game-script tilt
expected efficiency = player rolling efficiency × opponent-vs-position factor
mean               = volume × efficiency
```

Game script comes from the pre-game spread (favorites tilt run, dogs tilt
pass, capped ±12%). Each market gets a distribution family (gamma for yards,
negative binomial for counts, Poisson for TDs); the per-market SD is the
standard deviation of the engine's own **past** errors (walk-forward
residuals), so P(over line) is read off a distribution whose width reflects
how wrong the model has actually been. LLMs never touch any number
(`synthesis.py` enforces this with a contract violation exception).

## 4. Candidates and gates (`candidates.py`)

Every (player, market) pairing per game — then gates: ≥3 trailing games
(cold-start), minimum trailing usage (no scrubs), anytime-TD is YES-only.
~40 candidates per game survive. With no sportsbook line, the reference line
is the player's own trailing mean (floor + .5), rendered with a † and **never
allowed to mint an edge**.

## 5. Measured situational adjustments (`candidates.py`, constants from data, not vibes)

| Trigger | Adjustment | Evidence |
|---|---|---|
| Teammate in same usage family OUT | volume boost from that player's own historical with/without splits (cap ×1.35; halved if no absent-week sample) × efficiency dampening `1 − .29·(boost−1)` | n=297 absent player-weeks: beneficiaries gained volume, lost ~31% per-touch efficiency |
| Projected QB threw <50% of trailing attempts | pass-family means ×0.92 | n=162 backup weeks: volume flat, efficiency −8.4% |
| Team's WR1 / TE1 / RB1 OUT | QB passing markets ×0.921 / ×0.947 / ×0.971 (multiplicative, floor .85) | full absence matrix, n=1,146–1,514 per cause: `data/absence_matrix.json` |

## 6. Ranking (`composite.py` + `ml_ranker.py`)

Two rankers, both always computed:

- **Composite (auditable):** `100·(w_e·edge + w_c·confidence + w_m·matchup)`,
  weights tuned by walk-forward grid over 2019–2025 (each season scored with
  weights chosen only from prior seasons). Confidence = capped |z| from the
  line; matchup = opponent yards/EPA-allowed + game-script + pace, directional
  for the chosen side; edge (below) dominates when a real price exists.
- **ML ranker (orders the list, flag-gated):** a **calibrated random-forest**
  classifier predicting P(actual > line), stacked on the projection's own belief
  plus ~50 features (weather, PROE/pace, NGS separation/air-yards share, red-zone
  roles, defensive/O-line outs, QB chemistry, shotgun tilts, blitz/box rates,
  age, contract year, birthdays, revenge…). It weights features by outcomes —
  in practice it pruned birthdays/revenge to ~zero and leaned on red-zone
  share, NGS, and usage. Walk-forward out-of-sample 2021–25 it beat the
  composite 63–69% vs 57–59% at reference lines. **Calibration (Phase 7.1/7.2):**
  a per-market Platt map, fit strictly walk-forward, makes the probabilities
  *trustworthy as probabilities* — `P(over)=0.62` actually lands ~62% — which
  matters because edge and stake size read those numbers directly. RF was chosen
  over gradient boosting and over ensembling them, on a fixed calibrated
  log-loss + ECE + per-market-Brier metric suite (`decisions_p7.md` §7.1–7.2).
  Structural guard: the model **refuses to score any week it trained on**
  (`WalkForwardViolation`), and the calibrator never sees the fold it corrects.

## 7. The line, and the edge (`sources/oddsapi_props.py`, `composite.py`)

Lines are pulled (budget-capped, rotating) from the configured books —
DraftKings, BetMGM, Hard Rock. Per prop: consensus point → every book de-vigged
→ sharp-weighted **consensus fair probability**; best price per side kept
with the book named. **Edge = model P(side) − consensus fair P(side)**;
`ev_best_price` shows expected value at the best quote. No two-sided price →
`no_market`, and the pick ranks on model conviction alone, labeled as such.

## 8. Selection honesty + correlation-awareness (`shortlist.py`, `correlation.py`)

Top 5 per game, max 2 per player, deterministic tie-breaks, and the screen count
("5 of N") always printed — a top-5 from 40 candidates contains luck, and the
denominator keeps that visible. **Correlation-aware (Phase 7.5/7.6):** two leans
in one game are often not two independent edges, so selection is a greedy walk
that discounts a candidate by its largest positive correlation to what's already
chosen. The correlations are *measured* on standardized residuals across 2019–25,
shrunk toward zero, and used only where they clear an effect-size + stability bar:
a player's own two markets (ρ≈0.76) and a QB with his pass-catcher (ρ≈0.30) get
de-duplicated; genuinely independent or *hedging* legs (QB-pass vs RB-rush,
opposing RBs — measured negative) are never penalized. The popular two-same-team-
WR "stack" turned out to be **noise (ρ≈0.03)** and is treated as zero. So a top-5
can't be secretly five bets on one game outcome. The context panel (injuries,
birthdays, revenge, incentives, weather, mismatches) is assembled **after**
ranking from a scorer that cannot see it; tests prove score equality with and
without context.

## 8b. Advisory stake size (`staking.py`) — a recommendation, never a bet

Each lean gets an **advisory** unit size (Phase 7.7): fractional-Kelly off the
calibrated edge, but (a) the edge is first shrunk toward the market prior because
it's an estimate; (b) then quarter-Kelly; (c) then divided down for correlation
to the game's other leans (7.5); (d) then capped per-bet and per-slate. A bankroll
Monte Carlo at *plausible real-line* edges (52–58%, **not** the synthetic 66–68%)
shows this holds the worst-case drawdown to ~10–15% with near-zero ruin. **The
tool never places a bet, moves money, or initiates a transfer** — the size is
advice a human may act on, ignore, or override, and it says so. Plain-English
version: `docs/EXPLAINER_staking.md`.

## 9. Feedback (`prop_learning.py`, `clv.py`, `killcheck.py` — Tuesdays, automatic)

Every pick is graded; every miss attributed (volume miss / efficiency miss /
availability surprise / script flip / tail variance — queryable via the RAG
CLI). Per-market bias and reliability adjust next week's run (bounded,
walk-forward, from the full candidate pool to avoid selection bias). The ML
retrains weekly, and its training labels migrate from synthetic reference
lines to **real lines** as they accumulate — a row flips only once it has both a
real decision-time line *and* a graded outcome, proven un-leakable in the tests.
**The CLV referendum (Phase 7.3/7.4):** a budget-reserved entry snapshot and a
pre-kickoff close snapshot are captured per lean; closing-line value is logged in
de-vigged probability space (a stale snapshot outside the pre-kick window
resolves to *nothing*, never a faked ~0). After **n≥150 resolved** picks the
kill-check declares **GO** (avg CLV>0 AND ≥52% positive-CLV rate) or **NO-GO** —
and NO-GO means the pre-committed answer is *stop treating this as bettable*, in
writing, in the report and on the dashboard's CLV/Kill-Check tab (with a coverage
gauge so a thin close-snapshot budget can't quietly stall the referendum). What
flips this project from "entertainment" to "staked" is defined once, in advance,
in `decisions_p7.md` §7.9 — and it is CLV, never a synthetic-line hit rate.

## Known limitations (deliberately not hidden)

Backtest hit rates are measured at synthetic (trailing-mean) reference lines,
which structurally favor unders — real books price tighter; forward CLV is
the only edge proof accepted. Exact formations/personnel data is unavailable
free post-2023 (see `DATA_SOURCES.md`); the model uses formation-adjacent
signals instead. Candidate sets for live weeks carry forward from the prior
week, so debuting/just-traded players are invisible until they play. Player
props limit fast; the value here is the research, not scalable income.

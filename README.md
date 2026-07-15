# NFL Prop Lean Screener

A fully automated, free-data NFL player-prop research tool. Every Wednesday in
season it ranks the **top 5 value leans for every game** — deterministic
projections vs. real sportsbook lines — posts them to Discord with a writeup
(records, injuries, weather, birthdays, revenge games, contract incentives,
matchup mismatches), then grades itself, attributes every miss, and retrains.

**Leans, not locks.** This is research, not financial advice. It never places
bets. Gambling problem? **1-800-GAMBLER**.

## The one-paragraph version of how a pick is made

A player\'s projected stat = *(team volume × his usage share × game-script
tilt) × (his efficiency × opponent-vs-position factor)*, all from strictly
prior-week data, giving a distribution and P(over) for any line. Candidates
(≈40/game after usage and cold-start gates) are ranked by a gradient-boosted
classifier stacked on those projections plus ~50 walk-forward features
(matchups, weather, pace/PROE, NGS, red-zone roles, injuries both sides, QB
chemistry, formation tilts). Where a real line exists (DraftKings/BetMGM/Hard
Rock via The Odds API), the **edge = model probability − the de-vigged
cross-book consensus**, captured at the best available price. Top 5 per game,
max 2 per player, with the honest denominator ("5 of N screened") always
shown. Full detail: **[docs/HOW_A_PICK_IS_MADE.md](docs/HOW_A_PICK_IS_MADE.md)**.

## Quickstart

```bash
pip install -r requirements.txt
python3 pipeline_weekly.py --season 2025 --week 14 --mode historical   # replay a past week
python3 lean_backtest.py --season 2025 --learn                         # graded season replay
python3 -m nflvalue.rag.nl2sql "why did we miss in week 14"            # query the warehouse
```

Live setup: put `ODDS_API_KEY` and `DISCORD_WEBHOOK_URL` in the environment or
gitignored `config.local.json`. Scheduling, budgets, weekly cadence:
**[docs/phases_3-5.md](docs/phases_3-5.md)**.

GitHub Actions defines the offseason-safe Wednesday, T-90, and Tuesday loop in
`.github/workflows/live-weekly.yml`. A checksummed GitHub prerelease asset is
the durable model state; failed runs cannot overwrite it. Add repository
secrets `ODDS_API_KEY` and `DISCORD_WEBHOOK_URL` for live prices and
notifications. Successful runs deploy the generated dashboard to
[GitHub Pages](https://curtisdearing.github.io/fablesfable/). The schedule and
deployment become active after this workflow is present on the default branch.

## What\'s in the box

| Area | Files |
|---|---|
| Deterministic projections | `nflvalue/features.py`, `projection.py` (leakage-tested) |
| Candidates + adjustments | `nflvalue/candidates.py` (usage gates, synthetic lines, measured injury/backup-QB/absence adjustments) |
| Ranking | `nflvalue/composite.py` (auditable score), `ml_ranker.py` (GBDT/RF stacked classifier, walk-forward guarded) |
| Features | `advanced_features.py` (PROE/pace/NGS/RZ/weather/contract), `chemistry.py` (QB/teammate/formation tilts), `ftn_features.py` (blitz/box/PA/motion), `context_features.py` (birthdays/revenge/def-injuries) |
| Market | `sources/oddsapi_props.py` (budgeted, cross-book consensus + line shopping), `clv.py`, `killcheck.py` |
| Delivery | `pipeline_weekly.py` (two-clock), `report.py`, `document.py` (HTML drop), `notify.py` (Discord), dashboard Weekly Leans tab |
| Self-updating | `prop_learning.py` (grade→attribute→adjust), `context_study.py` (evidence-gated narrative tags), Tuesday ML retrain w/ real-line label migration |
| Data plumbing | `ingest.py` (auto-refresh), `scripts/auto_weekly.py` (self-scheduling jobs) |

The original game-line dashboard this grew from still works:
[docs/README_game_line_app.md](docs/README_game_line_app.md).

## Reviewer\'s map

- **[docs/HOW_A_PICK_IS_MADE.md](docs/HOW_A_PICK_IS_MADE.md)** — the full
  pipeline, every formula, every measured adjustment, and where each number
  on a pick comes from. Start here.
- **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** — every feed, coverage,
  trust grade; what\'s derived free vs. genuinely paywalled.
- **[docs/decisions_p3-5.md](docs/decisions_p3-5.md)** — the decision log:
  every default, every measured constant, every caught bug (including two
  data leaks the guardrails caught — documented, not buried).
- **[docs/phases_3-5.md](docs/phases_3-5.md)** — operations runbook.
- **[reports/all_data_factor_audit.md](reports/all_data_factor_audit.md)** —
  retraction of the non-reproducible factor counts and the replacement
  pregame-only, nested season-forward protocol.
- **[PREMORTEM.md](PREMORTEM.md) / [PROP_SHORTLISTER_SPEC.md](PROP_SHORTLISTER_SPEC.md)** —
  the design contracts the code is held to.

## Honesty invariants (enforced by ~200 tests)

No feature may see the week it predicts (leakage tests + structural
`AsOfLookup` / `WalkForwardViolation` guards). Narrative context (birthdays,
revenge, incentives) is displayed and *measured* but never scored until it
clears n≥100 and BH-q<0.05 — so far, none has. Backtest numbers are graded at
synthetic reference lines and say so; the only real edge test is forward
closing-line value, and a pre-committed kill-check (150 leans) says NO-GO in
plain language if the market wins. Selection counts ("5 of N") are never
hidden. The Odds API budget hard-stops at 450/500 monthly credits.

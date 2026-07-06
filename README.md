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
(≈40/game after usage and cold-start gates) are ranked by a **calibrated
random-forest classifier** stacked on those projections plus ~50 walk-forward
features (matchups, weather, pace/PROE, NGS, red-zone roles, injuries both
sides, QB chemistry, formation tilts); a per-market Platt calibration makes
`P(over)=0.62` actually mean 62% (Phase 7.1/7.2). Where a real line exists
(DraftKings/BetMGM/Hard Rock via The Odds API), the **edge = calibrated model
probability − the de-vigged cross-book consensus**, captured at the best
available price. Top 5 per game, max 2 per player, **correlation-aware** so a
slip isn\'t secretly five bets on one game (Phase 7.5/7.6), with the honest
denominator ("5 of N screened") always shown — and an **advisory** stake size
per lean (shrunk fractional-Kelly, correlation- and drawdown-aware; Phase 7.7).
Whether any of this is *real* edge is settled forward, by closing-line value,
not by these synthetic-line numbers. Full detail:
**[docs/HOW_A_PICK_IS_MADE.md](docs/HOW_A_PICK_IS_MADE.md)**.

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

## What\'s in the box

| Area | Files |
|---|---|
| Deterministic projections | `nflvalue/features.py`, `projection.py` (leakage-tested) |
| Candidates + adjustments | `nflvalue/candidates.py` (usage gates, synthetic lines, measured injury/backup-QB/absence adjustments) |
| Ranking | `nflvalue/composite.py` (auditable score), `ml_ranker.py` (calibrated RF stacked classifier, per-market Platt, walk-forward guarded) |
| Correlation + staking | `nflvalue/correlation.py` (measured same-game ρ, shrunk + leakage-tested), `shortlist.py` (correlation-aware selection), `staking.py` (advisory shrunk-Kelly sizing — never places a bet) |
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
- **[docs/decisions_p3-5.md](docs/decisions_p3-5.md)** / **[decisions_p6.md](docs/decisions_p6.md)** / **[decisions_p7.md](docs/decisions_p7.md)** — the decision logs:
  every default, every measured constant, every caught bug (including data
  leaks the guardrails caught — documented, not buried). p7 covers calibration,
  the ensemble/tuning verdict, the CLV referendum, correlation, staking, and the
  final go/no-go framework.
- **Plain-English explainers** (no stats background needed):
  [EXPLAINER_correlation.md](docs/EXPLAINER_correlation.md),
  [EXPLAINER_staking.md](docs/EXPLAINER_staking.md).
- **[docs/phases_3-5.md](docs/phases_3-5.md)** — operations runbook.
- **[PREMORTEM.md](PREMORTEM.md) / [PROP_SHORTLISTER_SPEC.md](PROP_SHORTLISTER_SPEC.md)** —
  the design contracts the code is held to.

## Honesty invariants (enforced by ~332 tests)

No feature, calibration fit, correlation estimate, or real-line re-label may see
the week/season it predicts — leakage tests poison the future and assert nothing
moves, backed by structural `AsOfLookup` / `WalkForwardViolation` guards.
Probabilities are **calibrated** and checked on a fixed walk-forward metric suite
(log-loss + ECE + per-market Brier); same-game correlations are **measured,
shrunk toward zero, and only used where they clear an effect-size + stability
bar** (`decisions_p7.md` §7.5). Narrative context (birthdays, revenge,
incentives) is displayed and *measured* but never scored until it clears n≥100
and BH-q<0.05 — so far, none has. Backtest numbers are graded at synthetic
reference lines and say so; the only real edge test is forward **closing-line
value**, and a pre-committed kill-check (n≥150 resolved, avg CLV>0, ≥52%
positive) says GO or NO-GO in plain language — NO-GO means *stop treating this as
bettable* (`decisions_p7.md` §7.9). Stake sizes are **advisory only; the tool
never places a bet or moves money.** Selection counts ("5 of N") are never
hidden. The Odds API budget hard-stops at 450/500 monthly credits.

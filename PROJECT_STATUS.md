# Project status — NFL prop lean screener

*Plain, current, and honest. Full detail: `docs/decisions_p7.md` §7.9.*
*Leans, not locks. This tool never places a bet or moves money. 1-800-GAMBLER.*

## Where the project stands today

**State: INSUFFICIENT_SAMPLE — paper-trade / research tool.** No live Odds API
key is configured and it's the offseason, so **0 leans have a resolved CLV**. By
the project's own pre-committed rule, that means: **keep logging, stake nothing,
conclude nothing** about real edge yet.

Phases 1–7 are built and green (332 tests). What that gives you is a *finished
instrument*, not a proven money-maker — and the difference is the whole point.

## What it demonstrably does (measured, walk-forward, reproducible)

- Ranks props with a **calibrated** random-forest model — `P(over)=0.62` really
  means ~62% (ECE ≈ 0.011).
- At **synthetic** trailing-mean reference lines, it shows real directional skill
  (63–69%/season vs 57–59% for the auditable composite). *These are practice-line
  numbers, not profit.*
- Selects **correlation-aware** top-5s (a slip isn't secretly five bets on one
  game) and attaches an **advisory** stake size that's shrunk, correlation- and
  drawdown-aware (worst-case drawdown ~10–15%, near-zero ruin at plausible edges).
- Captures real lines within a hard budget, logs closing-line value, and will
  render a GO/NO-GO verdict once enough resolve.

## What it does NOT prove

**Real-line profit.** Synthetic lines favor unders and price less than sharp real
markets; some of the model's edge is likely exploiting that. The honest chain is
*synthetic skill (shown) → real-line hit rate (unknown) → profit (variance-
dominated at NFL volumes)*. Nothing here is a profit claim.

## The pre-committed decision (this does not move)

Whether this becomes "staked" is decided **only** by closing-line value:

| Verdict | Condition | Consequence |
|---|---|---|
| **GO** | n ≥ 150 resolved **and** avg CLV > 0 **and** ≥ 52% positive-CLV **and** coverage ≥ 0.5 **and** ≥ 8 in-season weeks | Small, disciplined, monitored staking at the advisory sizes — kill-check stays armed |
| **NO-GO** | n ≥ 150 with avg CLV ≤ 0 **or** positive rate < 52% | **Revert to entertainment; stop staking.** In writing. |
| **INSUFFICIENT** | n < 150 *(today)* | Keep logging; stake nothing; conclude nothing |

A GO is "consistent with real edge," not a profit guarantee (prop CLV is a weaker
proxy; books limit winners fast; Wednesday entry sits near the close). Paid data
(the FTN API) is the one pre-authorized purchase — and **only after a GO**.

## What you can do now

- **Replay / research:** `python3 pipeline_weekly.py --season 2025 --week 14 --mode historical`
- **Run the honest evidence for yourself:** `python3 scripts/clv_worked_example.py`
  (CLV math on fixtures), `python3 scripts/staking_mc.py` (bankroll simulation),
  `python3 scripts/audit_calibration.py` (calibration audit),
  `python3 scripts/fit_correlation.py` (correlation structure).
- **Go live (when you choose):** put `ODDS_API_KEY` + `DISCORD_WEBHOOK_URL` in the
  environment or `config.local.json`; the scheduled jobs self-detect the season.
  Then let CLV accrue and read the dashboard's CLV/Kill-Check tab — it will tell
  you GO, NO-GO, or "not yet," on the rules above.

## Plain-English explainers (no stats needed)

- `docs/EXPLAINER_correlation.md` — why two picks in one game aren't two bets.
- `docs/EXPLAINER_staking.md` — how much to risk, and why it's cautious.
- `docs/HOW_A_PICK_IS_MADE.md` — the full, no-black-box walkthrough.

# Data sources — coverage, trust, and the paywall boundary

Everything the model consumes, what it costs (all currently $0), and what
genuinely sits behind paywalls. Policy: **no scraping paid data, no paywall
circumvention** (design premortem H11 — ToS drift kills hands-off tools and
taints the dataset). Where paid data matters, we either derive a free
equivalent from play-by-play or say plainly that we don't have it.

## Free feeds in production

| Feed | Source | Coverage | Trust | Used for |
|---|---|---|---|---|
| Play-by-play (397 cols incl. `xpass/pass_oe`, `cpoe`, `wp`, shotgun, box-score ids) | nflverse | 2019→now, nightly in season | A | Everything: usage, efficiency, PROE/pace, red-zone, pressure, chemistry splits |
| Schedules + pre-game spread/total + projected starter QBs + rest days | nflverse | 2019→now | A | Slate, game script, kickoff times, QB continuity, records |
| Weekly rosters (real positions) | nflverse | 2019→now | A− | Positions, revenge-game history, teammate ranks |
| Official injury reports (status + position) | nflverse | 2019→now (empirically, incl. 2025) | B+ | Defensive/O-line out counts, historical absence effects |
| Live injuries + per-event actives | ESPN (undocumented) | current | C+ | Wednesday statuses; T-90 void gate (schema-validated, freshness-gated) |
| NGS tracking (separation, intended-air-yards share, xYAC) | nflverse | 2016→now, weekly, qualifiers only | B+ | Efficiency-vs-luck features |
| FTN charting (play-action, motion, blitzers, pass rushers, defenders-in-box) | nflverse free subset | 2022→now, weekly | B | Formation-adjacent + defensive-aggression features |
| Contracts (year signed, length, APY) | nflverse/OTC | current + history | B | Contract-year flag |
| Player DOBs | nflverse | all | A | Birthday weeks, age |
| Weather forecasts (kickoff-hour wind/temp/precip) | Open-Meteo | live, keyless | B+ | Live weather features + writeup (historical = observed values from schedules; small train/serve gap, documented) |
| Sleeper projections | Sleeper API | current | B− | Divergence cross-check only — never a target (H5) |
| Prop lines + prices | The Odds API (free tier, 500 credits/mo) | live only | A for quotes | Edge vs consensus, line shopping, CLV. Hard-stopped at 450/mo |

## The paywall boundary (what we genuinely don't have)

| Data | Who sells it | Price (July 2026) | Our stance |
|---|---|---|---|
| Exact formations, personnel groupings, the 22 on field | NGS participation was free 2016–**2023**, then discontinued. FTN Data API (participation + charting since 2019) | CSV $599; API tier custom-priced; site sub $69.99/yr (no API) | Derived free proxies instead: shotgun/no-huddle rates, per-player shotgun-vs-under-center usage tilts, FTN PA/motion/blitz/box. The FTN API is the **first — and only pre-authorized — paid purchase, and only once live CLV clears the pre-committed GO** (`decisions_p7.md` §7.9). No paid source enters before that. |
| Alignment/slot rates, per-route data, PFF grades | PFF+ | $79.99/yr or $9.99/mo (browsable, no API) | Not used. Closest free proxies: NGS separation + air-yards share; pbp `pass_location` is a further untapped free derivation. |
| Historical prop **lines/prices** | SportsDataIO, others | enterprise | Cannot be reconstructed free — this is why backtests grade at synthetic reference lines (labeled †) and why forward CLV is the only accepted edge proof. |
| Real-time beat-reporter news | X/Twitter API | prohibitive | Accepted gap (premortem H4): ESPN editorial news feeds the context panel; the tool is honest that sharps see news first. |

## Derivation ledger (paid-adjacent signals we rebuilt free)

- **"Who benefits when X sits"** → per-player historical with/without splits +
  the pooled absence matrix (`data/absence_matrix.json`), from pbp + rosters.
- **"Defense anticipates the run with a backup QB"** → measured directly:
  volume flat, efficiency −8.4% (n=162) — encoded as an efficiency, not
  volume, adjustment.
- **Formation effects on a player** → shotgun-vs-under-center usage tilt per
  player (every snap has a shotgun flag) × the team's live formation
  tendencies.
- **Defensive front/scheme** → blitz rate, defenders-in-box, pressure rate
  (sacks+hits per dropback) — all from free per-play data.

Every cache lives under `historical/` and is refreshed automatically by
`nflvalue/ingest.py` before each live run.

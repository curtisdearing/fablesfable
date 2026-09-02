# NFL Team Intelligence: Local Practice and Team Context

**Product:** Fablesfable (game/team and player-prop research)  
**Status:** version-2 collector is implemented; context-only; not load-bearing; not automatically invoked by `pipeline_weekly.py`. Version 2 (2026-09-02) adds three source classes -- `independent_blog`, `reddit`, `x_twitter` -- across all 32 teams, per the owner's ask to exhaust the local/niche-blog, Reddit, and X avenue for individual player-performance insight (see the vault's fablesfable project notes for the 2026-09-02 feasibility writeup this follows on from).  
**Machine-readable registry:** `config/team_sources.json`  
**Collector:** `scripts/collect_team_intel.py`  
**Parser/schema:** `nflvalue/sources/team_intel.py`

## Non-negotiable rule

Local reporting can tell us **what to verify**. It cannot silently tell the model
**what number to use**. The collector never scrapes article bodies and every item
is emitted with source, URL, publication/fetch timestamps, collection method,
signal labels, and `performance_use: context_only`.

Structured official injury/inactive data remains authoritative for availability
gates. A local practice report can corroborate a designation, expose a role
question, lower confidence, or open a preregistered research hypothesis. It may
not change a projection, edge, shortlist rank, price, or bet tier by itself.

## Fast path for an agent

```bash
# From the repository root
python3 scripts/collect_team_intel.py --team BUF --team MIA --hours 72

# Inspect planned source queries without touching the network
python3 scripts/collect_team_intel.py --team PHI,DAL --dry-run

# Validate the collector and all 32 registry entries
python3 -m pytest -q tests/test_team_intel.py
```

Generated outputs are intentionally gitignored:

- `data/team_intel_latest.json` — versioned evidence packet for machines.
- `reports/team_intel_latest.md` — linked briefing for an agent/human.

Use `--all` only when a league-wide packet is actually needed; the normal unit
of work is the teams in one matchup or slate.

## Evidence hierarchy

| Tier | Source | Permitted use |
|---|---|---|
| A | NFL/team official injury report, transaction, inactive list | Availability gate after schema and freshness checks |
| B | Team-owned practice report or press conference | Corroboration and context; remember team media can be selective |
| C | Credentialed local outlet with regular team/practice access | Role, rep, lineup, staff, and locker-room leads; corroborate before escalation |
| D | ESPN/national reporting | Cross-check and broader context |
| D+ | `independent_blog` (e.g. a single-author, team-dedicated SB Nation site) | Same corroboration posture as Tier C/D local reporting; ranked with `local_outlet` in dedup, not above it |
| E | Google News RSS or another aggregator | Discovery only; open and verify the named publisher. Unregistered publisher domains are counted and dropped. |
| F | `reddit`, `x_twitter`, or any other social post/fan account/anonymous rumor | Lead only; `requires_corroboration` is hardcoded `true` on every item and it never enters the packet as established fact |

`config/team_sources.json` gives every franchise at least one official team source
and one established local outlet. A source URL is a discovery surface, not a
claim that every article is free or that every headline is correct.

As of 2026-09-02 every franchise also carries one `independent_blog` (a
single-author or small-team niche site focused solely on that franchise --
the pattern is `ebonybird.com` for the Ravens), one `reddit` source (the
team's primary active subreddit), and 2-4 `x_twitter` sources (beat writers,
team insiders, or independent team-dedicated analysts -- never generic
national NFL accounts). A handful of `independent_blog` feed URLs were not
independently fetch-verified (robots.txt blocked the verification request) --
each carries `feed_verified_2026_09_02: false` and a `note` saying so; treat
those as pattern-matched, not confirmed, until a live run proves the feed.

## Weekly practice intelligence clock

1. **Monday–Tuesday:** treatment updates, roster moves, coaching explanations,
   and new staff/play-caller changes. Record as context; do not infer Wednesday
   participation.
2. **Wednesday:** first official participation signal. Compare local observations
   with the structured injury report; record conflicts rather than choosing the
   more exciting account.
3. **Thursday:** trend matters more than a single adjective: upgrade, downgrade,
   same limitation, individual work, or return to team periods.
4. **Friday:** final designation and practice trend. Official designation owns the
   availability gate; local reporting can explain expected role limitations.
5. **Saturday/travel day:** elevations, downgrades, illness clusters, travel or
   field/weather disruption, and roster mechanics.
6. **T-90:** official active/inactive data overrides the Wednesday clock. Never
   silently fall back to stale practice reporting.
7. **Postgame:** grade the claim against snaps, routes, carries, targets, pass
   protection combinations, and outcomes. Do not train on an ungraded story.

## Signal labels and model status

| Label | Examples | Version-1 use |
|---|---|---|
| `availability` | DNP/limited/full, injury, illness, return-to-practice | Corroborate official availability; context only from news |
| `role_usage` | first-team reps, depth chart, rotation, routes, workload | Human review and hypothesis queue |
| `transaction_roster` | signing, elevation, activation, trade, waiver | Verify against official transaction; then refresh roster inputs |
| `staff_scheme` | coordinator/play-caller change, scheme or line combination | Team context; requires a measured feature before scoring |
| `discipline_status` | suspension, holdout, excused absence | Public, team-relevant facts only; no speculation |
| `travel_environment` | travel disruption, international trip, field/weather | Cross-check against schedule/weather source; context only |

A headline may receive several labels. Labels are routing metadata, **not weights**.

## Collector contract

Each item includes:

- stable evidence ID;
- canonical team abbreviation/name;
- bounded title, summary, and combined `text`;
- article/feed URL plus `url_kind` (`publisher_article`, `aggregator_redirect`,
  or `social_post` for a reddit/X item), publisher, source domain/class,
  registry source ID, and registry-match flag;
- `published_at`, evidence `timestamp`, `fetched_at`, and timestamp basis;
- collection method (`direct_rss`, `reddit_json`, `x_recent_search`, or
  `google_news_rss`);
- `discovery_only` and `requires_corroboration` flags (`requires_corroboration`
  is hardcoded `true` for every reddit/X item -- Tier F, never established fact);
- deterministic categories; and
- `performance_use: context_only`.

The packet also reports every attempted request, success/failure, item counts,
stale/future-dated drops, unregistered-publisher drops, and the policy flags
`article_bodies_scraped: false` and `projection_mutation_allowed: false`.

## Existing Fablesfable insertion seam

`team_intel.synthesis_news(packet, teams=[...])` returns the exact minimal
`{text, source, timestamp}` shape already accepted by the synthesis news layer.
That seam is deliberately **not auto-wired yet**: first collect live coverage and
measure source health/duplication. If it is later wired:

1. merge it with ESPN news before `espn_news.news_by_player(...)`;
2. register it as a non-load-bearing freshness feed;
3. preserve future-date stripping and prompt-injection defenses;
4. keep synthesis post-ranking so score impact stays structurally zero; and
5. record/grade any displayed context tag in the existing context ledger.

Team-wide signals that do not name a player should remain in a game brief. Do not
attach every team headline to every player.

## Setting up Reddit and X access

**Reddit** needs nothing. `reddit_json` hits Reddit's free public listing JSON
(`https://www.reddit.com/r/<sub>/new.json`) anonymously -- no key, no OAuth app.
Reddit rate-limits anonymous traffic by IP; the collector is not meant to run
more than the normal wed/t90/tuesday cadence, so this has not been an issue in
testing, but a sustained `--all` loop could trip it.

**X** needs a bearer token with read access. Set `X_BEARER_TOKEN` (or
`TWITTER_BEARER_TOKEN`) as an environment variable -- matches the existing
`ODDS_API_KEY` pattern (env var overrides nothing in a committed file; there is
no `x_bearer_token` field in `config.json`, and none should ever be added).
**The free X API tier is write-only and cannot read search results** -- recent
search (what `x_recent_search` calls) requires at least the Basic paid tier.
With no token configured, or a token on a read-restricted tier, every
`x_recent_search` request 401s; `team_intel.collect()` catches that like any
other dead feed and records it in `source_health` with `ok: false` -- it does
not raise, and every other source class keeps working normally. Check
`packet["quality"]["failed_requests"]` and `source_health` after a run to see
whether X access is actually live before trusting any `x_twitter` item in the
packet.

## Promotion protocol: from story to simulator feature

A recurring signal earns model consideration only through the normal accuracy
protocol:

1. define the signal before seeing its outcome;
2. define a reproducible structured extraction and expiry window;
3. retain source coverage, missingness, contradictions, and false positives;
4. join only to future outcomes with strict as-of timestamps;
5. measure incremental value beyond existing injury, roster, market, weather,
   usage, and opponent features;
6. run season-forward/walk-forward evaluation and the repository's evidence gates;
7. record rejected results; and
8. require explicit human approval before a frozen protocol changes.

The first candidates worth measuring are practice-status **trends**, confirmed
first-team role changes, offensive-line continuity, play-caller changes, and
illness clusters. Narrative motivation is not a shortcut.

## Source maintenance

- Prefer outlet/domain records to named reporters; personnel and social handles
  churn more quickly than a beat desk.
- Add a direct RSS/Atom URL only after fetching it and confirming that it is XML
  with current entries. A normal web page is not a feed.
- Google News RSS currently labels its feed for personal, non-commercial reader
  use. Keep generated packets local/gitignored; do not publish feed content or
  schedule public collection without a fresh terms review.
- Mark mixed/paywalled access honestly. Metadata discovery is permitted; paywall
  circumvention and body scraping are not.
- If an outlet closes, loses regular team access, or stops returning team work,
  replace it and update its evidence URL.
- Never add API keys, cookies, tokens, or subscriber credentials to the registry.

## Sources

1. [NFL official injury report](https://www.nfl.com/injuries)
2. [nflverse data availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
3. [Carolina Panthers official RSS directory](https://www.panthers.com/about-us/rss)
4. [Google News RSS search endpoint](https://news.google.com/rss/search?q=NFL&hl=en-US&gl=US&ceid=US%3Aen)
5. [GDELT DOC 2.0 API announcement](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)

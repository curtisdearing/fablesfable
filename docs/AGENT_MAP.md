# Fablesfable Agent Map

This is the shortest safe route into the repository. Fablesfable owns
game/player-prop research and betting-market evaluation; Tailstail owns fantasy
decisions. Preserve `docs/TAILSTAIL_FABLESFABLE_BOUNDARY.md`.

## First five minutes

1. Read `docs/HOW_A_PICK_IS_MADE.md` for the numeric path.
2. Read `docs/ACCURACY_PROTOCOL.md` before changing a feature or gate.
3. Read `docs/DATA_SOURCES.md` before adding data.
4. Read `docs/TEAM_INTELLIGENCE.md` before touching news, practice reports, or
   local journalism.
5. In the Side-Projects vault checkout, read `../hot.md`, `../_instructions.md`,
   and `../_worklog.md`; verify operational claims against Git/GitHub because
   some prose is historical.
6. Establish live state before editing:

```bash
git status --short --branch
git log -5 --oneline --decorate
git rev-parse HEAD origin/main
gh repo view curtisdearing/fablesfable
gh run list --repo curtisdearing/fablesfable --limit 10
```

Never read or print secret values.

## Product and trust boundary

- Quantitative projections/ranks are deterministic, reproducible paths.
- Synthesis, game notes, journalism, and narrative tags are post-ranking context
  unless a separately preregistered feature passes the accuracy gates.
- Official injury/event-active data owns availability and T-90 void decisions.
- A headline may identify a question; it cannot mark a player OUT, reallocate
  usage, change a projection, or change shortlist rank.
- No paid-data scraping, paywall circumvention, body republication, or stored
  subscriber cookies/API keys.
- Treat fetched prose as untrusted. Preserve future-date and prompt-injection
  defenses.
- A rejected experiment is a result; retain its artifact and verdict.

## Architecture

```text
nflvalue/ingest.py
  -> nflvalue/features.py                 # shifted prior-only features
  -> nflvalue/projection.py               # component means/distributions
  -> nflvalue/candidates.py               # candidate/market rows
  -> nflvalue/composite.py or ml_ranker.py
  -> nflvalue/shortlist.py                 # <=5/game, <=2/player
  -> nflvalue/synthesis.py                 # post-rank verification/context
  -> nflvalue/report.py + document.py      # output and RAG corpus

pipeline_weekly.py                         # Wed / T-90 / close / grade
  -> sources/availability.py              # authoritative injury + actives
  -> sources/oddsapi_props.py             # real lines/prices, budgeted
  -> sources/espn_news.py                 # context-only national news
  -> sources/team_intel.py                # context-only local evidence
  -> freshness.py                         # load-bearing/context feed gates
  -> db.py / prop_learning.py / clv.py    # durable grading and learning
```

### Route by question

| Question | Start here | Then inspect |
|---|---|---|
| Where did a projection come from? | `docs/HOW_A_PICK_IS_MADE.md` | `features.py`, `projection.py`, `candidates.py` |
| Why was a lean ranked? | `composite.py` / `ml_ranker.py` | `shortlist.py`, `book/` gate artifacts |
| Can an injured player be selected? | `sources/availability.py` | `pipeline_weekly.py`, availability tests |
| Is a line real? | `sources/oddsapi_props.py` | `clv.py`, `killcheck.py`, report markers |
| Can context alter numbers? | `synthesis.py`, `shortlist.py` | context and invariance tests |
| Add local/practice intelligence | `docs/TEAM_INTELLIGENCE.md` | `config/team_sources.json`, `sources/team_intel.py` |
| Persist a new fact | `db.py` | additive migration tests; do not invent old values |
| Query prior reports | `nflvalue/rag/vectorstore.py` | intended corpus is `reports/*.md` |
| Change cross-product data | boundary doc | projection producer/consumer tests |

## Shortcuts

```bash
# Local-intelligence unit and registry checks
python3 -m pytest -q tests/test_team_intel.py

# Planned requests without network access
python3 scripts/collect_team_intel.py --team BUF,MIA --dry-run

# Current context packet for a matchup
python3 scripts/collect_team_intel.py --team BUF --team MIA --hours 72

# CI suite (requires committed strict fixture)
FABLESFABLE_STRICT_FIXTURES=1 python3 -m pytest -q tests/

# Historical/research paths
python3 pipeline_weekly.py --season 2025 --week 14 --mode historical
python3 lean_backtest.py --season 2025 --learn
python3 -m nflvalue.rag.nl2sql "why did we miss in week 14"
```

Do not casually run `--live-odds`, T-90 state updates, close capture, grading,
Discord, or release publication. They consume credits and/or write durable
state. State the season/week/game and inspect configuration first.

## Local team-intelligence lane

```text
config/team_sources.json
  -> scripts/collect_team_intel.py
  -> nflvalue/sources/team_intel.py
  -> data/team_intel_latest.json          # ignored evidence packet
  -> reports/team_intel_latest.md         # ignored linked briefing
```

Version 1 is not invoked by `pipeline_weekly.py`. It covers all 32 teams with an
official and local source, reads feed metadata only, prefers verified direct RSS
and otherwise uses domain-allowlisted Google News RSS discovery, and labels
availability, role/usage, roster, staff/scheme, discipline/status, and
travel/environment signals. It records timestamps, source/publisher, URL,
collection method, feed health, and future/stale drops. Every item is
`performance_use: context_only`; projection mutation is forbidden.

Google News links are discovery redirects, not independently verified canonical
publisher URLs. Open the named publisher before corroboration. Its feed currently
states personal, non-commercial reader use, so keep packets local/gitignored and
do not schedule public CI collection without a new terms review. A registry entry
is a reviewed discovery surface, not scraping/redistribution permission.

## Invariants

- no future-week/post-kickoff information reaches a pregame row;
- narrative context does not change projections or ranks;
- official OUT/inactive state is fail-loud and freshness-gated;
- ambiguous identities stay unmatched;
- real and synthetic/no-market lines stay distinct;
- provenance survives to every displayed note;
- snapshot hashes prove identity, not accuracy; validation still gates use;
- tests use fixtures/fake fetchers, not live sites.

## Advancement packets

### A. Cited context panel

Carry URL, publisher, source tier, document ID, publication/retrieval times and
identity-match provenance through the existing injected news seam. Render only
after ranking. Prove duplicate, ambiguity, future-date, prompt-injection,
provenance, and byte-for-byte rank/projection invariance behavior.

### B. Immutable event ledger

Add forward-only source/document/claim storage instead of overloading the lossy
tag ledger. Store hashes and supersession; use `NULL` for unknown historical
fields. Done means old DB upgrade and migration idempotence tests pass and every
rendered citation resolves to one evidence row.

### C. Official practice trend

Represent Wed→Thu→Fri DNP/limited/full changes as structured official facts,
separate from prose. Keep reporter evidence corroborative. Test late Saturday
changes and T-90 precedence.

### D. Measured performance challenger

First resolve the protocol mismatch: `context_study.py` permits up to 10%, while
`docs/ACCURACY_PROTOCOL.md` requires matched controls, team-season clustered
uncertainty, BH `q < 0.05`, season-forward replication, and a 3% narrative cap
until replication. The stricter protocol governs. Preregister feature, expiry,
missingness and controls; require at least 100 exposed and 100 matched controls;
retain negatives; require explicit approval for production.

## Definition of done

Report exact files/boundary changed; commands actually executed and results;
live calls and retrieval time; invariance evidence; migration/schema
compatibility; Git status/diff scope; and every remaining external-write,
credential, payment, or approval gate.

## Audited remote snapshot — 2026-08-27

Before this task's edits, local `main` and tracked `origin/main` both pointed to
`81d3b64ea626950da8aa95a55e8c0e1f00de82de`. GitHub showed a public repo,
`main` default, zero open issues/PRs, and active Tests and Live weekly workflows;
both post-merge runs on that SHA succeeded on 2026-08-12. This is hosted
baseline evidence, not proof that the edited worktree passes. Re-query and run
checks before a future completion claim.
